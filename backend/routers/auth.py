from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from authlib.integrations.base_client.errors import MismatchingStateError
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt

from backend.db.queries import FREE_PLAN_LIMIT, get_or_create_user, get_user_by_id


router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_NAME = "deepread_token"
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
COOKIE_SECURE = APP_ENV in {"prod", "production"}
COOKIE_SAMESITE_RAW = os.getenv("COOKIE_SAMESITE", "lax").strip().lower()
if COOKIE_SAMESITE_RAW not in {"lax", "strict", "none"}:
    COOKIE_SAMESITE = "lax"
elif COOKIE_SAMESITE_RAW == "none" and not COOKIE_SECURE:
    COOKIE_SAMESITE = "lax"
else:
    COOKIE_SAMESITE = COOKIE_SAMESITE_RAW

oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


def _get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "").strip()
    if APP_ENV in {"prod", "production"} and len(secret) < 32:
        raise RuntimeError("JWT_SECRET must be set and at least 32 characters in production.")
    if not secret:
        raise RuntimeError("JWT_SECRET is not configured")
    return secret


def _encode_jwt(payload: dict[str, Any]) -> str:
    jwt_secret = _get_jwt_secret()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    enriched = {**payload, "exp": expires_at}
    return jwt.encode(enriched, jwt_secret, algorithm=JWT_ALGORITHM)


def _decode_jwt(token: str) -> dict[str, Any]:
    jwt_secret = _get_jwt_secret()
    return jwt.decode(token, jwt_secret, algorithms=[JWT_ALGORITHM])


async def get_current_user(request: Request) -> Any:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        payload = _decode_jwt(token)
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid authentication token") from exc

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@router.get("/google")
async def auth_google(request: Request):
    # Remove stale pending OAuth state to avoid collisions on repeated sign-in attempts.
    for key in list(request.session.keys()):
        if key.startswith("_state_google_"):
            request.session.pop(key, None)
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI") or str(request.url_for("auth_google_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback")
async def auth_google_callback(request: Request):
    frontend_base = os.getenv("NEXTAUTH_URL", "http://localhost:3000")
    try:
        token = await oauth.google.authorize_access_token(request)
    except MismatchingStateError:
        # Clear invalid state and force a clean OAuth restart from frontend.
        for key in list(request.session.keys()):
            if key.startswith("_state_google_"):
                request.session.pop(key, None)
        return RedirectResponse(url=f"{frontend_base}/?auth_error=state_mismatch", status_code=307)
    user_info = token.get("userinfo", {})
    if not user_info:
        response = await oauth.google.get("https://openidconnect.googleapis.com/v1/userinfo", token=token)
        user_info = response.json()

    google_sub = str(user_info.get("sub", "")).strip()
    email = str(user_info.get("email", "")).strip()
    name = str(user_info.get("name", "")).strip() or email
    avatar_url = user_info.get("picture")

    if not google_sub or not email:
        raise HTTPException(status_code=400, detail="Failed to retrieve Google profile")

    user = await get_or_create_user(
        google_sub=google_sub,
        email=email,
        name=name,
        avatar_url=avatar_url,
    )
    jwt_token = _encode_jwt({"user_id": user.id, "email": user.email, "plan": user.plan})

    response = RedirectResponse(url=f"{frontend_base}/upload")
    response.set_cookie(
        key=COOKIE_NAME,
        value=jwt_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=JWT_EXPIRE_MINUTES * 60,
    )
    return response


@router.post("/logout")
async def auth_logout() -> Response:
    response = Response(content='{"ok":true}', media_type="application/json")
    response.delete_cookie(COOKIE_NAME)
    return response


@router.get("/me")
async def auth_me(current_user: Any = Depends(get_current_user)) -> dict[str, Any]:
    limit: int | str = "unlimited"
    if current_user.plan == "FREE":
        limit = FREE_PLAN_LIMIT
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "avatar_url": current_user.avatar_url,
        "plan": current_user.plan,
        "papers_analyzed": int(current_user.papers_analyzed),
        "limit": limit,
    }
