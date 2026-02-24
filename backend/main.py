from __future__ import annotations

import asyncio
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse


def _clean_env_value(raw: str) -> str:
    value = raw.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1].strip()
    elif " #" in value:
        # Support inline comments for unquoted values: KEY=value # comment
        value = value.split(" #", 1)[0].strip()
    return value


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    app_env_hint = os.getenv("APP_ENV", "development").strip().lower()
    allow_override = app_env_hint not in {"prod", "production"}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_env_value(value)
        if allow_override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


_load_local_env()

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
JWT_SECRET_VALUE = os.getenv("JWT_SECRET", "").strip()
SESSION_SECRET_VALUE = os.getenv("SESSION_SECRET", "").strip()


def _is_local_hostname(hostname: str | None) -> bool:
    if not hostname:
        return True
    return hostname.lower() in {"localhost", "127.0.0.1", "::1"}


def _validate_public_url(var_name: str) -> str:
    raw = os.getenv(var_name, "").strip()
    if not raw:
        raise RuntimeError(f"{var_name} must be set in production.")
    parsed = urlparse(raw)
    if parsed.scheme != "https":
        raise RuntimeError(f"{var_name} must use https in production.")
    if _is_local_hostname(parsed.hostname):
        raise RuntimeError(f"{var_name} must not use localhost in production.")
    return raw


def _normalize_samesite(raw: str) -> str:
    value = raw.strip().lower()
    if value not in {"lax", "strict", "none"}:
        return "lax"
    if value == "none" and APP_ENV not in {"prod", "production"}:
        # Browsers reject SameSite=None cookies unless Secure=true.
        return "lax"
    return value

if APP_ENV in {"prod", "production"}:
    if len(JWT_SECRET_VALUE) < 32:
        raise RuntimeError("JWT_SECRET must be set and at least 32 characters in production.")
    if len(SESSION_SECRET_VALUE) < 32:
        raise RuntimeError("SESSION_SECRET must be set and at least 32 characters in production.")
    _validate_public_url("NEXTAUTH_URL")
    _validate_public_url("GOOGLE_REDIRECT_URI")
    if os.getenv("NEXT_PUBLIC_API_BASE"):
        _validate_public_url("NEXT_PUBLIC_API_BASE")
elif not JWT_SECRET_VALUE:
    # Dev fallback is generated per-process to avoid shipping a static weak secret.
    JWT_SECRET_VALUE = secrets.token_urlsafe(32)
    os.environ["JWT_SECRET"] = JWT_SECRET_VALUE

if not SESSION_SECRET_VALUE:
    # Stable dev fallback so OAuth state cookies survive auto-reload cycles.
    SESSION_SECRET_VALUE = "deepread-dev-session-secret"
    os.environ["SESSION_SECRET"] = SESSION_SECRET_VALUE

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from backend.db.prisma import prisma_lifespan
from backend.memory.session_memory import SessionMemoryManager
from backend.routers.auth import router as auth_router
from backend.routers.conversation import router as conversation_router
from backend.routers.ingest import router as ingest_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ingest_queues = {}
    app.state.ingest_concurrency = max(1, int(os.getenv("INGEST_CONCURRENCY_LIMIT", "2")))
    app.state.ingest_queue_limit = max(1, int(os.getenv("INGEST_PENDING_LIMIT", "20")))
    app.state.ingest_semaphore = asyncio.Semaphore(app.state.ingest_concurrency)
    app.state.ingest_pending = 0
    app.state.ingest_pending_lock = asyncio.Lock()
    app.state.memory_manager = SessionMemoryManager()
    async with prisma_lifespan():
        yield


app = FastAPI(title="DeepRead API", version="0.2.0", lifespan=lifespan)

frontend_origin = os.getenv("NEXTAUTH_URL", "http://localhost:3000").strip()
extra_frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]
if APP_ENV in {"prod", "production"}:
    allow_origins = [frontend_origin, *extra_frontend_origins]
else:
    allow_origins = [
        frontend_origin,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        *extra_frontend_origins,
    ]

# Deduplicate while preserving order.
allow_origins = list(dict.fromkeys(allow_origins))
cookie_secure = APP_ENV in {"prod", "production"}
cookie_samesite = _normalize_samesite(os.getenv("COOKIE_SAMESITE", "lax"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_VALUE,
    same_site=cookie_samesite,
    https_only=cookie_secure,
)

app.include_router(auth_router)
app.include_router(ingest_router)
app.include_router(conversation_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
