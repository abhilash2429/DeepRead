from __future__ import annotations

import asyncio
import os
import secrets
import time
from collections import deque
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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw.strip()))
    except Exception:
        return default


def _client_identifier(request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        # First hop is the originating client.
        return forwarded_for.split(",", 1)[0].strip() or "unknown"
    if request.client and request.client.host:
        return request.client.host
    return "unknown"

if APP_ENV in {"prod", "production"}:
    if len(JWT_SECRET_VALUE) < 32:
        raise RuntimeError("JWT_SECRET must be set and at least 32 characters in production.")
    if len(SESSION_SECRET_VALUE) < 32:
        raise RuntimeError("SESSION_SECRET must be set and at least 32 characters in production.")
    _validate_public_url("NEXTAUTH_URL")
    _validate_public_url("GOOGLE_REDIRECT_URI")
    github_client_id = os.getenv("GITHUB_CLIENT_ID", "").strip()
    github_client_secret = os.getenv("GITHUB_CLIENT_SECRET", "").strip()
    if github_client_id or github_client_secret:
        if not github_client_id or not github_client_secret:
            raise RuntimeError("Set both GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET in production.")
        _validate_public_url("GITHUB_REDIRECT_URI")
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
from fastapi.responses import JSONResponse
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
    app.state.ingest_rate_limit_per_minute = _env_int("INGEST_RATE_LIMIT_PER_MINUTE", 15)
    app.state.conversation_rate_limit_per_minute = _env_int("CONVERSATION_RATE_LIMIT_PER_MINUTE", 90)
    app.state.rate_limit_window_seconds = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
    app.state.rate_limit_lock = asyncio.Lock()
    app.state.rate_limit_buckets = {}
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
app.state.capacity_lock = _env_bool("CAPACITY_LOCK", False)

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


@app.middleware("http")
async def capacity_lock_guard(request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    if app.state.capacity_lock:
        path = request.url.path
        if path.startswith("/ingest") or path.startswith("/conversation"):
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Live analysis is temporarily paused due to API capacity. Please explore example walkthroughs.",
                },
            )
    return await call_next(request)


@app.middleware("http")
async def request_rate_limit_guard(request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.method != "POST":
        return await call_next(request)

    path = request.url.path
    scope = None
    if path.startswith("/ingest/"):
        scope = "ingest"
        limit = int(app.state.ingest_rate_limit_per_minute)
    elif path.startswith("/conversation/"):
        scope = "conversation"
        limit = int(app.state.conversation_rate_limit_per_minute)
    else:
        return await call_next(request)

    window = int(app.state.rate_limit_window_seconds)
    key = f"{scope}:{_client_identifier(request)}"
    now = time.monotonic()

    lock: asyncio.Lock = app.state.rate_limit_lock
    async with lock:
        buckets: dict[str, deque[float]] = app.state.rate_limit_buckets
        entries = buckets.get(key)
        if entries is None:
            entries = deque()
            buckets[key] = entries
        while entries and now - entries[0] >= window:
            entries.popleft()
        if len(entries) >= limit:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Too many {scope} requests from this client. "
                        f"Please retry in about {window} seconds."
                    ),
                },
            )
        entries.append(now)

        if len(buckets) > 5000:
            stale_keys = [bucket_key for bucket_key, values in buckets.items() if not values]
            for bucket_key in stale_keys[:500]:
                buckets.pop(bucket_key, None)

    return await call_next(request)

app.include_router(auth_router)
app.include_router(ingest_router)
app.include_router(conversation_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
