from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


_load_local_env()

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
    app.state.memory_manager = SessionMemoryManager()
    async with prisma_lifespan():
        yield


app = FastAPI(title="PaperLens API", version="0.2.0", lifespan=lifespan)

frontend_origin = os.getenv("NEXTAUTH_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("JWT_SECRET", "dev-secret"),
    same_site="lax",
    https_only=False,
)

app.include_router(auth_router)
app.include_router(ingest_router)
app.include_router(conversation_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
