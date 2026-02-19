from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.memory.session_memory import SessionMemoryManager
from backend.routers.conversation import router as conversation_router
from backend.routers.ingest import router as ingest_router
from backend.services.gemini_client import GeminiClient
from backend.store.session_store import SessionStore, cleanup_loop


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session_store = SessionStore(ttl_seconds=7200)
    app.state.ingest_queues = {}
    app.state.gemini = GeminiClient(model_name="gemini-flash-latest")
    app.state.memory_manager = SessionMemoryManager(session_dir=os.getenv("SESSION_DIR", "sessions"))
    cleanup_task = asyncio.create_task(cleanup_loop(app.state.session_store))
    try:
        yield
    finally:
        cleanup_task.cancel()


app = FastAPI(title="DeepRead API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(conversation_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
