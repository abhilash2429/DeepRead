from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path


logger = logging.getLogger(__name__)
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "prisma" / "schema.prisma"


def _run_prisma_generate() -> None:
    if not SCHEMA_PATH.exists():
        raise RuntimeError(f"Prisma schema not found at: {SCHEMA_PATH}")

    cache_dir = Path(__file__).resolve().parents[2] / ".prisma-python-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PRISMA_BINARY_CACHE_DIR", str(cache_dir))

    completed = subprocess.run(
        [sys.executable, "-m", "prisma", "generate", "--schema", str(SCHEMA_PATH)],
        check=False,
        text=True,
        capture_output=True,
        cwd=str(SCHEMA_PATH.parent),
        env=env,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = stderr or stdout or "unknown error"
        raise RuntimeError(f"`prisma generate` failed: {details}")


def _load_prisma_class():
    logger.info("Ensuring Prisma client is generated...")
    _run_prisma_generate()

    # Import from generated package output (prisma/schema.prisma -> ../prisma_client).
    sys.modules.pop("prisma_client", None)
    prisma_client_module = importlib.import_module("prisma_client")
    prisma_class = getattr(prisma_client_module, "Prisma", None)
    if prisma_class is None:
        raise RuntimeError("Generated prisma_client package does not expose Prisma.")
    return prisma_class


Prisma = _load_prisma_class()


prisma = Prisma()


@asynccontextmanager
async def prisma_lifespan():
    await prisma.connect()
    try:
        yield prisma
    finally:
        await prisma.disconnect()
