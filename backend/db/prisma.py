from __future__ import annotations

import logging
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path


logger = logging.getLogger(__name__)
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "prisma" / "schema.prisma"


def _run_prisma_generate() -> None:
    if not SCHEMA_PATH.exists():
        raise RuntimeError(f"Prisma schema not found at: {SCHEMA_PATH}")
    subprocess.run(
        [sys.executable, "-m", "prisma", "generate", "--schema", str(SCHEMA_PATH)],
        check=True,
    )


def _load_prisma_class():
    try:
        from prisma import Prisma as PrismaClient
    except RuntimeError as exc:
        if "hasn't been generated yet" not in str(exc):
            raise
        logger.warning("Prisma client not generated. Running `prisma generate` automatically...")
        _run_prisma_generate()
        from prisma import Prisma as PrismaClient
    return PrismaClient


Prisma = _load_prisma_class()


prisma = Prisma()


@asynccontextmanager
async def prisma_lifespan():
    await prisma.connect()
    try:
        yield prisma
    finally:
        await prisma.disconnect()
