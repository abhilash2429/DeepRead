from __future__ import annotations

from contextlib import asynccontextmanager

from prisma import Prisma


prisma = Prisma()


@asynccontextmanager
async def prisma_lifespan():
    await prisma.connect()
    try:
        yield prisma
    finally:
        await prisma.disconnect()

