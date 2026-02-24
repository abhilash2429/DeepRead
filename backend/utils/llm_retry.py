from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw.strip()))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(0.0, float(raw.strip()))
    except Exception:
        return default


def is_retryable_llm_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retryable_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "resource_exhausted",
        "quota",
        "rate limit",
        "temporarily unavailable",
        "unavailable",
        "deadline",
        "timeout",
        "connection reset",
        "service unavailable",
        "internal error",
    )
    return any(marker in message for marker in retryable_markers)


async def call_with_llm_retry(call: Callable[[], Awaitable[T]]) -> T:
    attempts = _env_int("LLM_RETRY_MAX_ATTEMPTS", 3)
    base_delay = _env_float("LLM_RETRY_BASE_DELAY_SECONDS", 0.8)
    max_delay = _env_float("LLM_RETRY_MAX_DELAY_SECONDS", 8.0)

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts or not is_retryable_llm_error(exc):
                raise
            # Exponential backoff with jitter to spread retries under burst traffic.
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 1.0 + random.uniform(-0.2, 0.2)
            await asyncio.sleep(max(0.0, delay))

    assert last_error is not None
    raise last_error
