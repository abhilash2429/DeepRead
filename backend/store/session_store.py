from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.models.artifacts import CodeSnippet
from backend.models.conversation import ConversationState, InternalRepresentation
from backend.models.paper import ParsedPaper


@dataclass
class SessionRecord:
    data: dict[str, Any] = field(default_factory=dict)
    last_accessed: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_seconds: int = 7200, session_dir: str | None = None) -> None:
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()
        self._session_root = Path(session_dir or os.getenv("SESSION_DIR", "sessions"))
        self._session_root.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        path = self._session_root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_to_disk(self, session_id: str, key: str, value: Any) -> None:
        session_path = self._session_path(session_id)
        if isinstance(value, bytes):
            (session_path / f"{key}.bin").write_bytes(value)
            return

        payload: dict[str, Any] = {"type": "raw", "value": value}
        if isinstance(value, ParsedPaper):
            payload = {"type": "ParsedPaper", "value": value.model_dump(mode="json")}
        elif isinstance(value, ConversationState):
            payload = {"type": "ConversationState", "value": value.model_dump(mode="json")}
        elif isinstance(value, InternalRepresentation):
            payload = {"type": "InternalRepresentation", "value": value.model_dump(mode="json")}
        elif isinstance(value, list) and value and isinstance(value[0], CodeSnippet):
            payload = {"type": "CodeSnippetList", "value": [v.model_dump(mode="json") for v in value]}
        (session_path / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _load_from_disk(self, session_id: str, key: str, default: Any = None) -> Any:
        session_path = self._session_root / session_id
        bin_path = session_path / f"{key}.bin"
        json_path = session_path / f"{key}.json"
        if bin_path.exists():
            return bin_path.read_bytes()
        if not json_path.exists():
            return default
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return default
        kind = payload.get("type")
        value = payload.get("value")
        try:
            if kind == "ParsedPaper":
                return ParsedPaper.model_validate(value)
            if kind == "ConversationState":
                return ConversationState.model_validate(value)
            if kind == "InternalRepresentation":
                return InternalRepresentation.model_validate(value)
            if kind == "CodeSnippetList":
                return [CodeSnippet.model_validate(v) for v in value or []]
            return value
        except Exception:
            return default

    async def create(self) -> str:
        async with self._lock:
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = SessionRecord()
            self._session_path(session_id)
            return session_id

    async def set(self, session_id: str, key: str, value: Any) -> None:
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                record = SessionRecord()
                self._sessions[session_id] = record
            record.data[key] = value
            record.last_accessed = time.time()
            self._save_to_disk(session_id, key, value)

    async def get(self, session_id: str, key: str, default: Any = None) -> Any:
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                value = self._load_from_disk(session_id, key, default=default)
                if value is default:
                    return default
                self._sessions[session_id] = SessionRecord(data={key: value})
                return value
            record.last_accessed = time.time()
            if key in record.data:
                return record.data[key]
            value = self._load_from_disk(session_id, key, default=default)
            if value is not default:
                record.data[key] = value
            return value

    async def get_all(self, session_id: str) -> dict[str, Any] | None:
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            record.last_accessed = time.time()
            return dict(record.data)

    async def cleanup_expired(self) -> None:
        cutoff = time.time() - self._ttl_seconds
        async with self._lock:
            to_delete = [sid for sid, rec in self._sessions.items() if rec.last_accessed < cutoff]
            for sid in to_delete:
                del self._sessions[sid]
                session_path = self._session_root / sid
                if session_path.exists():
                    shutil.rmtree(session_path, ignore_errors=True)


async def cleanup_loop(store: SessionStore, interval_seconds: int = 300) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        await store.cleanup_expired()
