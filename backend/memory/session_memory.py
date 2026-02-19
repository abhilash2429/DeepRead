from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

try:
    from langchain.memory import ConversationSummaryBufferMemory
except Exception:  # pragma: no cover
    try:
        from langchain_classic.memory import ConversationSummaryBufferMemory  # type: ignore[no-redef]
    except Exception:
        ConversationSummaryBufferMemory = None  # type: ignore[assignment]


class SessionMemoryManager:
    def __init__(self, session_dir: str | None = None) -> None:
        self.session_dir = Path(session_dir or os.getenv("SESSION_DIR", "sessions"))
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        path = self.session_dir / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _memory_path(self, session_id: str) -> Path:
        return self._session_path(session_id) / "memory.json"

    def create_or_load(self, session_id: str, llm: Any):
        if ConversationSummaryBufferMemory is None:  # pragma: no cover
            raise RuntimeError("ConversationSummaryBufferMemory is unavailable in this LangChain version")

        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=4000,
            return_messages=True,
            memory_key="chat_history",
        )
        mem_path = self._memory_path(session_id)
        if mem_path.exists():
            try:
                payload = json.loads(mem_path.read_text(encoding="utf-8"))
                messages = messages_from_dict(payload.get("messages", []))
                memory.chat_memory.messages = messages
            except Exception:
                memory.chat_memory.messages = []
        return memory

    def save(self, session_id: str, memory: Any) -> None:
        messages = getattr(memory.chat_memory, "messages", [])
        payload = {"messages": messages_to_dict(messages)}
        self._memory_path(session_id).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def extract_messages(memory: Any) -> list[BaseMessage]:
        messages = getattr(memory.chat_memory, "messages", [])
        return list(messages)
