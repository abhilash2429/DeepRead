from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.db.queries import append_qa_message, get_qa_history

try:
    from langchain.memory import ConversationSummaryBufferMemory
except Exception:  # pragma: no cover
    from langchain_classic.memory import ConversationSummaryBufferMemory  # type: ignore[no-redef]


MEMORY_MODEL = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)


class SessionMemoryManager:
    def __init__(self, max_token_limit: int = 4000) -> None:
        self.max_token_limit = max_token_limit

    async def load_memory(self, paper_id: str) -> Any:
        memory = ConversationSummaryBufferMemory(
            llm=MEMORY_MODEL,
            max_token_limit=self.max_token_limit,
            return_messages=True,
            memory_key="chat_history",
        )
        records = await get_qa_history(paper_id)
        messages: list[BaseMessage] = []
        for row in records:
            if row.role == "user":
                messages.append(HumanMessage(content=row.content))
            else:
                messages.append(AIMessage(content=row.content))
        memory.chat_memory.messages = messages
        return memory

    async def save_turn(self, paper_id: str, user_message: str, assistant_message: str) -> None:
        await append_qa_message(paper_id, "user", user_message)
        await append_qa_message(paper_id, "assistant", assistant_message)

    @staticmethod
    def extract_messages(memory: Any) -> list[BaseMessage]:
        return list(getattr(memory.chat_memory, "messages", []))
