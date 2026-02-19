from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.agents.graph import run_graph_turn
from backend.memory.session_memory import SessionMemoryManager
from backend.models.artifacts import CodeSnippet
from backend.models.conversation import ChatMessage, ConversationState, Stage
from backend.models.paper import ParsedPaper
from backend.services.gemini_client import GeminiClient


def _state_to_messages(state: ConversationState) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for msg in state.message_history:
        if msg.role == "user":
            converted.append(HumanMessage(content=msg.content))
        else:
            converted.append(AIMessage(content=msg.content))
    return converted


def _messages_to_state_history(messages: list[BaseMessage]) -> list[ChatMessage]:
    history: list[ChatMessage] = []
    for msg in messages:
        role = "assistant"
        if isinstance(msg, HumanMessage):
            role = "user"
        history.append(ChatMessage(role=role, content=str(msg.content)))
    return history


async def run_turn(
    gemini: GeminiClient,
    state: ConversationState,
    user_message: str,
    stage_override: str | None = None,
    parsed_paper: ParsedPaper | None = None,
    memory_manager: SessionMemoryManager | None = None,
    memory: Any | None = None,
) -> tuple[str, list[CodeSnippet], str | None]:
    if parsed_paper is None:
        raise ValueError("parsed_paper is required for conversation turns")

    if memory_manager and memory:
        chat_history = memory_manager.extract_messages(memory)
        if not chat_history:
            chat_history = _state_to_messages(state)
    else:
        chat_history = _state_to_messages(state)

    answer, snippets, clarifying, next_stage, updated_messages = await run_graph_turn(
        gemini=gemini,
        conversation_state=state,
        parsed_paper=parsed_paper,
        user_message=user_message,
        stage_override=stage_override,
        chat_history=chat_history,
    )

    try:
        state.current_stage = Stage(next_stage)
    except ValueError:
        pass

    if updated_messages:
        state.message_history = _messages_to_state_history(updated_messages)
    else:
        state.message_history.append(ChatMessage(role="user", content=user_message))
        state.message_history.append(ChatMessage(role="assistant", content=answer))

    if memory_manager and memory:
        memory.chat_memory.messages = updated_messages
        memory_manager.save(state.session_id, memory)

    state.pending_question = clarifying
    if any(token in user_message.lower() for token in ["gradient", "optimizer", "ablation", "derivation"]):
        state.user_level = "practitioner"
    else:
        state.user_level = state.user_level or "student"

    return answer, snippets, clarifying


async def stream_text_tokens(text: str) -> AsyncIterator[str]:
    for token in re.split(r"(\s+)", text):
        if token:
            yield token
