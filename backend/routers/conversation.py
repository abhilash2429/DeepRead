from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.agents.conversation_agent import run_turn, stream_text_tokens
from backend.services.artifact_builder import build_artifacts

router = APIRouter(prefix="/conversation", tags=["conversation"])


class MessageRequest(BaseModel):
    message: str
    stage_override: str | None = None


class ResolveAmbiguityRequest(BaseModel):
    ambiguity_id: str
    resolution: str


@router.post("/{session_id}/message")
async def conversation_message(request: Request, session_id: str, payload: MessageRequest) -> EventSourceResponse:
    store = request.app.state.session_store
    gemini = request.app.state.gemini
    memory_manager = request.app.state.memory_manager
    state = await store.get(session_id, "conversation_state")
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    parsed_paper = await store.get(session_id, "parsed_paper")
    if parsed_paper is None:
        raise HTTPException(status_code=404, detail="Parsed paper not found for session")

    async def event_gen() -> AsyncIterator[dict]:
        try:
            yield {"event": "progress", "data": json.dumps({"message": "Analyzing request..."})}
            memory = memory_manager.create_or_load(session_id, gemini.llm)
            answer, snippets, clarifying = await run_turn(
                gemini=gemini,
                state=state,
                user_message=payload.message,
                stage_override=payload.stage_override,
                parsed_paper=parsed_paper,
                memory_manager=memory_manager,
                memory=memory,
            )
            await store.set(session_id, "conversation_state", state)

            existing_snippets = await store.get(session_id, "code_snippets", [])
            existing_snippets.extend(snippets)
            await store.set(session_id, "code_snippets", existing_snippets)

            yield {
                "event": "stage",
                "data": json.dumps({"current_stage": state.current_stage.value, "reason": "intent_router"}),
            }
            if clarifying:
                yield {"event": "clarifying", "data": json.dumps({"question": clarifying})}
            async for tok in stream_text_tokens(answer):
                yield {"event": "token", "data": json.dumps({"text": tok})}
            yield {"event": "done", "data": json.dumps({"message_id": len(state.message_history)})}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}
            yield {"event": "done", "data": json.dumps({"failed": True})}

    return EventSourceResponse(event_gen())


@router.get("/{session_id}/state")
async def conversation_state(request: Request, session_id: str) -> dict:
    store = request.app.state.session_store
    state = await store.get(session_id, "conversation_state")
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state.model_dump()


@router.post("/{session_id}/resolve-ambiguity")
async def resolve_ambiguity(
    request: Request, session_id: str, payload: ResolveAmbiguityRequest
) -> dict:
    store = request.app.state.session_store
    state = await store.get(session_id, "conversation_state")
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    target = None
    for ambiguity in state.internal_representation.ambiguity_log:
        if ambiguity.ambiguity_id == payload.ambiguity_id:
            ambiguity.resolved = True
            ambiguity.user_resolution = payload.resolution
            state.resolved_ambiguities[payload.ambiguity_id] = payload.resolution
            target = ambiguity
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Ambiguity not found")

    await store.set(session_id, "conversation_state", state)
    return {"updated": target.model_dump(), "current_stage": state.current_stage.value}


@router.get("/{session_id}/artifacts")
async def conversation_artifacts(request: Request, session_id: str) -> dict:
    store = request.app.state.session_store
    state = await store.get(session_id, "conversation_state")
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    snippets = await store.get(session_id, "code_snippets", [])
    manifest = build_artifacts(state, snippets)
    return manifest.model_dump()
