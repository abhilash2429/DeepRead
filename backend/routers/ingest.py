from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from langchain_core.messages import AIMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.agents.comprehension_agent import run_comprehension
from backend.agents.ingestion_agent import run_ingestion
from backend.models.conversation import ChatMessage, ConversationState, Stage
from backend.services.arxiv_fetcher import fetch_arxiv_pdf_with_progress

router = APIRouter(prefix="/ingest", tags=["ingest"])


class ArxivIngestRequest(BaseModel):
    arxiv_ref: str


def _queue_event(queue: asyncio.Queue[dict[str, Any]], event: str, data: dict[str, Any]) -> None:
    queue.put_nowait({"event": event, "data": json.dumps(data)})


async def _run_pipeline(
    request: Request,
    session_id: str,
    pdf_bytes: bytes,
    title: str,
    authors: list[str],
    abstract: str,
) -> None:
    queue = request.app.state.ingest_queues[session_id]
    store = request.app.state.session_store
    gemini = request.app.state.gemini
    memory_manager = request.app.state.memory_manager

    def emit_status(step: str, message: str, progress: int) -> None:
        _queue_event(queue, "status", {"step": step, "message": message, "progress": progress})

    try:
        emit_status("start", "Ingestion started", 5)
        await store.set(session_id, "pdf_bytes", pdf_bytes)
        parsed = await run_ingestion(gemini, pdf_bytes, title, authors, abstract, emit_status)
        await store.set(session_id, "parsed_paper", parsed)

        emit_status("comprehension", "Running comprehension pass", 80)
        internal = await run_comprehension(gemini, parsed)
        await store.set(session_id, "internal_representation", internal)
        state = ConversationState(session_id=session_id, current_stage=Stage.ORIENTATION, internal_representation=internal)
        orientation = (
            "Orientation\n\n"
            f"Problem: {internal.problem_statement}\n\n"
            f"Method: {internal.method_summary}\n\n"
            f"Novelty: {internal.novelty}"
        )
        state.message_history.append(ChatMessage(role="assistant", content=orientation))
        await store.set(session_id, "conversation_state", state)
        await store.set(session_id, "code_snippets", [])

        memory = memory_manager.create_or_load(session_id, gemini.llm)
        memory.chat_memory.messages = [AIMessage(content=orientation)]
        memory_manager.save(session_id, memory)

        emit_status("done", "Session ready", 100)
        _queue_event(queue, "done", {"session_id": session_id})
    except Exception as exc:  # noqa: BLE001
        _queue_event(queue, "error", {"message": str(exc)})
        _queue_event(queue, "done", {"session_id": session_id, "failed": True})


@router.post("/upload")
async def ingest_upload(request: Request, pdf: UploadFile = File(...)) -> dict[str, str]:
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    content = await pdf.read()
    max_size_mb = int(os.getenv("MAX_PAPER_SIZE_MB", "20"))
    if len(content) > max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"PDF exceeds MAX_PAPER_SIZE_MB={max_size_mb}")

    session_id = await request.app.state.session_store.create()
    request.app.state.ingest_queues[session_id] = asyncio.Queue()
    asyncio.create_task(_run_pipeline(request, session_id, content, pdf.filename, [], ""))
    return {"session_id": session_id, "status_stream_url": f"/ingest/{session_id}/events"}


@router.post("/arxiv")
async def ingest_arxiv(request: Request, payload: ArxivIngestRequest) -> dict[str, str]:
    session_id = await request.app.state.session_store.create()
    request.app.state.ingest_queues[session_id] = asyncio.Queue()

    async def arxiv_pipeline() -> None:
        queue = request.app.state.ingest_queues[session_id]
        _queue_event(queue, "status", {"step": "download", "message": "Fetching from arXiv", "progress": 10})
        try:
            max_size_mb = int(os.getenv("MAX_PAPER_SIZE_MB", "20"))
            paper = await fetch_arxiv_pdf_with_progress(
                payload.arxiv_ref,
                max_size_mb=max_size_mb,
                progress_cb=lambda msg: _queue_event(
                    queue, "status", {"step": "download", "message": msg, "progress": 15}
                ),
            )
            _queue_event(
                queue,
                "status",
                {"step": "download", "message": "arXiv fetch complete. Starting ingestion", "progress": 18},
            )
            await _run_pipeline(
                request=request,
                session_id=session_id,
                pdf_bytes=paper.pdf_bytes,
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
            )
        except Exception as exc:  # noqa: BLE001
            _queue_event(queue, "error", {"message": str(exc)})
            _queue_event(queue, "done", {"session_id": session_id, "failed": True})

    asyncio.create_task(arxiv_pipeline())
    return {"session_id": session_id, "status_stream_url": f"/ingest/{session_id}/events"}


@router.get("/{session_id}/events")
async def ingest_events(request: Request, session_id: str) -> EventSourceResponse:
    queue = request.app.state.ingest_queues.get(session_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                event = await queue.get()
                yield event
                if event.get("event") == "done":
                    break
        finally:
            request.app.state.ingest_queues.pop(session_id, None)

    return EventSourceResponse(event_gen())


@router.get("/{session_id}/pdf")
async def session_pdf(request: Request, session_id: str) -> Response:
    store = request.app.state.session_store
    pdf_bytes = await store.get(session_id, "pdf_bytes")
    if pdf_bytes is None:
        raise HTTPException(status_code=404, detail="PDF not found for session")
    return Response(content=pdf_bytes, media_type="application/pdf")
