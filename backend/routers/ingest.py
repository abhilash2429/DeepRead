from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.agents.briefing_agent import run_briefing_pipeline
from backend.agents.comprehension_agent import run_comprehension
from backend.agents.ingestion_agent import run_ingestion
from backend.db.queries import (
    check_user_limit,
    create_paper,
    get_paper_by_id,
    get_user_limit_details,
    increment_paper_count,
    save_internal_rep,
    save_parsed_paper,
    update_paper_status,
)
from backend.routers.auth import get_current_user
from backend.services.arxiv_fetcher import fetch_arxiv_pdf_with_progress


router = APIRouter(prefix="/ingest", tags=["ingest"])


class ArxivIngestRequest(BaseModel):
    arxiv_ref: str


def _queue_event(queue: asyncio.Queue[dict[str, Any]], event: str, data: dict[str, Any]) -> None:
    queue.put_nowait({"event": event, "data": json.dumps(data)})


async def _run_pipeline(
    request: Request,
    paper_id: str,
    user_id: str,
    pdf_bytes: bytes | None = None,
    title: str | None = None,
    authors: list[str] | None = None,
    abstract: str | None = None,
    arxiv_ref: str | None = None,
) -> None:
    queue = request.app.state.ingest_queues[paper_id]

    async def emit_event(event: str, data: dict[str, Any]) -> None:
        _queue_event(queue, event, data)

    def emit_thinking(message: str) -> None:
        _queue_event(queue, "thinking", {"message": message})

    try:
        if arxiv_ref:
            _queue_event(queue, "status", {"message": "Fetching paper metadata from arXiv...", "progress": 5})
            payload = await fetch_arxiv_pdf_with_progress(
                arxiv_ref,
                max_size_mb=int(os.getenv("MAX_PAPER_SIZE_MB", "20")),
                progress_cb=lambda msg: _queue_event(queue, "thinking", {"message": msg}),
            )
            pdf_bytes = payload.pdf_bytes
            title = payload.title
            authors = payload.authors
            abstract = payload.abstract

        if pdf_bytes is None:
            raise ValueError("PDF payload is required")

        _queue_event(queue, "status", {"message": "Running ingestion pipeline...", "progress": 15})
        parsed_paper = await run_ingestion(
            pdf_bytes=pdf_bytes,
            title=title or "Untitled",
            authors=authors or [],
            abstract=abstract or "",
            emit_thinking=emit_thinking,
        )
        parsed_paper.pdf_bytes_b64 = base64.b64encode(pdf_bytes).decode("ascii")
        await save_parsed_paper(paper_id, parsed_paper)

        _queue_event(queue, "status", {"message": "Building internal representation...", "progress": 35})
        internal_rep = await run_comprehension(parsed_paper)
        await save_internal_rep(paper_id, internal_rep)

        _queue_event(queue, "status", {"message": "Generating six-section briefing...", "progress": 45})
        await run_briefing_pipeline(
            session_id=paper_id,
            paper_id=paper_id,
            parsed_paper=parsed_paper,
            internal_rep=internal_rep,
            emit_event=emit_event,
        )
        await increment_paper_count(user_id)
        _queue_event(queue, "done", {"paper_id": paper_id, "progress": 100})
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        await update_paper_status(paper_id, "FAILED")
        _queue_event(queue, "error", {"message": str(exc)})
        _queue_event(queue, "done", {"paper_id": paper_id, "failed": True})


@router.post("/upload")
async def ingest_upload(
    request: Request,
    pdf: UploadFile = File(...),
    current_user: Any = Depends(get_current_user),
) -> dict[str, str]:
    if not await check_user_limit(current_user.id):
        details = await get_user_limit_details(current_user.id)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Plan limit reached",
                "message": "Free plan allows 3 lifetime paper analyses. Upgrade to continue.",
                **details,
            },
        )

    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    content = await pdf.read()
    max_size_mb = int(os.getenv("MAX_PAPER_SIZE_MB", "20"))
    if len(content) > max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"PDF exceeds MAX_PAPER_SIZE_MB={max_size_mb}")

    paper = await create_paper(
        user_id=current_user.id,
        title=pdf.filename,
        authors=[],
        arxiv_id=None,
    )
    request.app.state.ingest_queues[paper.id] = asyncio.Queue()
    asyncio.create_task(
        _run_pipeline(
            request=request,
            paper_id=paper.id,
            user_id=current_user.id,
            pdf_bytes=content,
            title=pdf.filename,
            authors=[],
            abstract="",
        )
    )
    return {"paper_id": paper.id, "status_stream_url": f"/ingest/{paper.id}/events"}


@router.post("/arxiv")
async def ingest_arxiv(
    request: Request,
    payload: ArxivIngestRequest,
    current_user: Any = Depends(get_current_user),
) -> dict[str, str]:
    if not await check_user_limit(current_user.id):
        details = await get_user_limit_details(current_user.id)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Plan limit reached",
                "message": "Free plan allows 3 lifetime paper analyses. Upgrade to continue.",
                **details,
            },
        )

    paper = await create_paper(
        user_id=current_user.id,
        title=f"arXiv: {payload.arxiv_ref}",
        authors=[],
        arxiv_id=payload.arxiv_ref,
    )
    request.app.state.ingest_queues[paper.id] = asyncio.Queue()
    asyncio.create_task(
        _run_pipeline(
            request=request,
            paper_id=paper.id,
            user_id=current_user.id,
            arxiv_ref=payload.arxiv_ref,
        )
    )
    return {"paper_id": paper.id, "status_stream_url": f"/ingest/{paper.id}/events"}


@router.get("/{paper_id}/events")
async def ingest_events(
    request: Request,
    paper_id: str,
    current_user: Any = Depends(get_current_user),
) -> EventSourceResponse:
    paper = await get_paper_by_id(paper_id)
    if not paper or paper.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Paper not found")

    queue = request.app.state.ingest_queues.get(paper_id)
    if queue is None:
        async def already_done():
            yield {"event": "done", "data": json.dumps({"paper_id": paper_id})}

        return EventSourceResponse(already_done())

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
            request.app.state.ingest_queues.pop(paper_id, None)

    return EventSourceResponse(event_gen())


@router.get("/{paper_id}/pdf")
async def paper_pdf(
    paper_id: str,
    current_user: Any = Depends(get_current_user),
) -> Response:
    paper = await get_paper_by_id(paper_id)
    if not paper or paper.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Paper not found")

    parsed = paper.parsed_paper or {}
    pdf_b64 = parsed.get("pdf_bytes_b64") if isinstance(parsed, dict) else None
    if not pdf_b64:
        raise HTTPException(status_code=404, detail="PDF bytes unavailable")
    try:
        pdf_bytes = base64.b64decode(pdf_b64.encode("ascii"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Corrupt PDF payload in storage") from exc

    return Response(content=pdf_bytes, media_type="application/pdf")

