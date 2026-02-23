from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any

import fitz
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
    update_paper_metadata,
    update_paper_status,
)
from backend.routers.auth import get_current_user
from backend.services.arxiv_fetcher import fetch_arxiv_pdf_with_progress


router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = logging.getLogger(__name__)


class ArxivIngestRequest(BaseModel):
    arxiv_ref: str


def _friendly_pipeline_error(exc: Exception) -> str:
    raw = str(exc)
    lowered = raw.lower()
    if "api_key_invalid" in lowered or "api key not valid" in lowered:
        return "Invalid GEMINI_API_KEY. Set a valid key in .env and restart the backend."
    if "gemini_api_key" in lowered and ("not configured" in lowered or "missing" in lowered):
        return "Missing GEMINI_API_KEY. Set it in .env and restart the backend."
    if "invalid_argument" in lowered and "gemini" in lowered:
        return "Gemini request failed due to invalid configuration. Verify GEMINI_API_KEY and model access."
    return raw


async def _try_reserve_ingestion_slot(request: Request) -> bool:
    lock: asyncio.Lock = request.app.state.ingest_pending_lock
    async with lock:
        pending = int(request.app.state.ingest_pending)
        limit = int(request.app.state.ingest_queue_limit)
        if pending >= limit:
            return False
        request.app.state.ingest_pending = pending + 1
        return True


async def _release_ingestion_slot(request: Request) -> None:
    lock: asyncio.Lock = request.app.state.ingest_pending_lock
    async with lock:
        pending = int(request.app.state.ingest_pending)
        request.app.state.ingest_pending = max(0, pending - 1)


def _validate_pdf_payload(pdf_bytes: bytes) -> None:
    if not pdf_bytes:
        raise ValueError("Empty PDF payload.")
    if not pdf_bytes.lstrip().startswith(b"%PDF-"):
        raise ValueError("Uploaded file is not a valid PDF (missing PDF signature).")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = doc.page_count
        doc.close()
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Uploaded file is not a parseable PDF.") from exc
    if page_count <= 0:
        raise ValueError("Uploaded PDF has no pages.")


def _validate_gemini_key_preflight() -> None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError("Missing GEMINI_API_KEY. Set it in .env and restart the backend.")
    if "your_gemini_api_key" in key.lower():
        raise ValueError("Invalid GEMINI_API_KEY placeholder detected. Set a real key in .env and restart.")


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
        semaphore: asyncio.Semaphore = request.app.state.ingest_semaphore
        async with semaphore:
            _validate_gemini_key_preflight()
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
            _validate_pdf_payload(pdf_bytes)

            _queue_event(queue, "status", {"message": "Running ingestion pipeline...", "progress": 15})
            parsed_paper = await run_ingestion(
                pdf_bytes=pdf_bytes,
                title=title or "Untitled",
                authors=authors or [],
                abstract=abstract or "",
                emit_thinking=emit_thinking,
            )
            parsed_paper.pdf_bytes_b64 = base64.b64encode(pdf_bytes).decode("ascii")
            resolved_title = (parsed_paper.title or "").strip() or (title or "Untitled")
            resolved_authors = parsed_paper.authors or (authors or [])
            await update_paper_metadata(
                paper_id=paper_id,
                title=resolved_title,
                authors=resolved_authors,
            )
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
        logger.exception("Ingestion pipeline failed for paper_id=%s", paper_id)
        await update_paper_status(paper_id, "FAILED")
        _queue_event(queue, "error", {"message": _friendly_pipeline_error(exc)})
        _queue_event(queue, "done", {"paper_id": paper_id, "failed": True})
    finally:
        await _release_ingestion_slot(request)


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
    try:
        _validate_pdf_payload(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not await _try_reserve_ingestion_slot(request):
        raise HTTPException(
            status_code=429,
            detail="Ingestion queue is full. Please retry in a moment.",
        )
    try:
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
    except Exception:
        await _release_ingestion_slot(request)
        raise


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
    if not await _try_reserve_ingestion_slot(request):
        raise HTTPException(
            status_code=429,
            detail="Ingestion queue is full. Please retry in a moment.",
        )
    try:
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
    except Exception:
        await _release_ingestion_slot(request)
        raise


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
