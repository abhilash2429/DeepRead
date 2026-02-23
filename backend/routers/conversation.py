from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.agents.qa_agent import run_qa_turn
from backend.db.queries import get_paper_with_briefing, update_briefing_ambiguities
from backend.memory.session_memory import SessionMemoryManager
from backend.models.briefing import InternalRepresentation
from backend.models.paper import ParsedPaper
from backend.routers.auth import get_current_user


router = APIRouter(prefix="/conversation", tags=["conversation"])


class MessageRequest(BaseModel):
    message: str


class ResolveAmbiguityRequest(BaseModel):
    ambiguity_id: str
    resolution: str


def _briefing_markdown(briefing: Any) -> str:
    if not briefing:
        return ""
    sections = [
        ("1. What This Paper Actually Does", briefing.section_1),
        ("2. The Mechanism", briefing.section_2),
        ("3. What You Need To Already Know", briefing.section_3),
        ("4. The Full Implementation Map", briefing.section_4),
        ("5. What The Paper Left Out", briefing.section_5),
        ("6. How To Train It", briefing.section_6),
    ]
    lines: list[str] = []
    for title, content in sections:
        if content:
            lines.append(f"## {title}\n{content}")
    return "\n\n".join(lines)


def _split_tokens(text: str) -> list[str]:
    return [token for token in re.split(r"(\s+)", text) if token]


def _status_payload(paper: Any) -> dict[str, Any]:
    briefing = paper.briefing
    internal_rep = paper.internal_rep or {}
    prerequisites = []
    if isinstance(internal_rep, dict):
        prerequisites = list(internal_rep.get("prerequisite_concepts", []) or [])
    sections = {
        "section_1": bool(briefing and briefing.section_1),
        "section_2": bool(briefing and briefing.section_2),
        "section_3": bool(briefing and briefing.section_3),
        "section_4": bool(briefing and briefing.section_4),
        "section_5": bool(briefing and briefing.section_5),
        "section_6": bool(briefing and briefing.section_6),
    }
    ambiguities = list(briefing.ambiguities or []) if briefing else []
    resolved = {
        item.get("ambiguity_id"): item.get("user_resolution")
        for item in ambiguities
        if item.get("resolved") and item.get("ambiguity_id")
    }
    return {
        "paper_id": paper.id,
        "status": paper.status,
        "sections": sections,
        "hyperparameters": list((briefing.hyperparameters if briefing else []) or []),
        "ambiguities": ambiguities,
        "prerequisites": prerequisites,
        "resolved_ambiguities": resolved,
    }


@router.post("/{paper_id}/message")
async def conversation_message(
    request: Request,
    paper_id: str,
    payload: MessageRequest,
    current_user: Any = Depends(get_current_user),
) -> EventSourceResponse:
    paper = await get_paper_with_briefing(paper_id, current_user.id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    parsed_paper = ParsedPaper.model_validate(paper.parsed_paper or {})
    internal_rep = InternalRepresentation.model_validate(paper.internal_rep or {})
    briefing_md = _briefing_markdown(paper.briefing)
    resolved = _status_payload(paper).get("resolved_ambiguities", {})

    async def event_gen() -> AsyncIterator[dict[str, Any]]:
        memory_manager: SessionMemoryManager = request.app.state.memory_manager
        try:
            yield {"event": "progress", "data": json.dumps({"message": "Loading Q&A context..."})}
            memory = await memory_manager.load_memory(paper_id)
            history = memory_manager.extract_messages(memory)
            answer, updated_history = await run_qa_turn(
                user_message=payload.message,
                parsed_paper=parsed_paper,
                internal_rep=internal_rep,
                briefing_markdown=briefing_md,
                chat_history=history,
                resolved_ambiguities=resolved,
            )
            memory.chat_memory.messages = updated_history
            await memory_manager.save_turn(paper_id, payload.message, answer)

            for token in _split_tokens(answer):
                yield {"event": "token", "data": json.dumps({"text": token})}
            yield {"event": "done", "data": json.dumps({"paper_id": paper_id})}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}
            yield {"event": "done", "data": json.dumps({"paper_id": paper_id, "failed": True})}

    return EventSourceResponse(event_gen())


@router.get("/{paper_id}/state")
async def conversation_state(
    paper_id: str,
    current_user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    paper = await get_paper_with_briefing(paper_id, current_user.id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    payload = _status_payload(paper)
    payload["briefing"] = {
        "section_1": paper.briefing.section_1 if paper.briefing else None,
        "section_2": paper.briefing.section_2 if paper.briefing else None,
        "section_3": paper.briefing.section_3 if paper.briefing else None,
        "section_4": paper.briefing.section_4 if paper.briefing else None,
        "section_5": paper.briefing.section_5 if paper.briefing else None,
        "section_6": paper.briefing.section_6 if paper.briefing else None,
    }
    return payload


@router.get("/{paper_id}/artifacts")
async def conversation_artifacts(
    paper_id: str,
    current_user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    paper = await get_paper_with_briefing(paper_id, current_user.id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    briefing = paper.briefing
    if not briefing:
        return {"items": []}

    section_md = _briefing_markdown(briefing)

    snippets = list((briefing.code_snippets or []))
    merged_code = []
    for snippet in snippets:
        name = snippet.get("component_name", "component")
        merged_code.append(f"\n# {name}\n")
        merged_code.append(snippet.get("code", ""))

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["Name", "Value", "Source", "Status", "Suggested Default"])
    for row in list((briefing.hyperparameters or [])):
        writer.writerow(
            [
                row.get("name", ""),
                row.get("value", ""),
                row.get("source_section", ""),
                row.get("status", ""),
                row.get("suggested_default", ""),
            ]
        )

    ambiguity_lines = ["# Ambiguity Report\n"]
    for row in list((briefing.ambiguities or [])):
        ambiguity_lines.append(
            f"## {row.get('ambiguity_id', '')}: {row.get('title', '')}\n"
            f"- Type: {row.get('ambiguity_type', '')}\n"
            f"- Section: {row.get('section', '')}\n"
            f"- Ambiguous point: {row.get('ambiguous_point', '')}\n"
            f"- Impact: {row.get('implementation_consequence', '')}\n"
            f"- Resolution: {row.get('agent_resolution', '')}\n"
            f"- Confidence: {row.get('confidence', '')}\n"
            f"- User override: {row.get('user_resolution', '')}\n"
        )

    items = [
        {
            "kind": "briefing",
            "filename": "briefing.md",
            "content_type": "text/markdown",
            "content": section_md,
        },
        {
            "kind": "code",
            "filename": "annotated_code.py",
            "content_type": "text/x-python",
            "content": "\n".join(merged_code),
        },
        {
            "kind": "hyperparams",
            "filename": "hyperparameters.csv",
            "content_type": "text/csv",
            "content": csv_buf.getvalue(),
        },
        {
            "kind": "ambiguity_report",
            "filename": "ambiguity_report.md",
            "content_type": "text/markdown",
            "content": "\n".join(ambiguity_lines),
        },
    ]
    return {"items": items}


@router.post("/{paper_id}/resolve-ambiguity")
async def resolve_ambiguity(
    paper_id: str,
    payload: ResolveAmbiguityRequest,
    current_user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    paper = await get_paper_with_briefing(paper_id, current_user.id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    briefing = paper.briefing
    if not briefing:
        raise HTTPException(status_code=404, detail="Briefing not found")

    ambiguities = list((briefing.ambiguities or []))
    updated: dict[str, Any] | None = None
    for item in ambiguities:
        if item.get("ambiguity_id") != payload.ambiguity_id:
            continue
        item["resolved"] = True
        item["user_resolution"] = payload.resolution
        updated = item
        break

    if updated is None:
        raise HTTPException(status_code=404, detail="Ambiguity not found")

    await update_briefing_ambiguities(paper_id, ambiguities)
    return {"updated": updated}
