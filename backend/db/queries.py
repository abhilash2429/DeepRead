from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from backend.db.prisma import prisma

try:
    from prisma import Json  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    try:
        from prisma_client import Json  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        def Json(value: Any) -> Any:  # type: ignore[misc]
            return value


FREE_PLAN_LIMIT = 3


def _to_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_to_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json(item) for key, item in value.items()}
    return value


async def get_or_create_user(google_sub: str, email: str, name: str, avatar_url: str | None) -> Any:
    return await prisma.user.upsert(
        where={"google_sub": google_sub},
        data={
            "create": {
                "google_sub": google_sub,
                "email": email,
                "name": name,
                "avatar_url": avatar_url,
                "plan": "FREE",
                "papers_analyzed": 0,
            },
            "update": {
                "email": email,
                "name": name,
                "avatar_url": avatar_url,
            },
        },
    )


async def get_or_create_user_from_github(github_id: str, email: str, name: str, avatar_url: str | None) -> Any:
    existing_by_email = await prisma.user.find_unique(where={"email": email})
    if existing_by_email:
        return await prisma.user.update(
            where={"id": existing_by_email.id},
            data={
                "name": name,
                "avatar_url": avatar_url,
            },
        )

    return await prisma.user.upsert(
        where={"google_sub": f"github:{github_id}"},
        data={
            "create": {
                "google_sub": f"github:{github_id}",
                "email": email,
                "name": name,
                "avatar_url": avatar_url,
                "plan": "FREE",
                "papers_analyzed": 0,
            },
            "update": {
                "email": email,
                "name": name,
                "avatar_url": avatar_url,
            },
        },
    )


async def get_user_by_id(user_id: str) -> Any:
    return await prisma.user.find_unique(where={"id": user_id})


async def check_user_limit(user_id: str) -> bool:
    user = await get_user_by_id(user_id)
    if not user:
        return False
    if user.plan == "PRO":
        return True
    return int(user.papers_analyzed) < FREE_PLAN_LIMIT


async def get_user_limit_details(user_id: str) -> dict[str, Any]:
    user = await get_user_by_id(user_id)
    if not user:
        return {"plan": "FREE", "limit": FREE_PLAN_LIMIT, "used": FREE_PLAN_LIMIT}
    if user.plan == "PRO":
        return {"plan": "PRO", "limit": None, "used": int(user.papers_analyzed)}
    return {"plan": "FREE", "limit": FREE_PLAN_LIMIT, "used": int(user.papers_analyzed)}


async def increment_paper_count(user_id: str) -> Any:
    return await prisma.user.update(where={"id": user_id}, data={"papers_analyzed": {"increment": 1}})


async def create_paper(user_id: str, title: str, authors: list[str], arxiv_id: str | None = None) -> Any:
    return await prisma.paper.create(
        data={
            "user": {"connect": {"id": user_id}},
            "title": title,
            "authors": authors,
            "arxiv_id": arxiv_id,
            "status": "PROCESSING",
            "parsed_paper": Json({}),
            "internal_rep": Json({}),
        }
    )


async def update_paper_status(paper_id: str, status: str) -> Any:
    return await prisma.paper.update(where={"id": paper_id}, data={"status": status})


async def update_paper_metadata(paper_id: str, title: str | None = None, authors: list[str] | None = None) -> Any:
    data: dict[str, Any] = {}
    if title is not None:
        data["title"] = title
    if authors is not None:
        data["authors"] = authors
    if not data:
        return await get_paper_by_id(paper_id)
    return await prisma.paper.update(where={"id": paper_id}, data=data)


async def save_parsed_paper(paper_id: str, parsed_paper: Any) -> Any:
    return await prisma.paper.update(
        where={"id": paper_id},
        data={"parsed_paper": Json(_to_json(parsed_paper))},
    )


async def save_internal_rep(paper_id: str, internal_rep: Any) -> Any:
    return await prisma.paper.update(
        where={"id": paper_id},
        data={"internal_rep": Json(_to_json(internal_rep))},
    )


async def _ensure_briefing(paper_id: str) -> Any:
    existing = await prisma.briefing.find_unique(where={"paper_id": paper_id})
    if existing:
        return existing
    return await prisma.briefing.create(data={"paper_id": paper_id})


async def save_briefing_section(paper_id: str, section_number: int, content: str) -> Any:
    await _ensure_briefing(paper_id)
    field_name = f"section_{section_number}"
    return await prisma.briefing.update(
        where={"paper_id": paper_id},
        data={field_name: content},
    )


async def save_briefing_structured_data(
    paper_id: str,
    hyperparameters: list[Any],
    ambiguities: list[Any],
    code_snippets: list[Any],
) -> Any:
    await _ensure_briefing(paper_id)
    return await prisma.briefing.update(
        where={"paper_id": paper_id},
        data={
            "hyperparameters": Json(_to_json(hyperparameters)),
            "ambiguities": Json(_to_json(ambiguities)),
            "code_snippets": Json(_to_json(code_snippets)),
        },
    )


async def get_paper_with_briefing(paper_id: str, user_id: str) -> Any:
    return await prisma.paper.find_first(
        where={"id": paper_id, "user_id": user_id},
        include={"briefing": True},
    )


async def get_paper_by_id(paper_id: str) -> Any:
    return await prisma.paper.find_unique(where={"id": paper_id}, include={"briefing": True})


async def get_user_papers(user_id: str) -> list[Any]:
    return await prisma.paper.find_many(
        where={"user_id": user_id},
        order={"created_at": "desc"},
    )


async def append_qa_message(paper_id: str, role: str, content: str) -> Any:
    return await prisma.qamessage.create(
        data={
            "paper_id": paper_id,
            "role": role,
            "content": content,
        }
    )


async def get_qa_history(paper_id: str) -> list[Any]:
    return await prisma.qamessage.find_many(
        where={"paper_id": paper_id},
        order={"created_at": "asc"},
    )


async def update_briefing_ambiguities(paper_id: str, ambiguities: list[Any]) -> Any:
    await _ensure_briefing(paper_id)
    return await prisma.briefing.update(
        where={"paper_id": paper_id},
        data={"ambiguities": Json(_to_json(ambiguities))},
    )
