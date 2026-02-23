from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.agents.code_agent import generate_component_code
from backend.agents.graph import BriefingState, build_briefing_graph
from backend.db.queries import (
    save_briefing_section,
    save_briefing_structured_data,
    update_paper_status,
)
from backend.models.artifacts import CodeSnippet
from backend.models.briefing import InternalRepresentation
from backend.models.paper import ElementType, ParsedPaper
from backend.prompts.briefing_sections import SECTION_PROMPTS


BRIEFING_MODEL_PRO = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
)

BRIEFING_MODEL_FLASH = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
)


EventEmitter = Callable[[str, dict[str, Any]], Awaitable[None] | None]


async def _emit(emit_event: EventEmitter | None, event: str, data: dict[str, Any]) -> None:
    if emit_event is None:
        return
    result = emit_event(event, data)
    if isawaitable(result):
        await result


def _component_order(internal_rep: InternalRepresentation) -> list[str]:
    components = {edge.parent for edge in internal_rep.component_graph} | {
        edge.child for edge in internal_rep.component_graph
    }
    return sorted(components)


def _briefing_model_for_section(section_number: int) -> ChatGoogleGenerativeAI:
    if section_number in {2, 4}:
        return BRIEFING_MODEL_PRO
    return BRIEFING_MODEL_FLASH


def _figure_context(parsed_paper: ParsedPaper) -> str:
    figure_lines = []
    for element in parsed_paper.elements:
        if element.element_type != ElementType.FIGURE:
            continue
        figure_lines.append(
            f"[{element.id} | p.{element.page_number}] "
            f"Caption: {element.caption or '(none)'} | "
            f"Interpretation: {element.figure_description or '(none)'}"
        )
    return "\n".join(figure_lines) if figure_lines else "(none)"


def _section_context(section_number: int, parsed_paper: ParsedPaper, internal_rep: InternalRepresentation) -> str:
    if section_number == 1:
        return (
            f"Problem statement: {internal_rep.problem_statement}\n"
            f"Method summary: {internal_rep.method_summary}\n"
            f"Novelty: {internal_rep.novelty}"
        )
    if section_number == 2:
        equations = [
            f"[{element.equation_label or element.id}] {element.content}"
            for element in parsed_paper.elements
            if element.element_type == ElementType.EQUATION
        ]
        return (
            f"Method summary:\n{internal_rep.method_summary}\n\n"
            f"Component graph:\n{json.dumps([edge.model_dump() for edge in internal_rep.component_graph], indent=2)}\n\n"
            f"Equations:\n{chr(10).join(equations) if equations else '(none)'}\n\n"
            f"Figure interpretations:\n{_figure_context(parsed_paper)}"
        )
    if section_number == 3:
        return json.dumps([item.model_dump() for item in internal_rep.prerequisite_concepts], indent=2)
    if section_number == 4:
        return json.dumps([edge.model_dump() for edge in internal_rep.component_graph], indent=2)
    if section_number == 5:
        return json.dumps([item.model_dump() for item in internal_rep.ambiguity_log], indent=2)
    return (
        f"Training procedure:\n{internal_rep.training_procedure}\n\n"
        f"Hyperparameters:\n{json.dumps([item.model_dump() for item in internal_rep.hyperparameter_registry], indent=2)}"
    )


async def run_briefing_pipeline(
    session_id: str,
    paper_id: str,
    parsed_paper: ParsedPaper,
    internal_rep: InternalRepresentation,
    emit_event: EventEmitter | None = None,
) -> dict[str, str]:
    def make_section_handler(section_number: int):
        async def _handler(state: BriefingState) -> BriefingState:
            await _emit(
                emit_event,
                "thinking",
                {"message": f"Generating briefing section {section_number}..."},
            )

            code_snippets = state.get("code_snippets", [])
            if section_number == 4:
                import asyncio
                
                async def _gen_and_emit(component: str):
                    await _emit(
                        emit_event,
                        "thinking",
                        {"message": f"Generating PyTorch snippet for {component}..."},
                    )
                    snippet = await generate_component_code(
                        component_name=component,
                        component_description=f"Implementation for {component}",
                        relevant_sections=[component],
                        relevant_equations=[],
                        resolved_ambiguities={
                            item.ambiguity_id: (item.user_resolution or item.agent_resolution)
                            for item in internal_rep.ambiguity_log
                        },
                        reasoning_tier="pro",
                    )
                    await _emit(
                        emit_event,
                        "code_snippet",
                        {"snippet": snippet.model_dump()},
                    )
                    return snippet

                components = _component_order(internal_rep)
                snippets = await asyncio.gather(*[_gen_and_emit(c) for c in components])
                code_snippets.extend(snippets)

            section_prompt = ChatPromptTemplate.from_template(
                "{section_prompt}\n\n"
                "Paper title: {title}\n"
                "InternalRepresentation:\n{internal_rep_json}\n\n"
                "Section-specific context:\n{section_context}\n\n"
                "Code snippets (if any):\n{code_snippets}\n\n"
                "Write clean markdown."
            )
            chain = section_prompt | _briefing_model_for_section(section_number) | StrOutputParser()
            payload = {
                "section_prompt": SECTION_PROMPTS[section_number],
                "title": parsed_paper.title,
                "internal_rep_json": internal_rep.model_dump_json(indent=2),
                "section_context": _section_context(section_number, parsed_paper, internal_rep),
                "code_snippets": json.dumps([item.model_dump() for item in code_snippets], indent=2),
            }

            chunks: list[str] = []
            async for event in chain.astream_events(payload, version="v2"):
                if event.get("event") != "on_chat_model_stream":
                    continue
                chunk = event.get("data", {}).get("chunk")
                text = ""
                if hasattr(chunk, "content"):
                    raw_content = getattr(chunk, "content")
                    if isinstance(raw_content, list):
                        text = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in raw_content
                        )
                    else:
                        text = str(raw_content or "")
                if not text:
                    continue
                chunks.append(text)
                await _emit(
                    emit_event,
                    "section_token",
                    {"section_number": section_number, "text": text},
                )

            content = "".join(chunks).strip()
            state.setdefault("completed_sections", {})[f"section_{section_number}"] = content
            state["generation_progress"] = section_number
            state["code_snippets"] = code_snippets
            await save_briefing_section(paper_id, section_number, content)
            await _emit(
                emit_event,
                "progress",
                {"generation_progress": section_number},
            )
            return state

        return _handler

    section_handlers = {
        1: make_section_handler(1),
        2: make_section_handler(2),
        3: make_section_handler(3),
        4: make_section_handler(4),
        5: make_section_handler(5),
        6: make_section_handler(6),
    }
    graph = build_briefing_graph(section_handlers)
    initial_state: BriefingState = {
        "session_id": session_id,
        "paper_id": paper_id,
        "internal_rep": internal_rep,
        "parsed_paper": parsed_paper,
        "completed_sections": {},
        "generation_progress": 0,
        "code_snippets": [],
    }
    final_state = await graph.ainvoke(initial_state)
    await save_briefing_structured_data(
        paper_id=paper_id,
        hyperparameters=internal_rep.hyperparameter_registry,
        ambiguities=internal_rep.ambiguity_log,
        code_snippets=final_state.get("code_snippets", []),
    )
    await update_paper_status(paper_id, "COMPLETE")
    return dict(final_state.get("completed_sections", {}))
