from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypedDict

from langgraph.graph import END, StateGraph

from backend.models.artifacts import CodeSnippet
from backend.models.briefing import InternalRepresentation
from backend.models.paper import ParsedPaper


class BriefingState(TypedDict, total=False):
    session_id: str
    paper_id: str
    internal_rep: InternalRepresentation
    parsed_paper: ParsedPaper
    completed_sections: dict[str, str]
    generation_progress: int
    code_snippets: list[CodeSnippet]


SectionHandler = Callable[[BriefingState], Awaitable[BriefingState]]


def build_briefing_graph(section_handlers: dict[int, SectionHandler]):
    workflow = StateGraph(BriefingState)

    for section_number in range(1, 7):
        node_name = f"section_{section_number}"
        workflow.add_node(node_name, section_handlers[section_number])

    workflow.set_entry_point("section_1")
    workflow.add_edge("section_1", "section_2")
    workflow.add_edge("section_2", "section_3")
    workflow.add_edge("section_3", "section_4")
    workflow.add_edge("section_4", "section_5")
    workflow.add_edge("section_5", "section_6")
    workflow.add_edge("section_6", END)
    return workflow.compile()

