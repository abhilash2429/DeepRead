from __future__ import annotations

import os

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.models.briefing import InternalRepresentation
from backend.models.paper import ElementType, ParsedPaper
from backend.prompts.comprehension import COMPREHENSION_PROMPT
from backend.utils.llm_retry import call_with_llm_retry


COMPREHENSION_MODEL = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)


async def run_comprehension(parsed_paper: ParsedPaper) -> InternalRepresentation:
    parser = PydanticOutputParser(pydantic_object=InternalRepresentation)
    figure_lines = [
        f"- [{element.id} | p.{element.page_number}] {element.figure_description or element.caption or ''}"
        for element in parsed_paper.elements
        if element.element_type == ElementType.FIGURE
    ]
    prompt = ChatPromptTemplate.from_template(
        "{base_prompt}\n\n"
        "Title: {title}\n"
        "Authors: {authors}\n"
        "Abstract: {abstract}\n"
        "Primary task hint: {primary_task}\n"
        "Prerequisite hints: {prerequisites}\n\n"
        "Figure interpretations:\n{figure_descriptions}\n\n"
        "Full paper text:\n{full_text}\n\n"
        "Format instructions:\n{format_instructions}"
    )
    chain = prompt | COMPREHENSION_MODEL | parser
    payload = {
        "base_prompt": COMPREHENSION_PROMPT,
        "title": parsed_paper.title,
        "authors": ", ".join(parsed_paper.authors),
        "abstract": parsed_paper.abstract,
        "primary_task": parsed_paper.primary_task or "",
        "prerequisites": ", ".join(parsed_paper.prerequisites_raw),
        "figure_descriptions": "\n".join(figure_lines) if figure_lines else "(none)",
        "full_text": parsed_paper.full_text,
        "format_instructions": parser.get_format_instructions(),
    }
    try:
        return await call_with_llm_retry(lambda: chain.ainvoke(payload))
    except Exception:
        return InternalRepresentation(
            problem_statement=parsed_paper.primary_task or "Problem statement unavailable.",
            method_summary="Method summary unavailable due to temporary model failure.",
            novelty="Novelty extraction unavailable.",
            component_graph=[],
            hyperparameter_registry=[],
            ambiguity_log=[],
            training_procedure="Training procedure unavailable.",
            prerequisite_concepts=[],
        )
