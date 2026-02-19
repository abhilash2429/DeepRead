from __future__ import annotations

import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.models.conversation import InternalRepresentation
from backend.models.paper import ElementType, ParsedPaper
from backend.prompts.comprehension import COMPREHENSION_PROMPT
from backend.services.gemini_client import GeminiClient


async def run_comprehension(gemini: GeminiClient, parsed: ParsedPaper) -> InternalRepresentation:
    figure_lines = [
        f"- [p.{el.page_number}] {el.figure_description or el.caption or ''}"
        for el in parsed.elements
        if el.element_type == ElementType.FIGURE
    ]
    parser = PydanticOutputParser(pydantic_object=InternalRepresentation)
    prompt = ChatPromptTemplate.from_template(
        "{base_prompt}\n\n"
        "Title: {title}\n"
        "Authors: {authors}\n"
        "Abstract: {abstract}\n"
        "Primary task hint: {primary_task}\n"
        "Prerequisite hint list: {prereq_hint}\n\n"
        "Figure descriptions:\n{figure_descriptions}\n\n"
        "Full paper text:\n{full_text}\n\n"
        "{format_instructions}"
    )
    chain = prompt | gemini.llm | parser
    try:
        return await chain.ainvoke(
            {
                "base_prompt": COMPREHENSION_PROMPT,
                "title": parsed.title,
                "authors": ", ".join(parsed.authors),
                "abstract": parsed.abstract,
                "primary_task": parsed.primary_task or "",
                "prereq_hint": json.dumps(parsed.prerequisites_raw),
                "figure_descriptions": "\n".join(figure_lines) if figure_lines else "(none)",
                "full_text": parsed.full_text,
                "format_instructions": parser.get_format_instructions(),
            }
        )
    except Exception:
        fallback_prompt = (
            f"{COMPREHENSION_PROMPT}\n\n"
            f"Title: {parsed.title}\n"
            f"Authors: {', '.join(parsed.authors)}\n"
            f"Abstract: {parsed.abstract}\n\n"
            "Figure descriptions:\n"
            + ("\n".join(figure_lines) if figure_lines else "(none)")
            + "\n\nFull paper text:\n"
            + parsed.full_text
        )
        raw = await gemini.generate_json(fallback_prompt)
        return InternalRepresentation.model_validate(raw)
