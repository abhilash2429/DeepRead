from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from backend.models.paper import ElementType, ParsedPaper
from backend.services.gemini_client import GeminiClient
from backend.services.pdf_parser import parse_pdf
from backend.services.vision_service import describe_figure


StatusEmitter = Callable[[str, str, int], None]


class IngestionSummary(BaseModel):
    primary_task: str
    prerequisites: list[str] = Field(default_factory=list)


async def run_ingestion(
    gemini: GeminiClient,
    pdf_bytes: bytes,
    title: str,
    authors: list[str],
    abstract: str,
    emit_status: StatusEmitter,
) -> ParsedPaper:
    emit_status("parse", "Parsing PDF elements", 20)
    parsed = parse_pdf(pdf_bytes=pdf_bytes, title=title, authors=authors, abstract=abstract)

    figures = [el for el in parsed.elements if el.element_type == ElementType.FIGURE and el.image_bytes_b64]
    total = max(len(figures), 1)
    for i, fig in enumerate(figures, start=1):
        emit_status("vision", f"Interpreting figure {i}/{len(figures)}", 20 + int(40 * i / total))
        fig.figure_description = await describe_figure(gemini, fig.image_bytes_b64 or "", fig.caption)

    emit_status("parse", "Extracting primary task and prerequisites", 65)
    parser = PydanticOutputParser(pydantic_object=IngestionSummary)
    prompt = ChatPromptTemplate.from_template(
        "Read the full ML paper text and extract a concise implementation-oriented summary.\n"
        "Output strictly as JSON.\n{format_instructions}\n\n"
        "Title: {title}\n"
        "Abstract: {abstract}\n"
        "Full paper text:\n{full_text}"
    )
    chain = prompt | gemini.llm | parser
    try:
        summary = await chain.ainvoke(
            {
                "format_instructions": parser.get_format_instructions(),
                "title": parsed.title,
                "abstract": parsed.abstract,
                "full_text": parsed.full_text,
            }
        )
        parsed.primary_task = summary.primary_task
        parsed.prerequisites_raw = summary.prerequisites
    except Exception:
        parsed.primary_task = "Paper objective extraction failed; review abstract and introduction manually."
        parsed.prerequisites_raw = []

    emit_status("parse", "Ingestion complete", 70)
    return parsed


async def event_stream(events: list[dict]) -> AsyncIterator[dict]:
    for event in events:
        yield event
