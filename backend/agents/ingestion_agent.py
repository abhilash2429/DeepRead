from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from backend.models.paper import ElementType, ParsedPaper
from backend.services.pdf_parser import parse_pdf
from backend.services.vision_service import describe_figure
from backend.utils.llm_retry import call_with_llm_retry


INGESTION_MODEL = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)


class IngestionSummary(BaseModel):
    primary_task: str
    prerequisites: list[str] = Field(default_factory=list)


StatusEmitter = Callable[[str], None]


async def run_ingestion(
    pdf_bytes: bytes,
    title: str,
    authors: list[str],
    abstract: str,
    emit_thinking: StatusEmitter | None = None,
) -> ParsedPaper:
    if emit_thinking:
        emit_thinking("Extracting text blocks and structural elements from PDF...")
    parsed = parse_pdf(pdf_bytes=pdf_bytes, title=title, authors=authors, abstract=abstract)

    figures = [item for item in parsed.elements if item.element_type == ElementType.FIGURE and item.image_bytes_b64]
    if figures:
        if emit_thinking:
            emit_thinking(f"Interpreting {len(figures)} figures in parallel...")

        async def _describe(figure):
            return await describe_figure(
                image_b64=figure.image_bytes_b64 or "",
                caption=figure.caption,
            )

        descriptions = await asyncio.gather(*[_describe(fig) for fig in figures])
        for figure, description in zip(figures, descriptions):
            figure.figure_description = description
        if emit_thinking:
            emit_thinking(f"All {len(figures)} figures interpreted.")

    parser = PydanticOutputParser(pydantic_object=IngestionSummary)
    prompt = ChatPromptTemplate.from_template(
        "Read this paper context and identify:\n"
        "1) the primary ML task solved,\n"
        "2) foundational prerequisite concepts required to understand implementation.\n"
        "Return structured JSON.\n"
        "{format_instructions}\n\n"
        "Title: {title}\n"
        "Abstract: {abstract}\n"
        "Full text:\n{full_text}"
    )
    chain = prompt | INGESTION_MODEL | parser

    if emit_thinking:
        emit_thinking("Inferring primary task and prerequisite concepts from full text...")
    try:
        payload = {
            "format_instructions": parser.get_format_instructions(),
            "title": parsed.title,
            "abstract": parsed.abstract,
            "full_text": parsed.full_text,
        }
        summary = await call_with_llm_retry(lambda: chain.ainvoke(payload))
        parsed.primary_task = summary.primary_task
        parsed.prerequisites_raw = summary.prerequisites
    except Exception:
        parsed.primary_task = "Primary task extraction failed."
        parsed.prerequisites_raw = []

    return parsed
