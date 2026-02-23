from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    SECTION = "Section"
    EQUATION = "Equation"
    FIGURE = "Figure"
    TABLE = "Table"
    PSEUDOCODE = "Pseudocode"


class PaperElement(BaseModel):
    id: str
    element_type: ElementType
    section_heading: str
    page_number: int
    content: str = ""
    caption: str | None = None
    equation_label: str | None = None
    image_bytes_b64: str | None = None
    figure_description: str | None = None


class ParsedPaper(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str
    full_text: str
    elements: list[PaperElement] = Field(default_factory=list)
    primary_task: str | None = None
    prerequisites_raw: list[str] = Field(default_factory=list)
    pdf_bytes_b64: str | None = None


ProvenanceLabel = Literal["paper-stated", "inferred", "assumed", "missing"]

