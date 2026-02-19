from __future__ import annotations

from pydantic import BaseModel, Field

from backend.models.paper import ProvenanceLabel


class CodeSnippet(BaseModel):
    component_name: str
    code: str
    provenance: ProvenanceLabel
    assumption_notes: list[str] = Field(default_factory=list)
    source_sections: list[str] = Field(default_factory=list)


class ArtifactItem(BaseModel):
    kind: str
    filename: str
    content_type: str
    content: str


class ArtifactManifest(BaseModel):
    items: list[ArtifactItem] = Field(default_factory=list)

