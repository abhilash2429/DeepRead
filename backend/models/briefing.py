from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


HyperparameterStatus = Literal["paper-stated", "inferred", "missing"]
AmbiguityType = Literal[
    "missing_hyperparameter",
    "undefined_notation",
    "underspecified_architecture",
    "missing_training_detail",
    "ambiguous_loss_function",
]


class DependencyEdge(BaseModel):
    parent: str
    child: str


class HyperparameterEntry(BaseModel):
    name: str
    value: str | None = None
    source_section: str
    status: HyperparameterStatus
    suggested_default: str | None = None
    suggested_reasoning: str | None = None


class AmbiguityEntry(BaseModel):
    ambiguity_id: str
    ambiguity_type: AmbiguityType
    title: str
    ambiguous_point: str
    section: str
    implementation_consequence: str
    agent_resolution: str
    reasoning: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    resolved: bool = False
    user_resolution: str | None = None


class ConceptExplanation(BaseModel):
    concept: str
    problem: str
    solution: str
    usage_in_paper: str


class InternalRepresentation(BaseModel):
    problem_statement: str
    method_summary: str
    novelty: str
    component_graph: list[DependencyEdge] = Field(default_factory=list)
    hyperparameter_registry: list[HyperparameterEntry] = Field(default_factory=list)
    ambiguity_log: list[AmbiguityEntry] = Field(default_factory=list)
    training_procedure: str
    prerequisite_concepts: list[ConceptExplanation] = Field(default_factory=list)


class BriefingSection(BaseModel):
    section_number: int
    section_name: str
    content: str

