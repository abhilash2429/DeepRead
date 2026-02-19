from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.models.paper import ProvenanceLabel


class Stage(str, Enum):
    ORIENTATION = "orientation"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    AMBIGUITY = "ambiguity"
    TRAINING = "training"


class DependencyEdge(BaseModel):
    parent: str
    child: str


class HyperparameterEntry(BaseModel):
    name: str
    value: str | None = None
    source_section: str
    status: ProvenanceLabel
    suggested_default: str | None = None


class AmbiguityEntry(BaseModel):
    ambiguity_id: str
    ambiguous_point: str
    section: str
    implementation_consequence: str
    best_guess_resolution: str
    reasoning: str
    resolved: bool = False
    user_resolution: str | None = None


class ConceptExplanation(BaseModel):
    concept: str
    explanation: str


class InternalRepresentation(BaseModel):
    problem_statement: str
    method_summary: str
    novelty: str
    component_graph: list[DependencyEdge] = Field(default_factory=list)
    hyperparameter_registry: list[HyperparameterEntry] = Field(default_factory=list)
    ambiguity_log: list[AmbiguityEntry] = Field(default_factory=list)
    training_procedure: str
    prerequisite_concepts: list[ConceptExplanation] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str
    content: str


class ConversationState(BaseModel):
    session_id: str
    current_stage: Stage = Stage.ORIENTATION
    message_history: list[ChatMessage] = Field(default_factory=list)
    resolved_ambiguities: dict[str, str] = Field(default_factory=dict)
    prerequisites_explained: set[str] = Field(default_factory=set)
    internal_representation: InternalRepresentation
    last_component_focus: str | None = None
    current_component_index: int = 0
    user_level: str = "student"
    pending_question: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
