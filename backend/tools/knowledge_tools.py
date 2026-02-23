from __future__ import annotations

from langchain_core.tools import tool

from backend.background_knowledge.landmark_papers import LANDMARK_PAPERS
from backend.models.briefing import InternalRepresentation


def build_knowledge_tools(internal_rep: InternalRepresentation):
    concept_map = {c.concept.lower(): c for c in internal_rep.prerequisite_concepts}

    @tool("prerequisite_expander")
    def prerequisite_expander(concept_name: str) -> str:
        """Expand a prerequisite concept using problem -> solution -> paper usage."""
        key = concept_name.strip().lower()
        if key in concept_map:
            concept = concept_map[key]
            return (
                f"Problem: {concept.problem}\n"
                f"Solution: {concept.solution}\n"
                f"Usage in this paper: {concept.usage_in_paper}"
            )
        for concept, payload in concept_map.items():
            if key in concept:
                return (
                    f"Problem: {payload.problem}\n"
                    f"Solution: {payload.solution}\n"
                    f"Usage in this paper: {payload.usage_in_paper}"
                )
        return (
            f"No explicit prerequisite entry for '{concept_name}'. "
            "Treat this as assumed background and explain it from first principles before coding."
        )

    @tool("background_knowledge_lookup")
    def background_knowledge_lookup(paper_name: str) -> str:
        """Fetch built-in background knowledge for landmark ML papers/concepts."""
        key = paper_name.strip().lower()
        if key in LANDMARK_PAPERS:
            return LANDMARK_PAPERS[key]
        for name, summary in LANDMARK_PAPERS.items():
            if key in name:
                return summary
        return f"No built-in background note found for '{paper_name}'."

    return [prerequisite_expander, background_knowledge_lookup]
