from __future__ import annotations

from langchain_core.tools import tool

from backend.models.briefing import InternalRepresentation


def build_analysis_tools(internal_rep: InternalRepresentation):
    @tool("hyperparameter_extractor")
    def hyperparameter_extractor() -> str:
        """Return the hyperparameter registry as a readable table-like text."""
        if not internal_rep.hyperparameter_registry:
            return "No hyperparameters extracted."
        lines = ["Name | Value | Source | Status | Suggested Default | Reasoning"]
        for hp in internal_rep.hyperparameter_registry:
            lines.append(
                f"{hp.name} | {hp.value or ''} | {hp.source_section} | {hp.status} | {hp.suggested_default or ''} | {hp.suggested_reasoning or ''}"
            )
        return "\n".join(lines)

    @tool("ambiguity_detector")
    def ambiguity_detector(section_text: str) -> str:
        """Find implementation-impactful ambiguity hints in provided section text."""
        text = section_text.lower()
        markers = []
        keywords = ["standard", "we follow", "details omitted", "not shown", "supplementary", "appendix", "footnote"]
        for kw in keywords:
            if kw in text:
                markers.append(kw)
        if not markers:
            return "No explicit ambiguity markers detected in the provided section text."
        return (
            "Potential ambiguity markers detected: "
            + ", ".join(sorted(set(markers)))
            + ". Verify the exact implementation choices before coding."
        )

    return [hyperparameter_extractor, ambiguity_detector]
