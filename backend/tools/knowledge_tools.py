from __future__ import annotations

from langchain_core.tools import tool

from backend.models.conversation import InternalRepresentation


BACKGROUND_KNOWLEDGE = {
    "attention is all you need": "Introduced the Transformer architecture with self-attention, removing recurrence and enabling parallel sequence modeling.",
    "resnet": "Residual connections add identity shortcuts so deep networks train stably without vanishing gradients.",
    "adam": "Adaptive Moment Estimation combines momentum and per-parameter adaptive learning rates for faster, stable optimization.",
    "batch normalization": "Normalizes activations across a minibatch to stabilize training and allow higher learning rates.",
    "dropout": "Randomly zeros activations during training to reduce co-adaptation and improve generalization.",
    "bert": "Bidirectional Transformer pretraining using masked language modeling for contextual representations.",
    "gpt-2": "Decoder-only Transformer language model trained autoregressively at scale.",
    "vit": "Vision Transformer that tokenizes image patches and applies a Transformer encoder.",
    "ddpm": "Denoising Diffusion Probabilistic Models generate data by reversing a learned noising process.",
}


def build_knowledge_tools(internal_rep: InternalRepresentation):
    concept_map = {c.concept.lower(): c.explanation for c in internal_rep.prerequisite_concepts}

    @tool("prerequisite_expander")
    def prerequisite_expander(concept_name: str) -> str:
        """Expand a prerequisite concept in student-friendly implementation terms."""
        key = concept_name.strip().lower()
        if key in concept_map:
            return concept_map[key]
        for concept, explanation in concept_map.items():
            if key in concept:
                return explanation
        return (
            f"No explicit prerequisite entry for '{concept_name}'. "
            "Treat this as assumed background and explain it from first principles before coding."
        )

    @tool("background_knowledge_lookup")
    def background_knowledge_lookup(paper_name: str) -> str:
        """Fetch built-in background knowledge for landmark ML papers/concepts."""
        key = paper_name.strip().lower()
        if key in BACKGROUND_KNOWLEDGE:
            return BACKGROUND_KNOWLEDGE[key]
        for name, summary in BACKGROUND_KNOWLEDGE.items():
            if key in name:
                return summary
        return f"No built-in background note found for '{paper_name}'."

    return [prerequisite_expander, background_knowledge_lookup]
