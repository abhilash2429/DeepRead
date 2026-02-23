SECTION_PROMPTS: dict[int, str] = {
    1: """
You are writing Briefing Section 1: What This Paper Actually Does.
One paragraph, plain language, no equations and no architecture jargon.
Explain:
- what problem is being solved,
- what the paper proposes,
- why it matters.
Assume no prior ML background.
""",
    2: """
You are writing Briefing Section 2: The Mechanism.
Explain how the method works step-by-step.
Requirements:
- Decode every equation symbol at point of first use.
- Weave in figure interpretation where relevant.
- Explain prerequisites inline when needed.
- Chain explanations downward: why each component is necessary given prior steps.
- Cite sections/equations inline.
""",
    3: """
You are writing Briefing Section 3: What You Need To Already Know.
Produce a dependency-ordered list of concepts from foundational to paper-specific.
For each concept use exactly this 3-part structure:
1) Problem
2) Solution
3) Usage in this paper
Be explicit and practical for implementation understanding.
""",
    4: """
You are writing Briefing Section 4: The Full Implementation Map.
Enumerate all components in dependency order (leaf components first).
For each component include:
- plain-English role,
- annotated PyTorch snippet with provenance labels,
- implementation notes: pitfalls, dimension checks, common mistakes.
Group dependent components together.
""",
    5: """
You are writing Briefing Section 5: What The Paper Left Out.
This is the full ambiguity report and must be prominent.
For each ambiguity include:
- what is missing,
- section source,
- implementation consequence,
- agent resolution with reasoning and confidence.
Organize by ambiguity type:
missing_hyperparameter, undefined_notation, underspecified_architecture,
missing_training_detail, ambiguous_loss_function.
Be more aggressive than the comprehension pass in surfacing concerns.
""",
    6: """
You are writing Briefing Section 6: How To Train It.
Provide:
- full hyperparameter table (name, value, source, status),
- complete training loop recipe (optimizer, scheduler, loss, batch size, steps),
- preprocessing requirements,
- tricks/implementation details affecting results.
For every missing value, propose a default and justify it.
""",
}

