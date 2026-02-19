from __future__ import annotations

import csv
import io

from backend.models.artifacts import ArtifactItem, ArtifactManifest, CodeSnippet
from backend.models.conversation import ConversationState


def build_artifacts(state: ConversationState, snippets: list[CodeSnippet]) -> ArtifactManifest:
    items: list[ArtifactItem] = []

    summary = (
        "# Architecture Summary\n\n"
        f"## Problem\n{state.internal_representation.problem_statement}\n\n"
        f"## Method\n{state.internal_representation.method_summary}\n\n"
        f"## Novelty\n{state.internal_representation.novelty}\n"
    )
    items.append(
        ArtifactItem(
            kind="architecture_summary",
            filename="architecture_summary.md",
            content_type="text/markdown",
            content=summary,
        )
    )

    merged_parts = ["# Annotated Component Implementations\n"]
    for snip in snippets:
        merged_parts.append(f"\n## {snip.component_name}\n```python\n{snip.code}\n```\n")
        items.append(
            ArtifactItem(
                kind="component_code",
                filename=f"{snip.component_name.lower().replace(' ', '_')}.py",
                content_type="text/x-python",
                content=snip.code,
            )
        )
    items.append(
        ArtifactItem(
            kind="annotated_code_merged",
            filename="annotated_code.py",
            content_type="text/x-python",
            content="\n".join(merged_parts),
        )
    )

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["Name", "Value", "Source Section", "Status", "Suggested Default"])
    for h in state.internal_representation.hyperparameter_registry:
        writer.writerow([h.name, h.value or "", h.source_section, h.status, h.suggested_default or ""])
    items.append(
        ArtifactItem(
            kind="hyperparameter_table",
            filename="hyperparameters.csv",
            content_type="text/csv",
            content=csv_buf.getvalue(),
        )
    )

    ambiguity_md = ["# Ambiguity Report\n"]
    for a in state.internal_representation.ambiguity_log:
        ambiguity_md.append(
            f"\n## {a.ambiguity_id}\n"
            f"- Ambiguous point: {a.ambiguous_point}\n"
            f"- Section: {a.section}\n"
            f"- Impact: {a.implementation_consequence}\n"
            f"- Best guess: {a.best_guess_resolution}\n"
            f"- Resolved: {a.resolved}\n"
            f"- User resolution: {a.user_resolution or ''}\n"
        )
    items.append(
        ArtifactItem(
            kind="ambiguity_report",
            filename="ambiguity_report.md",
            content_type="text/markdown",
            content="".join(ambiguity_md),
        )
    )
    return ArtifactManifest(items=items)

