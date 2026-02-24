from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bootstrap_local_env_override() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


_bootstrap_local_env_override()

from backend.agents.briefing_agent import (
    _briefing_model_for_section,
    _component_order,
    _section_context,
)
from backend.agents.code_agent import generate_component_code
from backend.agents.comprehension_agent import run_comprehension
from backend.agents.ingestion_agent import run_ingestion
from backend.models.artifacts import CodeSnippet
from backend.models.briefing import InternalRepresentation
from backend.models.paper import ElementType, PaperElement, ParsedPaper
from backend.prompts.briefing_sections import SECTION_PROMPTS
from backend.services.arxiv_fetcher import fetch_arxiv_pdf_with_progress


ARTIFACT_ROOT = ROOT / "frontend" / "public" / "demo-artifacts"
EXAMPLES_TS = ROOT / "frontend" / "lib" / "examples.ts"
CONTEXT_MODEL = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
)


@dataclass(frozen=True)
class DemoPaper:
    slug: str
    title: str
    paper_title: str
    arxiv_id: str
    badges: list[str]


PAPERS: list[DemoPaper] = [
    DemoPaper(
        slug="transformer",
        title="Transformer",
        paper_title="Attention Is All You Need",
        arxiv_id="1706.03762",
        badges=["architecture", "code map", "training recipe"],
    ),
    DemoPaper(
        slug="resnet",
        title="ResNet",
        paper_title="Deep Residual Learning for Image Recognition",
        arxiv_id="1512.03385",
        badges=["residual blocks", "optimization", "implementation order"],
    ),
    DemoPaper(
        slug="bert",
        title="BERT",
        paper_title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        arxiv_id="1810.04805",
        badges=["pretraining", "fine-tuning", "nlp pipeline"],
    ),
]


SECTION_META: list[tuple[int, str, str]] = [
    (1, "problem", "1. What It Does"),
    (2, "mechanism", "2. The Mechanism"),
    (3, "prereqs", "3. Prerequisites"),
    (4, "implementation", "4. Implementation Map"),
    (5, "ambiguity", "5. Missing Details"),
    (6, "training", "6. Training Recipe"),
]


def _normalize_ascii(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _clean_markdown_for_card(text: str, max_len: int = 460) -> str:
    cleaned = re.sub(r"```[\s\S]*?```", " ", text)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _normalize_ascii(cleaned)
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _summary_from_section_1(section_1: str) -> str:
    compact = _clean_markdown_for_card(section_1, max_len=320)
    split = re.split(r"(?<=[.!?])\s+", compact)
    if split and split[0]:
        first = split[0].strip()
        if len(first) >= 40:
            return first
    return compact


def _build_section_chain(section_number: int):
    prompt = ChatPromptTemplate.from_template(
        "{section_prompt}\n\n"
        "Paper title: {title}\n"
        "InternalRepresentation:\n{internal_rep_json}\n\n"
        "Section-specific context:\n{section_context}\n\n"
        "Code snippets (if any):\n{code_snippets}\n\n"
        "Write clean markdown."
    )
    return prompt | _briefing_model_for_section(section_number) | StrOutputParser()


async def _generate_code_snippets(internal_rep: InternalRepresentation) -> list[CodeSnippet]:
    components = _component_order(internal_rep)
    if not components:
        return []
    selected = components[: min(3, len(components))]
    resolved = {
        item.ambiguity_id: (item.user_resolution or item.agent_resolution)
        for item in internal_rep.ambiguity_log
    }
    snippets: list[CodeSnippet] = []
    for component in selected:
        snippet = await generate_component_code(
            component_name=component,
            component_description=f"Implementation for {component}",
            relevant_sections=[component],
            relevant_equations=[],
            resolved_ambiguities=resolved,
            reasoning_tier="flash",
        )
        snippets.append(snippet)
    return snippets


async def _generate_sections(
    parsed_paper: ParsedPaper,
    internal_rep: InternalRepresentation,
    code_snippets: list[CodeSnippet],
) -> dict[int, str]:
    sections: dict[int, str] = {}
    snippet_json = json.dumps([item.model_dump() for item in code_snippets], indent=2)
    for section_number, _, _ in SECTION_META:
        chain = _build_section_chain(section_number)
        payload = {
            "section_prompt": SECTION_PROMPTS[section_number],
            "title": parsed_paper.title,
            "internal_rep_json": internal_rep.model_dump_json(indent=2),
            "section_context": _section_context(section_number, parsed_paper, internal_rep),
            "code_snippets": snippet_json,
        }
        content = await chain.ainvoke(payload)
        sections[section_number] = _normalize_ascii(content.strip())
    return sections


async def _build_parsed_paper_from_model(paper: DemoPaper) -> ParsedPaper:
    prompt = ChatPromptTemplate.from_template(
        "You are preparing machine-readable paper context for an ML tutoring system.\n"
        "Paper title: {paper_title}\n\n"
        "Generate structured plain text with these sections and high factual density:\n"
        "1) Abstract-style summary\n"
        "2) Method overview\n"
        "3) Key equations with symbol definitions\n"
        "4) Implementation details\n"
        "5) Training recipe and hyperparameters\n"
        "6) Known ambiguities/underspecified details\n"
        "7) Prerequisite concepts\n\n"
        "Do not use markdown tables. Keep output <= 2500 words."
    )
    chain = prompt | CONTEXT_MODEL | StrOutputParser()
    text = await chain.ainvoke({"paper_title": paper.paper_title})
    compact = _normalize_ascii(text.strip())
    abstract = _clean_markdown_for_card(compact, max_len=600)
    return ParsedPaper(
        title=paper.paper_title,
        authors=[],
        abstract=abstract,
        full_text=compact,
        elements=[
            PaperElement(
                id=f"{paper.slug}-context",
                element_type=ElementType.SECTION,
                section_heading="Generated Context",
                page_number=1,
                content=compact,
            )
        ],
        primary_task="Derived from model-generated canonical paper context",
        prerequisites_raw=[],
    )


def _write_artifacts(
    paper: DemoPaper,
    sections: dict[int, str],
    internal_rep: InternalRepresentation,
    code_snippets: list[CodeSnippet],
) -> None:
    out_dir = ARTIFACT_ROOT / paper.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    architecture_summary = (
        f"# {paper.title} Architecture Summary\n\n"
        f"## 1. What This Paper Actually Does\n{sections[1]}\n\n"
        f"## 2. The Mechanism\n{sections[2]}\n\n"
        f"## 3. What You Need To Already Know\n{sections[3]}\n"
    )
    (out_dir / "architecture_summary.md").write_text(architecture_summary, encoding="utf-8")

    code_parts: list[str] = []
    for snippet in code_snippets:
        code_parts.append(f"# Component: {snippet.component_name}")
        code_parts.append(f"# Provenance: {snippet.provenance}")
        if snippet.assumption_notes:
            for note in snippet.assumption_notes:
                code_parts.append(f"# Assumption: {_normalize_ascii(note)}")
        code_parts.append(snippet.code.strip())
        code_parts.append("")
    merged_code = "\n".join(code_parts).strip() + "\n"
    (out_dir / "annotated_code.py").write_text(merged_code, encoding="utf-8")

    with (out_dir / "hyperparameters.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "value", "source", "status", "suggested_default"])
        for row in internal_rep.hyperparameter_registry:
            writer.writerow(
                [
                    _normalize_ascii(row.name),
                    _normalize_ascii(row.value or ""),
                    _normalize_ascii(row.source_section),
                    _normalize_ascii(row.status),
                    _normalize_ascii(row.suggested_default or ""),
                ]
            )

    ambiguity_lines: list[str] = [f"# {paper.title} Ambiguity Report", ""]
    for item in internal_rep.ambiguity_log:
        ambiguity_lines.extend(
            [
                f"## {item.ambiguity_id}: {_normalize_ascii(item.title)}",
                f"- Type: {_normalize_ascii(item.ambiguity_type)}",
                f"- Section: {_normalize_ascii(item.section)}",
                f"- Ambiguous point: {_normalize_ascii(item.ambiguous_point)}",
                f"- Implementation consequence: {_normalize_ascii(item.implementation_consequence)}",
                f"- Agent resolution: {_normalize_ascii(item.agent_resolution)}",
                f"- Confidence: {item.confidence}",
                "",
            ]
        )
    if len(ambiguity_lines) == 2:
        ambiguity_lines.append("No material ambiguities were detected by the model.")
    (out_dir / "ambiguity_report.md").write_text("\n".join(ambiguity_lines), encoding="utf-8")


def _build_walkthrough_entry(
    paper: DemoPaper,
    sections: dict[int, str],
) -> dict[str, Any]:
    section_rows: list[dict[str, str]] = []
    for section_number, section_id, section_title in SECTION_META:
        section_rows.append(
            {
                "id": section_id,
                "title": section_title,
                "content": _clean_markdown_for_card(sections[section_number]),
            }
        )
    return {
        "slug": paper.slug,
        "title": paper.title,
        "paperTitle": paper.paper_title,
        "summary": _summary_from_section_1(sections[1]),
        "badges": paper.badges,
        "sections": section_rows,
        "downloads": [
            {
                "label": "architecture_summary.md",
                "href": f"/demo-artifacts/{paper.slug}/architecture_summary.md",
            },
            {
                "label": "annotated_code.py",
                "href": f"/demo-artifacts/{paper.slug}/annotated_code.py",
            },
            {
                "label": "hyperparameters.csv",
                "href": f"/demo-artifacts/{paper.slug}/hyperparameters.csv",
            },
            {
                "label": "ambiguity_report.md",
                "href": f"/demo-artifacts/{paper.slug}/ambiguity_report.md",
            },
        ],
    }


def _write_examples_ts(entries: dict[str, Any]) -> None:
    payload_json = json.dumps(entries, indent=2, ensure_ascii=False)
    content = (
        'export type ExampleSlug = "transformer" | "resnet" | "bert";\n\n'
        "export type ExampleSection = {\n"
        "  id: string;\n"
        "  title: string;\n"
        "  content: string;\n"
        "};\n\n"
        "export type ExampleDownload = {\n"
        "  label: string;\n"
        "  href: string;\n"
        "};\n\n"
        "export type ExampleWalkthrough = {\n"
        "  slug: ExampleSlug;\n"
        "  title: string;\n"
        "  paperTitle: string;\n"
        "  summary: string;\n"
        "  badges: string[];\n"
        "  sections: ExampleSection[];\n"
        "  downloads: ExampleDownload[];\n"
        "};\n\n"
        "export const EXAMPLE_WALKTHROUGHS: Record<ExampleSlug, ExampleWalkthrough> = "
        f"{payload_json};\n\n"
        "export const EXAMPLE_LIST: ExampleWalkthrough[] = Object.values(EXAMPLE_WALKTHROUGHS);\n"
    )
    EXAMPLES_TS.write_text(content, encoding="utf-8")


def _load_existing_entries() -> dict[str, Any]:
    if not EXAMPLES_TS.exists():
        return {}
    raw = EXAMPLES_TS.read_text(encoding="utf-8")
    marker = "export const EXAMPLE_WALKTHROUGHS: Record<ExampleSlug, ExampleWalkthrough> = "
    start = raw.find(marker)
    if start < 0:
        return {}
    start += len(marker)
    end = raw.find(";\n\nexport const EXAMPLE_LIST", start)
    if end < 0:
        return {}
    payload = raw[start:end].strip()
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001
        return {}
    return {}


async def _run_one(paper: DemoPaper) -> dict[str, Any]:
    parsed: ParsedPaper
    try:
        print(f"[{paper.slug}] Fetching arXiv {paper.arxiv_id}")
        payload = await fetch_arxiv_pdf_with_progress(
            paper.arxiv_id,
            max_size_mb=30,
            progress_cb=lambda msg: print(f"[{paper.slug}] {msg}"),
        )
        print(f"[{paper.slug}] Running ingestion")
        parsed = await run_ingestion(
            pdf_bytes=payload.pdf_bytes,
            title=payload.title,
            authors=payload.authors,
            abstract=payload.abstract,
            emit_thinking=lambda msg: print(f"[{paper.slug}] {msg}"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[{paper.slug}] arXiv fetch failed ({exc}). Using model-generated context fallback.")
        parsed = await _build_parsed_paper_from_model(paper)

    print(f"[{paper.slug}] Running comprehension")
    internal_rep = await run_comprehension(parsed)
    print(f"[{paper.slug}] Generating code snippets")
    code_snippets = await _generate_code_snippets(internal_rep)
    if not code_snippets:
        fallback_component = {
            "transformer": "Transformer Encoder Block",
            "resnet": "Residual Block",
            "bert": "BERT Encoder Layer",
        }.get(paper.slug, "Core Model Block")
        fallback_snippet = await generate_component_code(
            component_name=fallback_component,
            component_description=f"Fallback implementation snippet for {paper.paper_title}",
            relevant_sections=["model architecture"],
            relevant_equations=[],
            resolved_ambiguities={},
            reasoning_tier="flash",
        )
        code_snippets = [fallback_snippet]
    print(f"[{paper.slug}] Generating briefing sections")
    sections = await _generate_sections(parsed, internal_rep, code_snippets)
    print(f"[{paper.slug}] Writing artifacts")
    _write_artifacts(paper, sections, internal_rep, code_snippets)
    return _build_walkthrough_entry(paper, sections)


async def main() -> None:
    if not os.getenv("GEMINI_API_KEY", "").strip():
        raise RuntimeError("GEMINI_API_KEY is required.")

    requested_slugs = {
        slug.strip().lower()
        for slug in os.getenv("DEMO_SLUGS", "").split(",")
        if slug.strip()
    }
    selected_papers = [paper for paper in PAPERS if not requested_slugs or paper.slug in requested_slugs]
    if not selected_papers:
        valid = ", ".join(p.slug for p in PAPERS)
        raise RuntimeError(f"No matching papers for DEMO_SLUGS. Valid values: {valid}")

    entries: dict[str, Any] = _load_existing_entries()
    for paper in selected_papers:
        try:
            entries[paper.slug] = await _run_one(paper)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed generating walkthrough for {paper.slug}: {exc}") from exc

    print("[all] Updating frontend/lib/examples.ts")
    _write_examples_ts(entries)
    print("[all] Done")


if __name__ == "__main__":
    asyncio.run(main())
