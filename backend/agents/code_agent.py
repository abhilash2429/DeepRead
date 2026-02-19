from __future__ import annotations

import json
import re

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.models.artifacts import CodeSnippet
from backend.prompts.code_gen import CODE_GEN_PROMPT
from backend.services.gemini_client import GeminiClient


def _extract_python_block(raw: str) -> str:
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw.strip()


async def generate_component_code(
    gemini: GeminiClient,
    component_name: str,
    component_description: str,
    relevant_sections: list[str],
    relevant_equations: list[str],
    resolved_ambiguities: dict[str, str],
) -> CodeSnippet:
    parser = PydanticOutputParser(pydantic_object=CodeSnippet)
    prompt = ChatPromptTemplate.from_template(
        "{code_rules}\n\n"
        "Generate a CodeSnippet JSON object for this component.\n"
        "Component: {component_name}\n"
        "Description: {component_description}\n"
        "Relevant sections: {relevant_sections}\n"
        "Relevant equations: {relevant_equations}\n"
        "Resolved ambiguities: {resolved_ambiguities}\n\n"
        "{format_instructions}"
    )
    chain = prompt | gemini.llm | parser
    try:
        snippet = await chain.ainvoke(
            {
                "code_rules": CODE_GEN_PROMPT,
                "component_name": component_name,
                "component_description": component_description,
                "relevant_sections": relevant_sections,
                "relevant_equations": relevant_equations,
                "resolved_ambiguities": json.dumps(resolved_ambiguities),
                "format_instructions": parser.get_format_instructions(),
            }
        )
        snippet.code = _extract_python_block(snippet.code)
        if not snippet.component_name:
            snippet.component_name = component_name
        if not snippet.source_sections:
            snippet.source_sections = relevant_sections
        return snippet
    except Exception:
        fallback_prompt = (
            f"{CODE_GEN_PROMPT}\n\n"
            f"Component: {component_name}\n"
            f"Description: {component_description}\n"
            f"Sections: {relevant_sections}\n"
            f"Equations: {relevant_equations}\n"
            f"Resolved ambiguities: {resolved_ambiguities}\n"
        )
        code = _extract_python_block(await gemini.generate_text(fallback_prompt))
        notes = [line.strip() for line in code.splitlines() if "# ASSUMED:" in line or "# INFERRED:" in line]
        provenance = "paper-stated"
        lowered = code.lower()
        if "# assumed:" in lowered:
            provenance = "assumed"
        elif "# inferred:" in lowered:
            provenance = "inferred"
        return CodeSnippet(
            component_name=component_name,
            code=code,
            provenance=provenance,  # type: ignore[arg-type]
            assumption_notes=notes,
            source_sections=relevant_sections,
        )
