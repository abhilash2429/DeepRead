from __future__ import annotations

import os
import re
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.models.artifacts import CodeSnippet
from backend.prompts.code_gen import CODE_GEN_PROMPT
from backend.utils.llm_retry import call_with_llm_retry


CODE_MODEL_FLASH = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)

CODE_MODEL_PRO = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)


def _extract_python(raw: str) -> str:
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw.strip()


async def generate_component_code(
    component_name: str,
    component_description: str,
    relevant_sections: list[str],
    relevant_equations: list[str],
    resolved_ambiguities: dict[str, str],
    reasoning_tier: Literal["flash", "pro"] = "flash",
) -> CodeSnippet:
    parser = PydanticOutputParser(pydantic_object=CodeSnippet)
    prompt = ChatPromptTemplate.from_template(
        "{rules}\n\n"
        "Generate a CodeSnippet JSON object.\n"
        "Component: {component_name}\n"
        "Description: {component_description}\n"
        "Relevant sections: {relevant_sections}\n"
        "Relevant equations: {relevant_equations}\n"
        "Resolved ambiguities: {resolved_ambiguities}\n\n"
        "{format_instructions}"
    )
    code_model = CODE_MODEL_PRO if reasoning_tier == "pro" else CODE_MODEL_FLASH
    chain = prompt | code_model | parser
    payload = {
        "rules": CODE_GEN_PROMPT,
        "component_name": component_name,
        "component_description": component_description,
        "relevant_sections": ", ".join(relevant_sections),
        "relevant_equations": ", ".join(relevant_equations),
        "resolved_ambiguities": resolved_ambiguities,
        "format_instructions": parser.get_format_instructions(),
    }
    try:
        snippet = await call_with_llm_retry(lambda: chain.ainvoke(payload))
    except Exception:
        fallback_name = re.sub(r"[^0-9a-zA-Z_]", "_", component_name.strip()) or "component"
        return CodeSnippet(
            component_name=component_name,
            code=(
                f"class {fallback_name}:\n"
                "    def __init__(self) -> None:\n"
                "        raise NotImplementedError('Code generation is temporarily unavailable.')\n"
            ),
            provenance="missing",
            assumption_notes=[
                "Automatic code generation failed due to a temporary model error.",
            ],
            source_sections=list(relevant_sections),
            equation_references=list(relevant_equations),
        )
    snippet.code = _extract_python(snippet.code)
    if not snippet.component_name:
        snippet.component_name = component_name
    if not snippet.source_sections:
        snippet.source_sections = relevant_sections
    if not snippet.equation_references:
        snippet.equation_references = relevant_equations
    return snippet
