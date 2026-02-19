from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiClient:
    """LangChain-based Gemini wrapper used across the backend."""

    def __init__(self, model_name: str = "gemini-flash-latest") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
            top_p=0.9,
        )

    async def generate_text(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        content = getattr(response, "content", "")
        if isinstance(content, list):
            return "\n".join(str(part) for part in content if part)
        return str(content or "")

    async def stream_text(self, prompt: str):
        async for chunk in self.llm.astream(prompt):
            content = getattr(chunk, "content", "")
            if isinstance(content, list):
                for part in content:
                    if part:
                        yield str(part)
            elif content:
                yield str(content)

    async def generate_multimodal_text(self, prompt: str, image_b64: str) -> str:
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
        )
        response = await self.llm.ainvoke([message])
        content = getattr(response, "content", "")
        if isinstance(content, list):
            return "\n".join(str(part) for part in content if part)
        return str(content or "")

    @staticmethod
    def extract_json_candidate(text: str) -> str:
        stripped = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", stripped, flags=re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return stripped
        start = min([idx for idx in [stripped.find("{"), stripped.find("[")] if idx >= 0], default=-1)
        if start >= 0:
            return stripped[start:]
        return stripped

    async def generate_json(self, prompt: str) -> dict[str, Any]:
        text = await self.generate_text(prompt)
        try:
            parsed = json.loads(self.extract_json_candidate(text))
        except json.JSONDecodeError:
            repair_prompt = (
                "Return ONLY valid JSON from this content with no markdown fences.\n\n"
                f"{text}"
            )
            repaired = await self.generate_text(repair_prompt)
            parsed = json.loads(self.extract_json_candidate(repaired))
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object from model")
        return parsed
