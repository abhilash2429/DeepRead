from __future__ import annotations

from backend.prompts.figure import FIGURE_PROMPT
from backend.services.gemini_client import GeminiClient


async def describe_figure(gemini: GeminiClient, image_b64: str, caption: str | None) -> str:
    prompt = (
        f"{FIGURE_PROMPT}\n\nCaption:\n{caption or '(none)'}\n\n"
        "Use plain text and keep it implementation-focused."
    )
    # Fallback to caption-only text generation if multimodal parsing fails.
    try:
        response_text = await gemini.generate_multimodal_text(prompt=prompt, image_b64=image_b64)
        return response_text or "Figure description unavailable."
    except Exception:  # noqa: BLE001
        return await gemini.generate_text(
            prompt + "\n\nImage was provided but could not be processed; infer only from caption."
        )
