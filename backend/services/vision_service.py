from __future__ import annotations

import os

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.prompts.figure import FIGURE_PROMPT


VISION_MODEL = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)


async def describe_figure(image_b64: str, caption: str | None) -> str:
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    f"{FIGURE_PROMPT}\n\n"
                    f"Caption:\n{caption or '(none)'}\n\n"
                    "Return implementation-focused plain text."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
        ]
    )
    try:
        response = await VISION_MODEL.ainvoke([message])
        return str(response.content or "").strip() or "Figure interpretation unavailable."
    except Exception:
        fallback = await VISION_MODEL.ainvoke(
            [
                HumanMessage(
                    content=(
                        f"{FIGURE_PROMPT}\n\n"
                        f"Caption:\n{caption or '(none)'}\n\n"
                        "Image could not be processed. Infer only from caption."
                    )
                )
            ]
        )
        return str(fallback.content or "").strip() or "Figure interpretation unavailable."
