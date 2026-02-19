from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.tools import tool


def build_code_tools(code_callback: Callable[[str], Awaitable[str]]):
    @tool("code_snippet_generator")
    async def code_snippet_generator(component_name: str) -> str:
        """Generate an annotated PyTorch snippet for a target component."""
        return await code_callback(component_name)

    return [code_snippet_generator]
