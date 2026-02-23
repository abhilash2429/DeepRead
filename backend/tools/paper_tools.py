from __future__ import annotations

from langchain_core.tools import tool

from backend.models.paper import ElementType, ParsedPaper


def build_paper_tools(parsed_paper: ParsedPaper):
    @tool("paper_section_lookup")
    def paper_section_lookup(section_name: str) -> str:
        """Retrieve matching section text blocks from the parsed paper, including appendix/footnotes."""
        needle = section_name.strip().lower()
        matches = [
            f"[p.{el.page_number}] {el.content}"
            for el in parsed_paper.elements
            if el.content and needle in el.section_heading.lower()
        ]
        if not matches:
            matches = [
                f"[{el.section_heading} | p.{el.page_number}] {el.content}"
                for el in parsed_paper.elements
                if el.content and needle in el.content.lower()
            ]
        if not matches:
            return f"No section content matched '{section_name}'."
        return "\n".join(matches[:12])

    @tool("equation_decoder")
    def equation_decoder(equation_id: str) -> str:
        """Return equation text and contextual section for a given equation label."""
        target = equation_id.strip().lower()
        equations = [
            el for el in parsed_paper.elements if el.element_type == ElementType.EQUATION and (el.content or "")
        ]
        for eq in equations:
            label = (eq.equation_label or "").lower()
            if target and (target == label or target in (eq.content or "").lower()):
                return f"[{eq.section_heading} | p.{eq.page_number}] {eq.content}"
        if not equations:
            return "No equations were extracted from this paper."
        sample = equations[0]
        return f"Exact equation not found. Closest extracted equation: [{sample.section_heading} | p.{sample.page_number}] {sample.content}"

    @tool("figure_interpreter")
    def figure_interpreter(figure_ref: str) -> str:
        """Retrieve interpreted figure description by id/section/caption keyword."""
        needle = figure_ref.strip().lower()
        figures = [el for el in parsed_paper.elements if el.element_type == ElementType.FIGURE]
        for fig in figures:
            if needle in fig.id.lower():
                return fig.figure_description or fig.caption or "No figure interpretation available."
            caption = (fig.caption or "").lower()
            heading = (fig.section_heading or "").lower()
            if needle and (needle in caption or needle in heading):
                return fig.figure_description or fig.caption or "No figure interpretation available."
        if not figures:
            return "No figure elements were extracted."
        first = figures[0]
        return first.figure_description or first.caption or "No figure interpretation available."

    return [paper_section_lookup, equation_decoder, figure_interpreter]
