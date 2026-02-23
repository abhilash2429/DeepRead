from __future__ import annotations

import base64
import io
import re
from typing import Iterable

import fitz
from PIL import Image

from backend.models.paper import ElementType, PaperElement, ParsedPaper


GREEK_CHARS = set(
    "".join(
        [
            "\u03b1\u03b2\u03b3\u03b4\u03b5\u03b6\u03b7\u03b8\u03b9\u03ba\u03bb\u03bc\u03bd\u03be\u03bf\u03c0\u03c1\u03c3\u03c4\u03c5\u03c6\u03c7\u03c8\u03c9",
            "\u0391\u0392\u0393\u0394\u0395\u0396\u0397\u0398\u0399\u039a\u039b\u039c\u039d\u039e\u039f\u03a0\u03a1\u03a3\u03a4\u03a5\u03a6\u03a7\u03a8\u03a9",
        ]
    )
)
MATH_TOKENS = ("sum", "prod", "argmax", "argmin", "softmax", "cross-entropy", "||", "=")


def _avg_font_size(lines: Iterable[dict]) -> float:
    sizes: list[float] = []
    for line in lines:
        for span in line.get("spans", []):
            size = float(span.get("size", 0.0))
            if size > 0:
                sizes.append(size)
    return sum(sizes) / len(sizes) if sizes else 0.0


def _looks_like_heading(text: str, avg_size: float, body_size: float, bold_ratio: float) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    if len(normalized) > 140:
        return False
    if normalized.endswith("."):
        return False
    if normalized.lower().startswith(("figure", "fig.", "table")):
        return False
    return avg_size >= max(12.0, body_size + 1.0) or bold_ratio >= 0.8


def _looks_like_equation(text: str) -> bool:
    normalized = text.strip()
    if re.search(r"\(\d+\)\s*$", normalized):
        return True
    greek_count = sum(ch in GREEK_CHARS for ch in normalized)
    math_token_hits = sum(token in normalized.lower() for token in MATH_TOKENS)
    return greek_count >= 2 or math_token_hits >= 2


def _looks_like_pseudocode(text: str, is_monospace: bool) -> bool:
    normalized = text.strip().lower()
    if "algorithm" in normalized:
        return True
    if is_monospace and len(normalized) > 20:
        return True
    return "for " in normalized and "end" in normalized and len(normalized) < 1500


def _looks_like_table(text: str) -> bool:
    normalized = text.strip()
    if normalized.lower().startswith("table"):
        return True
    if normalized.count("|") >= 2:
        return True
    return bool(re.search(r"\b(top-1|top-5|f1|bleu|rouge|accuracy|precision|recall)\b", normalized, re.IGNORECASE))


def _to_png_bytes(raw_bytes: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG")
            return output.getvalue()
    except Exception:
        return raw_bytes


def parse_pdf(pdf_bytes: bytes, title: str = "", authors: list[str] | None = None, abstract: str = "") -> ParsedPaper:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    elements: list[PaperElement] = []
    full_text_parts: list[str] = []
    current_heading = "Unknown Section"
    element_index = 0

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            text_dict = page.get_text("dict")
            blocks = sorted(
                text_dict.get("blocks", []),
                key=lambda block: (block.get("bbox", [0, 0, 0, 0])[1], block.get("number", 0)),
            )
            page_text_blocks: list[dict] = []
            caption_candidates: list[tuple[float, str]] = []

            for block in blocks:
                if block.get("type") != 0:
                    continue
                lines = block.get("lines", [])
                spans = [span for line in lines for span in line.get("spans", [])]
                text = " ".join(span.get("text", "").strip() for span in spans if span.get("text", "").strip()).strip()
                if not text:
                    continue

                avg_size = _avg_font_size(lines)
                bold_spans = [span for span in spans if (int(span.get("flags", 0)) & 16)]
                mono_spans = [span for span in spans if "mono" in str(span.get("font", "")).lower()]
                page_text_blocks.append(
                    {
                        "text": text,
                        "avg_size": avg_size,
                        "bold_ratio": len(bold_spans) / max(1, len(spans)),
                        "is_monospace": len(mono_spans) / max(1, len(spans)) > 0.45,
                    }
                )
                if re.match(r"^(figure|fig\.|table)\s*\d*", text, flags=re.IGNORECASE):
                    caption_candidates.append((float(block.get("bbox", [0, 0, 0, 0])[1]), text))

            sizes = sorted(entry["avg_size"] for entry in page_text_blocks if entry["avg_size"] > 0)
            body_size = sizes[len(sizes) // 2] if sizes else 10.5

            for entry in page_text_blocks:
                text = entry["text"]
                full_text_parts.append(text)
                element_type = ElementType.SECTION
                equation_label: str | None = None

                if _looks_like_heading(text, entry["avg_size"], body_size, entry["bold_ratio"]):
                    current_heading = text

                if _looks_like_equation(text):
                    element_type = ElementType.EQUATION
                    label_match = re.search(r"(\(\d+\))\s*$", text)
                    equation_label = label_match.group(1) if label_match else None
                elif _looks_like_pseudocode(text, entry["is_monospace"]):
                    element_type = ElementType.PSEUDOCODE
                elif _looks_like_table(text):
                    element_type = ElementType.TABLE

                elements.append(
                    PaperElement(
                        id=f"el-{element_index}",
                        element_type=element_type,
                        section_heading=current_heading,
                        page_number=page_index + 1,
                        content=text,
                        equation_label=equation_label,
                    )
                )
                element_index += 1

            for image_index, image_info in enumerate(page.get_images(full=True)):
                xref = image_info[0]
                extracted = doc.extract_image(xref)
                image_bytes = extracted.get("image", b"")
                if not image_bytes:
                    continue
                png_bytes = _to_png_bytes(image_bytes)
                default_caption = f"Figure on page {page_index + 1}, image {image_index + 1}"
                caption = caption_candidates[image_index][1] if image_index < len(caption_candidates) else default_caption
                elements.append(
                    PaperElement(
                        id=f"el-{element_index}",
                        element_type=ElementType.FIGURE,
                        section_heading=current_heading,
                        page_number=page_index + 1,
                        caption=caption,
                        image_bytes_b64=base64.b64encode(png_bytes).decode("ascii"),
                    )
                )
                element_index += 1
    finally:
        doc.close()

    return ParsedPaper(
        title=title or "Untitled",
        authors=authors or [],
        abstract=abstract,
        full_text="\n".join(full_text_parts),
        elements=elements,
    )

