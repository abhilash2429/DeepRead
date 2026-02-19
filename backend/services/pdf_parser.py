from __future__ import annotations

import base64
import io
import re
from typing import Iterable

import fitz
from PIL import Image

from backend.models.paper import ElementType, PaperElement, ParsedPaper


GREEK_CHARS = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")


def _looks_like_heading(text: str, avg_size: float, body_size: float, bold_ratio: float) -> bool:
    t = text.strip()
    if not t or len(t) > 110 or t.endswith("."):
        return False
    if re.match(r"^(figure|fig\.|table)\b", t, flags=re.IGNORECASE):
        return False
    return avg_size >= max(12.0, body_size + 1.0) or bold_ratio >= 0.8


def _looks_like_equation(text: str) -> bool:
    t = text.strip()
    if re.search(r"\(\d+\)\s*$", t):
        return True
    greek_count = sum(ch in GREEK_CHARS for ch in t)
    equation_tokens = sum(tok in t for tok in ["∑", "∏", "∂", "argmax", "argmin", "||", "softmax", "="])
    return greek_count >= 2 or equation_tokens >= 2


def _looks_like_pseudocode(text: str, is_monospace: bool) -> bool:
    t = text.strip().lower()
    return "algorithm" in t or is_monospace or ("for " in t and "end" in t and len(t) < 1200)


def _looks_like_table(text: str) -> bool:
    t = text.strip()
    if t.lower().startswith("table"):
        return True
    if "|" in t and t.count("|") >= 2:
        return True
    return bool(re.search(r"\b(top-1|top-5|f1|bleu|rouge|accuracy|precision|recall)\b", t, flags=re.IGNORECASE))


def _avg_font_size(lines: Iterable[dict]) -> float:
    sizes: list[float] = []
    for line in lines:
        for span in line.get("spans", []):
            sizes.append(float(span.get("size", 0.0)))
    return sum(sizes) / len(sizes) if sizes else 0.0


def _to_png_bytes(raw_bytes: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            out = io.BytesIO()
            img.convert("RGB").save(out, format="PNG")
            return out.getvalue()
    except Exception:  # noqa: BLE001
        return raw_bytes


def parse_pdf(pdf_bytes: bytes, title: str = "", authors: list[str] | None = None, abstract: str = "") -> ParsedPaper:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    elements: list[PaperElement] = []
    full_text_parts: list[str] = []
    current_heading = "Unknown Section"
    el_idx = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text_dict = page.get_text("dict")
        blocks = sorted(text_dict.get("blocks", []), key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("number", 0)))
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
            bold_spans = [s for s in spans if (int(s.get("flags", 0)) & 16) or "bold" in str(s.get("font", "")).lower()]
            mono_spans = [s for s in spans if any(k in str(s.get("font", "")).lower() for k in ["mono", "courier", "code"])]
            page_text_blocks.append(
                {
                    "text": text,
                    "avg_size": avg_size,
                    "bold_ratio": len(bold_spans) / max(len(spans), 1),
                    "is_monospace": len(mono_spans) / max(len(spans), 1) > 0.4,
                }
            )
            if re.match(r"^(figure|fig\.|table)\s*\d*", text, flags=re.IGNORECASE):
                caption_candidates.append((float(block.get("bbox", [0, 0, 0, 0])[1]), text))

        body_size = 10.5
        if page_text_blocks:
            sorted_sizes = sorted(b["avg_size"] for b in page_text_blocks if b["avg_size"] > 0)
            if sorted_sizes:
                body_size = sorted_sizes[len(sorted_sizes) // 2]

        for block in page_text_blocks:
            text = block["text"]
            full_text_parts.append(text)
            el_type = ElementType.SECTION
            eq_label = None
            if _looks_like_heading(text, block["avg_size"], body_size, block["bold_ratio"]):
                current_heading = text
            if _looks_like_equation(text):
                el_type = ElementType.EQUATION
                match = re.search(r"(\(\d+\))\s*$", text)
                eq_label = match.group(1) if match else None
            elif _looks_like_pseudocode(text, block["is_monospace"]):
                el_type = ElementType.PSEUDOCODE
            elif _looks_like_table(text):
                el_type = ElementType.TABLE

            elements.append(
                PaperElement(
                    id=f"el-{el_idx}",
                    element_type=el_type,
                    section_heading=current_heading,
                    page_number=page_idx + 1,
                    content=text,
                    equation_label=eq_label,
                )
            )
            el_idx += 1

        for img_idx, image_info in enumerate(page.get_images(full=True)):
            xref = image_info[0]
            img = doc.extract_image(xref)
            img_bytes = img.get("image", b"")
            if not img_bytes:
                continue
            normalized = _to_png_bytes(img_bytes)
            fallback_caption = f"Figure on page {page_idx + 1}, image {img_idx + 1}"
            caption = caption_candidates[img_idx][1] if img_idx < len(caption_candidates) else fallback_caption
            elements.append(
                PaperElement(
                    id=f"el-{el_idx}",
                    element_type=ElementType.FIGURE,
                    section_heading=current_heading,
                    page_number=page_idx + 1,
                    content="",
                    caption=caption,
                    image_bytes_b64=base64.b64encode(normalized).decode("ascii"),
                )
            )
            el_idx += 1

    doc.close()
    return ParsedPaper(
        title=title or "Untitled",
        authors=authors or [],
        abstract=abstract,
        full_text="\n".join(full_text_parts),
        elements=elements,
    )
