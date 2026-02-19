from __future__ import annotations

import asyncio
import html
import re
from dataclasses import dataclass
from typing import Callable

import arxiv
import httpx


@dataclass
class ArxivPaperPayload:
    title: str
    authors: list[str]
    abstract: str
    pdf_bytes: bytes


@dataclass
class ArxivMetadata:
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str | None = None


def normalize_arxiv_id(arxiv_ref: str) -> str:
    candidate = arxiv_ref.strip()
    m = re.search(r"arxiv\.org/(abs|pdf)/([^/?#]+)", candidate)
    if m:
        extracted = m.group(2).replace(".pdf", "")
        return re.sub(r"v\d+$", "", extracted)
    cleaned = candidate.replace("arXiv:", "").strip()
    return re.sub(r"v\d+$", "", cleaned)


async def fetch_arxiv_pdf(arxiv_ref: str, max_size_mb: int = 20) -> ArxivPaperPayload:
    return await fetch_arxiv_pdf_with_progress(arxiv_ref, max_size_mb=max_size_mb)


async def fetch_arxiv_pdf_with_progress(
    arxiv_ref: str,
    max_size_mb: int = 20,
    progress_cb: Callable[[str], None] | None = None,
) -> ArxivPaperPayload:
    arxiv_id = normalize_arxiv_id(arxiv_ref)
    if progress_cb:
        progress_cb(f"Resolved arXiv id: {arxiv_id}")
    metadata_errors: list[str] = []
    metadata: ArxivMetadata | None = None

    if progress_cb:
        progress_cb("Fetching arXiv metadata")
    try:
        paper = await asyncio.wait_for(asyncio.to_thread(_fetch_metadata, arxiv_id), timeout=30)
        if paper is None:
            raise ValueError(f"arXiv paper not found: {arxiv_id}")
        metadata = ArxivMetadata(
            title=paper.title,
            authors=[a.name for a in paper.authors],
            abstract=paper.summary,
            pdf_url=paper.pdf_url,
        )
    except Exception as exc:  # noqa: BLE001
        metadata_errors.append(_format_network_error("fetching arXiv metadata", exc))
        if progress_cb:
            progress_cb("Metadata fetch failed, trying abs-page metadata fallback")
        metadata = await _fetch_abs_page_metadata(arxiv_id, progress_cb)

    pdf_errors: list[str] = []
    pdf_sources: list[str] = []
    if metadata and metadata.pdf_url:
        pdf_sources.append(metadata.pdf_url)
    pdf_sources.extend(
        [
            f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            f"https://export.arxiv.org/pdf/{arxiv_id}.pdf",
        ]
    )

    pdf_bytes: bytes | None = None
    seen: set[str] = set()
    for url in pdf_sources:
        if url in seen:
            continue
        seen.add(url)
        if progress_cb:
            progress_cb(f"Downloading PDF from {url}")
        try:
            pdf_bytes = await _download_pdf_bytes(url)
            break
        except Exception as exc:  # noqa: BLE001
            pdf_errors.append(_format_network_error(f"downloading PDF ({url})", exc))
            if progress_cb:
                progress_cb(f"Failed: {url}")

    if pdf_bytes is None:
        hint = (
            "Unable to fetch PDF from arXiv sources. Please upload the PDF manually "
            "using /ingest/upload, or check firewall/VPN/proxy rules for arxiv.org/export.arxiv.org."
        )
        details = "\n".join(metadata_errors + pdf_errors)
        raise ConnectionError(f"{hint}\n{details}")

    if progress_cb:
        progress_cb(f"Download complete ({len(pdf_bytes) // 1024} KB)")

    if len(pdf_bytes) > max_size_mb * 1024 * 1024:
        raise ValueError(f"PDF exceeds MAX_PAPER_SIZE_MB={max_size_mb}")

    if metadata is None:
        metadata = ArxivMetadata(title=f"arXiv {arxiv_id}", authors=[], abstract="")

    return ArxivPaperPayload(
        title=metadata.title,
        authors=metadata.authors,
        abstract=metadata.abstract,
        pdf_bytes=pdf_bytes,
    )


def _fetch_metadata(arxiv_id: str):
    results = arxiv.Client().results(arxiv.Search(id_list=[arxiv_id], max_results=1))
    return next(results, None)


def _format_network_error(stage: str, exc: Exception) -> str:
    message = str(exc)
    if "WinError 10013" in message:
        return (
            f"Network blocked while {stage}. "
            "Check firewall/VPN/proxy settings and allow outbound HTTPS to export.arxiv.org and arxiv.org."
        )
    return f"Network error while {stage}: {message}"


async def _download_pdf_bytes(url: str) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0), follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
    except httpx.TimeoutException as exc:
        raise TimeoutError("Timed out while downloading PDF from arXiv") from exc


async def _fetch_abs_page_metadata(
    arxiv_id: str,
    progress_cb: Callable[[str], None] | None = None,
) -> ArxivMetadata | None:
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True) as client:
            resp = await client.get(abs_url)
            resp.raise_for_status()
            page = resp.text
    except Exception:  # noqa: BLE001
        return None

    title = _extract_first(page, r'<meta name="citation_title" content="([^"]+)"')
    abstract = _extract_first(page, r'<meta name="citation_abstract" content="([^"]+)"')
    authors = re.findall(r'<meta name="citation_author" content="([^"]+)"', page)
    if not title and not abstract and not authors:
        return None
    if progress_cb:
        progress_cb("Metadata fallback succeeded via abs page")
    return ArxivMetadata(
        title=html.unescape(title) if title else f"arXiv {arxiv_id}",
        authors=[html.unescape(a) for a in authors],
        abstract=html.unescape(abstract) if abstract else "",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    )


def _extract_first(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None
