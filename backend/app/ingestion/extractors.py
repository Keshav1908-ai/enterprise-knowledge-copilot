"""Text extraction for PDF, DOCX, TXT and Markdown.

Each extractor returns a list of :class:`PageText` so page numbers survive all the way
into chunk metadata and, ultimately, citations.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass

from app.core.exceptions import ExtractionError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PageText:
    text: str
    page_number: int | None = None  # 1-based; None when the format has no pages


def extract_pdf(content: bytes) -> list[PageText]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise ExtractionError("PDF is password protected") from exc
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - a single bad page should not fail the document
                logger.warning("Failed to extract text from PDF page %d", i)
                text = ""
            pages.append(PageText(text=text, page_number=i))
        return pages
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Could not read PDF: {type(exc).__name__}") from exc


def extract_docx(content: bytes) -> list[PageText]:
    import docx

    try:
        document = docx.Document(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Could not read DOCX: {type(exc).__name__}") from exc

    parts: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style is not None else ""
        if style.startswith("heading"):
            parts.append(f"\n## {text}\n")
        elif "list" in style:
            parts.append(f"- {text}")
        else:
            parts.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return [PageText(text="\n".join(parts), page_number=None)]


def _decode(content: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ExtractionError("Unable to decode text file")  # pragma: no cover


def extract_text(content: bytes) -> list[PageText]:
    return [PageText(text=_decode(content), page_number=None)]


EXTRACTORS: dict[str, Callable[[bytes], list[PageText]]] = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".txt": extract_text,
    ".md": extract_text,
}


def extract(extension: str, content: bytes) -> list[PageText]:
    try:
        extractor = EXTRACTORS[extension.lower()]
    except KeyError as exc:
        raise ExtractionError(f"No extractor for '{extension}'") from exc
    return extractor(content)
