"""Chunking with LangChain's RecursiveCharacterTextSplitter, page-aware."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.extractors import PageText

_SEPARATORS = ["\n## ", "\n# ", "\n### ", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    chunk_index: int
    page_number: int | None
    metadata: dict = field(default_factory=dict)


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        keep_separator=True,
        strip_whitespace=True,
    )


def make_chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}-{index:05d}"


def chunk_pages(
    pages: list[PageText],
    *,
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
    base_metadata: dict | None = None,
    min_chunk_chars: int = 20,
) -> list[Chunk]:
    """Split each page separately so every chunk maps to exactly one page."""
    splitter = build_splitter(chunk_size, chunk_overlap)
    base = dict(base_metadata or {})
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if len(piece) < min_chunk_chars:
                continue
            idx = len(chunks)
            cid = make_chunk_id(document_id, idx)
            meta = {
                **base,
                "document_id": document_id,
                "chunk_id": cid,
                "chunk_index": idx,
                # Chroma metadata cannot hold None; -1 means "no page information"
                "page_number": page.page_number if page.page_number is not None else -1,
                "char_count": len(piece),
            }
            chunks.append(Chunk(chunk_id=cid, text=piece, chunk_index=idx, page_number=page.page_number, metadata=meta))
    return chunks
