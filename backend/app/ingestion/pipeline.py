"""End-to-end ingestion: extract -> clean -> chunk -> embed -> index.

Failure semantics: vectors for a document are only considered valid once the database
row is marked ``indexed``. Any exception removes partially written vectors and marks
the document ``failed`` with the error message.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import ExtractionError
from app.core.logging import get_logger
from app.embeddings.providers import get_embeddings
from app.ingestion.chunking import Chunk, chunk_pages
from app.ingestion.extractors import PageText, extract
from app.ingestion.preprocessing import clean_text
from app.retrieval.retriever import invalidate_keyword_index
from app.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)


@dataclass
class IngestionStats:
    num_pages: int | None
    num_characters: int
    num_chunks: int
    extraction_ms: float
    chunking_ms: float
    embedding_ms: float
    indexing_ms: float
    total_ms: float

    def as_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def prepare_chunks(
    *,
    document_id: str,
    filename: str,
    extension: str,
    content: bytes,
    uploaded_at: datetime,
    settings: Settings,
) -> tuple[list[Chunk], list[PageText], dict]:
    timings: dict[str, float] = {}
    t = time.perf_counter()
    raw_pages = extract(extension, content)
    pages = [PageText(text=clean_text(p.text), page_number=p.page_number) for p in raw_pages]
    timings["extraction_ms"] = (time.perf_counter() - t) * 1000
    if not any(p.text.strip() for p in pages):
        raise ExtractionError(
            "No extractable text found. Scanned/image-only PDFs need OCR, which is not enabled in v1.0."
        )
    t = time.perf_counter()
    chunks = chunk_pages(
        pages,
        document_id=document_id,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        base_metadata={
            "document_name": filename,
            "source_file": filename,
            "file_type": extension.lstrip("."),
            "upload_timestamp": uploaded_at.astimezone(timezone.utc).isoformat(),
        },
    )
    timings["chunking_ms"] = (time.perf_counter() - t) * 1000
    if not chunks:
        raise ExtractionError("Document produced no usable text chunks")
    return chunks, pages, timings


def ingest_document(
    *,
    document_id: str,
    filename: str,
    storage_path: str,
    uploaded_at: datetime,
    settings: Settings,
) -> IngestionStats:
    t0 = time.perf_counter()
    path = Path(storage_path)
    extension = path.suffix.lower()
    content = path.read_bytes()

    chunks, pages, timings = prepare_chunks(
        document_id=document_id,
        filename=filename,
        extension=extension,
        content=content,
        uploaded_at=uploaded_at,
        settings=settings,
    )

    embeddings = get_embeddings(settings)
    t = time.perf_counter()
    vectors: list[list[float]] = []
    bs = settings.embedding_batch_size
    for i in range(0, len(chunks), bs):
        vectors.extend(embeddings.embed_documents([c.text for c in chunks[i : i + bs]]))
    embedding_ms = (time.perf_counter() - t) * 1000

    store = get_vector_store(settings)
    t = time.perf_counter()
    store.delete_document(document_id)  # idempotent re-index
    try:
        store.add_chunks(chunks, vectors)
    except Exception:
        store.delete_document(document_id)
        raise
    finally:
        invalidate_keyword_index()
    indexing_ms = (time.perf_counter() - t) * 1000

    has_pages = any(p.page_number is not None for p in pages)
    stats = IngestionStats(
        num_pages=len(pages) if has_pages else None,
        num_characters=sum(len(p.text) for p in pages),
        num_chunks=len(chunks),
        extraction_ms=timings["extraction_ms"],
        chunking_ms=timings["chunking_ms"],
        embedding_ms=embedding_ms,
        indexing_ms=indexing_ms,
        total_ms=(time.perf_counter() - t0) * 1000,
    )
    logger.info("Indexed document=%s chunks=%d in %.0f ms", document_id, stats.num_chunks, stats.total_ms)
    return stats
