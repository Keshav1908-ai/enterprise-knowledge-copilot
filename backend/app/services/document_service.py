"""Document lifecycle: upload, de-duplication, background processing, delete, re-index."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.database.models import Document, DocumentStatus, new_id
from app.database.session import session_scope
from app.ingestion.pipeline import ingest_document
from app.ingestion.validation import validate_upload
from app.retrieval.retriever import invalidate_keyword_index
from app.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)


class DocumentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ---- queries -----------------------------------------------------------
    def list(self, db: Session, status: str | None = None) -> list[Document]:
        stmt = select(Document).order_by(Document.created_at.desc())
        if status:
            stmt = stmt.where(Document.status == status)
        return list(db.scalars(stmt))

    def get(self, db: Session, document_id: str) -> Document:
        doc = db.get(Document, document_id)
        if doc is None:
            raise NotFoundError(f"Document '{document_id}' not found")
        return doc

    def status_counts(self, db: Session) -> dict[str, int]:
        rows = db.execute(select(Document.status, func.count()).group_by(Document.status)).all()
        counts = {s.value: 0 for s in DocumentStatus}
        counts.update({status: n for status, n in rows})
        counts["total"] = sum(n for _, n in rows)
        return counts

    # ---- commands ----------------------------------------------------------
    def create_from_upload(self, db: Session, filename: str, content: bytes) -> tuple[Document, bool]:
        """Validate and persist an upload. Returns (document, is_duplicate)."""
        vf = validate_upload(filename, content, self.settings)
        existing = db.scalar(select(Document).where(Document.content_hash == vf.content_hash))
        if existing is not None:
            if existing.status != DocumentStatus.FAILED.value:
                return existing, True
            # A previously failed upload of the same bytes is retried in place.
            existing.status = DocumentStatus.PENDING.value
            existing.error_message = None
            existing.filename = vf.filename
            db.commit()
            return existing, False

        doc_id = new_id()
        upload_dir = Path(self.settings.upload_dir or "")
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / f"{doc_id}{vf.extension}"
        path.write_bytes(vf.content)

        doc = Document(
            id=doc_id,
            filename=vf.filename,
            file_type=vf.extension.lstrip("."),
            content_hash=vf.content_hash,
            file_size=vf.size,
            storage_path=str(path),
            status=DocumentStatus.PENDING.value,
            embedding_model=self.settings.embedding_model,
            collection=self.settings.collection_name,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc, False

    def mark_for_reindex(self, db: Session, document_id: str) -> Document:
        doc = self.get(db, document_id)
        if doc.status == DocumentStatus.PROCESSING.value:
            raise ConflictError("Document is already being processed")
        if not Path(doc.storage_path).exists():
            raise NotFoundError("Original file is missing from storage; please upload it again")
        doc.status = DocumentStatus.PENDING.value
        doc.error_message = None
        db.commit()
        return doc

    def delete(self, db: Session, document_id: str) -> None:
        doc = self.get(db, document_id)
        get_vector_store(self.settings).delete_document(doc.id)
        invalidate_keyword_index()
        try:
            Path(doc.storage_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove stored file for document %s", doc.id)
        db.delete(doc)
        db.commit()

    # ---- background processing --------------------------------------------
    def process(self, document_id: str) -> None:
        """Run the ingestion pipeline. Designed for FastAPI BackgroundTasks (own DB session)."""
        with session_scope() as db:
            doc = db.get(Document, document_id)
            if doc is None:
                return
            doc.status = DocumentStatus.PROCESSING.value
            doc.error_message = None
            filename, storage_path, created = doc.filename, doc.storage_path, doc.created_at
        try:
            stats = ingest_document(
                document_id=document_id,
                filename=filename,
                storage_path=storage_path,
                uploaded_at=created if created.tzinfo else created.replace(tzinfo=timezone.utc),
                settings=self.settings,
            )
        except Exception as exc:  # noqa: BLE001
            message = getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}"
            logger.warning("Ingestion failed for %s: %s", document_id, message)
            try:
                get_vector_store(self.settings).delete_document(document_id)
            except Exception:  # noqa: BLE001
                logger.exception("Cleanup after failed ingestion also failed")
            with session_scope() as db:
                doc = db.get(Document, document_id)
                if doc is not None:
                    doc.status = DocumentStatus.FAILED.value
                    doc.error_message = message[:2000]
                    doc.num_chunks = 0
            return

        with session_scope() as db:
            doc = db.get(Document, document_id)
            if doc is None:  # deleted while processing
                get_vector_store(self.settings).delete_document(document_id)
                return
            doc.status = DocumentStatus.INDEXED.value
            doc.num_chunks = stats.num_chunks
            doc.num_pages = stats.num_pages
            doc.num_characters = stats.num_characters
            doc.processing_time_ms = stats.total_ms
            doc.embedding_model = self.settings.embedding_model
            doc.collection = self.settings.collection_name
            doc.indexed_at = datetime.now(timezone.utc)

    def recover_interrupted(self, db: Session) -> list[str]:
        """Documents left 'processing'/'pending' by a crash or restart are re-queued."""
        stuck = list(
            db.scalars(
                select(Document).where(
                    Document.status.in_([DocumentStatus.PROCESSING.value, DocumentStatus.PENDING.value])
                )
            )
        )
        for d in stuck:
            d.status = DocumentStatus.PENDING.value
        db.commit()
        return [d.id for d in stuck]
