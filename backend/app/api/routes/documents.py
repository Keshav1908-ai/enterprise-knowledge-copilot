from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError, ValidationFailedError
from app.database.session import get_db
from app.retrieval.vector_store import get_vector_store
from app.schemas.documents import (
    ActionResponse,
    DocumentDetail,
    DocumentList,
    DocumentOut,
    UploadResponse,
    UploadResult,
)
from app.services.document_service import DocumentService

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

MAX_FILES_PER_REQUEST = 20


def _service(settings: Settings = Depends(get_settings)) -> DocumentService:
    return DocumentService(settings)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED,
             summary="Upload one or more documents (PDF, DOCX, TXT, MD)")
async def upload_documents(
    background: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Files to upload"),
    db: Session = Depends(get_db),
    service: DocumentService = Depends(_service),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """Validates each file, de-duplicates by SHA-256 content hash and queues indexing.

    Indexing runs asynchronously; poll ``GET /api/v1/documents/{id}`` for status.
    """
    if not files:
        raise ValidationFailedError("No files provided")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise ValidationFailedError(f"At most {MAX_FILES_PER_REQUEST} files per request")
    results: list[UploadResult] = []
    for f in files:
        # Read at most limit+1 bytes so oversized uploads are rejected without buffering everything.
        content = await f.read(settings.max_upload_bytes + 1)
        name = f.filename or "document"
        try:
            doc, duplicate = service.create_from_upload(db, name, content)
        except AppError as exc:
            results.append(UploadResult(filename=name, status="rejected", message=exc.message, error_code=exc.code))
            continue
        finally:
            await f.close()
        if duplicate:
            results.append(UploadResult(
                filename=name, status="duplicate", document=DocumentOut.model_validate(doc),
                message=f"Identical content already uploaded as '{doc.filename}' — not indexed again",
            ))
        else:
            background.add_task(service.process, doc.id)
            results.append(UploadResult(
                filename=name, status="accepted", document=DocumentOut.model_validate(doc),
                message="Upload accepted; indexing started",
            ))
    return UploadResponse(
        results=results,
        accepted=sum(r.status == "accepted" for r in results),
        duplicates=sum(r.status == "duplicate" for r in results),
        rejected=sum(r.status == "rejected" for r in results),
    )


@router.get("", response_model=DocumentList, summary="List documents")
def list_documents(
    status_filter: str | None = Query(default=None, alias="status", pattern="^(pending|processing|indexed|failed)$"),
    db: Session = Depends(get_db),
    service: DocumentService = Depends(_service),
) -> DocumentList:
    docs = service.list(db, status_filter)
    return DocumentList(items=[DocumentOut.model_validate(d) for d in docs], total=len(docs), counts=service.status_counts(db))


@router.get("/{document_id}", response_model=DocumentDetail, summary="Document details and processing status")
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    service: DocumentService = Depends(_service),
    settings: Settings = Depends(get_settings),
) -> DocumentDetail:
    doc = service.get(db, document_id)
    store = get_vector_store(settings)
    chunks = store.get_chunks({"document_id": doc.id})
    chunks.sort(key=lambda c: int(c.metadata.get("chunk_index", 0)))
    sample = [
        {"chunk_id": c.chunk_id, "page_number": c.page_number, "text": c.text[:300], "chars": len(c.text)}
        for c in chunks[:5]
    ]
    return DocumentDetail(
        **DocumentOut.model_validate(doc).model_dump(), chunks_in_vector_store=len(chunks), sample_chunks=sample
    )


@router.delete("/{document_id}", response_model=ActionResponse, summary="Delete a document and its vectors")
def delete_document(
    document_id: str, db: Session = Depends(get_db), service: DocumentService = Depends(_service)
) -> ActionResponse:
    service.delete(db, document_id)
    return ActionResponse(id=document_id, status="deleted", message="Document, file and vectors removed")


@router.post("/{document_id}/reindex", response_model=ActionResponse, status_code=status.HTTP_202_ACCEPTED,
             summary="Re-extract, re-chunk and re-embed a document")
def reindex_document(
    document_id: str,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    service: DocumentService = Depends(_service),
) -> ActionResponse:
    doc = service.mark_for_reindex(db, document_id)
    background.add_task(service.process, doc.id)
    return ActionResponse(id=doc.id, status=doc.status, message="Re-indexing started")
