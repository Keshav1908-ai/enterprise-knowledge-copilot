from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    file_size: int
    content_hash: str
    status: str
    error_message: str | None = None
    num_chunks: int
    num_pages: int | None = None
    num_characters: int
    processing_time_ms: float | None = None
    embedding_model: str | None = None
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None = None


class DocumentDetail(DocumentOut):
    chunks_in_vector_store: int
    sample_chunks: list[dict] = []


class UploadResult(BaseModel):
    filename: str
    status: str  # accepted | duplicate | rejected
    document: DocumentOut | None = None
    message: str
    error_code: str | None = None


class UploadResponse(BaseModel):
    results: list[UploadResult]
    accepted: int
    duplicates: int
    rejected: int


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
    counts: dict[str, int]


class ActionResponse(BaseModel):
    id: str
    status: str
    message: str
