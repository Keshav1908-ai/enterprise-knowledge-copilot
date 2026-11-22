"""Validation, extraction, cleaning, chunking and the ingestion pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.core.exceptions import ExtractionError, FileTooLargeError, UnsupportedFileError, ValidationFailedError
from app.database.models import Document
from app.ingestion.chunking import chunk_pages
from app.ingestion.extractors import PageText, extract
from app.ingestion.pipeline import prepare_chunks
from app.ingestion.preprocessing import clean_text
from app.ingestion.validation import sanitize_filename, validate_upload
from app.retrieval.vector_store import get_vector_store
from app.services.document_service import DocumentService
from tests.helpers import make_docx, make_pdf


# ---- validation --------------------------------------------------------------
@pytest.mark.parametrize("name", ["malware.exe", "data.csv", "noext", "image.png"])
def test_rejects_unsupported_extensions(settings, name):
    with pytest.raises(UnsupportedFileError):
        validate_upload(name, b"hello world", settings)


def test_rejects_fake_pdf(settings):
    with pytest.raises(UnsupportedFileError, match="not a valid PDF"):
        validate_upload("policy.pdf", b"this is plain text", settings)


def test_rejects_fake_docx(settings):
    with pytest.raises(UnsupportedFileError):
        validate_upload("policy.docx", b"PK\x03\x04garbage", settings)


def test_rejects_binary_text_file(settings):
    with pytest.raises(UnsupportedFileError):
        validate_upload("notes.txt", b"abc\x00\x01\x02", settings)


def test_rejects_empty_file(settings):
    with pytest.raises(ValidationFailedError):
        validate_upload("empty.txt", b"", settings)


def test_enforces_size_limit(settings, monkeypatch):
    monkeypatch.setattr(type(settings), "max_upload_bytes", property(lambda self: 100))
    with pytest.raises(FileTooLargeError):
        validate_upload("big.txt", b"a" * 101, settings)


def test_sanitizes_path_traversal():
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"
    assert sanitize_filename("C:\\evil\\..\\x.md").endswith("x.md")
    assert "/" not in sanitize_filename("a/b/c.pdf")


def test_accepts_valid_files_and_hashes(settings):
    vf = validate_upload("Policy.MD", b"# Title\nSome text", settings)
    assert vf.extension == ".md" and len(vf.content_hash) == 64


# ---- extraction --------------------------------------------------------------
def test_extract_pdf_preserves_page_numbers():
    pdf = make_pdf(["Annual leave is 24 days.", "Casual leave is 8 days."])
    pages = extract(".pdf", pdf)
    assert [p.page_number for p in pages] == [1, 2]
    assert "24 days" in pages[0].text and "8 days" in pages[1].text


def test_extract_docx_paragraphs_and_headings():
    data = make_docx(["Employees receive 12 days of sick leave."], heading="Leave Policy")
    pages = extract(".docx", data)
    assert len(pages) == 1 and pages[0].page_number is None
    assert "## Leave Policy" in pages[0].text and "12 days of sick leave" in pages[0].text


def test_extract_txt_and_md_handle_encodings():
    assert "café" in extract(".txt", "café menu".encode("utf-8"))[0].text
    assert "naïve" in extract(".md", "naïve".encode("latin-1"))[0].text


def test_extract_corrupt_pdf_raises():
    with pytest.raises(ExtractionError):
        extract(".pdf", b"%PDF-1.4 broken")


# ---- cleaning & chunking -----------------------------------------------------
def test_clean_text_normalises_artifacts():
    raw = "Infor-\nmation   with\t\tspaces\r\n\r\n\r\n\r\nNext \ufb01le\x07"
    out = clean_text(raw)
    assert "Information" in out and "with spaces" in out and "file" in out
    assert "\n\n\n" not in out and "\x07" not in out


def test_chunking_respects_size_overlap_and_metadata():
    text = " ".join(f"Sentence number {i} describes a policy detail." for i in range(120))
    chunks = chunk_pages([PageText(text, 3)], document_id="doc1", chunk_size=300, chunk_overlap=60,
                         base_metadata={"document_name": "x.pdf"})
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)
    assert chunks[0].chunk_id == "doc1-00000" and chunks[1].chunk_index == 1
    assert all(c.metadata["page_number"] == 3 and c.metadata["document_name"] == "x.pdf" for c in chunks)
    # overlap: the start of chunk 2 appears at the end of chunk 1
    assert chunks[1].text[:20] in chunks[0].text


def test_chunking_marks_missing_pages_with_minus_one():
    chunks = chunk_pages([PageText("A reasonably long paragraph of policy text here.", None)],
                         document_id="d", chunk_size=200, chunk_overlap=20)
    assert chunks[0].metadata["page_number"] == -1 and chunks[0].page_number is None


def test_chunk_size_is_configurable(settings, handbook_bytes):
    def count(size: int, overlap: int) -> int:
        settings.chunk_size, settings.chunk_overlap = size, overlap
        chunks, _, _ = prepare_chunks(document_id="d", filename="h.md", extension=".md", content=handbook_bytes,
                                      uploaded_at=datetime.now(timezone.utc), settings=settings)
        return len(chunks)

    assert count(300, 50) > count(1500, 100)


def test_prepare_chunks_rejects_textless_documents(settings):
    with pytest.raises(ExtractionError):
        prepare_chunks(document_id="d", filename="blank.pdf", extension=".pdf", content=make_pdf([""]),
                       uploaded_at=datetime.now(timezone.utc), settings=settings)


# ---- pipeline ----------------------------------------------------------------
def test_pipeline_indexes_and_updates_status(db, settings, handbook_bytes):
    service = DocumentService(settings)
    doc, dup = service.create_from_upload(db, "employee_handbook.md", handbook_bytes)
    assert not dup and doc.status == "pending"
    service.process(doc.id)
    db.expire_all()
    doc = db.get(Document, doc.id)
    assert doc.status == "indexed" and doc.num_chunks > 0 and doc.processing_time_ms is not None
    assert get_vector_store(settings).count_document(doc.id) == doc.num_chunks


def test_duplicate_upload_is_not_reindexed(db, settings, handbook_bytes):
    service = DocumentService(settings)
    first, _ = service.create_from_upload(db, "a.md", handbook_bytes)
    service.process(first.id)
    second, dup = service.create_from_upload(db, "renamed-copy.md", handbook_bytes)
    assert dup and second.id == first.id
    assert db.query(Document).count() == 1


def test_failed_ingestion_is_marked_failed_and_leaves_no_vectors(db, settings, monkeypatch, handbook_bytes):
    service = DocumentService(settings)
    doc, _ = service.create_from_upload(db, "h.md", handbook_bytes)

    def boom(*a, **k):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr("app.retrieval.vector_store.VectorStore.add_chunks", boom)
    service.process(doc.id)
    db.expire_all()
    doc = db.get(Document, doc.id)
    assert doc.status == "failed" and "embedding service down" in doc.error_message
    assert get_vector_store(settings).count_document(doc.id) == 0


def test_reindex_replaces_vectors_without_duplicates(db, settings, handbook_bytes):
    service = DocumentService(settings)
    doc, _ = service.create_from_upload(db, "h.md", handbook_bytes)
    service.process(doc.id)
    before = get_vector_store(settings).count_document(doc.id)
    service.mark_for_reindex(db, doc.id)
    service.process(doc.id)
    assert get_vector_store(settings).count_document(doc.id) == before


def test_delete_removes_vectors_file_and_row(db, settings, handbook_bytes):
    from pathlib import Path

    service = DocumentService(settings)
    doc, _ = service.create_from_upload(db, "h.md", handbook_bytes)
    service.process(doc.id)
    path = Path(doc.storage_path)
    service.delete(db, doc.id)
    assert get_vector_store(settings).count_document(doc.id) == 0
    assert not path.exists() and db.get(Document, doc.id) is None
