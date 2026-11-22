"""Shared fixtures. Every test gets an isolated data directory, offline embeddings and no paid API calls."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
PROJECT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

# Defaults for the whole session (individual tests may override via monkeypatch).
os.environ.setdefault("ENVIRONMENT", "test")
os.environ["EMBEDDING_PROVIDER"] = os.environ.get("TEST_EMBEDDING_PROVIDER", "hash")
os.environ["LLM_PROVIDER"] = "extractive"
os.environ["SIMILARITY_THRESHOLD"] = "0.05"
os.environ["LOG_LEVEL"] = "WARNING"
if os.environ.get("RUN_LIVE_TESTS") != "1":
    os.environ.pop("OPENAI_API_KEY", None)  # unit tests must never call a paid API

from app.core.config import get_settings  # noqa: E402
from app.database.session import init_db, reset_engine  # noqa: E402
from app.embeddings.providers import set_embeddings  # noqa: E402
from app.generation.llm import set_generator_override  # noqa: E402
from app.retrieval.retriever import invalidate_keyword_index  # noqa: E402
from app.retrieval.vector_store import reset_vector_store  # noqa: E402

DATASET_DOCS = PROJECT / "evaluation" / "dataset" / "documents"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch) -> Iterator[Path]:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EVALUATION_REPORTS_DIR", str(tmp_path / "reports"))
    get_settings.cache_clear()
    reset_engine()
    reset_vector_store()
    set_embeddings(None)
    set_generator_override(None)
    invalidate_keyword_index()
    init_db()
    yield tmp_path
    set_generator_override(None)
    reset_vector_store()
    reset_engine()
    get_settings.cache_clear()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def db():
    from app.database.session import get_sessionmaker

    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def handbook_bytes() -> bytes:
    return (DATASET_DOCS / "employee_handbook.md").read_bytes()


@pytest.fixture
def indexed_corpus(db, settings):
    """All four benchmark documents ingested synchronously. Returns {filename: document_id}."""
    from app.services.document_service import DocumentService

    service = DocumentService(settings)
    ids = {}
    for path in sorted(DATASET_DOCS.iterdir()):
        doc, _ = service.create_from_upload(db, path.name, path.read_bytes())
        service.process(doc.id)
        ids[path.name] = doc.id
    db.expire_all()
    return ids
