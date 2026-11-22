"""Live integration test against the providers configured in .env.

Skipped by default. Run with real providers (may incur API cost):

    RUN_LIVE_TESTS=1 pytest tests/test_integration_live.py -s

It uses whatever EMBEDDING_PROVIDER / LLM_PROVIDER / OPENAI_API_KEY are set in the environment.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to run")


@pytest.fixture
def live_settings(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("EMBEDDING_PROVIDER", os.environ.get("LIVE_EMBEDDING_PROVIDER", "sentence-transformers"))
    monkeypatch.setenv("LLM_PROVIDER", os.environ.get("LIVE_LLM_PROVIDER", "openai"))
    monkeypatch.setenv("SIMILARITY_THRESHOLD", os.environ.get("LIVE_SIMILARITY_THRESHOLD", "0.25"))
    get_settings.cache_clear()
    return get_settings()


def test_live_end_to_end(db, live_settings, handbook_bytes):
    from app.services.chat_service import ChatService
    from app.services.document_service import DocumentService

    docs = DocumentService(live_settings)
    doc, _ = docs.create_from_upload(db, "employee_handbook.md", handbook_bytes)
    docs.process(doc.id)
    db.expire_all()
    assert db.get(type(doc), doc.id).status == "indexed"

    res = ChatService(live_settings).ask(db, "How many casual leaves are allowed per year?")
    print("\nANSWER:", res.answer, "\nCITATIONS:", res.citations, "\nTOKENS:", res.token_usage, "\nLATENCY:", res.latency)
    assert res.status == "success"
    assert "8" in res.answer
    assert res.citations and res.citations[0]["document_name"] == "employee_handbook.md"
    if live_settings.llm_provider == "openai":
        assert res.token_usage["total_tokens"]

    unknown = ChatService(live_settings).ask(db, "What is the CEO's favourite colour?")
    assert unknown.grounding["abstained"] or unknown.status == "no_context"
