"""HTTP API: every SRS endpoint, status codes, structured errors and validation."""

from __future__ import annotations

import time

from app.generation.llm import set_generator_override
from tests.helpers import MockLLM, make_pdf


def _wait_indexed(client, doc_id, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        doc = client.get(f"/api/v1/documents/{doc_id}").json()
        if doc["status"] in ("indexed", "failed"):
            return doc
        time.sleep(0.05)
    raise AssertionError("document was not processed in time")


def _upload(client, name, content, mime="text/plain"):
    return client.post("/api/v1/documents/upload", files=[("files", (name, content, mime))])


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["checks"]["database"] == "ok" and body["checks"]["vector_store"] == "ok"
    assert "openai_api_key" not in str(body).lower()


def test_upload_list_get_reindex_delete_flow(client, handbook_bytes):
    r = _upload(client, "employee_handbook.md", handbook_bytes, "text/markdown")
    assert r.status_code == 202
    res = r.json()["results"][0]
    assert res["status"] == "accepted"
    doc_id = res["document"]["id"]

    doc = _wait_indexed(client, doc_id)
    assert doc["status"] == "indexed" and doc["chunks_in_vector_store"] == doc["num_chunks"] > 0
    assert doc["sample_chunks"][0]["chunk_id"].startswith(doc_id)

    listing = client.get("/api/v1/documents").json()
    assert listing["total"] == 1 and listing["counts"]["indexed"] == 1
    assert client.get("/api/v1/documents?status=failed").json()["total"] == 0

    assert client.post(f"/api/v1/documents/{doc_id}/reindex").status_code == 202
    assert _wait_indexed(client, doc_id)["status"] == "indexed"

    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 200
    assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404


def test_upload_pdf_reports_pages(client):
    pdf = make_pdf(["Annual leave policy: employees get 24 days.", "Casual leave: 8 days per year."])
    doc_id = _upload(client, "policy.pdf", pdf, "application/pdf").json()["results"][0]["document"]["id"]
    doc = _wait_indexed(client, doc_id)
    assert doc["status"] == "indexed" and doc["num_pages"] == 2


def test_upload_rejections_and_duplicates(client, handbook_bytes):
    r = client.post("/api/v1/documents/upload", files=[
        ("files", ("a.md", handbook_bytes, "text/markdown")),
        ("files", ("b.md", handbook_bytes, "text/markdown")),
        ("files", ("virus.exe", b"MZ", "application/octet-stream")),
        ("files", ("fake.pdf", b"not a pdf", "application/pdf")),
    ]).json()
    assert (r["accepted"], r["duplicates"], r["rejected"]) == (1, 1, 2)
    codes = {x["filename"]: x.get("error_code") for x in r["results"]}
    assert codes["virus.exe"] == "unsupported_file" and codes["fake.pdf"] == "unsupported_file"


def test_upload_without_files_is_422(client):
    r = client.post("/api/v1/documents/upload")
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_failed"


def test_chat_and_conversation_endpoints(client, handbook_bytes):
    set_generator_override(MockLLM(answer="Casual leave is 8 days per year [1]."))
    doc_id = _upload(client, "h.md", handbook_bytes).json()["results"][0]["document"]["id"]
    _wait_indexed(client, doc_id)

    r = client.post("/api/v1/chat", json={"question": "How many casual leaves are allowed?"})
    assert r.status_code == 200
    body = r.json()
    for key in ("answer", "citations", "retrieved_chunks", "conversation_id", "latency", "token_usage"):
        assert key in body
    assert body["citations"][0]["document_name"] == "h.md"
    assert body["retrieved_chunks"][0]["score"] is not None
    assert body["latency"]["total_ms"] >= body["latency"]["retrieval_ms"]
    cid = body["conversation_id"]

    r2 = client.post("/api/v1/chat", json={"question": "And sick leave?", "conversation_id": cid})
    assert r2.json()["conversation_id"] == cid

    convs = client.get("/api/v1/conversations").json()
    assert convs[0]["id"] == cid and convs[0]["message_count"] == 4
    detail = client.get(f"/api/v1/conversations/{cid}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][1]["citations"]

    assert client.delete(f"/api/v1/conversations/{cid}").status_code == 200
    assert client.get(f"/api/v1/conversations/{cid}").status_code == 404


def test_chat_validation_and_unknown_conversation(client):
    assert client.post("/api/v1/chat", json={"question": "   "}).status_code == 422
    assert client.post("/api/v1/chat", json={"question": "x", "top_k": 0}).status_code == 422
    r = client.post("/api/v1/chat", json={"question": "hello", "conversation_id": "doesnotexist"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_chat_provider_failure_returns_structured_502(client, handbook_bytes):
    set_generator_override(MockLLM(fail=True))
    _wait_indexed(client, _upload(client, "h.md", handbook_bytes).json()["results"][0]["document"]["id"])
    r = client.post("/api/v1/chat", json={"question": "annual leave days"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "provider_error"
    assert client.get("/api/v1/metrics").json()["summary"]["failed_queries"] == 1


def test_metrics_dashboard_and_config(client):
    m = client.get("/api/v1/metrics?hours=24&bucket=day").json()
    assert {"summary", "timeseries", "score_histogram", "api", "ingestion"} <= set(m)
    d = client.get("/api/v1/dashboard").json()
    assert {"ingestion", "queries", "config"} <= set(d)
    assert client.get("/api/v1/metrics?source=bogus").status_code == 422
    assert client.get("/api/v1/config").json()["embedding_provider"] == "hash"


def test_evaluation_endpoints(client):
    r = client.post("/api/v1/evaluation/run", json={"label": "api", "max_questions": 4, "top_k": 3})
    assert r.status_code == 200
    run = r.json()
    assert run["status"] == "completed" and run["is_synthetic"] is True and run["num_questions"] == 4
    results = client.get("/api/v1/evaluation/results").json()
    assert results["latest"]["id"] == run["id"] and results["dataset"]["available"]
    assert client.get(f"/api/v1/evaluation/results/{run['id']}").json()["results"]
    assert client.get("/api/v1/evaluation/results/nope").status_code == 404


def test_openapi_docs_available(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/health", "/api/v1/documents/upload", "/api/v1/documents", "/api/v1/documents/{document_id}",
              "/api/v1/documents/{document_id}/reindex", "/api/v1/chat", "/api/v1/conversations",
              "/api/v1/conversations/{conversation_id}", "/api/v1/metrics", "/api/v1/evaluation/run",
              "/api/v1/evaluation/results"):
        assert p in paths, p
