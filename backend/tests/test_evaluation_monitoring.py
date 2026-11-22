"""Evaluation metric maths, the benchmark runner, and monitoring aggregates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.database.models import EvaluationRun
from app.evaluation import metrics as M
from app.evaluation.runner import EvaluationRunner, load_dataset
from app.monitoring.metrics_service import get_metrics, record_query
from app.services.chat_service import ChatService
from tests.helpers import MockLLM


# ---- metric maths ---------------------------------------------------------------
def test_precision_recall_mrr():
    retrieved = ["a", "x", "b", "y"]
    relevant = {"a", "b", "c"}
    assert M.precision_at_k(retrieved, relevant, 4) == 0.5
    assert M.precision_at_k(retrieved, relevant, 1) == 1.0
    assert M.recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)
    assert M.reciprocal_rank(["x", "y", "b"], relevant) == pytest.approx(1 / 3)
    assert M.reciprocal_rank(["x"], relevant) == 0.0
    assert M.hit_rate(["x", "a"], relevant, 1) == 0.0 and M.hit_rate(["x", "a"], relevant, 2) == 1.0
    assert M.precision_at_k([], relevant, 3) == 0.0 and M.recall_at_k(["a"], set(), 3) == 0.0


def test_generation_metrics():
    ctx = ["Employees receive 8 days of casual leave per calendar year."]
    good = "Employees receive 8 days of casual leave per year [1]."
    bad = "Employees receive unlimited vacation and stock options [1]. Casual leave is 8 days per year."
    assert M.lexical_faithfulness(good, ctx) == 1.0
    assert M.lexical_faithfulness(bad, ctx) == 0.5
    assert M.citation_coverage(bad, {1}) == 0.5
    assert M.citation_coverage("Fact one is stated here [3].", {1}) == 0.0
    assert M.token_f1("8 days of casual leave", "8 days of casual leave") == 1.0
    assert M.token_f1("completely different", "8 days") == 0.0
    assert M.is_abstention("I couldn't find information about this") and not M.is_abstention("It is 8 days.")
    assert M.lexical_faithfulness("I couldn't find information about this.", ctx) is None


def test_dataset_loads_and_is_labelled_synthetic(settings):
    data = load_dataset(settings.evaluation_dataset_path)
    assert data["synthetic"] is True and len(data["questions"]) >= 20
    assert any(not q["answerable"] for q in data["questions"])


def test_runner_produces_metrics_and_reports(db, settings):
    runner = EvaluationRunner(settings)
    run = runner.create_run(db, {"top_k": 3, "ingest_dataset_documents": True}, "test-run")
    runner.execute(run.id)
    db.expire_all()
    run = db.get(EvaluationRun, run.id)
    assert run.status == "completed", run.error
    r, g = run.metrics["retrieval"], run.metrics["generation"]
    for v in (r["precision_at_k"], r["recall_at_k"], r["mrr"], g["faithfulness"], g["citation_coverage"]):
        assert 0.0 <= v <= 1.0
    assert r["evaluated_questions"] == 20 and run.metrics["abstention"]["unanswerable_questions"] == 3
    assert run.metrics["dataset"]["synthetic"] is True
    report = Path(run.report_path)
    assert report.exists() and report.with_suffix(".md").exists()
    assert "Synthetic / demo evaluation" in report.with_suffix(".md").read_text()
    assert json.loads(report.read_text())["run_id"] == run.id
    assert len(run.results) == 23


def test_runner_compares_configurations(db, settings):
    runner = EvaluationRunner(settings)
    a = runner.create_run(db, {"retrieval_mode": "vector", "max_questions": 5}, "vector")
    runner.execute(a.id)
    b = runner.create_run(db, {"retrieval_mode": "hybrid", "max_questions": 5, "ingest_dataset_documents": False}, "hybrid")
    runner.execute(b.id)
    db.expire_all()
    ra, rb = db.get(EvaluationRun, a.id), db.get(EvaluationRun, b.id)
    assert ra.config["mode"] == "vector" and rb.config["mode"] == "hybrid"
    assert ra.status == rb.status == "completed"


# ---- monitoring ------------------------------------------------------------------
def test_monitoring_aggregates_real_query_rows(db, settings, indexed_corpus):
    service = ChatService(settings, MockLLM())
    for q in ("annual leave days", "password length", "meal allowance"):
        service.ask(db, q)
    record_query(db, query_id="x", conversation_id=None, source="chat", status="error", retrieval_ms=None,
                 generation_ms=None, total_ms=5.0, input_tokens=None, output_tokens=None, scores=[],
                 llm_provider="mock", llm_model="mock", retrieval_mode="vector", error="boom")
    db.commit()
    m = get_metrics(db)
    s = m["summary"]
    assert s["total_queries"] == 4 and s["failed_queries"] == 1 and s["error_rate"] == 0.25
    assert s["total_tokens"] == 360 and s["total_input_tokens"] == 300 and s["total_output_tokens"] == 60
    assert s["avg_retrieval_latency_ms"] > 0 and s["avg_generation_latency_ms"] == 12.5
    assert s["avg_retrieved_chunks"] > 0
    assert sum(b["queries"] for b in m["timeseries"]) == 4
    assert m["ingestion"]["indexed_documents"] == 4 and m["ingestion"]["total_chunks"] > 0
    assert len(m["recent_queries"]) == 4


def test_monitoring_separates_evaluation_traffic(db, settings, indexed_corpus):
    ChatService(settings, MockLLM()).ask(db, "annual leave", persist=False, source="evaluation")
    assert get_metrics(db, source="chat")["summary"]["total_queries"] == 0
    assert get_metrics(db, source="evaluation")["summary"]["total_queries"] == 1
