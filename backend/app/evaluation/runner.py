"""Benchmark runner: executes the dataset through the real RAG pipeline and scores it."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import NotFoundError, ValidationFailedError
from app.core.logging import get_logger
from app.database.models import Document, DocumentStatus, EvaluationRun
from app.database.session import session_scope
from app.embeddings.providers import get_embeddings
from app.evaluation import metrics as M
from app.evaluation.judges import HeuristicJudge, LLMJudge
from app.generation.llm import get_generator
from app.retrieval.retriever import RetrievalParams
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService

logger = get_logger(__name__)


def load_dataset(path: Path) -> dict:
    if not path.exists():
        raise NotFoundError(f"Evaluation dataset not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions") or []
    if not questions:
        raise ValidationFailedError("Evaluation dataset has no questions")
    for i, q in enumerate(questions):
        if not q.get("question"):
            raise ValidationFailedError(f"Question #{i} is missing 'question'")
        q.setdefault("id", f"q{i + 1:03d}")
        q.setdefault("answerable", True)
        q.setdefault("relevant_documents", [])
        q.setdefault("expected_answer", "")
    data.setdefault("name", path.stem)
    data.setdefault("synthetic", True)
    return data


def dataset_summary(settings: Settings) -> dict:
    try:
        data = load_dataset(settings.evaluation_dataset_path)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(getattr(exc, "message", exc))}
    docs_dir = settings.evaluation_dataset_path.parent / data.get("documents_dir", "documents")
    return {
        "available": True,
        "name": data["name"],
        "synthetic": bool(data["synthetic"]),
        "description": data.get("description"),
        "num_questions": len(data["questions"]),
        "num_answerable": sum(1 for q in data["questions"] if q["answerable"]),
        "documents": sorted(p.name for p in docs_dir.glob("*") if p.is_file()) if docs_dir.exists() else [],
    }


class EvaluationRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ---- setup -------------------------------------------------------------
    def ensure_documents(self, db: Session, data: dict) -> list[str]:
        """Index any benchmark source documents that are not yet in the knowledge base."""
        docs_dir = self.settings.evaluation_dataset_path.parent / data.get("documents_dir", "documents")
        if not docs_dir.exists():
            return []
        service = DocumentService(self.settings)
        ingested = []
        for path in sorted(docs_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in self.settings.allowed_extension_set:
                continue
            doc, duplicate = service.create_from_upload(db, path.name, path.read_bytes())
            if not duplicate or doc.status != DocumentStatus.INDEXED.value:
                service.process(doc.id)
                ingested.append(path.name)
        db.expire_all()
        return ingested

    def _doc_name_map(self, db: Session) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = {}
        for d in db.scalars(select(Document).where(Document.status == DocumentStatus.INDEXED.value)):
            mapping.setdefault(d.filename.lower(), set()).add(d.id)
        return mapping

    # ---- run ---------------------------------------------------------------
    def create_run(self, db: Session, overrides: dict, label: str | None) -> EvaluationRun:
        data = load_dataset(self.settings.evaluation_dataset_path)
        judge = overrides.get("judge") or self.settings.evaluation_judge
        params = RetrievalParams.from_settings(
            self.settings,
            top_k=overrides.get("top_k"),
            similarity_threshold=overrides.get("similarity_threshold"),
            mode=overrides.get("retrieval_mode"),
            reranker=overrides.get("reranker"),
        )
        run = EvaluationRun(
            label=label,
            status="running",
            dataset_name=data["name"],
            is_synthetic=bool(data["synthetic"]),
            judge=judge,
            config={
                **params.as_dict(),
                "embedding_provider": self.settings.embedding_provider,
                "embedding_model": self.settings.embedding_model,
                "llm_provider": self.settings.llm_provider,
                "llm_model": self.settings.llm_model if self.settings.llm_provider == "openai" else "extractive",
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "max_questions": overrides.get("max_questions"),
                "ingest_dataset_documents": overrides.get("ingest_dataset_documents", True),
            },
            num_questions=0,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run

    def execute(self, run_id: str) -> None:
        t0 = time.perf_counter()
        with session_scope() as db:
            run = db.get(EvaluationRun, run_id)
            if run is None:
                return
            cfg = dict(run.config)
            judge_name = run.judge
        try:
            with session_scope() as db:
                result = self._run(db, cfg, judge_name)
            with session_scope() as db:
                run = db.get(EvaluationRun, run_id)
                assert run is not None
                run.status = "completed"
                run.num_questions = len(result["results"])
                run.metrics = result["metrics"]
                run.results = result["results"]
                run.duration_ms = (time.perf_counter() - t0) * 1000
                run.completed_at = datetime.now(timezone.utc)
                run.report_path = self._write_report(run, result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Evaluation run %s failed", run_id)
            with session_scope() as db:
                run = db.get(EvaluationRun, run_id)
                if run is not None:
                    run.status = "failed"
                    run.error = f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"[:2000]
                    run.duration_ms = (time.perf_counter() - t0) * 1000
                    run.completed_at = datetime.now(timezone.utc)

    def _run(self, db: Session, cfg: dict, judge_name: str) -> dict:
        data = load_dataset(self.settings.evaluation_dataset_path)
        ingested = self.ensure_documents(db, data) if cfg.get("ingest_dataset_documents", True) else []
        name_map = self._doc_name_map(db)
        questions = data["questions"][: cfg.get("max_questions") or None]
        k = int(cfg["top_k"])

        embeddings = get_embeddings(self.settings)
        generator = get_generator(self.settings)
        judge_used = judge_name
        if judge_name == "llm" and generator.provider == "openai":
            judge = LLMJudge(embeddings, generator)
        else:
            judge = HeuristicJudge(embeddings)
            if judge_name == "llm":
                judge_used = "heuristic (no LLM configured)"
        chat = ChatService(self.settings, generator)

        results = []
        for q in questions:
            relevant_ids: set[str] = set()
            missing_docs = []
            for name in q["relevant_documents"]:
                ids = name_map.get(name.lower())
                if ids:
                    relevant_ids |= ids
                else:
                    missing_docs.append(name)
            item: dict = {
                "id": q["id"],
                "question": q["question"],
                "category": q.get("category"),
                "answerable": bool(q["answerable"]),
                "expected_answer": q["expected_answer"],
                "relevant_documents": q["relevant_documents"],
                "missing_relevant_documents": missing_docs,
            }
            try:
                res = chat.ask(
                    db,
                    q["question"],
                    top_k=k,
                    similarity_threshold=cfg.get("similarity_threshold"),
                    retrieval_mode=cfg.get("mode"),
                    reranker=cfg.get("reranker"),
                    persist=False,
                    source="evaluation",
                )
            except Exception as exc:  # noqa: BLE001
                item.update({"error": f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"})
                results.append(item)
                continue

            retrieved_docs = [c["document_id"] for c in res.retrieved_chunks]
            unique_docs = list(dict.fromkeys(retrieved_docs))
            contexts = [c["text"] for c in res.retrieved_chunks]
            item.update(
                {
                    "answer": res.answer,
                    "status": res.status,
                    "retrieved": [
                        {"document_name": c["document_name"], "page_number": c["page_number"], "score": c["score"],
                         "chunk_id": c["chunk_id"], "relevant": c["document_id"] in relevant_ids}
                        for c in res.retrieved_chunks
                    ],
                    "citations": [{"source_id": c["source_id"], "document_name": c["document_name"],
                                   "page_number": c["page_number"]} for c in res.citations],
                    "latency": res.latency,
                    "token_usage": res.token_usage,
                    "abstained": M.is_abstention(res.answer),
                }
            )
            if q["answerable"] and relevant_ids:
                item["retrieval"] = {
                    "precision_at_k": M.precision_at_k(retrieved_docs, relevant_ids, k),
                    "recall_at_k": M.recall_at_k(unique_docs, relevant_ids, k),
                    "reciprocal_rank": M.reciprocal_rank(retrieved_docs, relevant_ids),
                    "hit": M.hit_rate(retrieved_docs, relevant_ids, k),
                }
            if q["answerable"]:
                scores = judge.score(q["question"], res.answer, contexts)
                scores["citation_coverage"] = M.citation_coverage(
                    res.answer, {c["source_id"] for c in res.retrieved_chunks}
                )
                scores["answer_f1"] = M.token_f1(res.answer, q["expected_answer"]) if q["expected_answer"] else None
                item["generation"] = scores
            else:
                item["abstention_correct"] = item["abstained"] or res.status == "no_context"
            results.append(item)

        return {"metrics": self._aggregate(results, k, judge_used, ingested, data), "results": results}

    @staticmethod
    def _aggregate(results: list[dict], k: int, judge_used: str, ingested: list[str], data: dict) -> dict:
        retr = [r["retrieval"] for r in results if "retrieval" in r]
        gen = [r["generation"] for r in results if "generation" in r]
        unans = [r for r in results if not r["answerable"] and "error" not in r]
        lat = [r["latency"] for r in results if "latency" in r]
        toks = [r["token_usage"]["total_tokens"] for r in results if r.get("token_usage", {}).get("total_tokens") is not None]

        def avg(key: str, rows: list[dict]) -> float | None:
            v = M.mean([row.get(key) for row in rows])
            return None if v is None else round(v, 4)

        return {
            "k": k,
            "retrieval": {
                "precision_at_k": avg("precision_at_k", retr),
                "recall_at_k": avg("recall_at_k", retr),
                "mrr": avg("reciprocal_rank", retr),
                "hit_rate": avg("hit", retr),
                "avg_retrieval_latency_ms": avg("retrieval_ms", lat),
                "evaluated_questions": len(retr),
                "method": "deterministic, ground truth = labelled relevant documents",
            },
            "generation": {
                "faithfulness": avg("faithfulness", gen),
                "answer_relevance": avg("answer_relevance", gen),
                "context_relevance": avg("context_relevance", gen),
                "citation_coverage": avg("citation_coverage", gen),
                "answer_f1": avg("answer_f1", gen),
                "evaluated_questions": len(gen),
                "judge": judge_used,
                "method": "automated estimates — not human-verified ground truth",
            },
            "abstention": {
                "unanswerable_questions": len(unans),
                "correct_abstentions": sum(1 for r in unans if r.get("abstention_correct")),
                "abstention_accuracy": round(sum(1 for r in unans if r.get("abstention_correct")) / len(unans), 4) if unans else None,
            },
            "performance": {
                "avg_total_latency_ms": avg("total_ms", lat),
                "avg_generation_latency_ms": avg("generation_ms", lat),
                "total_tokens": sum(toks) if toks else None,
            },
            "errors": sum(1 for r in results if "error" in r),
            "dataset": {"name": data["name"], "synthetic": bool(data["synthetic"]), "documents_ingested_for_run": ingested},
        }

    def _write_report(self, run: EvaluationRun, result: dict) -> str:
        out_dir = Path(self.settings.evaluation_reports_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = out_dir / f"eval_{stamp}_{run.id[:8]}"
        payload = {
            "run_id": run.id,
            "label": run.label,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "dataset": run.dataset_name,
            "synthetic": run.is_synthetic,
            "config": run.config,
            "metrics": result["metrics"],
            "results": result["results"],
        }
        base.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        base.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
        return str(base.with_suffix(".json"))


def _fmt(v) -> str:  # type: ignore[no-untyped-def]
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def render_markdown(p: dict) -> str:
    m = p["metrics"]
    r, g, a, perf = m["retrieval"], m["generation"], m["abstention"], m["performance"]
    banner = (
        "> **Synthetic / demo evaluation.** The benchmark documents are fictional sample policies created for "
        "this project. Scores describe this demo corpus only.\n\n"
        if p["synthetic"]
        else ""
    )
    lines = [
        f"# Evaluation report — {p['dataset']}",
        "",
        banner + f"Run `{p['run_id']}` · label: {p.get('label') or '-'} · created {p.get('created_at')}",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in p["config"].items()],
        "",
        f"## Retrieval (K = {m['k']}) — {r['method']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision@K | {_fmt(r['precision_at_k'])} |",
        f"| Recall@K | {_fmt(r['recall_at_k'])} |",
        f"| MRR | {_fmt(r['mrr'])} |",
        f"| Hit rate | {_fmt(r['hit_rate'])} |",
        f"| Avg retrieval latency (ms) | {_fmt(r['avg_retrieval_latency_ms'])} |",
        "",
        f"## Generation — judge: {g['judge']} ({g['method']})",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Faithfulness | {_fmt(g['faithfulness'])} |",
        f"| Answer relevance | {_fmt(g['answer_relevance'])} |",
        f"| Context relevance | {_fmt(g['context_relevance'])} |",
        f"| Citation coverage | {_fmt(g['citation_coverage'])} |",
        f"| Answer token-F1 vs reference | {_fmt(g['answer_f1'])} |",
        "",
        "## Unanswerable questions",
        "",
        f"Correct abstentions: {a['correct_abstentions']} / {a['unanswerable_questions']} "
        f"(accuracy {_fmt(a['abstention_accuracy'])})",
        "",
        "## Performance",
        "",
        f"Avg total latency: {_fmt(perf['avg_total_latency_ms'])} ms · "
        f"avg generation latency: {_fmt(perf['avg_generation_latency_ms'])} ms · total tokens: {_fmt(perf['total_tokens'])}",
        "",
        "## Per-question results",
        "",
        "| ID | Question | P@K | RR | Faithful | Cited | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in p["results"]:
        rr = item.get("retrieval", {})
        gg = item.get("generation", {})
        lines.append(
            f"| {item['id']} | {item['question'][:60]} | {_fmt(rr.get('precision_at_k'))} | "
            f"{_fmt(rr.get('reciprocal_rank'))} | {_fmt(gg.get('faithfulness'))} | {_fmt(gg.get('citation_coverage'))} | "
            f"{item.get('error') or item.get('status')} |"
        )
    return "\n".join(lines) + "\n"
