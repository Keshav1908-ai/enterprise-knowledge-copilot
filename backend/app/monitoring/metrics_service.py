"""Monitoring: persist per-query measurements and compute aggregates from them.

Every number shown on the monitoring dashboard is computed here from rows written during
real request execution. Nothing is hardcoded.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import Document, DocumentStatus, QueryMetric, RequestLog


def record_query(
    db: Session,
    *,
    query_id: str,
    conversation_id: str | None,
    source: str,
    status: str,
    retrieval_ms: float | None,
    generation_ms: float | None,
    total_ms: float | None,
    input_tokens: int | None,
    output_tokens: int | None,
    scores: list[float],
    llm_provider: str | None,
    llm_model: str | None,
    retrieval_mode: str | None,
    error: str | None = None,
) -> QueryMetric:
    total_tokens = None if input_tokens is None and output_tokens is None else (input_tokens or 0) + (output_tokens or 0)
    row = QueryMetric(
        query_id=query_id,
        conversation_id=conversation_id,
        source=source,
        status=status,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        total_ms=total_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        num_retrieved=len(scores),
        max_score=max(scores) if scores else None,
        avg_score=(sum(scores) / len(scores)) if scores else None,
        scores=[round(s, 4) for s in scores],
        llm_provider=llm_provider,
        llm_model=llm_model,
        retrieval_mode=retrieval_mode,
        error=error,
    )
    db.add(row)
    return row


def record_request(db: Session, method: str, path: str, status_code: int, duration_ms: float) -> None:
    db.add(RequestLog(method=method, path=path, status_code=status_code, duration_ms=duration_ms))


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def _r(v: float | None, n: int = 2) -> float | None:
    return None if v is None else round(float(v), n)


def _bucket_key(dt: datetime, bucket: str) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if bucket == "hour":
        return dt.strftime("%Y-%m-%dT%H:00")
    return dt.strftime("%Y-%m-%d")


def get_metrics(db: Session, *, hours: int = 24 * 7, source: str | None = "chat", bucket: str = "hour") -> dict:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    stmt = select(QueryMetric).where(QueryMetric.created_at >= since).order_by(QueryMetric.created_at)
    if source:
        stmt = stmt.where(QueryMetric.source == source)
    rows = list(db.scalars(stmt))

    ok = [r for r in rows if r.status != "error"]
    totals = [r.total_ms for r in ok if r.total_ms is not None]
    retr = [r.retrieval_ms for r in ok if r.retrieval_ms is not None]
    gen = [r.generation_ms for r in ok if r.generation_ms is not None and r.status == "success"]
    tokens_rows = [r for r in rows if r.total_tokens is not None]
    all_scores = [s for r in ok for s in (r.scores or [])]

    summary = {
        "total_queries": len(rows),
        "successful_queries": sum(1 for r in rows if r.status == "success"),
        "no_context_queries": sum(1 for r in rows if r.status == "no_context"),
        "failed_queries": sum(1 for r in rows if r.status == "error"),
        "error_rate": _r(sum(1 for r in rows if r.status == "error") / len(rows), 4) if rows else 0.0,
        "avg_total_latency_ms": _r(sum(totals) / len(totals)) if totals else None,
        "p50_total_latency_ms": _r(_percentile(totals, 0.5)),
        "p95_total_latency_ms": _r(_percentile(totals, 0.95)),
        "avg_retrieval_latency_ms": _r(sum(retr) / len(retr)) if retr else None,
        "p95_retrieval_latency_ms": _r(_percentile(retr, 0.95)),
        "avg_generation_latency_ms": _r(sum(gen) / len(gen)) if gen else None,
        "p95_generation_latency_ms": _r(_percentile(gen, 0.95)),
        "total_input_tokens": sum(r.input_tokens or 0 for r in tokens_rows),
        "total_output_tokens": sum(r.output_tokens or 0 for r in tokens_rows),
        "total_tokens": sum(r.total_tokens or 0 for r in tokens_rows),
        "queries_with_token_data": len(tokens_rows),
        "avg_tokens_per_query": _r(sum(r.total_tokens or 0 for r in tokens_rows) / len(tokens_rows)) if tokens_rows else None,
        "avg_retrieved_chunks": _r(sum(r.num_retrieved for r in ok) / len(ok)) if ok else None,
        "avg_relevance_score": _r(sum(all_scores) / len(all_scores), 4) if all_scores else None,
        "avg_top_score": _r(sum(r.max_score for r in ok if r.max_score is not None) / max(1, sum(1 for r in ok if r.max_score is not None)), 4) if ok else None,
    }

    # time series
    series: dict[str, dict] = {}
    for r in rows:
        key = _bucket_key(r.created_at, bucket)
        b = series.setdefault(key, {"bucket": key, "queries": 0, "errors": 0, "_tot": [], "_ret": [], "_gen": [], "tokens": 0})
        b["queries"] += 1
        if r.status == "error":
            b["errors"] += 1
        else:
            if r.total_ms is not None:
                b["_tot"].append(r.total_ms)
            if r.retrieval_ms is not None:
                b["_ret"].append(r.retrieval_ms)
            if r.generation_ms is not None and r.status == "success":
                b["_gen"].append(r.generation_ms)
        b["tokens"] += r.total_tokens or 0
    timeseries = []
    for key in sorted(series):
        b = series[key]
        timeseries.append(
            {
                "bucket": key,
                "queries": b["queries"],
                "errors": b["errors"],
                "tokens": b["tokens"],
                "avg_total_ms": _r(sum(b["_tot"]) / len(b["_tot"])) if b["_tot"] else None,
                "avg_retrieval_ms": _r(sum(b["_ret"]) / len(b["_ret"])) if b["_ret"] else None,
                "avg_generation_ms": _r(sum(b["_gen"]) / len(b["_gen"])) if b["_gen"] else None,
            }
        )

    # score distribution histogram (cosine similarity buckets)
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    histogram = []
    for lo, hi in zip(edges, edges[1:]):
        histogram.append({"range": f"{lo:.1f}-{min(hi, 1.0):.1f}", "count": sum(1 for s in all_scores if lo <= s < hi)})

    recent = [
        {
            "query_id": r.query_id,
            "created_at": r.created_at.isoformat(),
            "status": r.status,
            "total_ms": _r(r.total_ms),
            "retrieval_ms": _r(r.retrieval_ms),
            "generation_ms": _r(r.generation_ms),
            "total_tokens": r.total_tokens,
            "num_retrieved": r.num_retrieved,
            "max_score": _r(r.max_score, 4),
            "llm_model": r.llm_model,
            "retrieval_mode": r.retrieval_mode,
            "error": r.error,
        }
        for r in reversed(rows[-25:])
    ]

    # API layer
    req_rows = list(db.scalars(select(RequestLog).where(RequestLog.created_at >= since)))
    api = {
        "total_requests": len(req_rows),
        "error_responses": sum(1 for r in req_rows if r.status_code >= 400),
        "server_errors": sum(1 for r in req_rows if r.status_code >= 500),
        "avg_duration_ms": _r(sum(r.duration_ms for r in req_rows) / len(req_rows)) if req_rows else None,
        "by_status": {},
    }
    for r in req_rows:
        k = f"{r.status_code // 100}xx"
        api["by_status"][k] = api["by_status"].get(k, 0) + 1

    return {
        "window_hours": hours,
        "source": source or "all",
        "summary": summary,
        "timeseries": timeseries,
        "score_histogram": histogram,
        "recent_queries": recent,
        "api": api,
        "ingestion": get_ingestion_stats(db),
    }


def get_ingestion_stats(db: Session) -> dict:
    docs = list(db.scalars(select(Document)))
    indexed = [d for d in docs if d.status == DocumentStatus.INDEXED.value]
    times = [d.processing_time_ms for d in indexed if d.processing_time_ms is not None]
    return {
        "total_documents": len(docs),
        "indexed_documents": len(indexed),
        "failed_documents": sum(1 for d in docs if d.status == DocumentStatus.FAILED.value),
        "processing_documents": sum(1 for d in docs if d.status in (DocumentStatus.PENDING.value, DocumentStatus.PROCESSING.value)),
        "total_chunks": sum(d.num_chunks for d in indexed),
        "total_characters": sum(d.num_characters for d in indexed),
        "avg_processing_time_ms": _r(sum(times) / len(times)) if times else None,
        "max_processing_time_ms": _r(max(times)) if times else None,
        "by_type": {
            t: sum(1 for d in docs if d.file_type == t) for t in sorted({d.file_type for d in docs})
        },
    }


def count_queries(db: Session, source: str = "chat") -> int:
    return db.scalar(select(func.count()).select_from(QueryMetric).where(QueryMetric.source == source)) or 0
