"""Health, metrics, dashboard summary and evaluation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Query, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import NotFoundError
from app.database.models import Conversation, EvaluationRun
from app.database.session import get_db
from app.evaluation.runner import EvaluationRunner, dataset_summary
from app.monitoring.metrics_service import get_ingestion_stats, get_metrics
from app.retrieval.vector_store import get_vector_store
from app.schemas.evaluation import EvaluationResults, EvaluationRunDetail, EvaluationRunOut, EvaluationRunRequest

router = APIRouter(tags=["system"])


@router.get("/health", summary="Health check")
def health(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> dict:
    checks = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:  # noqa: BLE001
        checks["database"] = "error"
    try:
        store = get_vector_store(settings)
        checks["vector_store"] = "ok" if store.heartbeat() else "error"
        checks["indexed_chunks"] = store.count()
    except Exception:  # noqa: BLE001
        checks["vector_store"] = "error"
    healthy = checks.get("database") == "ok" and checks.get("vector_store") == "ok"
    return {
        "status": "ok" if healthy else "degraded",
        "version": settings.app_version,
        "checks": checks,
        "config": settings.public_dict(),
    }


@router.get("/api/v1/metrics", summary="Monitoring metrics")
def metrics(
    hours: int = Query(default=168, ge=1, le=24 * 365),
    source: str = Query(default="chat", pattern="^(chat|evaluation|all)$"),
    bucket: str = Query(default="hour", pattern="^(hour|day)$"),
    db: Session = Depends(get_db),
) -> dict:
    return get_metrics(db, hours=hours, source=None if source == "all" else source, bucket=bucket)


@router.get("/api/v1/dashboard", summary="Dashboard summary")
def dashboard(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> dict:
    m = get_metrics(db, hours=24 * 365, source="chat", bucket="day")
    latest = db.scalar(
        select(EvaluationRun).where(EvaluationRun.status == "completed").order_by(EvaluationRun.created_at.desc()).limit(1)
    )
    recent_convs = db.scalars(select(Conversation).order_by(Conversation.updated_at.desc()).limit(5))
    return {
        "ingestion": get_ingestion_stats(db),
        "queries": m["summary"],
        "timeseries": m["timeseries"][-14:],
        "recent_queries": m["recent_queries"][:8],
        "recent_conversations": [
            {"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat()} for c in recent_convs
        ],
        "latest_evaluation": None if latest is None else {
            "id": latest.id, "label": latest.label, "created_at": latest.created_at.isoformat(),
            "is_synthetic": latest.is_synthetic, "metrics": latest.metrics,
        },
        "config": settings.public_dict(),
    }


@router.post("/api/v1/evaluation/run", response_model=EvaluationRunDetail, summary="Execute the evaluation benchmark")
async def run_evaluation(
    req: EvaluationRunRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> EvaluationRunDetail:
    """Runs every benchmark question through the live RAG pipeline.

    Override ``top_k``, ``retrieval_mode`` (vector/hybrid) or ``reranker`` to compare
    configurations; each run is stored and listed in ``/evaluation/results``.
    """
    runner = EvaluationRunner(settings)
    run = runner.create_run(db, req.model_dump(), req.label)
    if req.run_in_background:
        background.add_task(runner.execute, run.id)
        return EvaluationRunDetail.model_validate(run)
    await run_in_threadpool(runner.execute, run.id)
    db.expire_all()
    return EvaluationRunDetail.model_validate(db.get(EvaluationRun, run.id))


@router.get("/api/v1/evaluation/results", response_model=EvaluationResults, summary="Evaluation history and latest report")
def evaluation_results(
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> EvaluationResults:
    runs = list(db.scalars(select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)))
    latest = next((r for r in runs if r.status == "completed"), None)
    return EvaluationResults(
        runs=[EvaluationRunOut.model_validate(r) for r in runs],
        latest=EvaluationRunDetail.model_validate(latest) if latest else None,
        dataset=dataset_summary(settings),
    )


@router.get("/api/v1/evaluation/results/{run_id}", response_model=EvaluationRunDetail, summary="One evaluation run")
def evaluation_run(run_id: str, db: Session = Depends(get_db)) -> EvaluationRunDetail:
    run = db.get(EvaluationRun, run_id)
    if run is None:
        raise NotFoundError(f"Evaluation run '{run_id}' not found")
    return EvaluationRunDetail.model_validate(run)


@router.get("/api/v1/config", summary="Active (non-secret) configuration", status_code=status.HTTP_200_OK)
def config(settings: Settings = Depends(get_settings)) -> dict:
    return settings.public_dict()
