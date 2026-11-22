from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluationRunRequest(BaseModel):
    label: str | None = Field(default=None, max_length=200)
    top_k: int | None = Field(default=None, ge=1, le=50)
    similarity_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    retrieval_mode: Literal["vector", "hybrid"] | None = None
    reranker: Literal["none", "cross-encoder"] | None = None
    judge: Literal["heuristic", "llm"] | None = None
    ingest_dataset_documents: bool = Field(
        default=True, description="Index the benchmark's source documents first if they are missing"
    )
    max_questions: int | None = Field(default=None, ge=1, le=1000)
    run_in_background: bool = False


class EvaluationRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    label: str | None
    status: str
    dataset_name: str
    is_synthetic: bool
    judge: str
    config: dict
    num_questions: int
    metrics: dict | None = None
    report_path: str | None = None
    duration_ms: float | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class EvaluationRunDetail(EvaluationRunOut):
    results: list[dict] | None = None


class EvaluationResults(BaseModel):
    runs: list[EvaluationRunOut]
    latest: EvaluationRunDetail | None = None
    dataset: dict
