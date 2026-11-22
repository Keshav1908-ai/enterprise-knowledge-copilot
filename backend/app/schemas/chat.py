from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=32)
    top_k: int | None = Field(default=None, ge=1, le=50)
    similarity_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    retrieval_mode: Literal["vector", "hybrid"] | None = None
    reranker: Literal["none", "cross-encoder"] | None = None
    document_ids: list[str] | None = Field(default=None, max_length=100)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class Citation(BaseModel):
    source_id: int
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None = None
    excerpt: str
    relevance_score: float | None = None


class RetrievedChunk(BaseModel):
    source_id: int
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None = None
    text: str
    score: float
    keyword_score: float | None = None
    rerank_score: float | None = None


class Latency(BaseModel):
    rewrite_ms: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    retrieval_breakdown: dict[str, float] = {}


class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    conversation_id: str | None
    message_id: str | None
    query_id: str
    standalone_question: str
    rewrite_method: str
    status: str
    generation_mode: str
    llm_model: str
    retrieval_params: dict
    latency: Latency
    token_usage: TokenUsage
    grounding: dict
    retrieval_debug: dict = {}


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    standalone_question: str | None = None
    citations: list[dict] | None = None
    retrieved_chunks: list[dict] | None = None
    metrics: dict | None = None
    created_at: datetime


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_preview: str | None = None


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]
