"""The RAG pipeline orchestrator.

question -> resolve follow-up -> fresh retrieval -> context -> LLM -> citations
         -> grounding check -> persist conversation -> record monitoring metrics
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.database.models import Conversation, Message, new_id, utcnow
from app.evaluation.metrics import citation_coverage, is_abstention, lexical_faithfulness
from app.generation.citations import build_citations
from app.generation.llm import BaseGenerator, get_generator
from app.generation.prompts import NO_CONTEXT_ANSWER
from app.generation.query_rewriter import resolve_question
from app.monitoring.metrics_service import record_query
from app.retrieval.retriever import RetrievalParams, Retriever
from app.retrieval.vector_store import StoredChunk

logger = get_logger(__name__)


@dataclass
class RAGResult:
    answer: str
    citations: list[dict]
    retrieved_chunks: list[dict]
    conversation_id: str | None
    message_id: str | None
    query_id: str
    standalone_question: str
    rewrite_method: str
    status: str
    generation_mode: str
    llm_model: str
    retrieval_params: dict
    latency: dict
    token_usage: dict
    grounding: dict
    retrieval_debug: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _chunk_to_source(i: int, c: StoredChunk) -> dict:
    return {
        "source_id": i,
        "chunk_id": c.chunk_id,
        "document_id": c.document_id,
        "document_name": str(c.metadata.get("document_name", "unknown")),
        "page_number": c.page_number,
        "text": c.text,
        "score": round(c.vector_score if c.vector_score is not None else c.score, 4),
        "keyword_score": None if c.keyword_score is None else round(c.keyword_score, 4),
        "rerank_score": None if c.rerank_score is None else round(c.rerank_score, 4),
    }


class ChatService:
    def __init__(self, settings: Settings, generator: BaseGenerator | None = None) -> None:
        self.settings = settings
        self._generator = generator

    @property
    def generator(self) -> BaseGenerator:
        if self._generator is None:
            self._generator = get_generator(self.settings)
        return self._generator

    # ---- conversation helpers ---------------------------------------------
    def _get_or_create_conversation(self, db: Session, conversation_id: str | None, question: str) -> Conversation:
        if conversation_id:
            conv = db.get(Conversation, conversation_id)
            if conv is None:
                raise NotFoundError(f"Conversation '{conversation_id}' not found")
            return conv
        title = question.strip().replace("\n", " ")
        conv = Conversation(title=(title[:77] + "...") if len(title) > 80 else title)
        db.add(conv)
        db.flush()
        return conv

    def _history(self, db: Session, conversation_id: str) -> list[dict]:
        n = self.settings.conversation_history_turns * 2
        if n == 0:
            return []
        rows = list(
            db.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(n)
            )
        )
        return [{"role": m.role, "content": m.content} for m in reversed(rows)]

    # ---- main entry point ---------------------------------------------------
    def ask(
        self,
        db: Session,
        question: str,
        *,
        conversation_id: str | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        retrieval_mode: str | None = None,
        reranker: str | None = None,
        document_ids: list[str] | None = None,
        persist: bool = True,
        source: str = "chat",
    ) -> RAGResult:
        t_start = time.perf_counter()
        query_id = new_id()
        params = RetrievalParams.from_settings(
            self.settings,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            mode=retrieval_mode,
            reranker=reranker,
            document_ids=document_ids or None,
        )
        conv: Conversation | None = None
        history: list[dict] = []
        retrieval_ms = generation_ms = rewrite_ms = None
        input_tokens = output_tokens = None
        num_retrieved = 0
        scores: list[float] = []

        try:
            if persist:
                conv = self._get_or_create_conversation(db, conversation_id, question)
                history = self._history(db, conv.id)

            # 1. Resolve follow-ups (history is used ONLY to clarify the query)
            t = time.perf_counter()
            standalone, rewrite_out, rewrite_method = resolve_question(
                question, history, self.generator, self.settings.rewrite_follow_up_questions
            )
            rewrite_ms = (time.perf_counter() - t) * 1000
            if rewrite_out is not None:
                input_tokens = (input_tokens or 0) + (rewrite_out.input_tokens or 0)
                output_tokens = (output_tokens or 0) + (rewrite_out.output_tokens or 0)

            # 2. Fresh retrieval for every turn
            retrieval = Retriever(self.settings).retrieve(standalone, params)
            retrieval_ms = retrieval.latency_ms
            sources = [_chunk_to_source(i, c) for i, c in enumerate(retrieval.chunks, start=1)]
            num_retrieved = len(sources)
            scores = [s["score"] for s in sources]

            # 3. Generation (skipped entirely when nothing relevant was retrieved)
            if not sources:
                answer, citations, status = NO_CONTEXT_ANSWER, [], "no_context"
                generation_ms = 0.0
                gen_mode, gen_model = self.generator.provider, self.generator.model
            else:
                # The LLM sees the original question + history; the standalone form drives retrieval.
                gen = self.generator.generate(question, sources, history)
                generation_ms = gen.latency_ms
                gen_mode, gen_model = gen.provider, gen.model
                if gen.input_tokens is not None or gen.output_tokens is not None:
                    input_tokens = (input_tokens or 0) + (gen.input_tokens or 0)
                    output_tokens = (output_tokens or 0) + (gen.output_tokens or 0)
                answer, citations = build_citations(gen.text, sources)
                status = "no_context" if (not citations and is_abstention(answer)) else "success"

            # 4. Post-hoc grounding check against the retrieved context
            grounding = {
                "faithfulness_estimate": lexical_faithfulness(answer, [s["text"] for s in sources]),
                "citation_coverage": citation_coverage(answer, {s["source_id"] for s in sources}),
                "abstained": is_abstention(answer),
                "method": "lexical overlap (automated estimate)",
            }

            total_ms = (time.perf_counter() - t_start) * 1000
            total_tokens = None if input_tokens is None and output_tokens is None else (input_tokens or 0) + (output_tokens or 0)
            latency = {
                "rewrite_ms": round(rewrite_ms or 0.0, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms or 0.0, 2),
                "total_ms": round(total_ms, 2),
                "retrieval_breakdown": {k: round(v, 2) for k, v in retrieval.timings.items()},
            }
            token_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}

            message_id = None
            if persist and conv is not None:
                db.add(Message(conversation_id=conv.id, role="user", content=question))
                db.flush()
                assistant = Message(
                    conversation_id=conv.id,
                    role="assistant",
                    content=answer,
                    standalone_question=standalone if standalone != question else None,
                    citations=citations,
                    retrieved_chunks=[{k: v for k, v in s.items()} for s in sources],
                    metrics={"latency": latency, "token_usage": token_usage, "status": status, "query_id": query_id,
                             "generation_mode": gen_mode, "grounding": grounding},
                )
                db.add(assistant)
                conv.updated_at = utcnow()
                db.flush()
                message_id = assistant.id

            record_query(
                db,
                query_id=query_id,
                conversation_id=conv.id if conv else None,
                source=source,
                status=status,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                scores=scores,
                llm_provider=gen_mode,
                llm_model=gen_model,
                retrieval_mode=params.mode,
            )
            db.commit()

            return RAGResult(
                answer=answer,
                citations=citations,
                retrieved_chunks=sources,
                conversation_id=conv.id if conv else None,
                message_id=message_id,
                query_id=query_id,
                standalone_question=standalone,
                rewrite_method=rewrite_method,
                status=status,
                generation_mode=gen_mode,
                llm_model=gen_model,
                retrieval_params=params.as_dict(),
                latency=latency,
                token_usage=token_usage,
                grounding=grounding,
                retrieval_debug={
                    "candidates_considered": retrieval.candidates_considered,
                    "filtered_below_threshold": retrieval.filtered_below_threshold,
                },
            )
        except Exception as exc:
            db.rollback()
            try:
                record_query(
                    db,
                    query_id=query_id,
                    conversation_id=conversation_id,
                    source=source,
                    status="error",
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    total_ms=(time.perf_counter() - t_start) * 1000,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    scores=scores,
                    llm_provider=self.settings.llm_provider,
                    llm_model=self.settings.llm_model,
                    retrieval_mode=params.mode,
                    error=f"{type(exc).__name__}: {getattr(exc, 'message', str(exc))}"[:1000],
                )
                db.commit()
            except Exception:  # noqa: BLE001
                db.rollback()
                logger.exception("Failed to record error metric")
            raise
