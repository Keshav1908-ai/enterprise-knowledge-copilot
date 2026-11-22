"""Semantic, hybrid (BM25 + vector) and reranked retrieval.

Pipeline:
    query -> embedding -> Chroma top-N (vector)
                       -> BM25 top-N (keyword, hybrid mode only)
          -> weighted Reciprocal Rank Fusion -> optional cross-encoder rerank
          -> similarity threshold -> top-K

The similarity threshold is always applied to the *cosine similarity* between the query
and the chunk, so it means the same thing in every mode.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from app.core.config import Settings
from app.core.logging import get_logger
from app.embeddings.providers import cosine, get_embeddings, tokenize
from app.retrieval.vector_store import StoredChunk, VectorStore, get_vector_store

logger = get_logger(__name__)

RRF_K = 60


@dataclass
class RetrievalParams:
    top_k: int
    similarity_threshold: float
    mode: str = "vector"
    hybrid_alpha: float = 0.5
    reranker: str = "none"
    candidates: int = 20
    document_ids: list[str] | None = None

    @classmethod
    def from_settings(cls, s: Settings, **overrides) -> "RetrievalParams":  # type: ignore[no-untyped-def]
        p = cls(
            top_k=s.top_k,
            similarity_threshold=s.similarity_threshold,
            mode=s.retrieval_mode,
            hybrid_alpha=s.hybrid_alpha,
            reranker=s.reranker,
            candidates=s.rerank_candidates,
        )
        for k, v in overrides.items():
            if v is not None and hasattr(p, k):
                setattr(p, k, v)
        p.candidates = max(p.candidates, p.top_k)
        return p

    def as_dict(self) -> dict:
        return {
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "mode": self.mode,
            "hybrid_alpha": self.hybrid_alpha,
            "reranker": self.reranker,
            "document_ids": self.document_ids,
        }


@dataclass
class RetrievalResult:
    chunks: list[StoredChunk]
    latency_ms: float
    params: RetrievalParams
    candidates_considered: int = 0
    filtered_below_threshold: int = 0
    timings: dict = field(default_factory=dict)


# ---- BM25 index cache ------------------------------------------------------
class _BM25Cache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._index: BM25Okapi | None = None
        self._chunks: list[StoredChunk] = []

    def get(self, store: VectorStore, where: dict | None) -> tuple[BM25Okapi | None, list[StoredChunk]]:
        key = (store.settings.collection_name, store.count(), repr(where), _BM25_VERSION[0])
        with self._lock:
            if key != self._key:
                chunks = store.get_chunks(where)
                corpus = [tokenize(c.text) or ["_empty_"] for c in chunks]
                self._index = BM25Okapi(corpus) if corpus else None
                self._chunks = chunks
                self._key = key
            return self._index, self._chunks


_BM25_VERSION = [0]
_bm25_cache = _BM25Cache()


def invalidate_keyword_index() -> None:
    _BM25_VERSION[0] += 1


# ---- Cross-encoder ---------------------------------------------------------
_cross_encoder = None
_cross_lock = threading.Lock()


def _get_cross_encoder(model_name: str):  # type: ignore[no-untyped-def]
    global _cross_encoder
    with _cross_lock:
        if _cross_encoder is None:
            from sentence_transformers import CrossEncoder

            _cross_encoder = CrossEncoder(model_name)
        return _cross_encoder


class Retriever:
    def __init__(self, settings: Settings, store: VectorStore | None = None) -> None:
        self.settings = settings
        self.store = store or get_vector_store(settings)
        self.embeddings = get_embeddings(settings)

    @staticmethod
    def _where(params: RetrievalParams) -> dict | None:
        if not params.document_ids:
            return None
        if len(params.document_ids) == 1:
            return {"document_id": params.document_ids[0]}
        return {"document_id": {"$in": list(params.document_ids)}}

    def retrieve(self, query: str, params: RetrievalParams) -> RetrievalResult:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        where = self._where(params)

        t = time.perf_counter()
        qvec = self.embeddings.embed_query(query)
        timings["embed_ms"] = (time.perf_counter() - t) * 1000

        n_candidates = params.candidates if (params.mode == "hybrid" or params.reranker != "none") else params.top_k
        t = time.perf_counter()
        vector_hits = self.store.query(qvec, n_candidates, where)
        timings["vector_search_ms"] = (time.perf_counter() - t) * 1000

        if params.mode == "hybrid":
            t = time.perf_counter()
            candidates = self._hybrid(query, qvec, vector_hits, params, where)
            timings["keyword_fusion_ms"] = (time.perf_counter() - t) * 1000
        else:
            candidates = vector_hits

        considered = len(candidates)
        kept = [c for c in candidates if (c.vector_score if c.vector_score is not None else c.score) >= params.similarity_threshold]
        filtered = considered - len(kept)

        if params.reranker == "cross-encoder" and kept:
            t = time.perf_counter()
            kept = self._rerank(query, kept)
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        final = kept[: params.top_k]
        latency = (time.perf_counter() - t0) * 1000
        return RetrievalResult(
            chunks=final,
            latency_ms=latency,
            params=params,
            candidates_considered=considered,
            filtered_below_threshold=filtered,
            timings=timings,
        )

    def _hybrid(
        self,
        query: str,
        qvec: list[float],
        vector_hits: list[StoredChunk],
        params: RetrievalParams,
        where: dict | None,
    ) -> list[StoredChunk]:
        index, corpus_chunks = _bm25_cache.get(self.store, where)
        keyword_ranked: list[tuple[StoredChunk, float]] = []
        q_tokens = tokenize(query)
        if index is not None and q_tokens:
            scores = index.get_scores(q_tokens)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: params.candidates]
            keyword_ranked = [(corpus_chunks[i], float(scores[i])) for i in order if scores[i] > 0]

        alpha = params.hybrid_alpha
        fused: dict[str, StoredChunk] = {}
        rrf: dict[str, float] = {}
        for rank, c in enumerate(vector_hits):
            fused[c.chunk_id] = c
            rrf[c.chunk_id] = rrf.get(c.chunk_id, 0.0) + alpha / (RRF_K + rank + 1)
        max_kw = keyword_ranked[0][1] if keyword_ranked else 1.0
        missing: list[str] = []
        for rank, (c, s) in enumerate(keyword_ranked):
            if c.chunk_id not in fused:
                fused[c.chunk_id] = StoredChunk(chunk_id=c.chunk_id, text=c.text, metadata=dict(c.metadata))
                missing.append(c.chunk_id)
            fused[c.chunk_id].keyword_score = s / max_kw if max_kw else 0.0
            rrf[c.chunk_id] = rrf.get(c.chunk_id, 0.0) + (1 - alpha) / (RRF_K + rank + 1)

        # Keyword-only hits need a cosine similarity so the threshold stays meaningful.
        if missing:
            res = self.store.collection.get(ids=missing, include=["embeddings"])
            for cid, emb in zip(res.get("ids") or [], res.get("embeddings") if res.get("embeddings") is not None else []):
                sim = cosine(qvec, list(emb))
                fused[cid].vector_score = sim
                fused[cid].score = sim

        ordered = sorted(fused.values(), key=lambda c: rrf[c.chunk_id], reverse=True)
        for c in ordered:
            c.metadata["_fused_score"] = round(rrf[c.chunk_id], 6)
        return ordered

    def _rerank(self, query: str, chunks: list[StoredChunk]) -> list[StoredChunk]:
        try:
            model = _get_cross_encoder(self.settings.reranker_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder unavailable (%s); skipping rerank", type(exc).__name__)
            return chunks
        scores = model.predict([(query, c.text) for c in chunks])
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
        return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
