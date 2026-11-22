"""ChromaDB persistence layer (cosine space, metadata filtering, per-document deletes)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.ingestion.chunking import Chunk

logger = get_logger(__name__)


@dataclass
class StoredChunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0  # cosine similarity in [-1, 1]; higher is better
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None

    @property
    def document_id(self) -> str:
        return str(self.metadata.get("document_id", ""))

    @property
    def page_number(self) -> int | None:
        p = self.metadata.get("page_number", -1)
        return None if p is None or int(p) < 0 else int(p)


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=settings.environment == "test"),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_provider": settings.embedding_provider,
                "embedding_model": settings.embedding_model,
            },
        )

    # ---- writes ------------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]], batch_size: int = 256) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for start in range(0, len(chunks), batch_size):
            part = chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[c.chunk_id for c in part],
                documents=[c.text for c in part],
                metadatas=[c.metadata for c in part],
                embeddings=vectors[start : start + batch_size],
            )

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    # ---- reads -------------------------------------------------------------
    def count(self) -> int:
        return self.collection.count()

    def count_document(self, document_id: str) -> int:
        res = self.collection.get(where={"document_id": document_id}, include=[])
        return len(res.get("ids") or [])

    def query(self, vector: list[float], top_k: int, where: dict | None = None) -> list[StoredChunk]:
        n = self.count()
        if n == 0:
            return []
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, n),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        out: list[StoredChunk] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            sim = 1.0 - float(dist)  # cosine distance -> cosine similarity
            out.append(StoredChunk(chunk_id=cid, text=doc or "", metadata=dict(meta or {}), score=sim, vector_score=sim))
        return out

    def get_chunks(self, where: dict | None = None) -> list[StoredChunk]:
        res = self.collection.get(where=where or None, include=["documents", "metadatas"])
        return [
            StoredChunk(chunk_id=cid, text=doc or "", metadata=dict(meta or {}))
            for cid, doc, meta in zip(res.get("ids") or [], res.get("documents") or [], res.get("metadatas") or [])
        ]

    def get_by_ids(self, ids: list[str]) -> list[StoredChunk]:
        if not ids:
            return []
        res = self.collection.get(ids=ids, include=["documents", "metadatas"])
        return [
            StoredChunk(chunk_id=cid, text=doc or "", metadata=dict(meta or {}))
            for cid, doc, meta in zip(res.get("ids") or [], res.get("documents") or [], res.get("metadatas") or [])
        ]

    def heartbeat(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception:  # noqa: BLE001
            return False


_lock = threading.Lock()
_store: VectorStore | None = None
_store_key: tuple | None = None


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    global _store, _store_key
    settings = settings or get_settings()
    key = (str(settings.chroma_persist_dir), settings.collection_name)
    with _lock:
        if _store is None or _store_key != key:
            _store = VectorStore(settings)
            _store_key = key
        return _store


def reset_vector_store() -> None:
    global _store, _store_key
    with _lock:
        _store = None
        _store_key = None
