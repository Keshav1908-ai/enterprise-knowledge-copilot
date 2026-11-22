"""Application configuration.

Every tunable value is read from environment variables (or a ``.env`` file) so the
application can be reconfigured without touching source code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Application -------------------------------------------------------
    app_name: str = "Enterprise Knowledge Copilot"
    app_version: str = "1.0.0"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://localhost:8080"

    # ---- Storage -----------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "storage"
    database_url: str | None = None  # defaults to sqlite in data_dir
    chroma_persist_dir: Path | None = None  # defaults to data_dir/chroma
    upload_dir: Path | None = None  # defaults to data_dir/uploads
    chroma_collection_prefix: str = "ekc"

    # ---- Upload validation ---------------------------------------------------
    max_upload_size_mb: int = Field(default=25, ge=1, le=500)
    allowed_extensions: str = ".pdf,.docx,.txt,.md"

    # ---- Chunking ----------------------------------------------------------
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)

    # ---- Embeddings --------------------------------------------------------
    # sentence-transformers: local model, no API key needed (default)
    # openai: OpenAI (or compatible) embeddings API
    # hash: deterministic lexical hashing embeddings (offline fallback / tests)
    embedding_provider: Literal["sentence-transformers", "openai", "hash"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    hash_embedding_dim: int = Field(default=384, ge=64, le=4096)

    # ---- Retrieval ---------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.25, ge=-1.0, le=1.0)
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)  # weight of vector vs keyword in RRF
    reranker: Literal["none", "cross-encoder"] = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=20, ge=1, le=200)

    # ---- LLM ---------------------------------------------------------------
    # openai: OpenAI or any OpenAI-compatible endpoint (Groq, OpenRouter, Ollama, vLLM...)
    # extractive: local, no-LLM mode. Answers are assembled from retrieved sentences.
    llm_provider: Literal["openai", "extractive"] = "extractive"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=800, ge=64, le=8192)
    llm_timeout_seconds: float = Field(default=60.0, ge=1.0)
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # ---- Conversation --------------------------------------------------------
    conversation_history_turns: int = Field(default=6, ge=0, le=50)
    rewrite_follow_up_questions: bool = True

    # ---- Evaluation ----------------------------------------------------------
    evaluation_dataset_path: Path = PROJECT_ROOT / "evaluation" / "dataset" / "benchmark.json"
    evaluation_reports_dir: Path = PROJECT_ROOT / "evaluation" / "reports"
    # heuristic: deterministic lexical/semantic estimates; llm: LLM-as-judge (needs llm_provider=openai)
    evaluation_judge: Literal["heuristic", "llm"] = "heuristic"

    @field_validator("data_dir", "chroma_persist_dir", "upload_dir", "evaluation_dataset_path",
                     "evaluation_reports_dir", mode="before")
    @classmethod
    def _expand(cls, v):  # type: ignore[no-untyped-def]
        if v in (None, ""):
            return None
        return Path(v).expanduser()

    @model_validator(mode="after")
    def _derive_paths(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        self.data_dir = self.data_dir.resolve()
        if self.chroma_persist_dir is None:
            self.chroma_persist_dir = self.data_dir / "chroma"
        if self.upload_dir is None:
            self.upload_dir = self.data_dir / "uploads"
        if not self.database_url:
            self.database_url = f"sqlite:///{(self.data_dir / 'copilot.db').as_posix()}"
        return self

    # ---- Derived helpers -----------------------------------------------------
    @property
    def allowed_extension_set(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider == "openai" and bool(self.openai_api_key)

    @property
    def collection_name(self) -> str:
        """Chroma collection name is bound to the embedding model.

        Vectors from different embedding models live in different spaces and must never
        be mixed, so switching EMBEDDING_MODEL automatically targets a fresh collection.
        """
        import hashlib

        key = f"{self.embedding_provider}:{self.embedding_model}"
        if self.embedding_provider == "hash":
            key += f":{self.hash_embedding_dim}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:10]
        return f"{self.chroma_collection_prefix}_{digest}"

    def ensure_directories(self) -> None:
        for p in (self.data_dir, self.chroma_persist_dir, self.upload_dir, self.evaluation_reports_dir):
            Path(p).mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict:
        """Configuration safe to expose to the UI (no secrets)."""
        return {
            "app_version": self.app_version,
            "environment": self.environment,
            "embedding_provider": self.embedding_provider,
            "embedding_model": (
                f"hash-{self.hash_embedding_dim}d (offline, lexical)" if self.embedding_provider == "hash" else self.embedding_model
            ),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model if self.llm_provider == "openai" else "extractive (local, no LLM)",
            "llm_configured": self.llm_enabled,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "retrieval_mode": self.retrieval_mode,
            "reranker": self.reranker,
            "max_upload_size_mb": self.max_upload_size_mb,
            "allowed_extensions": sorted(self.allowed_extension_set),
            "evaluation_judge": self.evaluation_judge,
            "collection": self.collection_name,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
