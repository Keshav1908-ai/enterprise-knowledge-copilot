"""Embedding providers behind LangChain's ``Embeddings`` interface.

Supported providers (``EMBEDDING_PROVIDER``):

* ``sentence-transformers`` – local model via ``langchain-huggingface`` (default, free).
* ``openai`` – OpenAI or any OpenAI-compatible embeddings endpoint.
* ``hash`` – deterministic feature-hashing embeddings. Lexical, not semantic; used for
  tests and as an offline fallback when no model can be downloaded.

The same provider instance embeds both documents and queries, which guarantees that
queries are compared against vectors from the same model that indexed the corpus.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading

from langchain_core.embeddings import Embeddings

from app.core.config import Settings, get_settings
from app.core.exceptions import ProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was were what "
    "when where which who will with do does can my our your you we they their there".split()
)


def _stem(tok: str) -> str:
    """Tiny, predictable suffix stripper (plural and common verb endings)."""
    if len(tok) <= 3:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("es") and tok[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    if tok.endswith("ing") and len(tok) > 5:
        return tok[:-3]
    if tok.endswith("ed") and len(tok) > 4:
        return tok[:-2]
    return tok


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with light stemming (shared by hash embeddings, BM25 and metrics)."""
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


class HashEmbeddings(Embeddings):
    """Signed feature hashing over unigrams and bigrams, L2-normalised."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            idx, sign = self._bucket(f)
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _build_sentence_transformers(settings: Settings) -> Embeddings:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ProviderError(
            "sentence-transformers provider requires 'langchain-huggingface' and 'sentence-transformers'. "
            "Install backend/requirements.txt or set EMBEDDING_PROVIDER=hash."
        ) from exc
    try:
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": settings.embedding_batch_size},
        )
    except Exception as exc:  # noqa: BLE001
        raise ProviderError(f"Failed to load embedding model '{settings.embedding_model}': {exc}") from exc


def _build_openai(settings: Settings) -> Embeddings:
    if not settings.openai_api_key:
        raise ProviderError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,  # type: ignore[arg-type]
        base_url=settings.openai_base_url,
        chunk_size=settings.embedding_batch_size,
    )


def create_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider
    logger.info("Initialising embedding provider=%s model=%s", provider, settings.embedding_model)
    if provider == "hash":
        return HashEmbeddings(settings.hash_embedding_dim)
    if provider == "openai":
        return _build_openai(settings)
    return _build_sentence_transformers(settings)


_lock = threading.Lock()
_instance: Embeddings | None = None
_instance_key: str | None = None


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Process-wide cached embedding model (loading a model is expensive)."""
    global _instance, _instance_key
    settings = settings or get_settings()
    key = settings.collection_name
    with _lock:
        if _instance is None or _instance_key != key:
            _instance = create_embeddings(settings)
            _instance_key = key
        return _instance


def set_embeddings(embeddings: Embeddings | None, key: str | None = None) -> None:
    """Override the cached provider (used by tests to inject fakes)."""
    global _instance, _instance_key
    with _lock:
        _instance = embeddings
        _instance_key = key


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
