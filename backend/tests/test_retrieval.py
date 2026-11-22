"""Embeddings, vector insertion and semantic / hybrid retrieval."""

from __future__ import annotations

import math

from app.embeddings.providers import HashEmbeddings, cosine, get_embeddings
from app.ingestion.chunking import Chunk
from app.retrieval.retriever import RetrievalParams, Retriever
from app.retrieval.vector_store import get_vector_store


def test_hash_embeddings_are_deterministic_and_normalised():
    e = HashEmbeddings(256)
    a, b = e.embed_query("annual leave policy"), e.embed_query("annual leave policy")
    assert a == b and len(a) == 256
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-6)


def test_embedding_similarity_orders_related_text_higher():
    e = HashEmbeddings(384)
    q = e.embed_query("How many days of casual leave?")
    related = e.embed_documents(["Employees receive 8 days of casual leave per year."])[0]
    unrelated = e.embed_documents(["Passwords must be at least 14 characters long."])[0]
    assert cosine(q, related) > cosine(q, unrelated)


def test_embedding_provider_is_cached_and_bound_to_collection(settings):
    assert get_embeddings(settings) is get_embeddings(settings)
    other = settings.model_copy(update={"hash_embedding_dim": 128})
    assert other.collection_name != settings.collection_name


def test_vector_insert_query_and_delete(settings):
    store = get_vector_store(settings)
    emb = get_embeddings(settings)
    chunks = [
        Chunk("d1-00000", "Casual leave is 8 days per year.", 0, None, {"document_id": "d1", "page_number": -1}),
        Chunk("d2-00000", "VPN requires multi-factor authentication.", 0, 2, {"document_id": "d2", "page_number": 2}),
    ]
    store.add_chunks(chunks, emb.embed_documents([c.text for c in chunks]))
    assert store.count() == 2
    hits = store.query(emb.embed_query("casual leave days"), top_k=2)
    assert hits[0].chunk_id == "d1-00000" and hits[0].score > hits[1].score
    assert hits[1].page_number == 2 and hits[0].page_number is None
    store.delete_document("d1")
    assert store.count() == 1


def test_semantic_retrieval_finds_relevant_document(indexed_corpus, settings):
    res = Retriever(settings).retrieve("How long must passwords be?", RetrievalParams.from_settings(settings, top_k=3))
    assert res.chunks and res.chunks[0].document_id == indexed_corpus["it_security_policy.md"]
    assert res.latency_ms > 0 and "embed_ms" in res.timings
    scores = [c.score for c in res.chunks]
    assert scores == sorted(scores, reverse=True)


def test_top_k_and_threshold_are_applied(indexed_corpus, settings):
    r = Retriever(settings)
    assert len(r.retrieve("leave", RetrievalParams.from_settings(settings, top_k=2)).chunks) <= 2
    strict = r.retrieve("leave", RetrievalParams.from_settings(settings, top_k=5, similarity_threshold=0.99))
    assert strict.chunks == [] and strict.filtered_below_threshold > 0


def test_document_filter_restricts_results(indexed_corpus, settings):
    only = indexed_corpus["travel_expense_policy.txt"]
    res = Retriever(settings).retrieve(
        "leave policy", RetrievalParams.from_settings(settings, top_k=5, document_ids=[only], similarity_threshold=-1)
    )
    assert res.chunks and all(c.document_id == only for c in res.chunks)


def test_hybrid_retrieval_combines_keyword_and_vector(indexed_corpus, settings):
    res = Retriever(settings).retrieve(
        "INR 12 per kilometre personal car", RetrievalParams.from_settings(settings, top_k=3, mode="hybrid")
    )
    assert res.chunks[0].document_id == indexed_corpus["travel_expense_policy.txt"]
    assert any(c.keyword_score is not None for c in res.chunks)
    assert all(c.vector_score is not None for c in res.chunks)


def test_empty_store_returns_no_chunks(settings):
    res = Retriever(settings).retrieve("anything", RetrievalParams.from_settings(settings))
    assert res.chunks == []
