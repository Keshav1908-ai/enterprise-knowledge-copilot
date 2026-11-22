"""Evaluation metrics.

Retrieval metrics (deterministic, ground truth = labelled relevant documents):
    precision_at_k, recall_at_k, reciprocal_rank (-> MRR), hit_rate

Generation metrics. Two families, always labelled in reports:
    * deterministic estimates — lexical faithfulness, citation coverage, token F1 vs. the
      reference answer, abstention correctness.
    * embedding estimates — answer relevance / context relevance as cosine similarity.
    * (optional) LLM-judge estimates — see ``judges.py``.
None of these are human ground truth; they are automated estimates.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from app.embeddings.providers import cosine, tokenize
from app.generation.citations import parse_markers
from app.generation.llm import split_sentences

ABSTAIN_PATTERNS = re.compile(
    r"(couldn'?t find|could not find|don'?t (specify|mention|contain|have)|do not (specify|mention|contain)|"
    r"no information|not (mentioned|specified|covered|available) in|unable to find|isn'?t (covered|mentioned))",
    re.I,
)
_MARKER = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


# ---- retrieval ---------------------------------------------------------------
def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(list(retrieved)[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def hit_rate(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(list(retrieved)[:k]) & relevant else 0.0


# ---- generation --------------------------------------------------------------
def is_abstention(answer: str) -> bool:
    return bool(ABSTAIN_PATTERNS.search(answer or ""))


def factual_sentences(answer: str) -> list[str]:
    """Answer sentences that make claims (drop headers like 'Here is what the documents say:')."""
    out = []
    for s in split_sentences(answer):
        bare = _MARKER.sub("", s).strip()
        if len(tokenize(bare)) < 3 or bare.endswith(":") or is_abstention(bare):
            continue
        out.append(s)
    return out


def lexical_faithfulness(answer: str, contexts: Sequence[str], support_threshold: float = 0.6) -> float | None:
    """Share of claim sentences whose content words are mostly present in the retrieved context."""
    sentences = factual_sentences(answer)
    if not sentences:
        return None
    ctx_tokens = set()
    for c in contexts:
        ctx_tokens.update(tokenize(c))
    supported = 0
    for s in sentences:
        toks = set(tokenize(_MARKER.sub("", s)))
        if toks and len(toks & ctx_tokens) / len(toks) >= support_threshold:
            supported += 1
    return supported / len(sentences)


def citation_coverage(answer: str, valid_source_ids: set[int]) -> float | None:
    """Share of claim sentences that carry at least one valid [n] citation."""
    sentences = factual_sentences(answer)
    if not sentences:
        return None
    cited = sum(1 for s in sentences if any(i in valid_source_ids for i in parse_markers(s)))
    return cited / len(sentences)


def token_f1(prediction: str, reference: str) -> float:
    p = tokenize(_MARKER.sub("", prediction))
    r = tokenize(reference)
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(r)
    return 2 * precision * recall / (precision + recall)


def embedding_relevance(question_vec: list[float], other_vecs: Sequence[list[float]]) -> float | None:
    if not other_vecs:
        return None
    sims = [max(0.0, cosine(question_vec, v)) for v in other_vecs]
    return sum(sims) / len(sims)


def mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
