"""Citation extraction and validation.

The LLM is shown numbered sources ``[1]..[n]`` that map 1:1 to retrieved chunks. After
generation we parse the markers, drop any that do not correspond to a retrieved chunk
(so a hallucinated "[7]" can never surface as a citation) and build citation objects
from the *actual* chunk metadata. Document names and page numbers therefore always come
from the vector store, never from model output.
"""

from __future__ import annotations

import re

from app.embeddings.providers import tokenize
from app.generation.llm import split_sentences

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_markers(text: str) -> list[int]:
    ids: list[int] = []
    for group in _MARKER.findall(text):
        for part in group.split(","):
            n = int(part.strip())
            if n not in ids:
                ids.append(n)
    return ids


def strip_invalid_markers(text: str, valid_ids: set[int]) -> str:
    def _fix(m: re.Match) -> str:
        keep = [p.strip() for p in m.group(1).split(",") if int(p.strip()) in valid_ids]
        return f"[{', '.join(keep)}]" if keep else ""

    return re.sub(r"\s+([.,;:])", r"\1", _MARKER.sub(_fix, text)).strip()


def best_excerpt(chunk_text: str, answer: str, max_chars: int = 420) -> str:
    """The chunk sentence window with the highest lexical overlap with the answer."""
    sentences = split_sentences(chunk_text) or [chunk_text]
    ans = set(tokenize(answer))
    best_i, best = 0, -1.0
    for i, s in enumerate(sentences):
        st = set(tokenize(s))
        score = len(ans & st) / (len(st) or 1)
        if score > best:
            best_i, best = i, score
    window = sentences[best_i]
    j = best_i + 1
    while j < len(sentences) and len(window) + len(sentences[j]) < max_chars:
        window += " " + sentences[j]
        j += 1
    return window if len(window) <= max_chars else window[: max_chars - 1] + "…"


def build_citations(answer: str, sources: list[dict]) -> tuple[str, list[dict]]:
    by_id = {s["source_id"]: s for s in sources}
    cited = [i for i in parse_markers(answer) if i in by_id]
    clean = strip_invalid_markers(answer, set(by_id))
    citations = []
    for sid in cited:
        s = by_id[sid]
        citations.append(
            {
                "source_id": sid,
                "chunk_id": s["chunk_id"],
                "document_id": s["document_id"],
                "document_name": s["document_name"],
                "page_number": s.get("page_number"),
                "excerpt": best_excerpt(s["text"], clean),
                "relevance_score": s.get("score"),
            }
        )
    return clean, citations
