"""Generation judges.

``HeuristicJudge``  deterministic: lexical faithfulness + embedding cosine relevance.
``LLMJudge``        LLM-as-judge (needs an LLM provider); falls back to heuristic per item
                    if the model output can't be parsed. Results are labelled with the
                    method actually used, so estimates are never passed off as ground truth.
"""

from __future__ import annotations

import json
import re

from langchain_core.embeddings import Embeddings

from app.core.logging import get_logger
from app.evaluation.metrics import embedding_relevance, lexical_faithfulness
from app.generation.llm import BaseGenerator
from app.generation.prompts import JUDGE_PROMPT

logger = get_logger(__name__)


class HeuristicJudge:
    name = "heuristic"

    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings

    def score(self, question: str, answer: str, contexts: list[str]) -> dict:
        qv = self.embeddings.embed_query(question)
        av = self.embeddings.embed_documents([answer])[0] if answer else None
        cvs = self.embeddings.embed_documents(contexts) if contexts else []
        return {
            "faithfulness": lexical_faithfulness(answer, contexts),
            "answer_relevance": embedding_relevance(qv, [av]) if av is not None else None,
            "context_relevance": embedding_relevance(qv, cvs),
            "method": "heuristic",
        }


class LLMJudge(HeuristicJudge):
    name = "llm"

    def __init__(self, embeddings: Embeddings, generator: BaseGenerator) -> None:
        super().__init__(embeddings)
        self.generator = generator

    def score(self, question: str, answer: str, contexts: list[str]) -> dict:
        context = "\n---\n".join(c[:1500] for c in contexts) or "(no context retrieved)"
        try:
            out = self.generator.complete(JUDGE_PROMPT.format_messages(question=question, context=context, answer=answer))
            match = re.search(r"\{.*\}", out.text, re.S)
            data = json.loads(match.group(0)) if match else {}
            result = {
                k: max(0.0, min(1.0, float(data[k])))
                for k in ("faithfulness", "answer_relevance", "context_relevance")
                if k in data
            }
            if len(result) == 3:
                result["method"] = "llm"
                result["judge_tokens"] = out.total_tokens
                return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM judge failed (%s); using heuristic for this item", type(exc).__name__)
        fallback = super().score(question, answer, contexts)
        fallback["method"] = "heuristic (llm judge fallback)"
        return fallback
