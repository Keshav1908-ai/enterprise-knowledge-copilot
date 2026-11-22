"""Answer generators.

* :class:`OpenAIGenerator` – LangChain ``ChatOpenAI``; works with OpenAI and any
  OpenAI-compatible server (set ``OPENAI_BASE_URL``: Groq, OpenRouter, Ollama, vLLM...).
* :class:`ExtractiveGenerator` – local, no-LLM mode. Picks the most relevant sentences
  from retrieved chunks and cites them. It never produces text that is not in a source,
  so it is a safe demo/testing mode, clearly labelled ``extractive`` in every response.
"""

from __future__ import annotations

import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.embeddings.providers import tokenize
from app.generation.prompts import NO_CONTEXT_ANSWER, RAG_PROMPT, REWRITE_PROMPT, format_history, format_sources

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?!\[)|\n+")


@dataclass
class GenerationOutput:
    text: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


class BaseGenerator(ABC):
    provider: str = "base"
    model: str = "base"

    @abstractmethod
    def generate(self, question: str, sources: list[dict], history: list[dict]) -> GenerationOutput: ...

    def rewrite_question(self, question: str, history: list[dict]) -> GenerationOutput | None:
        return None

    def complete(self, prompt_messages: list) -> GenerationOutput:  # pragma: no cover - judge only
        raise ProviderError(f"{self.provider} generator does not support free-form completion")


def split_sentences(text: str) -> list[str]:
    """Sentence split that keeps trailing citation markers ("... day. [2]") attached."""
    return [s.strip(" -•*\t") for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > 15]


def _is_heading(sentence: str) -> bool:
    s = sentence.strip()
    letters = [c for c in s if c.isalpha()]
    shouting = bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8
    # Short label-like lines are headings unless they carry a figure (table rows such as
    # "Bereavement leave 5 days" are content).
    short_label = len(s.split()) < 8 and not s.endswith(".") and not any(c.isdigit() for c in s)
    words = [w for w in re.findall(r"[A-Za-z]+", s) if len(w) > 3]
    title_case = (
        not s.endswith(".") and len(words) >= 2 and sum(w[0].isupper() for w in words) / len(words) >= 0.75
    )
    return s.startswith("#") or shouting or short_label or title_case


class ExtractiveGenerator(BaseGenerator):
    provider = "extractive"
    model = "extractive-v1 (local, no LLM)"

    def __init__(self, max_sentences: int = 4, min_query_coverage: float = 0.3) -> None:
        self.max_sentences = max_sentences
        # A sentence must cover this share of the question's content words to be used;
        # otherwise the generator abstains rather than returning loosely related text.
        self.min_query_coverage = min_query_coverage

    def generate(self, question: str, sources: list[dict], history: list[dict]) -> GenerationOutput:
        t0 = time.perf_counter()
        q_tokens = set(tokenize(question))
        candidates: list[tuple[int, int, str, set[str], float]] = []
        for s in sources:
            rank_weight = 1.0 + max(0.0, float(s.get("score") or 0.0))
            for pos, sent in enumerate(split_sentences(s["text"])):
                if _is_heading(sent):
                    continue
                st = set(tokenize(sent))
                if st:
                    candidates.append((s["source_id"], pos, sent, st, rank_weight))

        # IDF over candidate sentences: rare, specific words ("bereavement") outweigh
        # generic ones ("employees", "days") when deciding which sentence answers the question.
        n = len(candidates) or 1
        df: dict[str, int] = {}
        for *_, st, _ in candidates:
            for t in st & q_tokens:
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log(1 + n / (df.get(t, 0) + 0.5)) for t in q_tokens}
        q_weight = sum(idf.values()) or 1.0

        scored: list[tuple[float, int, str]] = []
        for sid, pos, sent, st, rank_weight in candidates:
            matched = q_tokens & st
            if not matched:
                continue
            # Blend plain and IDF-weighted coverage: IDF promotes specific terms, the plain share
            # keeps generic-but-central words ("annual leave") from being drowned out.
            coverage = 0.5 * (len(matched) / len(q_tokens)) + 0.5 * (sum(idf[t] for t in matched) / q_weight)
            if coverage < self.min_query_coverage:
                continue
            scored.append((coverage * rank_weight - pos * 0.001, sid, sent))
        scored.sort(key=lambda x: x[0], reverse=True)

        picked: list[tuple[int, str]] = []
        seen: set[str] = set()
        for _, sid, sent in scored:
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append((sid, sent))
            if len(picked) >= self.max_sentences:
                break

        if not picked:
            text = NO_CONTEXT_ANSWER
        else:
            lines = [f"- {sent.rstrip('. ')} [{sid}]." for sid, sent in picked]
            text = "Here is what the documents say:\n\n" + "\n".join(lines)
        return GenerationOutput(
            text=text, provider=self.provider, model=self.model, latency_ms=(time.perf_counter() - t0) * 1000
        )


class OpenAIGenerator(BaseGenerator):
    provider = "openai"

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ProviderError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set in .env")
        from langchain_openai import ChatOpenAI

        self.model = settings.llm_model
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
            base_url=settings.openai_base_url,
        )

    def _invoke(self, messages: list) -> GenerationOutput:
        t0 = time.perf_counter()
        try:
            msg = self.llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"LLM request failed: {type(exc).__name__}: {exc}") from exc
        usage = getattr(msg, "usage_metadata", None) or {}
        return GenerationOutput(
            text=str(msg.content).strip(),
            provider=self.provider,
            model=self.model,
            latency_ms=(time.perf_counter() - t0) * 1000,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )

    def generate(self, question: str, sources: list[dict], history: list[dict]) -> GenerationOutput:
        messages = RAG_PROMPT.format_messages(
            history=format_history(history), sources=format_sources(sources), question=question
        )
        return self._invoke(messages)

    def rewrite_question(self, question: str, history: list[dict]) -> GenerationOutput | None:
        messages = REWRITE_PROMPT.format_messages(history=format_history(history), question=question)
        return self._invoke(messages)

    def complete(self, prompt_messages: list) -> GenerationOutput:
        return self._invoke(prompt_messages)


def create_generator(settings: Settings) -> BaseGenerator:
    if settings.llm_provider == "openai":
        return OpenAIGenerator(settings)
    return ExtractiveGenerator()


_override: BaseGenerator | None = None


def get_generator(settings: Settings) -> BaseGenerator:
    if _override is not None:
        return _override
    return create_generator(settings)


def set_generator_override(gen: BaseGenerator | None) -> None:
    """Tests inject a mocked LLM here."""
    global _override
    _override = gen
