"""Follow-up question resolution.

With an LLM configured, the LLM rewrites the follow-up into a standalone query.
Without one, a transparent heuristic merges the follow-up with the previous user turn
when it looks context-dependent (short, or uses pronouns / ellipsis).
Either way the resolved question is used for *fresh retrieval* — history never replaces
document evidence.
"""

from __future__ import annotations

import re

from app.core.logging import get_logger
from app.generation.llm import BaseGenerator, GenerationOutput

logger = get_logger(__name__)

_DEPENDENT = re.compile(
    r"\b(it|its|that|this|those|these|they|them|their|there|he|she|his|her|same|above|previous|"
    r"what about|how about|and the|also|more|else|such|one)\b",
    re.I,
)


def needs_context(question: str) -> bool:
    words = re.findall(r"\w+", question)
    return len(words) <= 6 or bool(_DEPENDENT.search(question))


def heuristic_rewrite(question: str, history: list[dict]) -> str:
    previous = [m["content"] for m in history if m["role"] == "user"]
    if not previous or not needs_context(question):
        return question
    return f"{previous[-1].rstrip('?.! ')} — {question}"


def resolve_question(
    question: str, history: list[dict], generator: BaseGenerator, enabled: bool = True
) -> tuple[str, GenerationOutput | None, str]:
    """Returns (standalone_question, llm_output_if_any, method)."""
    if not history or not enabled:
        return question, None, "none"
    try:
        out = generator.rewrite_question(question, history)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM question rewrite failed (%s); falling back to heuristic", type(exc).__name__)
        out = None
    if out is not None and out.text.strip():
        rewritten = out.text.strip().strip('"').splitlines()[0][:500]
        return rewritten, out, "llm"
    rewritten = heuristic_rewrite(question, history)
    return rewritten, None, "heuristic" if rewritten != question else "none"
