"""Structured logging setup. Never logs API keys or document contents."""

from __future__ import annotations

import logging
import re
import sys

_SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]+)")


class RedactSecretsFilter(logging.Filter):
    """Masks anything that looks like an API key before it reaches a log sink."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _SECRET_PATTERN.sub("[REDACTED]", record.msg)
        if record.args:
            record.args = tuple(
                _SECRET_PATTERN.sub("[REDACTED]", a) if isinstance(a, str) else a for a in record.args
            )
        return True


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if getattr(root, "_ekc_configured", False):
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    handler.addFilter(RedactSecretsFilter())
    root.handlers = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "chromadb", "urllib3", "sentence_transformers", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root._ekc_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
