"""Text cleaning and normalisation applied before chunking."""

from __future__ import annotations

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# A single line break inside a sentence (PDF soft wrap): previous line ends mid-sentence,
# next line continues in lower case.
_SOFT_WRAP = re.compile(r"(?<=[a-z0-9,;:\-])\n(?=[a-z0-9(])")
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def clean_text(text: str) -> str:
    """Normalise unicode, repair PDF artefacts and collapse whitespace.

    Paragraph boundaries (blank lines) are preserved because the chunker uses them
    as preferred split points.
    """
    if not text:
        return ""
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # "infor-\nmation" -> "information"
    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _SOFT_WRAP.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
