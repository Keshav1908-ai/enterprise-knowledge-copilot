"""Test helpers: build real PDF/DOCX bytes without extra dependencies, and a mocked LLM."""

from __future__ import annotations

import io

from app.generation.llm import BaseGenerator, GenerationOutput


def make_pdf(pages: list[str]) -> bytes:
    """Minimal but valid multi-page PDF with extractable Helvetica text."""
    objs: list[bytes] = []
    n = len(pages)
    page_ids = [3 + i * 2 for i in range(n)]
    font_id = 3 + n * 2
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        lines = []
        y = 760
        for line in text.split("\n"):
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            lines.append(f"BT /F1 11 Tf 50 {y} Td ({safe}) Tj ET")
            y -= 16
        stream = "\n".join(lines).encode()
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {page_ids[i] + 1} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def make_docx(paragraphs: list[str], heading: str | None = None) -> bytes:
    import docx

    d = docx.Document()
    if heading:
        d.add_heading(heading, level=1)
    for p in paragraphs:
        d.add_paragraph(p)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


class MockLLM(BaseGenerator):
    """Deterministic LLM stand-in that records every call it receives."""

    provider = "mock"
    model = "mock-llm"

    def __init__(self, answer: str | None = None, rewrite: str | None = None, fail: bool = False) -> None:
        self.answer = answer
        self.rewrite = rewrite
        self.fail = fail
        self.calls: list[dict] = []
        self.rewrite_calls: list[dict] = []

    def generate(self, question: str, sources: list[dict], history: list[dict]) -> GenerationOutput:
        self.calls.append({"question": question, "sources": sources, "history": history})
        if self.fail:
            from app.core.exceptions import ProviderError

            raise ProviderError("mock LLM failure")
        text = self.answer
        if text is None:
            first = sources[0]["text"].split(".")[0]
            text = f"{first} [1]."
        return GenerationOutput(text=text, provider=self.provider, model=self.model, latency_ms=12.5,
                                input_tokens=100, output_tokens=20)

    def rewrite_question(self, question: str, history: list[dict]) -> GenerationOutput | None:
        self.rewrite_calls.append({"question": question, "history": history})
        if self.rewrite is None:
            return None
        return GenerationOutput(text=self.rewrite, provider=self.provider, model=self.model, latency_ms=3.0,
                                input_tokens=30, output_tokens=10)
