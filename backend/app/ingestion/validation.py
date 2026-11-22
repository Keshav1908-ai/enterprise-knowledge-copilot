"""Upload validation: extension, size, and content (magic bytes / encoding) checks."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import FileTooLargeError, UnsupportedFileError, ValidationFailedError

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\- ()]+")


@dataclass(frozen=True)
class ValidatedFile:
    filename: str
    extension: str
    content: bytes
    content_hash: str

    @property
    def size(self) -> int:
        return len(self.content)


def sanitize_filename(name: str) -> str:
    base = Path(name or "").name  # strip any directory components
    base = _SAFE_NAME.sub("_", base).strip(" .")
    return base[:200] or "document"


def _check_content(ext: str, content: bytes) -> None:
    if ext == ".pdf":
        if not content.lstrip()[:5].startswith(b"%PDF"):
            raise UnsupportedFileError("File has a .pdf extension but is not a valid PDF")
    elif ext == ".docx":
        if not content.startswith(b"PK"):
            raise UnsupportedFileError("File has a .docx extension but is not a valid DOCX archive")
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                if "word/document.xml" not in zf.namelist():
                    raise UnsupportedFileError("DOCX archive does not contain word/document.xml")
        except zipfile.BadZipFile as exc:
            raise UnsupportedFileError("Corrupted DOCX archive") from exc
    elif ext in {".txt", ".md"}:
        if b"\x00" in content[:4096]:
            raise UnsupportedFileError("Text file appears to be binary")
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content.decode("latin-1")
            except UnicodeDecodeError as exc:  # pragma: no cover - latin-1 decodes everything
                raise UnsupportedFileError("Text file is not valid UTF-8 or Latin-1") from exc


def validate_upload(filename: str, content: bytes, settings: Settings) -> ValidatedFile:
    safe = sanitize_filename(filename)
    ext = Path(safe).suffix.lower()
    if ext not in settings.allowed_extension_set:
        allowed = ", ".join(sorted(settings.allowed_extension_set))
        raise UnsupportedFileError(f"Unsupported file type '{ext or 'none'}'. Allowed: {allowed}")
    if not content:
        raise ValidationFailedError("Uploaded file is empty")
    if len(content) > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"File exceeds the maximum size of {settings.max_upload_size_mb} MB",
            details={"size_bytes": len(content), "limit_bytes": settings.max_upload_bytes},
        )
    _check_content(ext, content)
    return ValidatedFile(
        filename=safe, extension=ext, content=content, content_hash=hashlib.sha256(content).hexdigest()
    )
