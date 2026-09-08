"""Container extraction interface.

An archive is never an output. It is extracted, then its content is re-scanned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.detector.base import FileContext
from app.core.types import DetectionResult


@dataclass
class ExtractionResult:
    ok: bool = False
    files: list = field(default_factory=list)       # list[Path]
    dirs: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    extractor: str = ""
    entries: int = 0
    bytes_written: int = 0
    unsupported_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "extractor": self.extractor, "entries": self.entries,
            "files": len(self.files), "bytes": self.bytes_written,
            "errors": self.errors[:20], "skipped": self.skipped[:20],
            "unsupported_reason": self.unsupported_reason,
        }


class ContainerExtractor:
    """Base class for anything that turns one file into many files."""

    name: str = "extractor"
    priority: int = 50
    formats: tuple = ()          # format_name values it claims

    def can_extract(self, ctx: FileContext, detection: DetectionResult) -> bool:
        return detection.format_name in self.formats

    def extract(self, ctx: FileContext, dest: Path, budget, limits) -> ExtractionResult:  # pragma: no cover
        raise NotImplementedError

    def unsupported(self, reason: str) -> ExtractionResult:
        return ExtractionResult(ok=False, extractor=self.name, unsupported_reason=reason)
