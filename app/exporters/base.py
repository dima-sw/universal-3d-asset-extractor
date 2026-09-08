"""Exporter interface. Exporters only ever see the internal representation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.types import ModelAsset


@dataclass
class ExportResult:
    ok: bool = False
    files: list = field(default_factory=list)
    format: str = ""
    error: Optional[str] = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "format": self.format,
                "files": [str(f) for f in self.files],
                "error": self.error, "warnings": self.warnings}


class Exporter:
    name: str = "exporter"
    format: str = ""
    extension: str = ""
    supports_skeleton: bool = False
    supports_animation: bool = False
    priority: int = 50

    def export(self, model: ModelAsset, out_dir: Path, name: Optional[str] = None,
               options: Optional[dict] = None) -> ExportResult:  # pragma: no cover
        raise NotImplementedError

    def unsupported(self, reason: str) -> ExportResult:
        return ExportResult(ok=False, format=self.format, error=reason)
