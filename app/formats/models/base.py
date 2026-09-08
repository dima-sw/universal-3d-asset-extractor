"""Model parser interface. Parsers produce ModelAsset, nothing else."""
from __future__ import annotations

from typing import Optional

from app.core.detector.base import FileContext
from app.core.types import DetectionResult, ModelAsset


class ModelParser:
    name: str = "model_parser"
    priority: int = 50
    formats: tuple = ()

    def can_parse(self, ctx: FileContext, detection: Optional[DetectionResult] = None) -> bool:
        if detection is not None and detection.format_name in self.formats:
            return True
        return False

    def parse(self, ctx: FileContext) -> ModelAsset:  # pragma: no cover - interface
        raise NotImplementedError


class ParseError(Exception):
    pass
