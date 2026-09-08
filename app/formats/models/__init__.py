"""Model parser facade."""
from __future__ import annotations

from typing import Optional

from app.core.detector.base import FileContext
from app.core.logging_setup import get_logger
from app.core.registry import MODEL_PARSERS
from app.core.types import DetectionResult, ModelAsset
from app.formats.models.base import ModelParser, ParseError

log = get_logger("model")
_bootstrapped = False


def bootstrap() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    from app.formats.models.gltf import GltfParser
    from app.formats.models.simple import PlyParser, StlParser
    from app.formats.models.wavefront import ObjParser
    for cls in (GltfParser, ObjParser, StlParser, PlyParser):
        MODEL_PARSERS.register(cls())
    _bootstrapped = True


def find_parser(ctx: FileContext, detection: Optional[DetectionResult] = None):
    bootstrap()
    for parser in MODEL_PARSERS:
        try:
            if parser.can_parse(ctx, detection):
                return parser
        except Exception as exc:
            log.debug("parser %s can_parse failed: %s", parser.name, exc)
    return None


def parse_model(ctx: FileContext, detection: Optional[DetectionResult] = None) -> Optional[ModelAsset]:
    parser = find_parser(ctx, detection)
    if parser is None:
        return None
    model = parser.parse(ctx)
    if model is not None:
        model.metadata.setdefault("source", str(ctx.path))
        model.metadata.setdefault("parser", parser.name)
    return model


__all__ = ["ModelParser", "ParseError", "parse_model", "find_parser", "bootstrap"]
