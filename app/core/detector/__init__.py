"""Detection facade: bootstrap built-ins and run the registry over a file."""
from __future__ import annotations

from app.core.detector.base import FileContext, FormatDetector
from app.core.registry import DETECTORS
from app.core.types import Category, DetectionResult

_bootstrapped = False


def bootstrap() -> None:
    """Register built-in detectors once. Plugins register on top."""
    global _bootstrapped
    if _bootstrapped:
        return
    from app.core.detector import builtin, heuristics
    for cls in (builtin.MagicDetector, builtin.TextModelDetector,
                builtin.RenderWareDetector, builtin.UnityAssetsDetector,
                builtin.UnrealAssetDetector, builtin.DirectoryContextDetector,
                heuristics.UnknownBinaryDetector):
        DETECTORS.register(cls())
    _bootstrapped = True


def identify(ctx: FileContext, deep: bool = False) -> DetectionResult:
    """Run detectors in priority order; keep the strongest result.

    Weak results (extension-only, unknown-binary) never mask a strong one.
    """
    bootstrap()
    best = DetectionResult()
    alternatives = []
    for det in DETECTORS:
        try:
            res = det.detect(ctx)
        except Exception as exc:                      # a broken detector must not stop us
            from app.core.logging_setup import get_logger
            get_logger("detector").debug("%s failed on %s: %s", det.name, ctx.path, exc)
            continue
        if not res.detected:
            continue
        alternatives.append(res)
        if res.confidence > best.confidence:
            best = res
        if best.confidence >= 0.95 and not deep:
            break
    if alternatives:
        best.metadata.setdefault("alternatives",
                                 [a.to_dict() for a in alternatives if a is not best][:5])
    if not best.detected:
        best = DetectionResult(True, "Unknown", 0.1, Category.UNKNOWN, {}, "fallback")
    return best


__all__ = ["FileContext", "FormatDetector", "DetectionResult", "identify", "bootstrap"]
