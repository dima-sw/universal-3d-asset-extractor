"""Extraction facade."""
from __future__ import annotations

from pathlib import Path

from app.core.detector.base import FileContext
from app.core.extraction.base import ContainerExtractor, ExtractionResult
from app.core.fs_safety import ExtractionBudget
from app.core.logging_setup import get_logger, unsupported_logger
from app.core.registry import EXTRACTORS
from app.core.types import Category, DetectionResult

log = get_logger("extract")
_bootstrapped = False


def bootstrap() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    from app.core.extraction import archives, iso
    for cls in archives.BUILTIN + iso.BUILTIN:
        EXTRACTORS.register(cls())
    _bootstrapped = True


CONTAINER_CATEGORIES = {Category.ARCHIVE, Category.DISC_IMAGE, Category.COMPRESSED,
                        Category.GAME_CONTAINER}


def is_container(detection: DetectionResult) -> bool:
    return detection.category in CONTAINER_CATEGORIES


def find_extractor(ctx: FileContext, detection: DetectionResult):
    bootstrap()
    for ex in EXTRACTORS:
        try:
            if ex.can_extract(ctx, detection):
                return ex
        except Exception as exc:
            log.debug("extractor %s can_extract failed: %s", ex.name, exc)
    return None


def extract(ctx: FileContext, detection: DetectionResult, dest: Path, limits) -> ExtractionResult:
    """Extract one container into `dest`. Never raises for content problems."""
    ex = find_extractor(ctx, detection)
    if ex is None:
        unsupported_logger().info("no extractor for %s (%s)", ctx.path, detection.format_name)
        return ExtractionResult(ok=False, unsupported_reason="no extractor for %s"
                                % detection.format_name)
    dest.mkdir(parents=True, exist_ok=True)
    budget = ExtractionBudget(limits.max_extracted_size, limits.max_entries_per_archive,
                              limits.max_compression_ratio, compressed_size=ctx.size)
    log.info("extracting %s with %s", ctx.path.name, ex.name)
    try:
        res = ex.extract(ctx, dest, budget, limits)
    except Exception as exc:
        log.warning("extractor %s crashed on %s: %s", ex.name, ctx.path, exc)
        return ExtractionResult(ok=False, extractor=ex.name, errors=["extractor crashed: %s" % exc])
    res.extractor = res.extractor or ex.name
    if res.unsupported_reason:
        unsupported_logger().info("%s: %s", ctx.path, res.unsupported_reason)
    return res


__all__ = ["ContainerExtractor", "ExtractionResult", "extract", "is_container",
           "find_extractor", "bootstrap"]
