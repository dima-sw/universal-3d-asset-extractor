"""Animation parser facade."""
from __future__ import annotations

from app.core.detector.base import FileContext, FormatDetector
from app.core.registry import ANIMATION_PARSERS, DETECTORS
from app.core.types import Category, DetectionResult

_bootstrapped = False


class BvhDetector(FormatDetector):
    name = "bvh"
    priority = 86

    def detect(self, ctx: FileContext) -> DetectionResult:
        head = ctx.header[:64].lstrip()
        if head[:9].upper() != b"HIERARCHY":
            return self.nope()
        return self.result("BVH", 0.93, Category.ANIMATION)


def bootstrap() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    from app.formats.animations.bvh import BvhParser
    ANIMATION_PARSERS.register(BvhParser())
    DETECTORS.register(BvhDetector())
    _bootstrapped = True
