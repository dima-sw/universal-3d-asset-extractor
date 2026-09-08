"""Detector interface + the file context detectors inspect.

Detection order (highest signal first):
    1. magic bytes  2. header structure  3. file structure
    4. filename     5. directory structure
    6. engine signatures                 7. heuristics
Extension alone never decides.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from app.core.types import Category, DetectionResult


class FileContext:
    """Lazy, memory-safe view over one candidate file."""

    def __init__(self, path, header_size: int = 64 * 1024, siblings: Optional[list] = None,
                 depth: int = 0, container: Optional[str] = None):
        self.path = Path(path)
        self._header_size = header_size
        self._header: Optional[bytes] = None
        self._tail: Optional[bytes] = None
        self._size: Optional[int] = None
        self._siblings = siblings
        self.depth = depth
        self.container = container          # virtual path of the container we came from

    # --- basics ----------------------------------------------------------
    @property
    def name(self) -> str:
        return self.path.name

    @property
    def ext(self) -> str:
        return self.path.suffix.lower().lstrip(".")

    @property
    def size(self) -> int:
        if self._size is None:
            try:
                self._size = self.path.stat().st_size
            except OSError:
                self._size = 0
        return self._size

    # --- bytes -----------------------------------------------------------
    @property
    def header(self) -> bytes:
        if self._header is None:
            try:
                with open(self.path, "rb") as f:
                    self._header = f.read(self._header_size)
            except OSError:
                self._header = b""
        return self._header

    @property
    def tail(self) -> bytes:
        if self._tail is None:
            n = min(64 * 1024, self.size)
            try:
                with open(self.path, "rb") as f:
                    f.seek(max(0, self.size - n))
                    self._tail = f.read(n)
            except OSError:
                self._tail = b""
        return self._tail

    def read_at(self, offset: int, length: int) -> bytes:
        if offset < 0 or length <= 0 or offset >= self.size:
            return b""
        try:
            with open(self.path, "rb") as f:
                f.seek(offset)
                return f.read(length)
        except OSError:
            return b""

    def starts_with(self, magic: bytes, offset: int = 0) -> bool:
        return self.header[offset:offset + len(magic)] == magic

    # --- context ---------------------------------------------------------
    @property
    def siblings(self) -> list:
        if self._siblings is None:
            try:
                self._siblings = [e.name for e in os.scandir(self.path.parent)]
            except OSError:
                self._siblings = []
        return self._siblings

    def sibling_exists(self, name: str) -> bool:
        lowered = {s.lower() for s in self.siblings}
        return name.lower() in lowered

    def text_head(self, n: int = 4096, errors: str = "ignore") -> str:
        return self.header[:n].decode("utf-8", errors)

    def looks_textual(self, sample: int = 2048) -> bool:
        chunk = self.header[:sample]
        if not chunk:
            return False
        if b"\x00" in chunk:
            return False
        printable = sum(1 for b in chunk if 9 <= b <= 13 or 32 <= b < 127)
        return printable / len(chunk) > 0.92


class FormatDetector:
    """Base class. Subclasses override `detect`."""

    name: str = "detector"
    priority: int = 50          # higher runs first
    categories: tuple = ()

    def detect(self, ctx: FileContext) -> DetectionResult:  # pragma: no cover - interface
        raise NotImplementedError

    # convenience
    def result(self, format_name: str, confidence: float, category: Category,
               **metadata) -> DetectionResult:
        return DetectionResult(
            detected=True,
            format_name=format_name,
            confidence=max(0.0, min(1.0, confidence)),
            category=category,
            metadata=metadata,
            detector=self.name,
        )

    @staticmethod
    def nope() -> DetectionResult:
        return DetectionResult()
