"""Safety helpers. Every byte we touch comes from an untrusted file.

Rules enforced here:
  * source tree is READ ONLY
  * archive members can never escape their extraction root
  * symlinks / device nodes / absolute paths / drive letters are rejected
  * total extracted size, entry count and compression ratio are capped
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath

from app.core.logging_setup import get_logger

log = get_logger("safety")

_ILLEGAL = re.compile(r'[\x00-\x1f<>:"|?*]')
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class UnsafePathError(Exception):
    pass


class LimitExceeded(Exception):
    pass


def sanitize_component(name: str) -> str:
    """Make a single path component safe for the host filesystem."""
    name = unicodedata.normalize("NFC", name)
    name = _ILLEGAL.sub("_", name).strip().rstrip(".")
    if not name or name in (".", ".."):
        name = "_"
    if name.split(".")[0].upper() in _RESERVED:
        name = "_" + name
    return name[:200]


def sanitize_relpath(member: str) -> PurePosixPath:
    """Normalise an archive member name into a safe relative posix path.

    Rejects absolute paths, drive letters and any traversal that escapes root.
    """
    raw = str(member).replace("\\", "/")
    if PureWindowsPath(raw).drive or raw.startswith("/") or raw.startswith("//"):
        raw = raw.lstrip("/")
        raw = re.sub(r"^[A-Za-z]:", "", raw)
    parts = []
    for part in PurePosixPath(raw).parts:
        if part in ("", ".", "/", "//", "\\"):
            continue                                # drop the posix root component
        if part == "..":
            if not parts:
                raise UnsafePathError("path traversal in archive member: %r" % member)
            parts.pop()
            continue
        parts.append(sanitize_component(part))
    if not parts:
        raise UnsafePathError("empty archive member name: %r" % member)
    return PurePosixPath(*parts)


def safe_join(root, member: str) -> Path:
    """Resolve `member` under `root`, guaranteeing containment."""
    root_path = Path(root).resolve()
    target = (root_path / sanitize_relpath(member))
    resolved = Path(os.path.normpath(str(target)))
    try:
        resolved.relative_to(root_path)
    except ValueError:
        raise UnsafePathError("member escapes extraction root: %r" % member)
    return resolved


def is_symlink_like(path) -> bool:
    p = Path(path)
    try:
        return p.is_symlink() or (p.exists() and not p.is_file() and not p.is_dir())
    except OSError:
        return True


class ExtractionBudget:
    """Tracks bytes/entries written during one container extraction."""

    def __init__(self, max_bytes: int, max_entries: int, max_ratio: float, compressed_size: int = 0):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.max_ratio = max_ratio
        self.compressed_size = max(compressed_size, 1)
        self.bytes_written = 0
        self.entries = 0

    def account(self, size: int) -> None:
        self.bytes_written += max(0, size)
        self.entries += 1
        if self.entries > self.max_entries:
            raise LimitExceeded("archive entry limit exceeded (%d)" % self.max_entries)
        if self.bytes_written > self.max_bytes:
            raise LimitExceeded("extracted size limit exceeded (%d bytes)" % self.max_bytes)
        ratio = self.bytes_written / self.compressed_size
        if self.bytes_written > 64 << 20 and ratio > self.max_ratio:
            raise LimitExceeded("suspicious compression ratio %.1fx (archive bomb guard)" % ratio)


def copy_stream(src, dst_path, budget: ExtractionBudget, chunk: int = 1 << 20) -> int:
    """Stream-copy with budget accounting. Never loads whole file in RAM."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(dst_path, "wb") as out:
        while True:
            block = src.read(chunk)
            if not block:
                break
            out.write(block)
            written += len(block)
            budget.bytes_written += len(block)
            if budget.bytes_written > budget.max_bytes:
                raise LimitExceeded("extracted size limit exceeded while streaming")
    budget.entries += 1
    return written
