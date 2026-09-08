"""Content identity. Cheap key first (size+mtime), SHA-256 only when needed."""
from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1 << 20


def quick_key(path) -> str:
    st = Path(path).stat()
    return "%d:%d" % (st.st_size, int(st.st_mtime))


def sha256_file(path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
            read += len(block)
            if limit and read >= limit:
                break
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def head_signature(path, size: int = 4096) -> bytes:
    with open(path, "rb") as f:
        return f.read(size)
