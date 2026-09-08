"""Optional Oodle (Kraken/Mermaid/Selkie) decompression.

Oodle is a proprietary codec used by many modern engines - Unreal, Anvil,
Decima and others. It cannot be reimplemented here and is not shipped with this
application. What this module does is use a runtime library the user already
has, because games that ship Oodle-compressed data usually ship `oo2core_*.dll`
next to their executable.

Resolution order:

1. the path in settings (`oodle_dll`) or the `U3DE_OODLE_DLL` environment variable
2. any `oo2core_*` library inside the folder currently being scanned
3. nothing - callers then report the format as unsupported, with the reason

No key, no DRM and no protection is involved: this is a compression library
being asked to decompress data the user already owns.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

from app.core.logging_setup import get_logger

log = get_logger("oodle")

ENV_VAR = "U3DE_OODLE_DLL"
LIB_PATTERNS = ("oo2core_*.dll", "liboo2core*.so", "liboo2core*.dylib", "oo2core*.dylib")
MAX_SEARCH_DEPTH = 4

_cache: dict = {}


def _library_names() -> tuple:
    if sys.platform == "win32":
        return ("oo2core_*.dll",)
    return LIB_PATTERNS[1:]


def find_library(hints=None) -> Optional[Path]:
    """Locate an Oodle runtime. `hints` are folders to look inside (bounded)."""
    configured = os.environ.get(ENV_VAR)
    if configured and Path(configured).exists():
        return Path(configured)

    for hint in hints or []:
        base = Path(hint)
        if base.is_file():
            base = base.parent
        if not base.exists():
            continue
        for depth in range(MAX_SEARCH_DEPTH):
            pattern = "/".join(["*"] * depth) if depth else ""
            for name in _library_names():
                glob = ("%s/%s" % (pattern, name)) if pattern else name
                for candidate in base.glob(glob):
                    if candidate.is_file():
                        return candidate
    return None


class OodleCodec:
    """Thin ctypes wrapper around OodleLZ_Decompress."""

    def __init__(self, library: Path):
        self.path = Path(library)
        self._lib = ctypes.CDLL(str(self.path))
        try:
            self._decompress = self._lib.OodleLZ_Decompress
        except AttributeError as exc:
            raise OSError("%s exports no OodleLZ_Decompress" % self.path) from exc
        self._decompress.restype = ctypes.c_int64
        self._decompress.argtypes = [
            ctypes.c_char_p, ctypes.c_int64,          # compressed buffer + size
            ctypes.c_char_p, ctypes.c_int64,          # output buffer + size
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # fuzz, crc, verbosity
            ctypes.c_void_p, ctypes.c_int64,          # dst base + size
            ctypes.c_void_p, ctypes.c_void_p,         # callback + context
            ctypes.c_void_p, ctypes.c_int64,          # scratch + size
            ctypes.c_int,                             # thread phase
        ]

    def decompress(self, data: bytes, uncompressed_size: int) -> Optional[bytes]:
        if uncompressed_size <= 0 or uncompressed_size > (1 << 31):
            return None
        out = ctypes.create_string_buffer(uncompressed_size)
        written = self._decompress(data, len(data), out, uncompressed_size,
                                   0, 0, 0, None, 0, None, None, None, 0, 3)
        if written <= 0:
            return None
        return out.raw[:written]


def get_codec(hints=None) -> Optional[OodleCodec]:
    """Return a cached codec, or None when no runtime can be found."""
    key = tuple(str(h) for h in (hints or []))
    if key in _cache:
        return _cache[key]
    library = find_library(hints)
    codec = None
    if library is not None:
        try:
            codec = OodleCodec(library)
            log.info("Oodle runtime loaded from %s", library)
        except OSError as exc:
            log.warning("found %s but could not load it: %s", library, exc)
    _cache[key] = codec
    return codec


def set_library(path) -> Optional[OodleCodec]:
    """Point the app at a specific Oodle runtime (settings or CLI)."""
    os.environ[ENV_VAR] = str(path)
    _cache.clear()
    return get_codec()


def unavailable_reason() -> str:
    return ("Oodle-compressed data: the codec is proprietary and is not part of this "
            "application. Point the app at an oo2core library you already have "
            "(settings 'oodle_dll', or the %s environment variable) and rerun." % ENV_VAR)
