"""Heuristics for files no parser claims: entropy, embedded assets, compression."""
from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Iterator, Optional

from app.core.detector.base import FileContext, FormatDetector
from app.core.detector.signatures import EMBEDDABLE
from app.core.logging_setup import get_logger
from app.core.types import Category, DetectionResult

log = get_logger("heuristics")

CHUNK = 4 << 20
OVERLAP = 64


def shannon_entropy(data: bytes) -> float:
    """0..8 bits per byte."""
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    ent = 0.0
    for c in counts:
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return ent


def entropy_profile(path, blocks: int = 16, block_size: int = 65536) -> list:
    """Sample entropy across the file: catches 'header + compressed payload'."""
    import os
    size = os.path.getsize(path)
    if size == 0:
        return []
    step = max(block_size, size // max(1, blocks))
    out = []
    with open(path, "rb") as f:
        for i in range(blocks):
            off = min(size - 1, i * step)
            f.seek(off)
            data = f.read(min(block_size, size - off))
            if not data:
                break
            out.append((off, round(shannon_entropy(data), 3)))
    return out


@dataclass
class EmbeddedHit:
    offset: int
    format_name: str
    category: Category
    size: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "offset": self.offset,
            "offset_hex": hex(self.offset),
            "format": self.format_name,
            "category": self.category.value,
            "size": self.size,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# size resolution for carving
# ---------------------------------------------------------------------------
def _png_size(read) -> Optional[dict]:
    head = read(0, 33)
    if len(head) < 33:
        return None
    w, h = struct.unpack_from(">II", head, 16)
    bit_depth = head[24]
    color_type = head[25]
    return {"width": w, "height": h, "bit_depth": bit_depth, "color_type": color_type}


def _dds_size(read) -> Optional[dict]:
    head = read(0, 128)
    if len(head) < 128:
        return None
    h, w = struct.unpack_from("<II", head, 12)
    linear = struct.unpack_from("<I", head, 20)[0]
    mips = struct.unpack_from("<I", head, 28)[0]
    fourcc = head[84:88]
    return {"width": w, "height": h, "linear_size": linear, "mipmaps": mips,
            "fourcc": fourcc.decode("ascii", "ignore").strip("\x00")}


def _carve_end_png(data: bytes, start: int) -> Optional[int]:
    idx = data.find(b"IEND\xaeB`\x82", start)
    return idx + 8 if idx >= 0 else None


def _carve_end_jpeg(data: bytes, start: int) -> Optional[int]:
    idx = data.find(b"\xff\xd9", start + 2)
    return idx + 2 if idx >= 0 else None


class EmbeddedAssetScanner:
    """Streams a binary looking for known signatures at any offset.

    Used for unknown containers, old console data, executables with baked assets.
    """

    def __init__(self, signatures=None, max_bytes: int = 256 << 20, max_hits: int = 4096):
        self.signatures = signatures or EMBEDDABLE
        self.max_bytes = max_bytes
        self.max_hits = max_hits

    def scan(self, path, skip_offset_zero: bool = True) -> list:
        hits: list = []
        scanned = 0
        with open(path, "rb") as f:
            base = 0
            carry = b""
            while scanned < self.max_bytes:
                block = f.read(CHUNK)
                if not block:
                    break
                buf = carry + block
                buf_base = base - len(carry)
                for sig in self.signatures:
                    start = 0
                    while True:
                        idx = buf.find(sig.magic, start)
                        if idx < 0:
                            break
                        abs_off = buf_base + idx
                        start = idx + 1
                        if abs_off == 0 and skip_offset_zero:
                            continue
                        hits.append(EmbeddedHit(abs_off, sig.format_name, sig.category))
                        if len(hits) >= self.max_hits:
                            return self._finish(path, hits)
                carry = buf[-OVERLAP:]
                base += len(block)
                scanned += len(block)
        return self._finish(path, hits)

    def _finish(self, path, hits: list) -> list:
        hits.sort(key=lambda h: h.offset)
        for hit in hits:
            try:
                self._enrich(path, hit)
            except Exception as exc:               # never let a hit kill the scan
                log.debug("enrich failed at %s: %s", hex(hit.offset), exc)
        return hits

    def _enrich(self, path, hit: EmbeddedHit) -> None:
        def read(rel: int, n: int) -> bytes:
            with open(path, "rb") as f:
                f.seek(hit.offset + rel)
                return f.read(n)

        if hit.format_name == "PNG":
            info = _png_size(read)
            if info:
                hit.metadata.update(info)
        elif hit.format_name == "DirectDraw Surface":
            info = _dds_size(read)
            if info:
                hit.metadata.update(info)


def carve(path, hit: EmbeddedHit, out_path, max_size: int = 64 << 20) -> Optional[int]:
    """Extract one embedded asset to disk when its end can be determined."""
    with open(path, "rb") as f:
        f.seek(hit.offset)
        data = f.read(max_size)
    end = None
    if hit.format_name == "PNG":
        end = _carve_end_png(data, 0)
    elif hit.format_name == "JPEG":
        end = _carve_end_jpeg(data, 0)
    elif hit.format_name == "DirectDraw Surface" and len(data) > 128:
        info = _dds_size(lambda o, n: data[o:o + n])
        if info and info.get("linear_size"):
            end = 128 + info["linear_size"] * max(1, info.get("mipmaps", 1))
            end = min(end, len(data))
    if not end or end <= 0:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data[:end])
    hit.size = end
    return end


# ---------------------------------------------------------------------------
# compression
# ---------------------------------------------------------------------------
class CompressionDetector:
    """Recognises raw compressed blobs that carry no container magic."""

    name = "compression"

    def detect(self, data: bytes) -> Optional[str]:
        if len(data) < 4:
            return None
        if data[:2] in (b"\x78\x01", b"\x78\x5e", b"\x78\x9c", b"\x78\xda"):
            return "zlib"
        if data[:3] == b"\x1f\x8b\x08":
            return "gzip"
        if data[:4] == b"\x28\xb5\x2f\xfd":
            return "zstd"
        if data[:4] == b"\x04\x22\x4d\x18":
            return "lz4"
        if data[:1] == b"\x5d" and data[1:3] == b"\x00\x00":
            return "lzma_alone"
        return None

    def try_decompress(self, data: bytes, kind: Optional[str] = None,
                       max_out: int = 64 << 20) -> Optional[bytes]:
        kind = kind or self.detect(data)
        try:
            if kind == "zlib":
                return zlib.decompressobj().decompress(data, max_out)
            if kind == "gzip":
                return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(data, max_out)
            if kind == "lzma_alone":
                import lzma
                return lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(data, max_out)
            if kind == "zstd":
                import zstandard  # optional dependency
                return zstandard.ZstdDecompressor().decompress(data, max_output_size=max_out)
            if kind == "lz4":
                import lz4.frame  # optional dependency
                return lz4.frame.decompress(data)
        except Exception as exc:
            log.debug("decompress %s failed: %s", kind, exc)
        return None

    def find_deflate_streams(self, data: bytes, limit: int = 32) -> Iterator[tuple]:
        """Yield (offset, decompressed) for raw zlib streams inside a blob."""
        found = 0
        for i in range(len(data) - 2):
            if data[i] != 0x78 or data[i + 1] not in (0x01, 0x5E, 0x9C, 0xDA):
                continue
            try:
                out = zlib.decompressobj().decompress(data[i:], 16 << 20)
            except zlib.error:
                continue
            if len(out) >= 64:
                yield i, out
                found += 1
                if found >= limit:
                    return


class UnknownBinaryDetector(FormatDetector):
    """Last resort: classify the blob, never claim it is corrupt."""

    name = "unknown_binary"
    priority = 1

    def detect(self, ctx: FileContext) -> DetectionResult:
        if ctx.size == 0:
            return self.result("Empty file", 1.0, Category.OTHER, empty=True)
        head = ctx.header[:65536]
        ent = shannon_entropy(head)
        comp = CompressionDetector().detect(head)
        meta = {"entropy": round(ent, 3), "size": ctx.size}
        if comp:
            meta["compression"] = comp
            return self.result("Compressed blob (%s)" % comp, 0.55, Category.COMPRESSED, **meta)
        if ent > 7.5:
            meta["note"] = "high entropy: compressed or encrypted payload"
        elif ent < 1.0:
            meta["note"] = "very low entropy: padding or sparse data"
        if ctx.looks_textual():
            return self.result("Text data", 0.4, Category.OTHER, **meta)
        return self.result("Unknown binary", 0.2, Category.UNKNOWN, **meta)
