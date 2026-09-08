"""CRI Middleware plugin (PS2 / Dreamcast / arcade era).

AFS is the archive CRI shipped in a huge number of PS2 games (Dragon Ball
Budokai Tenkaichi, Sakura Taisen, Shenmue ports, ...). It is a flat container:
a table of (offset, size) pairs plus an optional name table at the end. Peeling
it open lets the pipeline keep descending into whatever the game stores inside.

Also identifies the CRI media formats that usually sit next to it (ADX audio,
PSS/SFD video) so they are labelled instead of ending up as "unknown binary".
"""
from __future__ import annotations

import struct
from pathlib import Path

from app.core.fs_safety import LimitExceeded, safe_join, sanitize_component
from app.core.logging_setup import get_logger
from app.plugins.api import (Category, ContainerExtractor, DetectionResult, ExtractionResult,
                             FileContext, FormatDetector, GameProfile)

log = get_logger("plugin.cri")

AFS_MAGIC = b"AFS\x00"
SECTOR = 2048


class AfsDetector(FormatDetector):
    name = "cri_afs"
    priority = 88

    def detect(self, ctx: FileContext) -> DetectionResult:
        if ctx.header[:4] != AFS_MAGIC or len(ctx.header) < 16:
            return self.nope()
        count = struct.unpack_from("<I", ctx.header, 4)[0]
        if not (0 < count < 1_000_000):
            return self.nope()
        first_offset, first_size = struct.unpack_from("<II", ctx.header, 8)
        if first_offset < 8 or first_offset > ctx.size or first_size > ctx.size:
            return self.nope()
        return self.result("CRI AFS archive", 0.95, Category.ARCHIVE, entries=count,
                           engine="CRI Middleware")


class CriMediaDetector(FormatDetector):
    name = "cri_media"
    priority = 84

    def detect(self, ctx: FileContext) -> DetectionResult:
        head = ctx.header
        if head[:2] == b"\x80\x00" and head[4:6] == b"\x03\x12":
            return self.result("CRI ADX audio", 0.9, Category.AUDIO, engine="CRI Middleware")
        if head[:4] == b"\x00\x00\x01\xba":
            fmt = "Sony PSS video" if ctx.ext == "pss" else "MPEG program stream"
            return self.result(fmt, 0.85, Category.OTHER,
                               note="video container; no 3D assets inside")
        if head[:4] == b"CRID":
            return self.result("CRI USM video", 0.9, Category.OTHER,
                               engine="CRI Middleware")
        if head[:4] == b"@UTF":
            return self.result("CRI UTF table", 0.85, Category.OTHER,
                               engine="CRI Middleware")
        return self.nope()


class AfsExtractor(ContainerExtractor):
    name = "cri_afs"
    priority = 90
    formats = ("CRI AFS archive",)

    def extract(self, ctx: FileContext, dest: Path, budget, limits) -> ExtractionResult:
        res = ExtractionResult(extractor=self.name)
        header = ctx.read_at(0, 8)
        if len(header) < 8 or header[:4] != AFS_MAGIC:
            return self.unsupported("not an AFS archive")
        count = struct.unpack_from("<I", header, 4)[0]
        if count > limits.max_entries_per_archive:
            return self.unsupported("AFS declares %d entries (limit %d)"
                                    % (count, limits.max_entries_per_archive))

        table = ctx.read_at(8, (count + 1) * 8)
        if len(table) < count * 8:
            return self.unsupported("AFS table truncated")
        entries = []
        for i in range(count):
            offset, size = struct.unpack_from("<II", table, i * 8)
            entries.append((offset, size))

        names = self._read_names(ctx, table, count)
        written = 0
        with open(ctx.path, "rb") as fh:
            for index, (offset, size) in enumerate(entries):
                if size == 0:
                    continue
                if offset + size > ctx.size or offset < 8:
                    res.errors.append("entry %d points outside the archive" % index)
                    continue
                if size > limits.max_single_file_size:
                    res.skipped.append("entry %d (%d bytes, too large)" % (index, size))
                    continue
                name = names.get(index) or "%05d.bin" % index
                try:
                    out = safe_join(dest, sanitize_component(name))
                    fh.seek(offset)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with open(out, "wb") as target:
                        remaining = size
                        while remaining > 0:
                            chunk = fh.read(min(1 << 20, remaining))
                            if not chunk:
                                break
                            target.write(chunk)
                            remaining -= len(chunk)
                    budget.account(size)
                    res.files.append(out)
                    res.entries += 1
                    written += size
                except LimitExceeded as exc:
                    res.errors.append(str(exc))
                    break
                except Exception as exc:
                    res.errors.append("entry %d: %s" % (index, exc))
        res.bytes_written = budget.bytes_written
        res.ok = bool(res.files)
        log.info("AFS %s: %d/%d entries, %.1f MB", ctx.path.name, res.entries, count,
                 written / 1e6)
        return res

    @staticmethod
    def _read_names(ctx: FileContext, table: bytes, count: int) -> dict:
        """The optional trailing block holds 48-byte records with real names."""
        names: dict = {}
        if len(table) < (count + 1) * 8:
            return names
        meta_offset, meta_size = struct.unpack_from("<II", table, count * 8)
        if meta_offset <= 0 or meta_offset >= ctx.size or meta_size < 48:
            return names
        if meta_size > 48 * count + 4096:
            return names
        block = ctx.read_at(meta_offset, min(meta_size, 48 * count))
        for i in range(min(count, len(block) // 48)):
            raw = block[i * 48: i * 48 + 32]
            name = raw.split(b"\x00")[0].decode("ascii", "ignore").strip()
            if name:
                names[i] = name
        return names


def register(api) -> None:
    api.add_detector(AfsDetector())
    api.add_detector(CriMediaDetector())
    api.add_extractor(AfsExtractor())
    api.add_game_profile(GameProfile(name="CRI Middleware", platforms=["ps2", "dreamcast"],
                                     detectors=[AfsDetector()], extractors=[AfsExtractor()]))
