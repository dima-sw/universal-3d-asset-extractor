"""ISO 9660 / disc-image layer.

Reads the filesystem directly (no full-image dump) and extracts only what the
pipeline asks for. Handles 2048-byte logical sectors and 2352-byte raw sectors
(PS1/PS2 BIN, Mode 1 and Mode 2 Form 1), plus Joliet names when present.

Unsupported filesystems (UDF, XDVDFS, GD-ROM, Wii/GC) are reported as
unsupported - never as corrupt - so a plugin can add them later.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterator, Optional

from app.core.detector.base import FileContext
from app.core.extraction.base import ContainerExtractor, ExtractionResult
from app.core.fs_safety import ExtractionBudget, LimitExceeded, safe_join
from app.core.logging_setup import get_logger

log = get_logger("extract.iso")

SECTOR_2048 = 2048
RAW_2352 = 2352
RAW_2336 = 2336
SYNC = b"\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00"


@dataclass
class IsoEntry:
    name: str
    path: str
    lba: int
    size: int
    is_dir: bool = False
    children: list = field(default_factory=list)


class UnsupportedImage(Exception):
    pass


class Iso9660:
    """Minimal, defensive ISO 9660 reader."""

    def __init__(self, path):
        self.path = Path(path)
        self.fh = open(self.path, "rb")
        self.image_size = self.path.stat().st_size
        self.sector_size, self.data_offset = self._probe_layout()
        self.volume_label = ""
        self.joliet = False
        self.root: Optional[IsoEntry] = None
        self._read_volume_descriptors()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- low level ---------------------------------------------------------
    def _probe_layout(self) -> tuple:
        for sector_size, offsets in ((SECTOR_2048, (0,)), (RAW_2352, (16, 24)), (RAW_2336, (8,))):
            for data_off in offsets:
                pos = 16 * sector_size + data_off + 1
                self.fh.seek(pos)
                if self.fh.read(5) == b"CD001":
                    return sector_size, data_off
        raise UnsupportedImage("no ISO 9660 volume descriptor found "
                               "(UDF/XDVDFS/GD-ROM images need a plugin)")

    def read_sector(self, lba: int) -> bytes:
        off = lba * self.sector_size + self.data_offset
        if off < 0 or off >= self.image_size:
            return b""
        self.fh.seek(off)
        return self.fh.read(SECTOR_2048)

    def read_range(self, lba: int, size: int) -> Iterator[bytes]:
        remaining = size
        cur = lba
        while remaining > 0:
            block = self.read_sector(cur)
            if not block:
                break
            take = min(len(block), remaining)
            yield block[:take]
            remaining -= take
            cur += 1

    # -- volume descriptors ------------------------------------------------
    def _read_volume_descriptors(self) -> None:
        primary_root = None
        joliet_root = None
        for i in range(16, 80):
            sector = self.read_sector(i)
            if len(sector) < 8 or sector[1:6] != b"CD001":
                break
            vd_type = sector[0]
            if vd_type == 255:
                break
            if vd_type == 1:      # primary volume descriptor
                self.volume_label = sector[40:72].decode("ascii", "ignore").strip()
                primary_root = sector[156:190]
            elif vd_type == 2:    # supplementary (Joliet if UCS-2 escape)
                if sector[88:91] in (b"%/@", b"%/C", b"%/E"):
                    joliet_root = sector[156:190]
        record = joliet_root or primary_root
        if record is None:
            raise UnsupportedImage("no primary volume descriptor")
        self.joliet = joliet_root is not None
        lba, size, _, _ = _parse_record_core(record)
        self.root = IsoEntry(name="", path="/", lba=lba, size=size, is_dir=True)

    # -- directory walking -------------------------------------------------
    def list_dir(self, entry: IsoEntry) -> list:
        data = b"".join(self.read_range(entry.lba, entry.size))
        out = []
        pos = 0
        while pos < len(data):
            length = data[pos]
            if length == 0:
                # move to next sector boundary
                pos = ((pos // SECTOR_2048) + 1) * SECTOR_2048
                if pos >= len(data):
                    break
                continue
            record = data[pos:pos + length]
            pos += length
            if len(record) < 33:
                continue
            lba, size, flags, name_len = _parse_record_core(record)
            raw_name = record[33:33 + name_len]
            if name_len == 1 and raw_name in (b"\x00", b"\x01"):
                continue                      # '.' and '..'
            name = _decode_name(raw_name, self.joliet)
            is_dir = bool(flags & 0x02)
            child_path = str(PurePosixPath(entry.path) / name)
            out.append(IsoEntry(name=name, path=child_path, lba=lba, size=size, is_dir=is_dir))
        return out

    def walk(self, max_entries: int = 200000, max_depth: int = 32) -> Iterator[IsoEntry]:
        assert self.root is not None
        stack = [(self.root, 0)]
        seen_lbas = set()
        count = 0
        while stack:
            node, depth = stack.pop()
            if depth > max_depth:
                continue
            try:
                children = self.list_dir(node)
            except Exception as exc:
                log.debug("iso dir read failed at %s: %s", node.path, exc)
                continue
            for child in children:
                count += 1
                if count > max_entries:
                    return
                if child.is_dir:
                    if child.lba in seen_lbas:
                        continue           # self-referential image guard
                    seen_lbas.add(child.lba)
                    stack.append((child, depth + 1))
                yield child

    def extract_file(self, entry: IsoEntry, out_path: Path, budget: ExtractionBudget) -> int:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(out_path, "wb") as out:
            for block in self.read_range(entry.lba, entry.size):
                out.write(block)
                written += len(block)
                budget.bytes_written += len(block)
                if budget.bytes_written > budget.max_bytes:
                    raise LimitExceeded("extracted size limit exceeded while reading ISO")
        budget.entries += 1
        return written

    # -- platform hints ----------------------------------------------------
    def platform_hint(self) -> Optional[str]:
        try:
            names = {e.name.upper() for e in self.list_dir(self.root)}  # type: ignore[arg-type]
        except Exception:
            return None
        if "SYSTEM.CNF" in names:
            return "PS1/PS2"
        if "PSX.EXE" in names:
            return "PS1"
        if any(n.startswith(("SLUS", "SCES", "SLES", "SCUS", "SLPS")) for n in names):
            return "PlayStation"
        if "UMD_DATA.BIN" in names:
            return "PSP"
        return None


def _parse_record_core(record: bytes) -> tuple:
    lba = struct.unpack_from("<I", record, 2)[0]
    size = struct.unpack_from("<I", record, 10)[0]
    flags = record[25]
    name_len = record[32]
    return lba, size, flags, name_len


def _decode_name(raw: bytes, joliet: bool) -> str:
    if joliet:
        try:
            name = raw.decode("utf-16-be")
        except UnicodeDecodeError:
            name = raw.decode("ascii", "ignore")
    else:
        name = raw.decode("ascii", "ignore")
    return name.split(";")[0] or name


class IsoExtractor(ContainerExtractor):
    """Extracts every file from a disc image, streaming, with budget guards."""

    name = "iso9660"
    priority = 90
    formats = ("ISO 9660", "ISO 9660 PVD", "ISO 9660 (2352 raw sectors)")

    def can_extract(self, ctx: FileContext, detection) -> bool:
        if detection.format_name in self.formats:
            return True
        return detection.category.value == "disc_image" and ctx.ext in (
            "iso", "bin", "img", "mdf", "cdi", "nrg")

    def extract(self, ctx: FileContext, dest: Path, budget: ExtractionBudget, limits) -> ExtractionResult:
        res = ExtractionResult(extractor=self.name)
        try:
            image = Iso9660(ctx.path)
        except UnsupportedImage as exc:
            return self.unsupported(str(exc))
        except Exception as exc:
            res.errors.append("iso open failed: %s" % exc)
            return res
        with image:
            hint = image.platform_hint()
            if hint:
                res.skipped.append("platform hint: %s" % hint)
            for entry in image.walk(max_entries=limits.max_entries_per_archive):
                if entry.is_dir:
                    continue
                if entry.size > limits.max_single_file_size:
                    res.skipped.append("%s (too large)" % entry.path)
                    continue
                try:
                    out = safe_join(dest, entry.path.lstrip("/"))
                    image.extract_file(entry, out, budget)
                    res.files.append(out)
                    res.entries += 1
                except LimitExceeded as exc:
                    res.errors.append(str(exc))
                    break
                except Exception as exc:
                    res.errors.append("%s: %s" % (entry.path, exc))
        res.bytes_written = budget.bytes_written
        res.ok = bool(res.files)
        return res


class CueSheetExtractor(ContainerExtractor):
    """A .cue points at a .bin track; hand the pipeline the referenced image."""

    name = "cue"
    priority = 70
    formats = ()

    def can_extract(self, ctx: FileContext, detection) -> bool:
        return ctx.ext == "cue" and ctx.size < 1 << 20

    def extract(self, ctx: FileContext, dest: Path, budget, limits) -> ExtractionResult:
        res = ExtractionResult(extractor=self.name)
        try:
            text = ctx.path.read_text(errors="ignore")
        except Exception as exc:
            res.errors.append(str(exc))
            return res
        for line in text.splitlines():
            line = line.strip()
            if not line.upper().startswith("FILE "):
                continue
            parts = line.split('"')
            target = parts[1] if len(parts) >= 2 else line.split()[1]
            candidate = ctx.path.parent / target
            if candidate.exists():
                res.files.append(candidate)     # already on disk: scan in place
                res.entries += 1
        res.ok = bool(res.files)
        if not res.ok:
            res.unsupported_reason = "cue sheet references no reachable track file"
        return res


BUILTIN = [IsoExtractor, CueSheetExtractor]
