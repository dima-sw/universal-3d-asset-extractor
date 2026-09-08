"""Generic offset-table containers, the workhorse format of PS1/PS2-era games.

Countless console games store their data as: a count, a table of offsets (or
offset/size pairs), then the payloads back to back. There is no magic to key
off, so this detector proves the structure instead:

  * the count must be plausible for the file size
  * every offset must be aligned, strictly increasing and inside the file
  * the entries must tile the file with no overlap and little slack

If any of that fails the file is left alone, so a random binary is never
shredded into meaningless pieces. What comes out is re-scanned by the pipeline
like anything else, which is how a nested game archive gets peeled open one
layer at a time.
"""
from __future__ import annotations

import struct
from pathlib import Path

from app.core.fs_safety import LimitExceeded, safe_join
from app.core.logging_setup import get_logger
from app.plugins.api import (Category, ContainerExtractor, DetectionResult, ExtractionResult,
                             FileContext, FormatDetector)

log = get_logger("plugin.retro")

MIN_ENTRIES = 4
MAX_ENTRIES = 65536
MIN_SIZE = 4096
MAX_SLACK = 4096            # trailing padding we tolerate


def _plan_offsets_only(data: bytes, size: int):
    """Layout: [count][offset x count][payloads]."""
    if len(data) < 8:
        return None
    count = struct.unpack_from("<I", data, 0)[0]
    if not (MIN_ENTRIES <= count <= MAX_ENTRIES):
        return None
    table_end = 4 + 4 * count
    if table_end > len(data) or table_end > size:
        return None
    offsets = list(struct.unpack_from("<%dI" % count, data, 4))
    if offsets[0] < table_end or offsets[-1] >= size:
        return None
    for i in range(count - 1):
        if offsets[i] >= offsets[i + 1] or offsets[i] % 4:
            return None
    if offsets[-1] % 4:
        return None
    entries = [(offsets[i], offsets[i + 1] - offsets[i]) for i in range(count - 1)]
    entries.append((offsets[-1], size - offsets[-1]))
    return entries


def _plan_offset_size_pairs(data: bytes, size: int):
    """Layout: [count][(offset, size) x count][payloads]."""
    if len(data) < 12:
        return None
    count = struct.unpack_from("<I", data, 0)[0]
    if not (MIN_ENTRIES <= count <= MAX_ENTRIES):
        return None
    table_end = 4 + 8 * count
    if table_end > len(data) or table_end > size:
        return None
    raw = struct.unpack_from("<%dI" % (count * 2), data, 4)
    entries = []
    previous_end = table_end
    for i in range(count):
        offset, length = raw[i * 2], raw[i * 2 + 1]
        if length == 0:
            continue
        if offset < table_end or offset % 4 or offset + length > size:
            return None
        if offset < previous_end - 2048:            # allow small alignment overlap only
            return None
        previous_end = offset + length
        entries.append((offset, length))
    if len(entries) < MIN_ENTRIES:
        return None
    return entries


def _score(entries, size: int) -> float:
    """How convincingly do the entries tile the file?"""
    if not entries:
        return 0.0
    covered = sum(length for _o, length in entries)
    last_end = max(o + length for o, length in entries)
    slack = size - last_end
    if slack < 0 or slack > MAX_SLACK:
        return 0.0
    if any(length <= 0 for _o, length in entries):
        return 0.0
    coverage = covered / max(1, size)
    if coverage < 0.80:
        return 0.0
    return min(0.75, 0.45 + coverage * 0.3)


def plan(ctx: FileContext):
    """Return (entries, confidence, layout name) or None."""
    if ctx.size < MIN_SIZE:
        return None
    head = ctx.header
    best = None
    for name, planner in (("offset table", _plan_offsets_only),
                          ("offset/size table", _plan_offset_size_pairs)):
        entries = planner(head, ctx.size)
        if not entries:
            continue
        confidence = _score(entries, ctx.size)
        if confidence and (best is None or confidence > best[1]):
            best = (entries, confidence, name)
    return best


class OffsetTableDetector(FormatDetector):
    name = "offset_table_container"
    priority = 30           # below every real signature: only unclaimed files

    def detect(self, ctx: FileContext) -> DetectionResult:
        if ctx.looks_textual():
            return self.nope()
        result = plan(ctx)
        if result is None:
            return self.nope()
        entries, confidence, layout = result
        return self.result("Offset-table container (%s)" % layout, confidence,
                           Category.ARCHIVE, entries=len(entries), layout=layout,
                           note="structure proven by validation, not by a magic number")


class OffsetTableExtractor(ContainerExtractor):
    name = "offset_table_container"
    priority = 35

    def can_extract(self, ctx: FileContext, detection: DetectionResult) -> bool:
        return detection.format_name.startswith("Offset-table container")

    def extract(self, ctx: FileContext, dest: Path, budget, limits) -> ExtractionResult:
        res = ExtractionResult(extractor=self.name)
        result = plan(ctx)
        if result is None:
            return self.unsupported("offset table no longer validates")
        entries, _confidence, layout = result
        if len(entries) > limits.max_entries_per_archive:
            return self.unsupported("container declares %d entries (limit %d)"
                                    % (len(entries), limits.max_entries_per_archive))

        stem = ctx.path.stem[:40]
        with open(ctx.path, "rb") as fh:
            for index, (offset, length) in enumerate(entries):
                if length > limits.max_single_file_size:
                    res.skipped.append("entry %d too large" % index)
                    continue
                try:
                    fh.seek(offset)
                    payload = fh.read(length)
                    if not payload:
                        continue
                    out = safe_join(dest, "%s_%05d.dat" % (stem, index))
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(payload)
                    budget.account(len(payload))
                    res.files.append(out)
                    res.entries += 1
                except LimitExceeded as exc:
                    res.errors.append(str(exc))
                    break
                except Exception as exc:
                    res.errors.append("entry %d: %s" % (index, exc))
        res.bytes_written = budget.bytes_written
        res.ok = bool(res.files)
        log.info("%s: %d entries via %s", ctx.path.name, res.entries, layout)
        return res


def register(api) -> None:
    api.add_detector(OffsetTableDetector())
    api.add_extractor(OffsetTableExtractor())
