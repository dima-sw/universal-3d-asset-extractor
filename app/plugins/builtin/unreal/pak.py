"""Unreal .pak reader, versions 1-11.

Two index layouts exist:

* up to version 9 - a flat list of (file name, FPakEntry)
* version 10 and 11 - a path-hash index plus a "full directory index", with the
  entries themselves bit-packed into one blob (FPakEntry::Encode)

Both are handled here. Encrypted indexes need the game's AES key, which this
build neither ships nor derives: those paks are reported as unsupported.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.logging_setup import get_logger

log = get_logger("plugin.unreal.pak")

PAK_MAGIC = 0x5A6F12E1
COMPRESSION_NAME_SIZE = 32

# footer fields after the magic: version, index offset, index size, index hash
FOOTER_TAIL = 4 + 8 + 8 + 20


class PakError(Exception):
    pass


class PakUnsupported(PakError):
    """Recognised, but this build cannot read it. Never means corrupt."""


@dataclass
class PakEntry:
    name: str = ""
    offset: int = 0
    size: int = 0                       # compressed / on disk
    uncompressed_size: int = 0
    compression_index: int = 0
    encrypted: bool = False
    blocks: list = field(default_factory=list)     # (start, end) absolute or entry-relative
    block_size: int = 0

    @property
    def is_compressed(self) -> bool:
        return self.compression_index != 0


@dataclass
class PakInfo:
    version: int = 0
    index_offset: int = 0
    index_size: int = 0
    encrypted_index: bool = False
    compression_methods: list = field(default_factory=list)
    footer_offset: int = 0


class _Reader:
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def read(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise PakError("index truncated at %d" % self.pos)
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read(8))[0]

    def string(self) -> str:
        length = self.i32()
        if length == 0:
            return ""
        if length < 0:
            raw = self.read(-length * 2)
            return raw.decode("utf-16-le", "replace").rstrip("\x00")
        raw = self.read(length)
        return raw.decode("utf-8", "replace").rstrip("\x00")


def read_info(path: Path) -> PakInfo:
    """Locate and decode the footer without assuming its exact size."""
    size = path.stat().st_size
    tail_len = min(size, 512)
    with open(path, "rb") as fh:
        fh.seek(size - tail_len)
        tail = fh.read(tail_len)
    needle = struct.pack("<I", PAK_MAGIC)
    idx = tail.rfind(needle)
    if idx < 0:
        raise PakUnsupported("no PAK footer magic found")
    magic_pos = size - (tail_len - idx)

    r = _Reader(tail, idx + 4)
    info = PakInfo(footer_offset=magic_pos)
    info.version = r.i32()
    info.index_offset = r.i64()
    info.index_size = r.i64()
    r.read(20)                                     # index sha1

    # everything between the hash and EOF is the compression-method name table
    remaining = len(tail) - r.pos
    slots = remaining // COMPRESSION_NAME_SIZE
    methods = ["None"]
    for _ in range(slots):
        raw = r.read(COMPRESSION_NAME_SIZE)
        name = raw.split(b"\x00")[0].decode("ascii", "ignore").strip()
        if name:
            methods.append(name)
    info.compression_methods = methods

    # the encrypted-index flag sits just before the magic (version 4 and up)
    if idx >= 1:
        info.encrypted_index = tail[idx - 1] == 1
    if info.index_offset <= 0 or info.index_size <= 0 or \
            info.index_offset + info.index_size > size:
        raise PakUnsupported("PAK index lies outside the file (encrypted or unknown layout)")
    return info


class PakFile:
    """Random-access reader over one .pak."""

    def __init__(self, path, oodle_hints=None):
        self.path = Path(path)
        # look for an Oodle runtime in the game folder this pak lives in
        self.oodle_hints = list(oodle_hints or []) or list(Path(path).parents)[:5]
        self.size = self.path.stat().st_size
        self.info = read_info(self.path)
        if self.info.encrypted_index:
            raise PakUnsupported(
                "PAK index is AES-encrypted; the game's key is required and this build "
                "neither ships nor derives keys")
        self.mount_point = ""
        self.entries: list = []
        self._read_index()

    # -- index ----------------------------------------------------------
    def _read_index(self) -> None:
        with open(self.path, "rb") as fh:
            fh.seek(self.info.index_offset)
            data = fh.read(self.info.index_size)
        r = _Reader(data)
        try:
            self.mount_point = r.string().replace("../../../", "")
            count = r.i32()
        except Exception as exc:
            raise PakUnsupported("PAK index unreadable (%s)" % exc)
        if count < 0 or count > 5_000_000:
            raise PakUnsupported("PAK index reports %d entries" % count)

        if self.info.version >= 10:
            self.entries = self._read_index_v10(r, data, count)
        else:
            self.entries = self._read_index_legacy(r, count)
        log.info("%s: pak v%d, %d entries, methods=%s", self.path.name, self.info.version,
                 len(self.entries), self.info.compression_methods[1:] or ["none"])

    def _read_index_legacy(self, r: _Reader, count: int) -> list:
        entries = []
        for _ in range(count):
            name = r.string()
            entry = self._read_full_entry(r)
            entry.name = name
            entries.append(entry)
        return entries

    def _read_full_entry(self, r: _Reader) -> PakEntry:
        """FPakEntry::Serialize - the form used in the index (<= v9) and in the
        header written in front of every payload."""
        version = self.info.version
        entry = PakEntry()
        entry.offset = r.i64()
        entry.size = r.i64()
        entry.uncompressed_size = r.i64()
        entry.compression_index = r.i32()              # uint32 in the file, not a byte
        if version <= 1:
            r.read(8)                                  # timestamp
        r.read(20)                                     # sha1
        if version >= 3:
            if entry.compression_index != 0:
                block_count = r.i32()
                for _ in range(block_count):
                    entry.blocks.append((r.i64(), r.i64()))
            entry.encrypted = r.read(1)[0] != 0
            entry.block_size = r.u32()
        return entry

    def _read_index_v10(self, r: _Reader, data: bytes, count: int) -> list:
        """Path-hash index layout (Unreal 4.26+ / 5.x)."""
        r.u64()                                        # path hash seed
        if r.i32() != 0:                               # has path hash index
            r.i64(), r.i64()
            r.read(20)
        has_full_dir = r.i32() != 0
        full_dir_offset = full_dir_size = 0
        if has_full_dir:
            full_dir_offset = r.i64()
            full_dir_size = r.i64()
            r.read(20)
        encoded_size = r.i32()
        encoded = r.read(max(0, encoded_size))
        if not has_full_dir:
            raise PakUnsupported("PAK has no full directory index; only the path-hash "
                                 "index is present and file names cannot be recovered")

        with open(self.path, "rb") as fh:
            fh.seek(full_dir_offset)
            directory_blob = fh.read(full_dir_size)
        dr = _Reader(directory_blob)
        entries = []
        directory_count = dr.i32()
        if directory_count < 0 or directory_count > 1_000_000:
            raise PakUnsupported("directory index looks encrypted (%d directories)"
                                 % directory_count)
        for _ in range(directory_count):
            directory = dr.string()
            file_count = dr.i32()
            for _f in range(file_count):
                filename = dr.string()
                encoded_offset = dr.i32()
                if encoded_offset < 0:
                    continue                           # entry stored elsewhere
                try:
                    entry = self._decode_entry(encoded, encoded_offset)
                except PakError as exc:
                    log.debug("entry %s%s: %s", directory, filename, exc)
                    continue
                entry.name = (directory + filename).lstrip("/")
                entries.append(entry)
        if len(entries) != count:
            log.debug("%s: index declared %d entries, decoded %d", self.path.name,
                      count, len(entries))
        return entries

    @staticmethod
    def _decode_entry(blob: bytes, offset: int) -> PakEntry:
        """FPakEntry::Encode - one bitfield header, then only what it needs."""
        if offset + 4 > len(blob):
            raise PakError("encoded entry offset out of range")
        r = _Reader(blob, offset)
        bits = r.u32()
        entry = PakEntry()
        entry.compression_index = (bits >> 23) & 0x3F
        entry.encrypted = bool(bits & (1 << 22))
        block_count = bits & 0xFFFF
        offset_32 = bool(bits & 0x80000000)
        uncompressed_32 = bool(bits & 0x40000000)
        size_32 = bool(bits & 0x20000000)

        entry.offset = r.u32() if offset_32 else r.u64()
        entry.uncompressed_size = r.u32() if uncompressed_32 else r.u64()
        if entry.compression_index != 0:
            entry.size = r.u32() if size_32 else r.u64()
        else:
            entry.size = entry.uncompressed_size

        block_size = (bits >> 6) & 0x3F
        entry.block_size = 0 if block_size == 0 else (
            r.u32() if block_size == 0x3F else block_size << 11)

        if block_count > 0:
            if block_count == 1 and entry.uncompressed_size <= entry.block_size:
                entry.blocks.append((0, entry.size))
            else:
                start = 0
                for _ in range(block_count):
                    end = start + r.u32()
                    entry.blocks.append((start, end))
                    start = end
        return entry

    # -- reading --------------------------------------------------------
    def _read_payload_header(self, fh, entry: PakEntry):
        """Every payload is preceded by its own FPakEntry. Reading it back is
        version-proof: it gives the authoritative block table and header size."""
        fh.seek(entry.offset)
        raw = fh.read(4096)
        for _ in range(4):                     # grow for entries with many blocks
            try:
                r = _Reader(raw)
                header_entry = self._read_full_entry(r)
                return header_entry, r.pos
            except PakError:
                fh.seek(entry.offset)
                raw = fh.read(len(raw) * 8)
                if len(raw) > (8 << 20):
                    break
        raise PakError("payload header does not fit in %d bytes" % len(raw))

    def method_name(self, index: int) -> str:
        methods = self.info.compression_methods
        if 0 < index < len(methods):
            return methods[index]
        if self.info.version < 8:
            # older paks store ECompressionFlags, not an index into a name table
            return {1: "Zlib", 2: "Gzip", 4: "Custom"}.get(index, "None" if index == 0
                                                            else "Unknown_%d" % index)
        return "None" if index == 0 else "Unknown_%d" % index

    def read_entry(self, entry: PakEntry) -> Optional[bytes]:
        if entry.encrypted:
            return None
        with open(self.path, "rb") as fh:
            try:
                header_entry, header_size = self._read_payload_header(fh, entry)
            except PakError as exc:
                log.debug("%s: %s", entry.name, exc)
                return None
            blocks = header_entry.blocks or entry.blocks
            method = self.method_name(header_entry.compression_index or
                                      entry.compression_index)
            size = header_entry.size or entry.size
            uncompressed = header_entry.uncompressed_size or entry.uncompressed_size

            if not blocks:
                fh.seek(entry.offset + header_size)
                return self._decompress(fh.read(size), method, uncompressed)

            block_size = header_entry.block_size or entry.block_size or 65536
            # Block offsets are relative to the entry from RelativeChunkOffsets (v5)
            # onwards and absolute before it; try the likely one, fall back to the
            # other rather than guessing the version rules wrong.
            for base in ((entry.offset, 0) if self.info.version >= 5 else (0, entry.offset)):
                out = bytearray()
                ok = True
                for start, end in blocks:
                    fh.seek(base + start)
                    chunk = fh.read(end - start)
                    block = self._decompress(chunk, method, block_size)
                    if block is None:
                        ok = False
                        break
                    out += block
                if ok and out:
                    return bytes(out[:uncompressed]) if uncompressed else bytes(out)
            return None

    def _decompress(self, data: bytes, method: str, expected: int) -> Optional[bytes]:
        name = (method or "None").lower()
        if name in ("none", ""):
            return data
        try:
            if name == "zlib":
                return zlib.decompress(data)
            if name == "gzip":
                return zlib.decompress(data, 16 + zlib.MAX_WBITS)
            if name == "zstd":
                import zstandard
                return zstandard.ZstdDecompressor().decompress(data, max_output_size=max(
                    expected, len(data) * 32))
            if name in ("lz4",):
                from app.formats.archives.lz4_block import decompress_block
                return decompress_block(data, expected)
            if name in ("oodle", "custom", "kraken", "mermaid"):
                from app.formats.archives.oodle import get_codec
                codec = get_codec(self.oodle_hints)
                return codec.decompress(data, expected) if codec else None
        except Exception as exc:
            log.debug("pak block decompression failed (%s): %s", method, exc)
            return None
        return None

    def uses_oodle(self) -> bool:
        methods = {m.lower() for m in self.info.compression_methods}
        if methods & {"oodle", "kraken", "mermaid"}:
            return True
        # older paks use ECompressionFlags; 4 = custom, which is Oodle in practice
        return self.info.version < 8 and any(e.compression_index == 4 for e in self.entries[:64])

    def unsupported_reason(self) -> Optional[str]:
        if self.uses_oodle():
            from app.formats.archives.oodle import get_codec, unavailable_reason
            if get_codec(self.oodle_hints) is None:
                return unavailable_reason()
        return None
