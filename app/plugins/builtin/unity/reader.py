"""Endian-aware binary reader used by the Unity SerializedFile parser."""
from __future__ import annotations

import struct


class ReadError(Exception):
    pass


class BinaryReader:
    __slots__ = ("data", "pos", "big", "_base")

    def __init__(self, data: bytes, big_endian: bool = False, pos: int = 0, base: int = 0):
        self.data = data
        self.pos = pos
        self.big = big_endian
        self._base = base

    # -- plumbing ----------------------------------------------------------
    @property
    def prefix(self) -> str:
        return ">" if self.big else "<"

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def seek(self, pos: int) -> None:
        self.pos = pos

    def skip(self, n: int) -> None:
        self.pos += n

    def align(self, boundary: int = 4) -> None:
        rel = self.pos - self._base
        pad = (-rel) % boundary
        self.pos += pad

    def read(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise ReadError("read past end of buffer (%d bytes at %d)" % (n, self.pos))
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def _unpack(self, code: str, size: int):
        value = struct.unpack_from(self.prefix + code, self.data, self.pos)[0]
        if self.pos + size > len(self.data):
            raise ReadError("read past end of buffer")
        self.pos += size
        return value

    # -- primitives --------------------------------------------------------
    def u8(self) -> int:
        return self._unpack("B", 1)

    def i8(self) -> int:
        return self._unpack("b", 1)

    def u16(self) -> int:
        return self._unpack("H", 2)

    def i16(self) -> int:
        return self._unpack("h", 2)

    def u32(self) -> int:
        return self._unpack("I", 4)

    def i32(self) -> int:
        return self._unpack("i", 4)

    def u64(self) -> int:
        return self._unpack("Q", 8)

    def i64(self) -> int:
        return self._unpack("q", 8)

    def f32(self) -> float:
        return self._unpack("f", 4)

    def f64(self) -> float:
        return self._unpack("d", 8)

    def boolean(self) -> bool:
        return self.u8() != 0

    def cstring(self, limit: int = 4096) -> str:
        end = self.data.find(b"\x00", self.pos, self.pos + limit)
        if end < 0:
            raise ReadError("unterminated string at %d" % self.pos)
        out = self.data[self.pos:end].decode("utf-8", "replace")
        self.pos = end + 1
        return out

    def sized_string(self) -> str:
        length = self.i32()
        if length < 0 or length > self.remaining():
            raise ReadError("bad string length %d" % length)
        value = self.read(length).decode("utf-8", "replace")
        self.align(4)
        return value
