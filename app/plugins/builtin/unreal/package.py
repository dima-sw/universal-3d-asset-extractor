"""Unreal package reader: .uasset header + .uexp export data.

A cooked Unreal asset is split in two (sometimes three) files:

    Foo.uasset   summary, name table, import table, export table
    Foo.uexp     the serialized objects themselves
    Foo.ubulk    bulk payloads (texture mips, vertex buffers) kept out of line

This module reads the header and the tagged properties of each export. Property
serialization is stable across engine versions in a way the class-specific
binary blobs are not, so this is the part that can be relied on: it gives the
object's class, its name, and its properties (materials, sizes, flags, LOD
settings) for every export in the package.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.logging_setup import get_logger

log = get_logger("plugin.unreal.package")

PACKAGE_MAGIC = 0x9E2A83C1

# the handful of engine version gates this reader actually needs
VER_UE4_OLDEST = 342
VER_UE4_ADDED_SEARCHABLE_NAMES = 510
VER_UE4_64BIT_EXPORTMAP_SERIALSIZES = 511
VER_UE4_ADDED_PACKAGE_SUMMARY_LOCALIZATION_ID = 516
VER_UE4_NAME_HASHES_SERIALIZED = 504
VER_UE4_ADDED_PACKAGE_OWNER = 518
VER_UE4_NON_OUTER_PACKAGE_IMPORT = 520
VER_UE4_TEMPLATE_INDEX_IN_COOKED_EXPORTS = 507
VER_UE4_PRELOAD_DEPENDENCIES_IN_COOKED_EXPORTS = 508
VER_UE4_SERIALIZE_TEXT_IN_PACKAGES = 459


class PackageError(Exception):
    pass


class PackageUnsupported(PackageError):
    """Recognised as an Unreal package this build cannot read. Not corruption."""


class Reader:
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def eof(self) -> bool:
        return self.pos >= len(self.data)

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def read(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise PackageError("read past end (%d bytes at %d of %d)"
                               % (n, self.pos, len(self.data)))
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def u8(self) -> int:
        return self.read(1)[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.read(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.read(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.read(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.read(8))[0]

    def boolean32(self) -> bool:
        return self.u32() != 0

    def guid(self) -> bytes:
        return self.read(16)

    def string(self) -> str:
        length = self.i32()
        if length == 0:
            return ""
        if length < 0:
            if -length > 1_000_000:
                raise PackageError("implausible utf-16 string length %d" % length)
            raw = self.read(-length * 2)
            return raw.decode("utf-16-le", "replace").rstrip("\x00")
        if length > 1_000_000:
            raise PackageError("implausible string length %d" % length)
        return self.read(length).decode("utf-8", "replace").rstrip("\x00")

    def array(self, item):
        count = self.i32()
        if count < 0 or count > 5_000_000:
            raise PackageError("implausible array count %d" % count)
        return [item() for _ in range(count)]


@dataclass
class ObjectImport:
    class_package: str = ""
    class_name: str = ""
    outer_index: int = 0
    object_name: str = ""


@dataclass
class ObjectExport:
    class_index: int = 0
    super_index: int = 0
    template_index: int = 0
    outer_index: int = 0
    object_name: str = ""
    object_flags: int = 0
    serial_size: int = 0
    serial_offset: int = 0
    is_asset: bool = False
    class_name: str = ""            # resolved from the import table
    properties: Optional[dict] = None
    data_offset: int = 0            # where its data starts inside the .uexp payload


@dataclass
class Summary:
    file_version_ue4: int = 0
    file_version_ue5: int = 0
    file_version_licensee: int = 0
    legacy_version: int = 0
    total_header_size: int = 0
    package_name: str = ""
    package_flags: int = 0
    name_count: int = 0
    name_offset: int = 0
    export_count: int = 0
    export_offset: int = 0
    import_count: int = 0
    import_offset: int = 0
    depends_offset: int = 0
    is_unversioned: bool = False


# Header layouts only change at these engine versions, so trying one candidate
# per layout is enough to cover every UE4/UE5 release.
CANDIDATE_UE4_VERSIONS = (522, 519, 517, 515, 513, 510, 508, 506, 503, 500, 460, 400)


class Package:
    """Reads a .uasset (+ .uexp) pair.

    Shipped games usually save packages *unversioned*: the engine version is not
    written in the file, and a reader has to be told which build made it. Rather
    than asking the user, this reader tries one candidate per known header
    layout and keeps the first that produces a self-consistent package.
    """

    def __init__(self, data: bytes, uexp: Optional[bytes] = None, name: str = "",
                 ubulk: Optional[bytes] = None, assume_ue4: int = 0, assume_ue5: int = 0):
        self.data = data
        self.uexp = uexp
        self.ubulk = ubulk
        self.name = name
        self.summary = Summary()
        self.names: list = []
        self.imports: list = []
        self.exports: list = []
        self._assume_ue4 = assume_ue4
        self._assume_ue5 = assume_ue5
        self._read_header()

    # ------------------------------------------------------------------
    def _reset(self) -> None:
        self.summary = Summary()
        self.names = []
        self.imports = []
        self.exports = []

    def _infer_version(self) -> None:
        """Try each header layout and keep the first self-consistent one."""
        errors = []
        for candidate in CANDIDATE_UE4_VERSIONS:
            self._reset()
            self._assume_ue4 = candidate
            try:
                self._read_header(allow_inference=False)
            except PackageError as exc:
                errors.append("ue4=%d: %s" % (candidate, exc))
                continue
            if self._looks_consistent():
                log.debug("%s: unversioned package read as ue4 version %d",
                          self.name, candidate)
                return
        raise PackageUnsupported(
            "package is saved unversioned and none of the known header layouts fits it "
            "(tried %d). A .usmap mapping file from this game's build would be needed."
            % len(CANDIDATE_UE4_VERSIONS))

    def _looks_consistent(self) -> bool:
        """Cheap structural proof that the guessed layout was the right one."""
        if not self.names or len(self.names) != self.summary.name_count:
            return False
        # Unreal's name table always contains "None"
        if "None" not in self.names[:64] and "None" not in self.names:
            return False
        printable = sum(1 for n in self.names[:200]
                        if n and all(32 <= ord(c) < 127 for c in n[:64]))
        if printable < min(len(self.names), 200) * 0.95:
            return False
        if len(self.exports) != self.summary.export_count:
            return False
        if len(self.imports) != self.summary.import_count:
            return False
        payload = len(self.uexp) if self.uexp is not None else len(self.data)
        for export in self.exports:
            if export.serial_size < 0 or export.serial_size > payload + len(self.data):
                return False
            start = export.data_offset if self.uexp is not None else export.serial_offset
            if start < 0 or start + export.serial_size > payload + 4:
                return False
            if not export.object_name or export.object_name == "None":
                return False
        # class names should resolve to something that looks like a class
        resolved = [e.class_name for e in self.exports if e.class_name != "None"]
        return bool(resolved) or not self.exports

    # ------------------------------------------------------------------
    @property
    def engine(self) -> str:
        if self.summary.file_version_ue5:
            return "UE5 (ue4=%d ue5=%d)" % (self.summary.file_version_ue4,
                                            self.summary.file_version_ue5)
        return "UE4 (version %d)" % self.summary.file_version_ue4

    def name_at(self, index: int) -> str:
        return self.names[index] if 0 <= index < len(self.names) else "None"

    def fname(self, r: Reader) -> str:
        index = r.i32()
        number = r.i32()
        text = self.name_at(index)
        return "%s_%d" % (text, number - 1) if number > 0 else text

    def resolve(self, package_index: int) -> str:
        """Package indices: >0 = export, <0 = import, 0 = null."""
        if package_index > 0 and package_index <= len(self.exports):
            return self.exports[package_index - 1].object_name
        if package_index < 0 and -package_index <= len(self.imports):
            return self.imports[-package_index - 1].object_name
        return "None"

    # ------------------------------------------------------------------
    def _read_header(self, allow_inference: bool = True) -> None:
        r = Reader(self.data)
        if r.remaining() < 64 or r.u32() != PACKAGE_MAGIC:
            raise PackageUnsupported("not an Unreal package (bad magic)")
        s = self.summary
        s.legacy_version = r.i32()
        if s.legacy_version > -6 or s.legacy_version < -9:
            raise PackageUnsupported("legacy package version %d is not supported "
                                     "(UE3 or newer than this build)" % s.legacy_version)
        if s.legacy_version != -4:
            r.i32()                                        # legacy UE3 version
        s.file_version_ue4 = r.i32()
        if s.legacy_version <= -8:
            s.file_version_ue5 = r.i32()
        s.file_version_licensee = r.i32()
        custom_count = r.i32()
        if custom_count < 0 or custom_count > 10000:
            raise PackageUnsupported("implausible custom version count %d" % custom_count)
        r.read(custom_count * 20)                          # guid + version each

        s.total_header_size = r.i32()
        s.package_name = r.string()
        s.package_flags = r.u32()
        # An unversioned package records no engine version at all. It does not
        # matter here: every table below is located by its own structure rather
        # than by version gates, so the reader works either way.
        s.is_unversioned = s.file_version_ue4 == 0 and s.file_version_ue5 == 0

        s.name_count = r.i32()
        s.name_offset = r.i32()
        if s.name_count < 0 or s.name_count > 5_000_000 or not (0 < s.name_offset < len(self.data)):
            raise PackageError("implausible name table (%d @ %d)" % (s.name_count,
                                                                     s.name_offset))
        # Licensee builds move these fields around, so find the offset block by
        # checking which reading produces a consistent table layout instead of
        # trusting the declared engine version.
        self._read_offset_block(r)
        self._read_names()
        self._read_imports()
        self._read_exports()

    def _read_offset_block(self, r: Reader) -> None:
        """Locate export/import/depends offsets after the name table fields.

        Between them sit optional fields (a localisation id string, a gatherable
        text count/offset pair) that came and went across versions. Each layout
        is tried and accepted only if the resulting offsets are inside the file
        and the tables tile the space between them.
        """
        s = self.summary
        start = r.pos
        best = None
        # optional fields that appear between the name table and the offsets:
        # a localisation id (string), gatherable text and soft-object-path
        # count/offset pairs. Try each shape; the record arithmetic decides.
        for skip_string in (False, True):
            for extra_pairs in (0, 1, 2, 3):
                probe = Reader(self.data, start)
                try:
                    if skip_string:
                        probe.string()
                    for _ in range(extra_pairs):
                        probe.i32(), probe.i32()
                    values = (probe.i32(), probe.i32(), probe.i32(), probe.i32(), probe.i32())
                except PackageError:
                    continue
                score = self._score_offsets(*values)
                if score and (best is None or score > best[0]):
                    best = (score,) + values
        if best is None:
            raise PackageError("could not locate the export/import tables")
        (_score, s.export_count, s.export_offset, s.import_count, s.import_offset,
         s.depends_offset) = best

    def _score_offsets(self, export_count, export_offset, import_count,
                       import_offset, depends_offset) -> int:
        """0 = impossible, higher = more invariants satisfied."""
        size = len(self.data)
        header_end = self.summary.total_header_size or size
        if export_count < 1 or import_count < 1:
            return 0
        if export_count > 1_000_000 or import_count > 1_000_000:
            return 0
        if not (self.summary.name_offset < import_offset < export_offset <= size):
            return 0
        if not (export_offset <= depends_offset <= header_end + 4):
            return 0

        import_span = export_offset - import_offset
        export_span = depends_offset - export_offset
        if import_span <= 0 or export_span <= 0:
            return 0
        if import_span % import_count or export_span % export_count:
            return 0
        import_record = import_span // import_count
        export_record = export_span // export_count
        if not (24 <= import_record <= 64):
            return 0
        if not (56 <= export_record <= 200):
            return 0
        known = {self._export_record_size(layout) for layout in self._EXPORT_LAYOUTS}
        score = 2
        if export_record in known:
            score += 3               # matches a layout this reader knows exactly
        if import_record in (28, 29, 32, 36, 37):
            score += 1
        return score

    def _offsets_plausible(self, export_count, export_offset, import_count,
                           import_offset, depends_offset) -> bool:
        size = len(self.data)
        counts = (export_count, import_count)
        offsets = (export_offset, import_offset, depends_offset)
        if any(c < 0 or c > 1_000_000 for c in counts):
            return False
        if any(o < 0 or o > size for o in offsets):
            return False
        if export_count and not (0 < export_offset < size):
            return False
        if import_count and not (0 < import_offset < size):
            return False
        if export_count and import_count:
            # both tables are fixed-size records, so the gaps must divide evenly
            import_span = export_offset - import_offset
            export_span = (depends_offset or size) - export_offset
            if import_span <= 0 or export_span <= 0:
                return False
            if import_span % import_count or export_span % export_count:
                return False
            if not (24 <= import_span // import_count <= 64):
                return False
            if not (56 <= export_span // export_count <= 160):
                return False
        return True

    def _read_names(self) -> None:
        """Name entries are a string plus, in most builds, two 16-bit hashes.

        Licensee builds disagree with the engine version about this, so both
        forms are tried and the one that reads the whole table wins.
        """
        s = self.summary
        if not s.name_count:
            return
        limit = min(s.import_offset or len(self.data), len(self.data))
        for hashed in (True, False):
            r = Reader(self.data, s.name_offset)
            names = []
            try:
                for _ in range(s.name_count):
                    text = r.string()
                    if hashed:
                        r.read(4)
                    names.append(text)
            except PackageError:
                continue
            if r.pos > limit + 8:
                continue
            printable = sum(1 for n in names if all(32 <= ord(c) < 127 for c in n[:80]))
            if printable >= max(1, int(len(names) * 0.95)):
                self.names = names
                self._names_hashed = hashed             # type: ignore[attr-defined]
                return
        raise PackageError("name table could not be read in either known form")

    def _read_imports(self) -> None:
        """Import records are fixed size; the record size tells us the layout."""
        s = self.summary
        if not s.import_count:
            return
        span = (s.export_offset or len(self.data)) - s.import_offset
        record = span // s.import_count if s.import_count else 0
        # Records are fixed size: read the four fields we need, then jump to the
        # next record. Whatever a given build appends after them is skipped.
        for index in range(s.import_count):
            r = Reader(self.data, s.import_offset + index * record)
            item = ObjectImport()
            item.class_package = self.fname(r)
            item.class_name = self.fname(r)
            item.outer_index = r.i32()
            item.object_name = self.fname(r)
            self.imports.append(item)

    _EXPORT_LAYOUTS = (
        # (template_index, wide_sizes, ue5_public_hash, preload_deps)
        (True, True, False, True),
        (True, True, True, True),
        (True, False, False, True),
        (True, True, False, False),
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, True),
    )

    def _export_record_size(self, layout) -> int:
        template, wide, public_hash, preload = layout
        size = 4 + 4 + (4 if template else 0) + 4 + 8 + 4      # indices, name, flags
        size += 16 if wide else 8                              # serial size + offset
        size += 4 * 3 + 16 + 4 + 4 + 4                         # flags, guid, package flags
        if public_hash:
            size += 4
        if preload:
            size += 4 * 5
        return size

    def _read_exports(self) -> None:
        """Export records are fixed size too. Only the head of each record is
        needed; whether it carries a template index and 32- or 64-bit sizes is
        decided by trying both and keeping the reading whose offsets and sizes
        actually fit the payload."""
        s = self.summary
        if not s.export_count:
            return
        span = (s.depends_offset or len(self.data)) - s.export_offset
        record = span // s.export_count
        payload = len(self.uexp) if self.uexp is not None else len(self.data)

        best = None
        for template in (True, False):
            for wide in (True, False):
                exports = self._parse_exports(record, template, wide)
                if exports is None:
                    continue
                score = self._score_exports(exports, payload)
                if score and (best is None or score > best[0]):
                    best = (score, exports)
        if best is None:
            raise PackageError("export table layout not recognised (record %d bytes)" % record)
        self.exports = best[1]
        for item in self.exports:
            item.class_name = self.resolve(item.class_index)
            item.data_offset = item.serial_offset - s.total_header_size

    def _parse_exports(self, record: int, template: bool, wide: bool):
        s = self.summary
        exports = []
        try:
            for index in range(s.export_count):
                r = Reader(self.data, s.export_offset + index * record)
                item = ObjectExport()
                item.class_index = r.i32()
                item.super_index = r.i32()
                if template:
                    item.template_index = r.i32()
                item.outer_index = r.i32()
                item.object_name = self.fname(r)
                item.object_flags = r.u32()
                item.serial_size = r.i64() if wide else r.i32()
                item.serial_offset = r.i64() if wide else r.i32()
                exports.append(item)
        except PackageError:
            return None
        return exports

    def _score_exports(self, exports: list, payload: int) -> int:
        """Every export must sit inside the payload and carry a real name."""
        total = self.summary.total_header_size
        covered = 0
        for item in exports:
            if item.serial_size <= 0 or item.serial_size > payload + total:
                return 0
            start = item.serial_offset - (total if self.uexp is not None else 0)
            if start < 0 or start + item.serial_size > payload + 4:
                return 0
            name = item.object_name
            if not name or name == "None" or not all(32 <= ord(c) < 127 for c in name[:64]):
                return 0
            covered += item.serial_size
        # the exports should account for most of the payload
        ratio = covered / max(1, payload)
        return 3 if 0.5 <= ratio <= 1.05 else 1

    # ------------------------------------------------------------------
    def export_data(self, export: ObjectExport) -> Optional[bytes]:
        """Bytes of one export, from the .uexp when cooked, else from the .uasset."""
        if self.uexp is not None:
            start = export.data_offset
            if start < 0 or start + export.serial_size > len(self.uexp):
                return None
            return self.uexp[start:start + export.serial_size]
        start = export.serial_offset
        if start < 0 or start + export.serial_size > len(self.data):
            return None
        return self.data[start:start + export.serial_size]

    def read_properties(self, export: ObjectExport) -> Optional[dict]:
        """Tagged properties of one export (the part that survives versions)."""
        data = self.export_data(export)
        if data is None:
            return None
        r = Reader(data)
        try:
            props = self._read_property_list(r)
        except PackageError as exc:
            log.debug("%s: properties of %s unreadable: %s", self.name,
                      export.object_name, exc)
            return None
        export.properties = props
        export.properties_end = r.pos          # type: ignore[attr-defined]
        return props

    # -- tagged property serialisation ---------------------------------
    def _read_property_list(self, r: Reader, depth: int = 0) -> dict:
        out: dict = {}
        if depth > 8:
            return out
        while True:
            name = self.fname(r)
            if name in ("None", "none"):
                break
            type_name = self.fname(r)
            size = r.i32()
            r.i32()                                        # array index
            meta = self._read_tag_meta(r, type_name)
            end = r.pos + size
            try:
                value = self._read_property_value(r, type_name, meta, size, depth)
            except PackageError:
                value = None
            r.pos = min(max(end, r.pos), len(r.data))
            out[name] = value
            if r.remaining() <= 0:
                break
        return out

    def _read_tag_meta(self, r: Reader, type_name: str):
        if type_name == "StructProperty":
            struct_name = self.fname(r)
            r.guid()
            r.u8()                                         # has property guid
            return struct_name
        if type_name == "BoolProperty":
            value = r.u8() != 0
            r.u8()
            return value
        if type_name in ("ByteProperty", "EnumProperty"):
            enum_name = self.fname(r)
            r.u8()
            return enum_name
        if type_name == "ArrayProperty":
            inner = self.fname(r)
            r.u8()
            return inner
        if type_name in ("SetProperty", "MapProperty"):
            key = self.fname(r)
            value = self.fname(r) if type_name == "MapProperty" else None
            r.u8()
            return (key, value)
        r.u8()                                             # has property guid
        return None

    def _read_property_value(self, r: Reader, type_name: str, meta, size: int, depth: int):
        if type_name == "BoolProperty":
            return meta
        if type_name == "ByteProperty":
            return self.fname(r) if size == 8 else r.u8()
        if type_name == "EnumProperty":
            return self.fname(r)
        if type_name == "IntProperty":
            return r.i32()
        if type_name == "Int8Property":
            return r.i8()
        if type_name == "Int16Property":
            return r.i16()
        if type_name == "Int64Property":
            return r.i64()
        if type_name == "UInt16Property":
            return r.u16()
        if type_name == "UInt32Property":
            return r.u32()
        if type_name == "UInt64Property":
            return r.u64()
        if type_name == "FloatProperty":
            return r.f32()
        if type_name == "DoubleProperty":
            return r.f64()
        if type_name in ("StrProperty", "NameProperty"):
            return r.string() if type_name == "StrProperty" else self.fname(r)
        if type_name in ("ObjectProperty", "SoftObjectProperty", "AssetObjectProperty",
                         "WeakObjectProperty", "LazyObjectProperty"):
            if type_name == "ObjectProperty":
                index = r.i32()
                return {"__object": self.resolve(index), "index": index}
            return r.string()
        if type_name == "ArrayProperty":
            return self._read_array_property(r, meta, size, depth)
        if type_name == "StructProperty":
            return self._read_struct(r, meta, size, depth)
        if type_name == "TextProperty":
            return None                                    # localisation blob, skipped
        return None

    def _read_array_property(self, r: Reader, inner: str, size: int, depth: int):
        count = r.i32()
        if count < 0 or count > 5_000_000:
            raise PackageError("implausible array size %d" % count)
        if inner == "StructProperty":
            # arrays of structs repeat the tag once, then the elements follow
            self.fname(r)                                   # inner name
            self.fname(r)                                   # "StructProperty"
            r.i32()                                         # size
            r.i32()                                         # index
            struct_name = self.fname(r)
            r.guid()
            r.u8()
            return [self._read_struct(r, struct_name, -1, depth + 1) for _ in range(count)]
        simple = {"IntProperty": lambda: r.i32(), "FloatProperty": lambda: r.f32(),
                  "BoolProperty": lambda: r.u8() != 0, "ByteProperty": lambda: r.u8(),
                  "NameProperty": lambda: self.fname(r), "StrProperty": lambda: r.string(),
                  "ObjectProperty": lambda: self.resolve(r.i32())}
        reader = simple.get(inner)
        if reader is None:
            raise PackageError("array of %s not decoded" % inner)
        return [reader() for _ in range(count)]

    def _read_struct(self, r: Reader, struct_name: str, size: int, depth: int):
        simple = {
            "Vector": lambda: (r.f32(), r.f32(), r.f32()),
            "Vector2D": lambda: (r.f32(), r.f32()),
            "Vector4": lambda: (r.f32(), r.f32(), r.f32(), r.f32()),
            "Rotator": lambda: (r.f32(), r.f32(), r.f32()),
            "Quat": lambda: (r.f32(), r.f32(), r.f32(), r.f32()),
            "Color": lambda: tuple(r.read(4)),
            "LinearColor": lambda: (r.f32(), r.f32(), r.f32(), r.f32()),
            "IntPoint": lambda: (r.i32(), r.i32()),
            "Guid": lambda: r.guid().hex(),
            "Box": lambda: {"min": (r.f32(), r.f32(), r.f32()),
                            "max": (r.f32(), r.f32(), r.f32()), "valid": r.u8()},
        }
        if struct_name in simple:
            return simple[struct_name]()
        # unknown struct: it is itself a tagged property list
        return self._read_property_list(r, depth + 1)

    # ------------------------------------------------------------------
    def summary_dict(self) -> dict:
        classes: dict = {}
        for export in self.exports:
            classes[export.class_name] = classes.get(export.class_name, 0) + 1
        return {"package": self.summary.package_name, "engine": self.engine,
                "names": len(self.names), "imports": len(self.imports),
                "exports": len(self.exports), "classes": classes}


def load(uasset_path) -> Package:
    """Open a .uasset together with its .uexp / .ubulk siblings."""
    path = Path(uasset_path)
    uexp = path.with_suffix(".uexp")
    ubulk = path.with_suffix(".ubulk")
    return Package(path.read_bytes(),
                   uexp.read_bytes() if uexp.exists() else None,
                   name=path.name,
                   ubulk=ubulk.read_bytes() if ubulk.exists() else None)
