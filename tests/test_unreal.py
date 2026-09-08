"""Unreal reader tests: pak container, package header, texture payload.

All fixtures are built here, so the suite needs no game files. (The readers were
additionally validated against four shipped titles: It Takes Two, Chained
Together, Days Gone and Split Fiction.)
"""
from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import pytest

from app.core.detector import FileContext, identify
from app.formats.textures import blocks
from app.plugins.builtin.unreal import pak as pak_module
from app.plugins.builtin.unreal.package import Package, PackageUnsupported, Reader
from app.plugins.builtin.unreal.pak import PakFile, PakUnsupported, read_info

PAK_MAGIC = 0x5A6F12E1


# ---------------------------------------------------------------------------
# helpers that build a minimal but real .pak
# ---------------------------------------------------------------------------
def _entry_record(offset, size, uncompressed, compression=0, blocks_=(), encrypted=0,
                  block_size=0):
    out = struct.pack("<qqq", offset, size, uncompressed)
    out += struct.pack("<i", compression)
    out += b"\x00" * 20                                   # sha1
    if compression:
        out += struct.pack("<i", len(blocks_))
        for start, end in blocks_:
            out += struct.pack("<qq", start, end)
    out += struct.pack("<B", encrypted) + struct.pack("<I", block_size)
    return out


def build_pak(path: Path, files: dict, version: int = 8) -> None:
    """Classic index layout, one uncompressed entry per file."""
    body = bytearray()
    records = []
    for name, payload in files.items():
        offset = len(body)
        header = _entry_record(0, len(payload), len(payload))
        body += header + payload
        records.append((name, offset, len(payload), header))

    index = bytearray()
    mount = b"../\x00"
    index += struct.pack("<i", len(mount)) + mount        # mount point
    index += struct.pack("<i", len(records))
    for name, offset, size, _header in records:
        encoded = name.encode("utf-8") + b"\x00"
        index += struct.pack("<i", len(encoded)) + encoded
        index += _entry_record(offset, size, size)

    index_offset = len(body)
    footer = struct.pack("<I", PAK_MAGIC) + struct.pack("<i", version)
    footer += struct.pack("<qq", index_offset, len(index))
    footer += hashlib.sha1(bytes(index)).digest()
    footer += b"\x00" * 32 * 4                            # compression name slots
    path.write_bytes(bytes(body) + bytes(index) + b"\x00" + footer)


def test_pak_footer_and_index(tmp_path):
    archive = tmp_path / "game.pak"
    build_pak(archive, {"Game/Content/mesh.uasset": b"\xc1\x83\x2a\x9e" + b"A" * 100,
                        "Game/Content/tex.ubulk": b"B" * 64})
    info = read_info(archive)
    assert info.version == 8
    pak = PakFile(archive)
    assert len(pak.entries) == 2
    names = {e.name for e in pak.entries}
    assert "Game/Content/mesh.uasset" in names
    entry = next(e for e in pak.entries if e.name.endswith(".uasset"))
    assert pak.read_entry(entry).startswith(b"\xc1\x83\x2a\x9e")


def test_pak_detected_by_footer(tmp_path):
    archive = tmp_path / "chunk.pak"
    build_pak(archive, {"a.uasset": b"\xc1\x83\x2a\x9e" + b"x" * 32})
    assert identify(FileContext(archive)).format_name == "Unreal PAK"


def test_pak_without_footer_is_unsupported_not_corrupt(tmp_path):
    archive = tmp_path / "broken.pak"
    archive.write_bytes(b"\x00" * 4096)
    with pytest.raises(PakUnsupported) as exc:
        PakFile(archive)
    assert "footer" in str(exc.value)


def test_encoded_entry_bitfield():
    """FPakEntry::Encode - the packed form used by the UE5 index."""
    # 32-bit offset, uncompressed size and size; compression method 1; no blocks
    bits = (1 << 31) | (1 << 30) | (1 << 29) | (1 << 23)
    blob = struct.pack("<I", bits) + struct.pack("<III", 4096, 2048, 1024)
    entry = PakFile._decode_entry(blob, 0)
    assert entry.offset == 4096
    assert entry.uncompressed_size == 2048
    assert entry.size == 1024
    assert entry.compression_index == 1
    assert entry.encrypted is False

    encrypted_bits = bits | (1 << 22)
    encrypted = PakFile._decode_entry(
        struct.pack("<I", encrypted_bits) + struct.pack("<III", 0, 16, 16), 0)
    assert encrypted.encrypted is True


def test_compression_method_names(tmp_path):
    archive = tmp_path / "c.pak"
    build_pak(archive, {"a.bin": b"x" * 16})
    pak = PakFile(archive)
    pak.info.compression_methods = ["None", "Zlib", "Oodle"]
    assert pak.method_name(1) == "Zlib"
    assert pak.method_name(2) == "Oodle"
    assert pak.uses_oodle() is True
    reason = pak.unsupported_reason()
    assert reason is None or "Oodle" in reason


def test_zlib_block_decompression(tmp_path):
    archive = tmp_path / "z.pak"
    build_pak(archive, {"a.bin": b"x" * 16})
    pak = PakFile(archive)
    payload = zlib.compress(b"hello world" * 20)
    assert pak._decompress(payload, "Zlib", 220) == b"hello world" * 20
    assert pak._decompress(b"raw", "None", 3) == b"raw"


# ---------------------------------------------------------------------------
# package header
# ---------------------------------------------------------------------------
def _fstring(text: str) -> bytes:
    raw = text.encode("utf-8") + b"\x00"
    return struct.pack("<i", len(raw)) + raw


def _fname(index: int, number: int = 0) -> bytes:
    return struct.pack("<ii", index, number)


def build_package(names, imports, exports, payload_size=64):
    """Minimal cooked package: summary, names (hashed), imports, exports."""
    name_blob = bytearray()
    for name in names:
        name_blob += _fstring(name) + b"\x00\x00\x00\x00"      # two 16-bit hashes

    import_blob = bytearray()
    for class_package, class_name, object_name in imports:
        import_blob += _fname(names.index(class_package)) + _fname(names.index(class_name))
        import_blob += struct.pack("<i", 0) + _fname(names.index(object_name))

    export_blob = bytearray()
    offset_cursor = 0
    offset_positions = []
    for class_index, object_name, size in exports:
        export_blob += struct.pack("<iii", class_index, 0, 0)       # class, super, template
        export_blob += struct.pack("<i", 0)                          # outer
        export_blob += _fname(names.index(object_name))
        export_blob += struct.pack("<I", 0)                          # object flags
        offset_positions.append((len(export_blob) + 8, offset_cursor))
        export_blob += struct.pack("<qq", size, offset_cursor)       # 64-bit size/offset
        export_blob += struct.pack("<iii", 0, 0, 0)                  # forced/client/server
        export_blob += b"\x00" * 16                                  # guid
        export_blob += struct.pack("<I", 0)                          # package flags
        export_blob += struct.pack("<ii", 0, 1)                      # notAlways, isAsset
        export_blob += struct.pack("<iiiii", 0, 0, 0, 0, 0)          # preload deps
        offset_cursor += size

    header = bytearray()
    header += struct.pack("<I", 0x9E2A83C1)
    header += struct.pack("<i", -7)                                  # legacy version
    header += struct.pack("<i", 864)                                 # legacy ue3
    header += struct.pack("<i", 522)                                 # ue4 version
    header += struct.pack("<i", 0)                                   # licensee
    header += struct.pack("<i", 0)                                   # custom versions
    total_header_placeholder = len(header)
    header += struct.pack("<i", 0)                                   # total header size
    header += _fstring("None")                                       # folder name
    header += struct.pack("<I", 0)                                   # package flags
    header += struct.pack("<i", len(names))
    name_offset_pos = len(header)
    header += struct.pack("<i", 0)                                   # name offset
    header += _fstring("")                                           # localisation id
    header += struct.pack("<ii", 0, 0)                               # gatherable text
    export_fields = len(header)
    header += struct.pack("<iiiii", 0, 0, 0, 0, 0)                   # counts/offsets

    name_offset = len(header)
    import_offset = name_offset + len(name_blob)
    export_offset = import_offset + len(import_blob)
    depends_offset = export_offset + len(export_blob)
    struct.pack_into("<i", header, name_offset_pos, name_offset)
    struct.pack_into("<iiiii", header, export_fields, len(exports), export_offset,
                     len(imports), import_offset, depends_offset)
    struct.pack_into("<i", header, total_header_placeholder, depends_offset)
    # cooked export offsets are absolute across .uasset + .uexp
    for position, cursor in offset_positions:
        struct.pack_into("<q", export_blob, position, depends_offset + cursor)
    return bytes(header + name_blob + import_blob + export_blob)


def test_package_header_names_imports_exports():
    names = ["None", "/Script/Engine", "Texture2D", "Class", "MyTexture", "Package"]
    imports = [("/Script/Engine", "Class", "Texture2D")]
    exports = [(-1, "MyTexture", 128)]
    data = build_package(names, imports, exports)
    package = Package(data, uexp=b"\x00" * 128, name="MyTexture.uasset")

    assert package.names == names
    assert len(package.imports) == 1
    assert package.imports[0].object_name == "Texture2D"
    assert len(package.exports) == 1
    export = package.exports[0]
    assert export.object_name == "MyTexture"
    assert export.class_name == "Texture2D"          # resolved through the import table
    assert export.serial_size == 128


def test_package_rejects_non_unreal_data():
    with pytest.raises(PackageUnsupported):
        Package(b"NOT A PACKAGE" + b"\x00" * 128)


def test_package_reader_bounds():
    r = Reader(b"\x01\x02\x03\x04")
    assert r.u32() == 0x04030201
    with pytest.raises(Exception):
        r.u32()


def test_tagged_property_reading():
    """A property list: one IntProperty, terminated by 'None'."""
    names = ["None", "IntProperty", "MyValue", "Texture2D", "Class", "/Script/Engine",
             "MyTexture"]
    body = bytearray()
    body += _fname(names.index("MyValue")) + _fname(names.index("IntProperty"))
    body += struct.pack("<i", 4) + struct.pack("<i", 0)          # size, array index
    body += b"\x00"                                              # has property guid
    body += struct.pack("<i", 1234)
    body += _fname(names.index("None"))

    data = build_package(names, [("/Script/Engine", "Class", "Texture2D")],
                         [(-1, "MyTexture", len(body))])
    package = Package(data, uexp=bytes(body), name="t.uasset")
    props = package.read_properties(package.exports[0])
    assert props == {"MyValue": 1234}


# ---------------------------------------------------------------------------
# texture decoding shared with the Unity plugin
# ---------------------------------------------------------------------------
def test_shared_block_decoders():
    white_black = struct.pack("<HHI", 0xFFFF, 0x0000, 0x00000000)
    image = blocks.decode_block(white_black, 4, 4, "DXT1")
    assert image.shape == (4, 4, 4)
    assert (image[:, :, :3] == 255).all()

    packed = blocks.decode_packed(bytes([10, 20, 30, 255] * 4), 2, 2, "RGBA32")
    assert packed[0, 0, 0] == 10 and packed[0, 0, 2] == 30

    with pytest.raises(blocks.UnsupportedTextureFormat):
        blocks.decode_block(b"\x00" * 64, 4, 4, "BC7")


def test_expected_texture_bytes_matches_block_math():
    from app.plugins.builtin.unreal.assets import Texture2DReader
    assert Texture2DReader._expected_bytes("PF_DXT1", 8, 8) == 32
    assert Texture2DReader._expected_bytes("PF_DXT5", 8, 8) == 64
    assert Texture2DReader._expected_bytes("PF_B8G8R8A8", 4, 4) == 64
    assert Texture2DReader._expected_bytes("PF_UNKNOWN", 4, 4) == 0


def test_oodle_reports_absence_clearly(monkeypatch, tmp_path):
    from app.formats.archives import oodle
    monkeypatch.setattr(oodle, "_cache", {})
    monkeypatch.delenv(oodle.ENV_VAR, raising=False)
    assert oodle.get_codec([tmp_path]) is None
    reason = oodle.unavailable_reason()
    assert "oo2core" in reason and "proprietary" in reason
