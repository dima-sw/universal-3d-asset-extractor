from __future__ import annotations

import struct

from app.core.detector import FileContext, identify
from app.core.extraction.iso import Iso9660, IsoExtractor, UnsupportedImage
from app.core.registry import summary
from app.formats.archives.lz4_block import decompress_block
from app.plugins.loader import discover, load_all

SECTOR = 2048


def build_iso(path, files: dict) -> None:
    """Minimal single-level ISO 9660 image (enough to exercise the reader)."""
    root_lba = 20
    data_lba = 21
    records = []
    entries = []
    for name, payload in files.items():
        entries.append((name, data_lba, len(payload), payload))
        data_lba += max(1, (len(payload) + SECTOR - 1) // SECTOR)

    def directory_record(name: str, lba: int, size: int, flags: int) -> bytes:
        identifier = name.encode("ascii")
        length = 33 + len(identifier)
        if length % 2:
            length += 1
        rec = bytearray(length)
        rec[0] = length
        struct.pack_into("<I", rec, 2, lba)
        struct.pack_into(">I", rec, 6, lba)
        struct.pack_into("<I", rec, 10, size)
        struct.pack_into(">I", rec, 14, size)
        rec[25] = flags
        rec[32] = len(identifier)
        rec[33:33 + len(identifier)] = identifier
        return bytes(rec)

    root_records = directory_record("\x00", root_lba, SECTOR, 0x02)
    root_records += directory_record("\x01", root_lba, SECTOR, 0x02)
    for name, lba, size, _payload in entries:
        root_records += directory_record(name, lba, size, 0x00)
    records.append(root_records)

    pvd = bytearray(SECTOR)
    pvd[0] = 1
    pvd[1:6] = b"CD001"
    pvd[6] = 1
    pvd[40:72] = b"TESTDISC".ljust(32)
    pvd[156:156 + 34] = directory_record("\x00", root_lba, SECTOR, 0x02)[:34]
    terminator = bytearray(SECTOR)
    terminator[0] = 255
    terminator[1:6] = b"CD001"

    with open(path, "wb") as f:
        f.write(b"\x00" * (16 * SECTOR))
        f.write(pvd)
        f.write(terminator)
        f.write(b"\x00" * ((root_lba - 18) * SECTOR))
        root_block = bytearray(SECTOR)
        root_block[:len(root_records)] = root_records
        f.write(root_block)
        for _name, _lba, _size, payload in entries:
            block = bytearray(max(SECTOR, ((len(payload) + SECTOR - 1) // SECTOR) * SECTOR))
            block[:len(payload)] = payload
            f.write(block)


def test_iso_reader_lists_and_extracts(tmp_path):
    iso_path = tmp_path / "GAME.ISO"
    build_iso(iso_path, {"SYSTEM.CNF": b"BOOT2 = cdrom0:\\SLUS_123.45;1\r\n",
                         "MODEL.BIN": b"MODELDATA" * 100})
    with Iso9660(iso_path) as image:
        names = sorted(e.name for e in image.walk())
        assert "SYSTEM.CNF" in names and "MODEL.BIN" in names
        assert image.platform_hint() == "PS1/PS2"

    ctx = FileContext(iso_path)
    detection = identify(ctx)
    assert "ISO 9660" in detection.format_name
    from app.config import Limits
    result = IsoExtractor().extract(ctx, tmp_path / "out", _budget(), Limits())
    assert result.ok
    assert (tmp_path / "out" / "MODEL.BIN").read_bytes().startswith(b"MODELDATA")


def _budget():
    from app.core.fs_safety import ExtractionBudget
    return ExtractionBudget(1 << 30, 10000, 1000.0, 1024)


def test_non_iso_file_reports_unsupported_not_corrupt(tmp_path):
    p = tmp_path / "weird.img"
    p.write_bytes(b"\x00" * (40 * SECTOR))
    try:
        Iso9660(p)
        assert False, "should not parse"
    except UnsupportedImage as exc:
        assert "plugin" in str(exc) or "ISO 9660" in str(exc)


def test_lz4_block_roundtrip():
    # literal-only block: token 0xF0.. with extended length
    payload = b"ABCDEFGH" * 4
    token = bytes([0xF0, len(payload) - 15])
    block = token + payload
    assert decompress_block(block, len(payload)) == payload


def test_lz4_block_with_match():
    # 4 literals "abcd", then a match of length 4 at offset 4 -> "abcdabcd"
    block = bytes([0x40]) + b"abcd" + bytes([0x04, 0x00])
    assert decompress_block(block, 8) == b"abcdabcd"


def test_plugins_discovered_and_registered():
    infos = discover()
    names = {i.name for i in infos}
    assert {"Unity", "Unreal", "PlayStation (PS1/PS2)"} <= names
    loaded = load_all()
    assert all(p.loaded for p in loaded), [p.error for p in loaded if not p.loaded]
    registries = summary()
    assert any("unity" in d for d in registries["extractors"])
    assert any("unreal" in d for d in registries["extractors"])
    assert "tim" in registries["texture_parsers"]


def test_unity_bundle_unsupported_message_is_actionable(tmp_path):
    fake = tmp_path / "assets.bundle"
    fake.write_bytes(b"UnityFS\x00" + b"\x00" * 64)
    ctx = FileContext(fake)
    detection = identify(ctx)
    assert detection.metadata.get("engine") == "Unity"
    from app.config import Limits
    from app.core.extraction import extract
    result = extract(ctx, detection, tmp_path / "out", Limits())
    assert result.ok is False
    assert result.unsupported_reason
    assert "corrupt" not in result.unsupported_reason.lower()
