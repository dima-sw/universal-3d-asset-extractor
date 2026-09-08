"""Tests for the stripped-type-tree layouts and the retro container plugins.

The Unity layout tests compare the hand-written layouts against type trees
dumped from a real Unity 2018.4 build (tests/data/ref_*.txt): if a field, an
order or an alignment flag drifts, these fail.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import pytest

from app.config import Limits
from app.core.detector import FileContext, identify
from app.core.detector.engine import detect_engine
from app.core.extraction import extract
from app.core.fs_safety import ExtractionBudget
from app.plugins.builtin.unity.layouts import (LayoutCache, expected_texture_bytes, mesh,
                                               parse_unity_version, texture2d,
                                               validate_texture2d)

DATA = Path(__file__).parent / "data"
REF_LINE = re.compile(r"^\s*(\d+)\s+(.+?)\s\s+(.+?)\s+size=(-?\d+)\s+flags=([0-9a-f]+)\s*$")


def _flatten(node, level=0, out=None):
    out = [] if out is None else out
    out.append((level, node.type, node.name, bool(node.meta_flag & 0x4000)))
    for child in node.children:
        _flatten(child, level + 1, out)
    return out


def _reference(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = REF_LINE.match(line)
        if not m:
            continue
        type_, name = m.group(2).strip(), m.group(3).strip()
        if not name:                       # "unsigned int  name" collapsed in the dump
            parts = type_.split()
            type_, name = " ".join(parts[:-1]), parts[-1]
        rows.append((int(m.group(1)), type_, name, bool(int(m.group(5), 16) & 0x4000)))
    return rows


@pytest.mark.parametrize("builder,fixture", [
    (texture2d, "ref_texture2d_2018.txt"),
    (mesh, "ref_mesh_2018.txt"),
])
def test_layout_matches_a_real_unity_type_tree(builder, fixture):
    reference = _reference(DATA / fixture)
    assert reference, "reference fixture missing"
    built = _flatten(builder((2018, 4, 21),
                             {"bones_aabb": False, "cooking_options": False}))
    assert len(built) == len(reference)
    for i, (mine, theirs) in enumerate(zip(built, reference)):
        assert mine == theirs, "node %d differs: %s != %s" % (i, mine, theirs)


def test_matrix_elements_are_row_major():
    """Bind poses are transposed if this order slips."""
    nodes = _flatten(mesh((2018, 4, 21), {"bones_aabb": False, "cooking_options": False}))
    names = [n for _l, t, n, _a in nodes if t == "float" and n.startswith("e")]
    assert names[:5] == ["e00", "e01", "e02", "e03", "e10"]


def test_parse_unity_version():
    assert parse_unity_version("2019.2.21f1") == (2019, 2, 21)
    assert parse_unity_version("5.6.0p4") == (5, 6, 0)
    assert parse_unity_version("") == (0, 0, 0)


def test_expected_texture_sizes():
    assert expected_texture_bytes(4, 4, "RGBA32", 1) == (64, 64)
    assert expected_texture_bytes(4, 4, "DXT1", 1) == (8, 8)
    base, full = expected_texture_bytes(8, 8, "DXT5", 4)
    assert base == 64 and full > base


def test_texture_validation_rejects_mismatched_payload():
    good = {"m_Name": "tex", "m_Width": 4, "m_Height": 4, "m_TextureFormat": 4,
            "m_MipCount": 1, "image data": b"\x00" * 64}
    assert validate_texture2d(good)
    bad_size = dict(good, **{"image data": b"\x00" * 9})
    assert not validate_texture2d(bad_size)
    bad_format = dict(good, m_TextureFormat=999)
    assert not validate_texture2d(bad_format)
    bad_name = dict(good, m_Name=123)
    assert not validate_texture2d(bad_name)


def test_layout_cache_rejects_garbage():
    cache = LayoutCache("2019.2.21f1")
    assert cache.read("Texture2D", b"\x7f" * 512, 0, 512, False) is None
    assert cache.read("Nonexistent", b"\x00" * 64, 0, 64, False) is None


# ---------------------------------------------------------------------------
# CRI AFS
# ---------------------------------------------------------------------------
def build_afs(path: Path, payloads: dict) -> None:
    count = len(payloads)
    header_size = 8 + (count + 1) * 8
    data_start = (header_size + 2047) // 2048 * 2048
    entries = []
    blob = bytearray()
    offset = data_start
    for _name, payload in payloads.items():
        entries.append((offset, len(payload)))
        blob += payload
        padding = (-len(payload)) % 2048
        blob += b"\x00" * padding
        offset += len(payload) + padding
    meta_offset = offset
    meta = bytearray()
    for name in payloads:
        record = bytearray(48)
        encoded = name.encode("ascii")[:31]
        record[:len(encoded)] = encoded
        meta += record
    out = bytearray(b"AFS\x00" + struct.pack("<I", count))
    for entry in entries:
        out += struct.pack("<II", *entry)
    out += struct.pack("<II", meta_offset, len(meta))
    out += b"\x00" * (data_start - len(out))
    out += blob
    out += meta
    path.write_bytes(bytes(out))


def test_afs_detection_and_extraction(tmp_path):
    archive = tmp_path / "DATA.AFS"
    build_afs(archive, {"model.bin": b"MDL0" + b"\x01" * 300,
                        "sound.adx": b"\x80\x00\x00\x20\x03\x12" + b"\x02" * 100,
                        "extra.dat": b"\x03" * 50})
    ctx = FileContext(archive)
    detection = identify(ctx)
    assert detection.format_name == "CRI AFS archive"
    assert detection.metadata["entries"] == 3

    dest = tmp_path / "out"
    result = extract(ctx, detection, dest, Limits())
    assert result.ok
    assert (dest / "model.bin").read_bytes().startswith(b"MDL0")
    assert (dest / "sound.adx").exists()
    assert identify(FileContext(dest / "sound.adx")).format_name == "CRI ADX audio"


# ---------------------------------------------------------------------------
# generic offset-table containers
# ---------------------------------------------------------------------------
def build_offset_container(path: Path, payloads: list) -> None:
    count = len(payloads)
    table_end = 4 + 4 * count
    start = (table_end + 15) // 16 * 16
    offsets = []
    blob = bytearray()
    offset = start
    for payload in payloads:
        offsets.append(offset)
        blob += payload
        padding = (-len(payload)) % 16
        blob += b"\x00" * padding
        offset += len(payload) + padding
    out = bytearray(struct.pack("<I", count))
    for value in offsets:
        out += struct.pack("<I", value)
    out += b"\x00" * (start - len(out))
    out += blob
    path.write_bytes(bytes(out))


def test_offset_table_container_roundtrip(tmp_path):
    payloads = [bytes([i]) * 2000 for i in range(1, 9)]
    archive = tmp_path / "levels.unk"
    build_offset_container(archive, payloads)
    ctx = FileContext(archive)
    detection = identify(ctx)
    assert detection.format_name.startswith("Offset-table container")
    assert detection.metadata["entries"] == len(payloads)

    dest = tmp_path / "out"
    result = extract(ctx, detection, dest, Limits())
    assert result.ok and result.entries == len(payloads)
    first = sorted(dest.glob("*.dat"))[0]
    assert first.read_bytes().startswith(b"\x01\x01\x01")


def test_offset_table_ignores_random_binaries(tmp_path):
    import os
    noise = tmp_path / "random.bin"
    noise.write_bytes(os.urandom(200000))
    assert not identify(FileContext(noise)).format_name.startswith("Offset-table")

    text = tmp_path / "notes.txt"
    text.write_text("hello world\n" * 2000)
    assert not identify(FileContext(text)).format_name.startswith("Offset-table")


# ---------------------------------------------------------------------------
# engine fingerprints
# ---------------------------------------------------------------------------
def test_frostbite_and_anvil_are_named(tmp_path):
    toc = tmp_path / "layout.toc"
    toc.write_bytes(b"\x00\xd1\xce\x01" + b"\x00" * 512)
    detection = identify(FileContext(toc))
    assert "Frostbite" in detection.format_name
    assert detection.metadata["engine"] == "Frostbite"
    assert "obfuscated" in detection.metadata["note"]

    forge = tmp_path / "DataPC.forge"
    forge.write_bytes(b"scimitar\x00" + struct.pack("<I", 28) + b"\x00" * 512)
    detection = identify(FileContext(forge))
    assert detection.format_name == "Ubisoft Anvil forge archive"
    assert detection.metadata["engine"] == "Anvil"

    pck = tmp_path / "sounds.pck"
    pck.write_bytes(b"AKPK" + b"\x00" * 256)
    assert identify(FileContext(pck)).metadata["engine"] == "Wwise"


def test_engine_scoring_does_not_saturate_on_one_extension(tmp_path):
    root = tmp_path / "Game"
    root.mkdir()
    for i in range(200):
        (root / ("audio_%03d.pck" % i)).write_bytes(b"AKPK" + b"\x00" * 16)
    (root / "DataPC.forge").write_bytes(b"scimitar\x00" + b"\x00" * 32)
    engines = {e.engine: e.confidence for e in detect_engine(root)}
    assert engines.get("Anvil", 0) > engines.get("Godot", 0)


def test_bin_extension_is_not_treated_as_a_disc_image(tmp_path):
    p = tmp_path / "character.bin"
    p.write_bytes(b"\x01\x02\x03\x04" * 100)
    detection = identify(FileContext(p))
    assert detection.category.value != "disc_image"
