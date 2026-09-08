from __future__ import annotations

import struct
import zipfile

from app.core.detector import FileContext, identify
from app.core.detector.engine import detect_engine
from app.core.detector.heuristics import (CompressionDetector, EmbeddedAssetScanner,
                                          shannon_entropy)
from app.core.types import Category
from tests.conftest import CUBE_OBJ, make_png


def test_zip_detected_by_magic_not_extension(tmp_path):
    p = tmp_path / "archive.dat"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("a.txt", "hello")
    result = identify(FileContext(p))
    assert result.format_name.startswith("ZIP")
    assert result.category == Category.ARCHIVE
    assert result.confidence > 0.9


def test_obj_detected_by_content(tmp_path):
    p = tmp_path / "model.bin"          # deliberately misleading extension
    p.write_text(CUBE_OBJ)
    result = identify(FileContext(p))
    assert result.format_name == "Wavefront OBJ"
    assert result.category == Category.MODEL


def test_png_detected(tmp_path):
    p = tmp_path / "tex"
    p.write_bytes(make_png())
    result = identify(FileContext(p))
    assert result.format_name == "PNG"
    assert result.category == Category.TEXTURE


def test_unknown_binary_is_classified_not_rejected(tmp_path):
    p = tmp_path / "game.dat"
    p.write_bytes(bytes(range(256)) * 40)
    result = identify(FileContext(p))
    assert result.detected
    assert result.category in (Category.UNKNOWN, Category.OTHER)
    assert "entropy" in result.metadata


def test_renderware_dff_signature(tmp_path):
    p = tmp_path / "player.dff"
    body = b"\x00" * 64
    p.write_bytes(struct.pack("<III", 0x10, len(body), 0x1803FFFF) + body)
    result = identify(FileContext(p))
    assert "RenderWare" in result.format_name
    assert result.metadata.get("engine") == "RenderWare"


def test_tim_texture_detected_by_plugin(tmp_path):
    p = tmp_path / "face.tim"
    header = struct.pack("<II", 0x10, 0x02)          # 16bpp, no CLUT
    block = struct.pack("<IHHHH", 12 + 8, 0, 0, 2, 2) + b"\x00" * 8
    p.write_bytes(header + block)
    result = identify(FileContext(p))
    assert result.format_name == "Sony TIM texture"
    assert result.metadata.get("engine") == "PS1"


def test_engine_detection_unity(tmp_path):
    game = tmp_path / "Game"
    (game / "Game_Data").mkdir(parents=True)
    (game / "Game_Data" / "globalgamemanagers").write_bytes(b"\x00" * 32)
    (game / "Game_Data" / "resources.assets").write_bytes(b"\x00" * 32)
    engines = detect_engine(game)
    assert engines and engines[0].engine == "Unity"
    assert engines[0].confidence > 0.5


def test_embedded_asset_scanner_finds_png_inside_blob(tmp_path):
    png = make_png(16, 16)
    blob = tmp_path / "container.bin"
    blob.write_bytes(b"HEADER" + b"\x00" * 100 + png + b"\xff" * 50)
    hits = EmbeddedAssetScanner().scan(blob)
    png_hits = [h for h in hits if h.format_name == "PNG"]
    assert png_hits
    assert png_hits[0].offset == 106
    assert png_hits[0].metadata.get("width") == 16


def test_compression_detector_roundtrip():
    import zlib
    data = zlib.compress(b"x" * 4096)
    cd = CompressionDetector()
    assert cd.detect(data) == "zlib"
    assert cd.try_decompress(data) == b"x" * 4096


def test_entropy_ranges():
    assert shannon_entropy(b"\x00" * 4096) < 0.1
    assert shannon_entropy(bytes(range(256)) * 16) > 7.9
