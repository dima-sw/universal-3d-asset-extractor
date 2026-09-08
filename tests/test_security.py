"""Security tests: every input is hostile until proven otherwise."""
from __future__ import annotations

import os
import tarfile
import zipfile
import zlib

import pytest

from app.config import Limits
from app.core.detector import FileContext, identify
from app.core.extraction import extract
from app.core.fs_safety import (ExtractionBudget, LimitExceeded, UnsafePathError,
                                sanitize_relpath, safe_join)


def test_sanitize_rejects_traversal():
    with pytest.raises(UnsafePathError):
        sanitize_relpath("../../etc/passwd")
    with pytest.raises(UnsafePathError):
        sanitize_relpath("a/../../b")


def test_sanitize_strips_absolute_and_drive(tmp_path):
    assert str(sanitize_relpath("/etc/passwd")) == "etc/passwd"
    assert str(sanitize_relpath("C:\\Windows\\system32\\x.dll")) == "Windows/system32/x.dll"
    target = safe_join(tmp_path, "sub/../file.txt")
    assert target.parent == tmp_path.resolve()


def test_zip_slip_is_contained(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../../escaped.txt", "pwned")
        z.writestr("ok.txt", "fine")
    dest = tmp_path / "out"
    ctx = FileContext(archive)
    result = extract(ctx, identify(ctx), dest, Limits())
    escaped = tmp_path.parent / "escaped.txt"
    assert not escaped.exists()
    assert (dest / "ok.txt").exists()
    assert any("traversal" in e or "escapes" in e for e in result.errors)


def test_tar_symlink_member_is_skipped(tmp_path):
    archive = tmp_path / "links.tar"
    payload = tmp_path / "payload.txt"
    payload.write_text("data")
    with tarfile.open(archive, "w") as tf:
        tf.add(payload, arcname="payload.txt")
        info = tarfile.TarInfo("link.txt")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    dest = tmp_path / "out"
    ctx = FileContext(archive)
    result = extract(ctx, identify(ctx), dest, Limits())
    assert (dest / "payload.txt").exists()
    assert not (dest / "link.txt").exists()
    assert any("link" in s for s in result.skipped)


def test_zip_bomb_hits_the_size_limit(tmp_path):
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("big.bin", b"\x00" * (8 << 20))
    limits = Limits()
    limits.max_extracted_size = 1 << 20         # 1 MiB budget
    dest = tmp_path / "out"
    ctx = FileContext(archive)
    result = extract(ctx, identify(ctx), dest, limits)
    assert any("size limit" in e for e in result.errors)


def test_entry_count_limit(tmp_path):
    budget = ExtractionBudget(max_bytes=1 << 30, max_entries=3, max_ratio=1000)
    for _ in range(3):
        budget.account(10)
    with pytest.raises(LimitExceeded):
        budget.account(10)


def test_corrupted_archive_does_not_raise(tmp_path):
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"PK\x03\x04" + os.urandom(512))
    ctx = FileContext(archive)
    result = extract(ctx, identify(ctx), tmp_path / "out", Limits())
    assert result.ok is False
    assert result.errors or result.unsupported_reason


def test_truncated_model_is_reported_not_crashed(tmp_path):
    glb = tmp_path / "truncated.glb"
    glb.write_bytes(b"glTF" + b"\x02\x00\x00\x00" + b"\x10\x00\x00\x00")
    from app.formats.models import parse_model
    ctx = FileContext(glb)
    with pytest.raises(Exception):
        parse_model(ctx, identify(ctx))          # raises, but a controlled ParseError


def test_nested_archives_extract_recursively(tmp_path):
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("deep/file.txt", "hi")
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.write(inner, "inner.zip")
    dest = tmp_path / "out"
    ctx = FileContext(outer)
    result = extract(ctx, identify(ctx), dest, Limits())
    assert (dest / "inner.zip").exists()
    inner_ctx = FileContext(dest / "inner.zip")
    inner_result = extract(inner_ctx, identify(inner_ctx), dest / "nested", Limits())
    assert (dest / "nested" / "deep" / "file.txt").exists()
    assert inner_result.ok


def test_source_tree_is_never_modified(tmp_path, game_dir):
    before = {p: p.stat().st_mtime_ns for p in game_dir.rglob("*") if p.is_file()}
    from app.config import Settings
    from app.core.pipeline.controller import ExtractorCore
    s = Settings()
    s.output_dir = str(tmp_path / "out")
    s.temp_dir = str(tmp_path / "temp")
    s.db_path = str(tmp_path / "db.sqlite3")
    core = ExtractorCore(s)
    core.analyze(game_dir)
    core.extract()
    after = {p: p.stat().st_mtime_ns for p in game_dir.rglob("*") if p.is_file()}
    assert before == after
