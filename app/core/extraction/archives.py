"""Generic archive extractors. All writes go through the safety layer."""
from __future__ import annotations

import bz2
import gzip
import lzma
import tarfile
import zipfile
from pathlib import Path

from app.core.detector.base import FileContext
from app.core.extraction.base import ContainerExtractor, ExtractionResult
from app.core.fs_safety import (ExtractionBudget, LimitExceeded, UnsafePathError,
                                copy_stream, safe_join, sanitize_component)
from app.core.logging_setup import get_logger
from app.core.types import DetectionResult

log = get_logger("extract.archive")


class ZipExtractor(ContainerExtractor):
    name = "zip"
    priority = 90
    formats = ("ZIP", "ZIP (empty)")

    def can_extract(self, ctx: FileContext, detection: DetectionResult) -> bool:
        return detection.format_name.startswith("ZIP") or (
            detection.category.value == "archive" and ctx.ext in ("zip", "pk3", "pk4", "love",
                                                                  "jar", "apk", "unitypackage"))

    def extract(self, ctx: FileContext, dest: Path, budget: ExtractionBudget, limits) -> ExtractionResult:
        res = ExtractionResult(extractor=self.name)
        try:
            zf = zipfile.ZipFile(ctx.path)
        except Exception as exc:
            res.errors.append("open failed: %s" % exc)
            return res
        with zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                try:
                    if info.file_size > limits.max_single_file_size:
                        res.skipped.append("%s (too large)" % info.filename)
                        continue
                    # symlinks are stored with S_IFLNK in external_attr high bits
                    if (info.external_attr >> 16) & 0xF000 == 0xA000:
                        res.skipped.append("%s (symlink)" % info.filename)
                        continue
                    out = safe_join(dest, info.filename)
                    with zf.open(info, "r") as src:
                        copy_stream(src, out, budget)
                    res.files.append(out)
                    res.entries += 1
                except (UnsafePathError, LimitExceeded) as exc:
                    res.errors.append("%s: %s" % (info.filename, exc))
                    if isinstance(exc, LimitExceeded):
                        break
                except Exception as exc:
                    res.errors.append("%s: %s" % (info.filename, exc))
        res.bytes_written = budget.bytes_written
        res.ok = bool(res.files) or not res.errors
        return res


class TarExtractor(ContainerExtractor):
    name = "tar"
    priority = 88
    formats = ("TAR",)

    def can_extract(self, ctx: FileContext, detection: DetectionResult) -> bool:
        if detection.format_name == "TAR":
            return True
        return ctx.name.lower().endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2",
                                          ".tar.xz", ".txz"))

    def extract(self, ctx: FileContext, dest: Path, budget: ExtractionBudget, limits) -> ExtractionResult:
        res = ExtractionResult(extractor=self.name)
        try:
            tf = tarfile.open(ctx.path, "r:*")
        except Exception as exc:
            res.errors.append("open failed: %s" % exc)
            return res
        with tf:
            for member in tf:
                if not member.isfile():
                    if member.issym() or member.islnk():
                        res.skipped.append("%s (link)" % member.name)
                    continue
                try:
                    if member.size > limits.max_single_file_size:
                        res.skipped.append("%s (too large)" % member.name)
                        continue
                    out = safe_join(dest, member.name)
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with src:
                        copy_stream(src, out, budget)
                    res.files.append(out)
                    res.entries += 1
                except (UnsafePathError, LimitExceeded) as exc:
                    res.errors.append("%s: %s" % (member.name, exc))
                    if isinstance(exc, LimitExceeded):
                        break
                except Exception as exc:
                    res.errors.append("%s: %s" % (member.name, exc))
        res.bytes_written = budget.bytes_written
        res.ok = bool(res.files)
        return res


class SingleStreamExtractor(ContainerExtractor):
    """gzip / bzip2 / xz holding one payload (often another archive or a tar)."""

    name = "single_stream"
    priority = 80
    formats = ("GZIP", "BZIP2", "XZ")

    OPENERS = {"GZIP": gzip.open, "BZIP2": bz2.open, "XZ": lzma.open}

    def extract(self, ctx: FileContext, dest: Path, budget: ExtractionBudget, limits) -> ExtractionResult:
        res = ExtractionResult(extractor=self.name)
        # a .tar.gz is handled by TarExtractor; here we unwrap a lone stream
        opener = self.OPENERS.get(_normalise(ctx, res))
        if opener is None:
            res.errors.append("unsupported stream type")
            return res
        stem = ctx.path.stem or "payload"
        out = dest / sanitize_component(stem)
        try:
            with opener(ctx.path, "rb") as src:
                copy_stream(src, out, budget)
            res.files.append(out)
            res.entries = 1
            res.ok = True
        except LimitExceeded as exc:
            res.errors.append(str(exc))
        except Exception as exc:
            res.errors.append("decompress failed: %s" % exc)
        res.bytes_written = budget.bytes_written
        return res


def _normalise(ctx: FileContext, res: ExtractionResult) -> str:
    head = ctx.header[:6]
    if head[:3] == b"\x1f\x8b\x08":
        return "GZIP"
    if head[:3] == b"BZh":
        return "BZIP2"
    if head[:6] == b"\xfd7zXZ\x00":
        return "XZ"
    return ""


class SevenZipExtractor(ContainerExtractor):
    name = "7z"
    priority = 85
    formats = ("7-Zip",)

    def extract(self, ctx: FileContext, dest: Path, budget: ExtractionBudget, limits) -> ExtractionResult:
        try:
            import py7zr
        except ImportError:
            return self.unsupported("7-Zip support needs the optional 'py7zr' package")
        res = ExtractionResult(extractor=self.name)
        try:
            with py7zr.SevenZipFile(ctx.path, "r") as archive:
                names = archive.getnames()
                if len(names) > limits.max_entries_per_archive:
                    return self.unsupported("archive has %d entries (limit %d)"
                                            % (len(names), limits.max_entries_per_archive))
                safe_dest = Path(dest)
                safe_dest.mkdir(parents=True, exist_ok=True)
                archive.extractall(path=safe_dest)
            for p in safe_dest.rglob("*"):
                if p.is_file():
                    try:
                        p.resolve().relative_to(safe_dest.resolve())
                    except ValueError:
                        res.skipped.append(str(p))
                        continue
                    res.files.append(p)
                    budget.account(p.stat().st_size)
            res.entries = len(res.files)
            res.bytes_written = budget.bytes_written
            res.ok = True
        except LimitExceeded as exc:
            res.errors.append(str(exc))
        except Exception as exc:
            res.errors.append("7z extraction failed: %s" % exc)
        return res


class RarExtractor(ContainerExtractor):
    name = "rar"
    priority = 85
    formats = ("RAR4", "RAR5")

    def extract(self, ctx: FileContext, dest: Path, budget: ExtractionBudget, limits) -> ExtractionResult:
        try:
            import rarfile
        except ImportError:
            return self.unsupported("RAR support needs the optional 'rarfile' package "
                                    "(and an unrar/bsdtar backend)")
        res = ExtractionResult(extractor=self.name)
        try:
            with rarfile.RarFile(ctx.path) as rf:
                for info in rf.infolist():
                    if info.is_dir():
                        continue
                    try:
                        out = safe_join(dest, info.filename)
                        with rf.open(info) as src:
                            copy_stream(src, out, budget)
                        res.files.append(out)
                        res.entries += 1
                    except (UnsafePathError, LimitExceeded) as exc:
                        res.errors.append("%s: %s" % (info.filename, exc))
                        if isinstance(exc, LimitExceeded):
                            break
                    except Exception as exc:
                        res.errors.append("%s: %s" % (info.filename, exc))
            res.ok = bool(res.files)
        except Exception as exc:
            res.errors.append("rar extraction failed: %s" % exc)
        res.bytes_written = budget.bytes_written
        return res


class CabExtractor(ContainerExtractor):
    name = "cab"
    priority = 60
    formats = ("Microsoft Cabinet",)

    def extract(self, ctx: FileContext, dest: Path, budget, limits) -> ExtractionResult:
        return self.unsupported("CAB extraction not implemented; install a plugin or "
                                "extract manually")


BUILTIN = [ZipExtractor, TarExtractor, SevenZipExtractor, RarExtractor,
           SingleStreamExtractor, CabExtractor]
