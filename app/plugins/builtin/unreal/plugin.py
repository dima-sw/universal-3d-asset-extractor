"""Unreal Engine plugin.

Implemented:
  * .pak reader, versions 1-11: both the classic index and the UE5 path-hash /
    full-directory index, Zlib/Gzip/LZ4/Zstd blocks, and Oodle when an oo2core
    runtime the user already owns can be found
  * asset-type classification of the extracted .uasset/.uexp/.ubulk trio

Reported as unsupported (never as corrupt):
  * AES-encrypted paks and encrypted indexes (the game key is required, and this
    build neither ships nor derives keys)
  * IoStore .utoc/.ucas containers (UE4.26+/UE5)
  * Oodle-compressed data when no oo2core runtime is available
  * .uasset object graph parsing (StaticMesh / SkeletalMesh / AnimSequence)
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

from app.core.fs_safety import LimitExceeded, safe_join
from app.core.logging_setup import get_logger
from app.plugins.builtin.unreal.pak import PakFile, PakUnsupported
from app.plugins.api import (Category, ContainerExtractor, DetectionResult, ExtractionResult,
                             FileContext, FormatDetector, GameProfile)

log = get_logger("plugin.unreal")

PAK_MAGIC = 0x5A6F12E1
MAX_SUPPORTED_VERSION = 8

ASSET_KIND_HINTS = {
    "SkeletalMesh": "skeletal mesh", "StaticMesh": "static mesh", "Skeleton": "skeleton",
    "AnimSequence": "animation sequence", "Texture2D": "texture", "Material": "material",
    "PhysicsAsset": "physics asset", "MaterialInstanceConstant": "material instance",
}


def _fstring(data: bytes, pos: int) -> tuple:
    if pos + 4 > len(data):
        raise ValueError("truncated string length")
    n = struct.unpack_from("<i", data, pos)[0]
    pos += 4
    if n == 0:
        return "", pos
    if n > 0:
        raw = data[pos:pos + n]
        pos += n
        return raw.split(b"\x00")[0].decode("utf-8", "replace"), pos
    n = -n * 2
    raw = data[pos:pos + n]
    pos += n
    return raw.decode("utf-16-le", "replace").split("\x00")[0], pos


class UnrealPakExtractor(ContainerExtractor):
    name = "unreal_pak"
    priority = 92
    formats = ("Unreal PAK",)

    def can_extract(self, ctx: FileContext, detection: DetectionResult) -> bool:
        if ctx.ext != "pak":
            return detection.format_name == "Unreal PAK"
        return struct.pack("<I", PAK_MAGIC) in ctx.tail[-256:]

    def extract(self, ctx: FileContext, dest: Path, budget, limits) -> ExtractionResult:
        """Read the pak with the full v1-v11 reader and write its files out."""
        res = ExtractionResult(extractor=self.name)
        try:
            pak = PakFile(ctx.path)
        except PakUnsupported as exc:
            return self.unsupported(str(exc))
        except Exception as exc:
            return self.unsupported("PAK could not be read (%s)" % exc)

        blocked = pak.unsupported_reason()
        if blocked:
            return self.unsupported(blocked)
        if len(pak.entries) > limits.max_entries_per_archive:
            return self.unsupported("PAK holds %d files (limit %d)"
                                    % (len(pak.entries), limits.max_entries_per_archive))

        mount = (pak.mount_point or "").strip("/")
        encrypted = 0
        for entry in pak.entries:
            if entry.encrypted:
                encrypted += 1
                continue
            if entry.uncompressed_size > limits.max_single_file_size:
                res.skipped.append("%s (too large)" % entry.name)
                continue
            try:
                data = pak.read_entry(entry)
                if data is None:
                    res.skipped.append("%s (%s not decodable)"
                                       % (entry.name, pak.method_name(entry.compression_index)))
                    continue
                out = safe_join(dest, "%s/%s" % (mount, entry.name) if mount else entry.name)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
                budget.account(len(data))
                res.files.append(out)
                res.entries += 1
            except LimitExceeded as exc:
                res.errors.append(str(exc))
                break
            except Exception as exc:
                res.errors.append("%s: %s" % (entry.name, exc))
        if encrypted:
            res.skipped.append("%d entries are AES-encrypted (game key required)" % encrypted)
        res.bytes_written = budget.bytes_written
        res.ok = bool(res.files)
        if not res.ok and not res.errors:
            res.unsupported_reason = "no readable entries in this PAK"
        return res


class UnrealAssetNotice(FormatDetector):
    """Classify .uasset/.uexp/.utoc so the report is precise about what is missing."""

    name = "unreal_asset_notice"
    priority = 12

    def detect(self, ctx: FileContext) -> DetectionResult:
        ext = ctx.ext
        if ext in ("uasset", "umap"):
            head = ctx.header[:65536]
            kind = next((v for k, v in ASSET_KIND_HINTS.items() if k.encode() in head), "unknown")
            return self.result("Unreal %s (.%s)" % (kind, ext), 0.75, Category.GAME_CONTAINER,
                               engine="Unreal", asset_kind=kind,
                               note="object graph parsing not implemented; add it in "
                                    "plugins/unreal without touching the core")
        if ext in ("uexp", "ubulk", "uptnl"):
            # payload for the sibling .uasset, not a container in its own right
            return self.result("Unreal bulk data (.%s)" % ext, 0.75, Category.OTHER,
                               engine="Unreal", note="payload read through its .uasset")
        if ext in ("utoc", "ucas"):
            return self.result("Unreal IoStore container (.%s)" % ext, 0.8,
                               Category.GAME_CONTAINER, engine="Unreal",
                               note="IoStore reading is not implemented in this build")
        return self.nope()


class UnrealAssetExtractor(ContainerExtractor):
    """Reads a .uasset package and writes out what it can decode.

    Textures become PNG next to the package, so the pipeline picks them up like
    any other image. Exports we cannot decode yet are reported by class name, so
    the report says exactly what is inside and what is missing.
    """

    name = "unreal_asset"
    priority = 90
    formats = ("Unreal Package",)

    def can_extract(self, ctx: FileContext, detection: DetectionResult) -> bool:
        if ctx.ext in ("uasset", "umap"):
            return True
        return detection.format_name == "Unreal Package"

    def extract(self, ctx: FileContext, dest: Path, budget, limits) -> ExtractionResult:
        from app.plugins.builtin.unreal.assets import Texture2DReader, describe_mesh
        from app.plugins.builtin.unreal.package import Package, PackageUnsupported

        res = ExtractionResult(extractor=self.name)
        try:
            data = ctx.path.read_bytes()
        except OSError as exc:
            res.errors.append("read failed: %s" % exc)
            return res

        sidecar = {}
        for suffix in (".uexp", ".ubulk", ".uptnl"):
            candidate = ctx.path.with_suffix(suffix)
            sidecar[suffix] = candidate.read_bytes() if candidate.exists() else None

        try:
            package = Package(data, sidecar[".uexp"], name=ctx.name)
        except PackageUnsupported as exc:
            return self.unsupported(str(exc))
        except Exception as exc:
            return self.unsupported("Unreal package could not be read (%s)" % exc)

        reader = Texture2DReader(package, sidecar[".ubulk"], sidecar[".uptnl"])
        undecoded = []
        for export in package.exports:
            if export.class_name == "Texture2D":
                texture = reader.read(export)
                if texture.png:
                    out = safe_join(dest, "%s.png" % (texture.name or export.object_name))
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(texture.png)
                    budget.account(len(texture.png))
                    res.files.append(out)
                    res.entries += 1
                else:
                    undecoded.append("%s (Texture2D): %s" % (export.object_name,
                                                             texture.reason))
            elif export.class_name in ("StaticMesh", "SkeletalMesh"):
                undecoded.append("%s - geometry decoding is not implemented yet"
                                 % describe_mesh(package, export))
            elif export.class_name not in ("Package", "None"):
                undecoded.append("%s (%s): no reader for this class"
                                 % (export.object_name, export.class_name))

        for note in undecoded[:50]:
            res.skipped.append(note)
            unsupported_logger().info("%s -> %s", ctx.path, note)
        res.bytes_written = budget.bytes_written
        res.ok = bool(res.files)
        if not res.ok:
            classes = sorted({e.class_name for e in package.exports})
            res.unsupported_reason = ("Unreal %s package read (%d exports: %s) but nothing "
                                      "could be decoded yet"
                                      % (package.engine, len(package.exports),
                                         ", ".join(classes[:6]) or "none"))
        return res


def register(api) -> None:
    api.add_extractor(UnrealPakExtractor())
    api.add_extractor(UnrealAssetExtractor())
    api.add_detector(UnrealAssetNotice())
    api.add_game_profile(GameProfile(name="Unreal Engine", platforms=["pc", "console"],
                                     extractors=[UnrealPakExtractor(),
                                                 UnrealAssetExtractor()]))
