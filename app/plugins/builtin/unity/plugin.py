"""Unity plugin.

Implemented here:
  * UnityFS bundle header + block table (LZ4 / LZMA / uncompressed), split back
    into its .assets / .resS members, which the pipeline then re-scans
  * SerializedFile reading (versions 13-22) with generic TypeTree decoding
  * Mesh -> GLB: vertex streams and compressed (PackedBitVector) meshes,
    skin weights, bone names and hierarchy from SkinnedMeshRenderer + Transform,
    bone transforms re-derived from the bind pose, Unity's left-handed space
    converted to glTF conventions
  * Texture2D -> PNG: uncompressed formats plus DXT1/DXT5/BC4/BC5, external
    .resS streams resolved, DXT5nm normal maps un-swizzled

Reported as unsupported (never as corruption):
  * files built without a type tree (many release builds) - that needs a
    class-layout database per Unity version
  * crunched, BC6H/BC7, ETC/ASTC/PVRTC textures
  * AnimationClip curves, Avatar rigs, MonoBehaviour scripts
"""
from __future__ import annotations

import struct
from pathlib import Path, PurePosixPath

from app.core.conversion.normalize import normalize
from app.core.fs_safety import LimitExceeded, safe_join, sanitize_component
from app.core.logging_setup import get_logger, unsupported_logger
from app.exporters import get_exporter
from app.formats.archives.lz4_block import Lz4Error, decompress_block, decompress_lzma_alone
from app.plugins.api import (Category, ContainerExtractor, DetectionResult, ExtractionResult,
                             FileContext, FormatDetector, GameProfile, TextureAsset)
from app.plugins.builtin.unity.objects import UnityAssetReader
from app.plugins.builtin.unity.serialized import SerializedFile, UnsupportedSerializedFile

log = get_logger("plugin.unity")

COMPRESSION_NONE = 0
COMPRESSION_LZMA = 1
COMPRESSION_LZ4 = 2
COMPRESSION_LZ4HC = 3


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        chunk = self.data[self.pos:self.pos + n]
        if len(chunk) < n:
            raise ValueError("unexpected end of bundle data")
        self.pos += n
        return chunk

    def u16(self) -> int:
        return struct.unpack(">H", self.read(2))[0]

    def u32(self) -> int:
        return struct.unpack(">I", self.read(4))[0]

    def i32(self) -> int:
        return struct.unpack(">i", self.read(4))[0]

    def i64(self) -> int:
        return struct.unpack(">q", self.read(8))[0]

    def cstring(self, limit: int = 512) -> str:
        end = self.data.find(b"\x00", self.pos, self.pos + limit)
        if end < 0:
            raise ValueError("unterminated string in bundle header")
        out = self.data[self.pos:end].decode("utf-8", "replace")
        self.pos = end + 1
        return out

    def align(self, boundary: int) -> None:
        pad = (-self.pos) % boundary
        self.pos += pad


def _decompress(payload: bytes, flags: int, uncompressed_size: int) -> bytes:
    kind = flags & 0x3F
    if kind == COMPRESSION_NONE:
        return payload[:uncompressed_size]
    if kind == COMPRESSION_LZMA:
        return decompress_lzma_alone(payload, uncompressed_size)
    if kind in (COMPRESSION_LZ4, COMPRESSION_LZ4HC):
        return decompress_block(payload, uncompressed_size)
    raise Lz4Error("unsupported bundle compression id %d" % kind)


class UnityBundleExtractor(ContainerExtractor):
    name = "unity_bundle"
    priority = 95
    formats = ("Unity AssetBundle (FS)", "Unity AssetBundle (Web)", "Unity AssetBundle (Raw)")

    def can_extract(self, ctx: FileContext, detection: DetectionResult) -> bool:
        return ctx.header[:7] == b"UnityFS"

    def extract(self, ctx: FileContext, dest: Path, budget, limits) -> ExtractionResult:
        res = ExtractionResult(extractor=self.name)
        if ctx.size > limits.max_single_file_size:
            return self.unsupported("bundle larger than the configured single-file limit")
        try:
            raw = ctx.path.read_bytes()
        except OSError as exc:
            res.errors.append("read failed: %s" % exc)
            return res

        try:
            r = _Reader(raw)
            signature = r.cstring()
            version = r.u32()
            unity_version = r.cstring()
            unity_revision = r.cstring()
            _bundle_size = r.i64()
            compressed_info = r.u32()
            uncompressed_info = r.u32()
            flags = r.u32()
            if signature != "UnityFS":
                return self.unsupported("unsupported bundle signature %r" % signature)
            if version >= 7:
                r.align(16)

            if flags & 0x80:                      # blocks info stored at file end
                info_raw = raw[-compressed_info:]
            else:
                info_raw = r.read(compressed_info)
            info = _decompress(info_raw, flags, uncompressed_info)
        except (ValueError, Lz4Error, Exception) as exc:
            return self.unsupported("Unity bundle header/version not supported by this "
                                    "build (%s)" % exc)

        try:
            ir = _Reader(info)
            ir.read(16)                            # uncompressed data hash
            block_count = ir.i32()
            blocks = []
            total_uncompressed = 0
            for _ in range(block_count):
                u_size = ir.u32()
                c_size = ir.u32()
                b_flags = ir.u16()
                blocks.append((u_size, c_size, b_flags))
                total_uncompressed += u_size
            if total_uncompressed > limits.max_extracted_size:
                raise LimitExceeded("bundle expands to %d bytes" % total_uncompressed)

            node_count = ir.i32()
            nodes = []
            for _ in range(node_count):
                offset = ir.i64()
                size = ir.i64()
                node_flags = ir.u32()
                path = ir.cstring(1024)
                nodes.append((offset, size, node_flags, path))
        except LimitExceeded as exc:
            return self.unsupported(str(exc))
        except Exception as exc:
            return self.unsupported("Unity bundle block table unreadable (%s)" % exc)

        # data blocks follow the (possibly relocated) block info
        data_start = r.pos if not (flags & 0x80) else r.pos
        payload = bytearray()
        pos = data_start
        try:
            for u_size, c_size, b_flags in blocks:
                chunk = raw[pos:pos + c_size]
                if len(chunk) < c_size:
                    raise ValueError("truncated block at offset %d" % pos)
                payload += _decompress(chunk, b_flags, u_size)
                pos += c_size
        except Exception as exc:
            return self.unsupported("Unity bundle block decompression failed (%s); "
                                    "compression id %d" % (exc, blocks[0][2] & 0x3F if blocks else -1))

        for offset, size, _node_flags, path in nodes:
            try:
                out = safe_join(dest, path)
                data = bytes(payload[offset:offset + size])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
                budget.account(len(data))
                res.files.append(out)
                res.entries += 1
            except LimitExceeded as exc:
                res.errors.append(str(exc))
                break
            except Exception as exc:
                res.errors.append("%s: %s" % (path, exc))
        res.bytes_written = budget.bytes_written
        res.ok = bool(res.files)
        log.info("unity bundle %s: %d member(s), unity %s (%s)", ctx.path.name, len(res.files),
                 unity_version, unity_revision)
        return res


class UnitySerializedNotice(FormatDetector):
    """Marks raw Unity resource streams (payload for a sibling SerializedFile)."""

    name = "unity_objects_notice"
    priority = 10

    def detect(self, ctx: FileContext) -> DetectionResult:
        if ctx.ext in ("ress", "resource") or ctx.name.lower().endswith(".ress"):
            # payload, not a container: the SerializedFile next to it reads it
            return self.result("Unity resource stream", 0.75, Category.OTHER,
                               engine="Unity",
                               note="raw pixel/vertex payload referenced by a SerializedFile")
        return self.nope()


class UnitySerializedExtractor(ContainerExtractor):
    """Reads a SerializedFile and writes its assets out in neutral formats.

    Meshes become GLB (mesh + skin + skeleton + material textures) and textures
    become PNG, both written into the extraction directory. The pipeline then
    re-scans them like any other model or texture, so the character grouping,
    viewer and exporters work with no Unity-specific code in the core.
    """

    name = "unity_serialized"
    priority = 94
    formats = ("Unity SerializedFile",)

    SERIALIZED_EXT = ("assets", "sharedassets", "resource", "unity3d")

    def can_extract(self, ctx: FileContext, detection: DetectionResult) -> bool:
        if detection.format_name in self.formats:
            return True
        if detection.metadata.get("engine") != "Unity":
            return False
        name = ctx.name.lower()
        return (name.endswith(self.SERIALIZED_EXT) or name.startswith("level")
                or name in ("globalgamemanagers", "mainData".lower()))

    def extract(self, ctx: FileContext, dest: Path, budget, limits) -> ExtractionResult:
        res = ExtractionResult(extractor=self.name)
        if ctx.size > limits.max_single_file_size:
            return self.unsupported("serialized file larger than the single-file limit")
        try:
            data = ctx.path.read_bytes()
        except OSError as exc:
            res.errors.append("read failed: %s" % exc)
            return res

        try:
            sf = SerializedFile(data, ctx.name)
        except UnsupportedSerializedFile as exc:
            return self.unsupported(str(exc))
        except Exception as exc:
            return self.unsupported("Unity SerializedFile could not be read (%s)" % exc)

        summary = sf.summary()
        log.info("%s: unity %s, %d objects %s", ctx.name, sf.unity_version,
                 len(sf.objects), summary["classes"])
        if not sf.enable_type_tree:
            # release build: the reader falls back to known class layouts, which
            # only cover Mesh / Texture2D / GameObject / Transform
            log.info("%s: no type tree, using class layouts for Unity %s",
                     ctx.name, sf.unity_version or "unknown")

        reader = UnityAssetReader(sf, stream_loader=lambda p, o, s:
                                  self._load_stream(ctx.path, p, o, s))
        dest.mkdir(parents=True, exist_ok=True)

        texture_files: dict = {}
        texture_assets: dict = {}
        for extracted in reader.textures():
            if extracted.asset is None or not extracted.asset.data:
                res.skipped.append("%s (%s): %s" % (extracted.name, extracted.format_name,
                                                    extracted.reason))
                continue
            out = safe_join(dest, "textures/%s.png" % extracted.name)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(extracted.asset.data)
            budget.account(len(extracted.asset.data))
            res.files.append(out)
            res.entries += 1
            texture_files[extracted.name] = out
            texture_assets[extracted.name] = extracted.asset

        by_path_id = {}
        for info in sf.objects_of("Texture2D"):
            obj = reader.object_dict(info.path_id) or {}
            name = obj.get("m_Name")
            if name in texture_assets:
                by_path_id[info.path_id] = texture_assets[name]

        exporter = get_exporter("glb")
        for extracted in reader.models():
            if extracted.model is None:
                res.skipped.append("%s: %s" % (extracted.name, extracted.reason))
                unsupported_logger().info("%s -> %s: %s", ctx.path, extracted.name,
                                          extracted.reason)
                continue
            for usage, path_id in extracted.texture_ids.items():
                tex = by_path_id.get(path_id)
                if tex is not None and extracted.model.materials:
                    copy = TextureAsset(name=tex.name, width=tex.width, height=tex.height,
                                        channels=tex.channels, has_alpha=tex.has_alpha,
                                        source_format=tex.source_format, data=tex.data,
                                        usage=usage,
                                        path=str(texture_files.get(tex.name, "")) or None)
                    extracted.model.materials[0].textures[usage] = copy
            # Unity is Y-up left-handed; convert once here so everything
            # downstream (viewer, exporters, reports) sees glTF conventions.
            extracted.model.coordinate_system = "y_up_lh"
            normalize(extracted.model, up_axis="y", handedness="right", unit_scale=1.0)
            result = exporter.export(extracted.model, dest / "models",
                                     sanitize_component(extracted.name))
            if not result.ok:
                res.errors.append("%s: %s" % (extracted.name, result.error))
                continue
            for warning in extracted.warnings:
                log.info("%s: %s", extracted.name, warning)
            for path in result.files:
                budget.account(path.stat().st_size if path.exists() else 0)
                res.files.append(path)
            res.entries += 1

        res.bytes_written = budget.bytes_written
        res.ok = bool(res.files)
        if not res.ok:
            if sf.enable_type_tree:
                res.unsupported_reason = ("no mesh or texture could be decoded from this "
                                          "SerializedFile (objects: %s)" % summary["classes"])
            else:
                res.unsupported_reason = (
                    "Unity %s release build without a type tree: no Mesh or Texture2D "
                    "matched a known class layout (objects: %s)"
                    % (sf.unity_version or "unknown", summary["classes"]))
        return res

    @staticmethod
    def _load_stream(source: Path, path: str, offset: int, size: int):
        """Resolve a StreamingInfo reference (.resS / .resource next to the file)."""
        if not path or size <= 0:
            return b""
        name = PurePosixPath(path.replace("\\", "/")).name
        candidates = [source.parent / name, source.parent / "Resources" / name]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                try:
                    with open(candidate, "rb") as f:
                        f.seek(offset)
                        return f.read(size)
                except OSError:
                    return b""
        return b""


def register(api) -> None:
    api.add_extractor(UnityBundleExtractor())
    api.add_extractor(UnitySerializedExtractor())
    api.add_detector(UnitySerializedNotice())
    api.add_game_profile(GameProfile(name="Unity", platforms=["pc", "mobile", "console"],
                                     extractors=[UnityBundleExtractor(),
                                                 UnitySerializedExtractor()]))
