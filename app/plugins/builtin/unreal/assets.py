"""Unreal object readers: UTexture2D today, mesh classes reported precisely.

After the tagged properties of an export come class-specific bytes. For
UTexture2D that payload is well defined and stable: strip flags, a cooked flag,
then one FTexturePlatformData per pixel format, each holding the mip chain. Mip
payloads live inline, at the end of the .uexp, or in the .ubulk sidecar.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging_setup import get_logger
from app.formats.textures import blocks
from app.plugins.builtin.unreal.package import Package, PackageError, Reader

log = get_logger("plugin.unreal.assets")

# EBulkDataFlags
BULKDATA_PayloadAtEndOfFile = 0x0001
BULKDATA_SerializeCompressedZLIB = 0x0002
BULKDATA_ForceInlinePayload = 0x0040
BULKDATA_PayloadInSeperateFile = 0x0100
BULKDATA_OptionalPayload = 0x0800
BULKDATA_Size64Bit = 0x2000

# Unreal pixel format -> how to decode it
BLOCK_FORMATS = {
    "PF_DXT1": "DXT1", "PF_DXT3": "DXT3", "PF_DXT5": "DXT5",
    "PF_BC4": "BC4", "PF_BC5": "BC5",
}
PACKED_FORMATS = {
    "PF_B8G8R8A8": "BGRA32", "PF_R8G8B8A8": "RGBA32", "PF_A8R8G8B8": "ARGB32",
    "PF_G8": "R8", "PF_A8": "Alpha8", "PF_R5G6B5_UNORM": "RGB565",
}
KNOWN_UNDECODED = {"PF_BC6H", "PF_BC7", "PF_ASTC_4x4", "PF_ETC2_RGB", "PF_ETC2_RGBA",
                   "PF_FloatRGBA", "PF_FloatRGB", "PF_R16F", "PF_G16", "PF_A32B32G32R32F"}


@dataclass
class BulkData:
    flags: int = 0
    element_count: int = 0
    size_on_disk: int = 0
    offset_in_file: int = 0
    inline: Optional[bytes] = None

    @property
    def in_separate_file(self) -> bool:
        return bool(self.flags & BULKDATA_PayloadInSeperateFile)

    @property
    def at_end_of_file(self) -> bool:
        return bool(self.flags & BULKDATA_PayloadAtEndOfFile)


@dataclass
class Mip:
    width: int = 0
    height: int = 0
    bulk: BulkData = field(default_factory=BulkData)


@dataclass
class TextureData:
    name: str = ""
    width: int = 0
    height: int = 0
    pixel_format: str = ""
    mips: list = field(default_factory=list)
    reason: str = ""
    png: Optional[bytes] = None
    srgb: bool = True


def _read_bulk(r: Reader) -> BulkData:
    bulk = BulkData()
    bulk.flags = r.i32()
    wide = bool(bulk.flags & BULKDATA_Size64Bit)
    bulk.element_count = r.i64() if wide else r.i32()
    bulk.size_on_disk = r.i64() if wide else r.i32()
    bulk.offset_in_file = r.i64()
    if bulk.flags & BULKDATA_ForceInlinePayload:
        bulk.inline = r.read(max(0, bulk.size_on_disk))
    return bulk


class Texture2DReader:
    """Decodes UTexture2D exports into PNG."""

    def __init__(self, package: Package, ubulk: Optional[bytes] = None,
                 uptnl: Optional[bytes] = None):
        self.package = package
        self.ubulk = ubulk if ubulk is not None else package.ubulk
        self.uptnl = uptnl

    def read(self, export) -> TextureData:
        out = TextureData(name=export.object_name)
        # UE 4.25+ can store properties unversioned (no names), in which case the
        # tagged reader cannot be trusted - the cooked block is found by structure
        # below either way.
        props = self.package.read_properties(export) or {}
        out.srgb = bool(props.get("SRGB", True))
        properties_end = getattr(export, "properties_end", 0)
        if properties_end >= (export.serial_size * 0.9):
            properties_end = 0                      # properties were unversioned
        data = self.package.export_data(export)
        if data is None:
            out.reason = "export payload is missing"
            return out
        found = self._find_platform_data(data, properties_end)
        if found is None:
            out.reason = ("no cooked texture data found in this export (editor-only "
                          "asset, or a build layout this reader does not know)")
            return out
        start, gap = found
        try:
            self._read_platform_data(Reader(data, start), out, gap)
        except PackageError as exc:
            out.reason = "texture payload unreadable: %s" % exc
            return out

        if not out.mips:
            out.reason = out.reason or "no mip data found"
            return out
        self._decode(out)
        return out

    def _find_platform_data(self, data: bytes, after: int) -> Optional[int]:
        """Locate FTexturePlatformData without trusting a fixed field order.

        Between the properties and the cooked data sit a few flags whose number
        differs between engine builds. The block itself is unmistakable: an FName
        naming a pixel format, then a skip offset, sizes, and the same format
        name repeated as a string. Find that and the rest reads cleanly.
        """
        names = self.package.names
        limit = min(len(data), max(after, 0) + (8192 if after else 1 << 20))
        for pos in range(max(0, after - 4), limit - 24):
            index, number = struct.unpack_from("<ii", data, pos)
            if number != 0 or not (0 <= index < len(names)):
                continue
            name = names[index]
            if not name.startswith("PF_"):
                continue
            # UE5 inserts extra fields between the skip offset and the sizes;
            # try each plausible gap and keep the one whose repeated format
            # string matches the FName we started from.
            for gap in (0, 4, 8, 12, 16, 20, 24):
                try:
                    probe = Reader(data, pos + 8)
                    skip_offset = probe.i64()
                    probe.read(gap)
                    size_x = probe.i32()
                    size_y = probe.i32()
                    probe.i32()                         # packed slices/flags
                    format_string = probe.string()
                except PackageError:
                    continue
                if format_string != name or skip_offset < 0:
                    continue
                if not (0 < size_x <= 32768 and 0 < size_y <= 32768):
                    continue
                return pos, gap
        return None

    def _read_platform_data(self, r: Reader, out: TextureData, gap: int = 0) -> None:
        """One block per pixel format; a skip offset lets us jump the rest."""
        guard = 0
        while guard < 8:
            guard += 1
            pixel_format = self.package.fname(r)
            if pixel_format in ("None", "none", ""):
                break
            skip_offset = r.i64()
            r.read(gap)                                 # build-specific extra fields
            size_x = r.i32()
            size_y = r.i32()
            r.i32()                                     # packed slices + flags
            format_string = r.string()
            r.i32()                                     # first mip to serialize
            mip_count = r.i32()
            if mip_count < 0 or mip_count > 32:
                raise PackageError("implausible mip count %d" % mip_count)
            mips = self._read_mips(r, mip_count, format_string or pixel_format,
                                   size_x, size_y)
            if not out.mips:
                out.width, out.height = size_x, size_y
                out.pixel_format = format_string or pixel_format
                out.mips = mips
            if skip_offset > 0 and skip_offset < len(r.data):
                r.pos = skip_offset
            else:
                break

    def _read_mips(self, r: Reader, count: int, fmt: str, size_x: int, size_y: int) -> list:
        """Mip records gained fields between engine versions. Try the known
        shapes and keep the one whose declared payload sizes match what the
        pixel format says a mip of that size must weigh."""
        start = r.pos
        best = None
        for strip_flags in (4, 0, 2):
            for extra_tail in (0, 4, 8):
                r.pos = start
                mips = []
                try:
                    for _ in range(count):
                        r.read(strip_flags)
                        bulk = _read_bulk(r)
                        mip = Mip(width=r.i32(), height=r.i32(), bulk=bulk)
                        r.i32()                             # SizeZ
                        r.read(extra_tail)
                        mips.append(mip)
                except PackageError:
                    continue
                score = self._score_mips(mips, fmt, size_x, size_y)
                if score and (best is None or score > best[0]):
                    best = (score, mips, r.pos)
        if best is None:
            raise PackageError("mip records match no known layout")
        r.pos = best[2]
        return best[1]

    @staticmethod
    def _expected_bytes(fmt: str, width: int, height: int) -> int:
        block = {"PF_DXT1": 8, "PF_BC4": 8, "PF_DXT3": 16, "PF_DXT5": 16, "PF_BC5": 16,
                 "PF_BC6H": 16, "PF_BC7": 16}.get(fmt)
        if block:
            return ((max(1, width) + 3) // 4) * ((max(1, height) + 3) // 4) * block
        per_pixel = {"PF_B8G8R8A8": 4, "PF_R8G8B8A8": 4, "PF_A8R8G8B8": 4, "PF_G8": 1,
                     "PF_A8": 1, "PF_FloatRGBA": 8}.get(fmt)
        return width * height * per_pixel if per_pixel else 0

    def _score_mips(self, mips: list, fmt: str, size_x: int, size_y: int) -> int:
        if not mips:
            return 0
        score = 0
        for mip in mips:
            if not (0 < mip.width <= 32768 and 0 < mip.height <= 32768):
                return 0
            if mip.bulk.size_on_disk < 0 or mip.bulk.size_on_disk > (256 << 20):
                return 0
            expected = self._expected_bytes(fmt, mip.width, mip.height)
            if expected and mip.bulk.size_on_disk == expected:
                score += 2                       # sizes agree with the pixel format
        if mips[0].width == size_x and mips[0].height == size_y:
            score += 3
        return score

    # ------------------------------------------------------------------
    def _payload(self, mip: Mip) -> Optional[bytes]:
        bulk = mip.bulk
        if bulk.inline is not None:
            return bulk.inline
        size = bulk.size_on_disk
        if size <= 0:
            return None
        if bulk.in_separate_file:
            source = self.uptnl if (bulk.flags & BULKDATA_OptionalPayload) else self.ubulk
            if source is None:
                return None
            if bulk.offset_in_file + size > len(source):
                return None
            return source[bulk.offset_in_file:bulk.offset_in_file + size]
        if bulk.at_end_of_file and self.package.uexp is not None:
            start = bulk.offset_in_file - self.package.summary.total_header_size
            if 0 <= start and start + size <= len(self.package.uexp):
                return self.package.uexp[start:start + size]
        return None

    def _decode(self, out: TextureData) -> None:
        name = out.pixel_format
        mip = out.mips[0]
        payload = self._payload(mip)
        if payload is None:
            for candidate in out.mips[1:]:
                payload = self._payload(candidate)
                if payload is not None:
                    mip = candidate
                    break
        if payload is None:
            out.reason = ("pixel data is stored outside the package (.ubulk) and was not "
                          "available")
            return

        width = mip.width or out.width
        height = mip.height or out.height
        try:
            if name in BLOCK_FORMATS:
                image = blocks.decode_block(payload, width, height, BLOCK_FORMATS[name])
            elif name in PACKED_FORMATS:
                image = blocks.decode_packed(payload, width, height, PACKED_FORMATS[name])
            elif name in KNOWN_UNDECODED:
                out.reason = ("pixel format %s is not decoded by this build "
                              "(add a decoder in plugins/unreal)" % name)
                return
            else:
                out.reason = "unknown pixel format %s" % name
                return
        except blocks.UnsupportedTextureFormat as exc:
            out.reason = str(exc)
            return
        except Exception as exc:
            out.reason = "decode failed: %s" % exc
            return

        out.width, out.height = width, height
        out.png = blocks.to_png(image)
        if out.png is None:
            out.reason = "decoded but PNG encoding failed"


MESH_CLASSES = {"StaticMesh", "SkeletalMesh"}


def describe_mesh(package: Package, export) -> str:
    """What we can say about a mesh export without decoding its render data."""
    props = package.read_properties(export) or {}
    materials = props.get("StaticMaterials") or props.get("Materials") or []
    bits = ["%s '%s'" % (export.class_name, export.object_name)]
    if isinstance(materials, list) and materials:
        bits.append("%d material slot(s)" % len(materials))
    if "LODGroup" in props:
        bits.append("LOD group %s" % props["LODGroup"])
    if export.class_name == "SkeletalMesh" and props.get("Skeleton"):
        skeleton = props["Skeleton"]
        if isinstance(skeleton, dict):
            bits.append("skeleton %s" % skeleton.get("__object"))
    return ", ".join(bits)
