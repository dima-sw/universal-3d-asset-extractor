"""Built-in content detectors. Registered by app.core.detector.bootstrap()."""
from __future__ import annotations

import re
import struct

from app.core.detector.base import FileContext, FormatDetector
from app.core.detector.signatures import match_at_head
from app.core.types import Category, DetectionResult


class MagicDetector(FormatDetector):
    """Signature table lookup. Highest priority: bytes beat names."""

    name = "magic"
    priority = 100

    def detect(self, ctx: FileContext) -> DetectionResult:
        sig = match_at_head(ctx.header, ctx.read_at)
        if not sig:
            return self.nope()
        conf = sig.confidence
        # extension agreement is a weak bonus, disagreement is not a penalty
        if sig.ext and ctx.ext == sig.ext:
            conf = min(1.0, conf + 0.03)
        meta = {"magic_offset": sig.offset, "expected_ext": sig.ext}
        if sig.engine:
            meta["engine"] = sig.engine
        return self.result(sig.format_name, conf, sig.category, **meta)


class TextModelDetector(FormatDetector):
    """ASCII 3D formats: OBJ, MTL, DAE, glTF-json, ascii PLY/STL, SMD."""

    name = "text_model"
    priority = 90

    _OBJ = re.compile(r"^\s*(v|vn|vt|f|usemtl|mtllib|o|g)\s", re.M)

    def detect(self, ctx: FileContext) -> DetectionResult:
        if not ctx.looks_textual():
            return self.nope()
        text = ctx.text_head(8192)
        stripped = text.lstrip()
        low = stripped.lower()

        if low.startswith("<?xml") and "collada" in low[:4000]:
            return self.result("COLLADA (DAE)", 0.96, Category.MODEL)
        if stripped.startswith("{") and '"asset"' in text and '"version"' in text:
            return self.result("glTF (JSON)", 0.95, Category.MODEL)
        if stripped.startswith("ply"):
            return self.result("PLY (ascii)", 0.95, Category.MODEL)
        if low.startswith("solid ") and "facet normal" in low:
            return self.result("STL (ascii)", 0.94, Category.MODEL)
        if low.startswith("version 1") and ("skeleton" in low or "triangles" in low):
            return self.result("Valve SMD", 0.9, Category.MODEL, engine="Source")
        hits = len(self._OBJ.findall(text))
        if hits >= 4 and ("v " in text or "f " in text):
            return self.result("Wavefront OBJ", min(0.97, 0.75 + hits * 0.01), Category.MODEL)
        if low.startswith("newmtl") or "\nnewmtl " in low:
            return self.result("Wavefront MTL", 0.9, Category.OTHER)
        return self.nope()


class RenderWareDetector(FormatDetector):
    """RenderWare chunk stream (GTA III/VC/SA DFF/TXD/IFP and friends)."""

    name = "renderware"
    priority = 85

    TYPES = {0x10: ("RenderWare Clump (DFF)", Category.MODEL),
             0x16: ("RenderWare Texture Dictionary (TXD)", Category.TEXTURE),
             0x1B: ("RenderWare Animation (ANM)", Category.ANIMATION),
             0x24: ("RenderWare Delta Morph", Category.MODEL)}

    def detect(self, ctx: FileContext) -> DetectionResult:
        if len(ctx.header) < 12:
            return self.nope()
        ctype, size, libid = struct.unpack_from("<III", ctx.header, 0)
        if ctype not in self.TYPES:
            return self.nope()
        # library id: either packed version (>= 0x30000) or old style small value
        plausible_lib = libid >= 0x30000 or 0 < libid < 0x40
        if not plausible_lib:
            return self.nope()
        if not (0 < size <= max(ctx.size, 1)):
            return self.nope()
        fmt, cat = self.TYPES[ctype]
        return self.result(fmt, 0.9, cat, engine="RenderWare", chunk_type=ctype,
                           library_id=hex(libid))


class UnityAssetsDetector(FormatDetector):
    """Unity serialized files (.assets/.sharedAssets/level*/resources.assets).

    They have no magic; identify via the SerializedFile header + Unity version string.
    """

    name = "unity_serialized"
    priority = 84

    _VER = re.compile(rb"\b([2-6]\.[0-9x]+\.[0-9]+[a-z0-9]*)\b")
    _UNITY_VER = re.compile(rb"\b(20[0-2][0-9]|[3-6])\.[0-9]+\.[0-9]+[a-zA-Z0-9]*\b")

    def detect(self, ctx: FileContext) -> DetectionResult:
        h = ctx.header
        if len(h) < 32 or ctx.size < 64:
            return self.nope()
        if h[:7] in (b"UnityFS", b"UnityWe", b"UnityRa"):
            return self.nope()   # handled by MagicDetector
        try:
            meta_size, file_size, version, data_offset = struct.unpack_from(">IIII", h, 0)
        except struct.error:
            return self.nope()
        if not (5 <= version <= 30):
            return self.nope()
        if file_size not in (ctx.size, 0) and abs(file_size - ctx.size) > 4096:
            return self.nope()
        if data_offset >= max(ctx.size, 1) or meta_size == 0:
            return self.nope()
        m = self._UNITY_VER.search(h[16:256])
        unity_version = m.group(0).decode("ascii", "ignore") if m else None
        conf = 0.9 if unity_version else 0.72
        return self.result("Unity SerializedFile", conf, Category.GAME_CONTAINER,
                           engine="Unity", unity_version=unity_version,
                           format_version=version)


class UnrealAssetDetector(FormatDetector):
    """UE4/UE5 .uasset/.umap/.pak/.utoc packages."""

    name = "unreal_package"
    priority = 84
    PKG_MAGIC = 0x9E2A83C1
    PAK_MAGIC = b"\xE1\x12\x6F\x5A"      # trailing footer magic (UE4 pak)

    def detect(self, ctx: FileContext) -> DetectionResult:
        if len(ctx.header) >= 4:
            magic = struct.unpack_from("<I", ctx.header, 0)[0]
            if magic == self.PKG_MAGIC:
                legacy = struct.unpack_from("<i", ctx.header, 4)[0]
                return self.result("Unreal Package", 0.95, Category.GAME_CONTAINER,
                                   engine="Unreal", legacy_version=legacy)
        if ctx.ext == "utoc" or ctx.header[:4] == b"\x2d\x3d\x3d\x2d":
            return self.result("Unreal IoStore TOC", 0.9, Category.GAME_CONTAINER,
                               engine="Unreal")
        tail = ctx.tail
        if self.PAK_MAGIC in tail[-256:]:
            return self.result("Unreal PAK", 0.93, Category.GAME_CONTAINER, engine="Unreal")
        return self.nope()


class DirectoryContextDetector(FormatDetector):
    """Weak signals from filename + neighbouring files. Never decides alone."""

    name = "context"
    priority = 20

    MODEL_EXT = {"obj", "fbx", "gltf", "glb", "dae", "ply", "stl", "3ds", "blend", "md5mesh"}
    TEX_EXT = {"png", "jpg", "jpeg", "tga", "bmp", "dds", "ktx", "ktx2", "psd", "tif"}
    ANIM_EXT = {"bvh", "anim", "md5anim", "smd"}
    ARCHIVE_EXT = {"zip", "7z", "rar", "tar", "gz", "bz2", "xz", "cab", "pak", "pk3", "pk4"}
    # ".bin" is deliberately absent: it is generic. A real BIN track is caught
    # by the CD001 magic at the raw-sector offset, or through its .cue sheet.
    DISC_EXT = {"iso", "img", "cue", "cdi", "mdf", "mds", "nrg", "gcm", "wbfs"}

    def detect(self, ctx: FileContext) -> DetectionResult:
        e = ctx.ext
        if e in self.MODEL_EXT:
            return self.result("%s (by extension)" % e.upper(), 0.35, Category.MODEL, weak=True)
        if e in self.TEX_EXT:
            return self.result("%s (by extension)" % e.upper(), 0.35, Category.TEXTURE, weak=True)
        if e in self.ANIM_EXT:
            return self.result("%s (by extension)" % e.upper(), 0.35, Category.ANIMATION, weak=True)
        if e in self.ARCHIVE_EXT:
            return self.result("%s (by extension)" % e.upper(), 0.3, Category.ARCHIVE, weak=True)
        if e in self.DISC_EXT:
            return self.result("%s (by extension)" % e.upper(), 0.3, Category.DISC_IMAGE, weak=True)
        return self.nope()
