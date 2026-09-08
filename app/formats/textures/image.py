"""Texture identification + conversion to modern formats.

Header parsing is dependency-free; conversion uses Pillow when installed.
DDS/KTX keep their compressed payload when Pillow cannot decode them - the raw
file is still exported so no data is lost.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Optional

from app.core.logging_setup import get_logger
from app.core.types import TextureAsset

log = get_logger("texture")

USAGE_PATTERNS = [
    ("normal", re.compile(r"(_n|_nrm|_norm|normal|_nm)\b|_n$", re.I)),
    ("roughness", re.compile(r"(rough|_r$|_rgh)", re.I)),
    ("metallic", re.compile(r"(metal|_m$|_mtl)", re.I)),
    ("ao", re.compile(r"(ambient|_ao\b|occlusion)", re.I)),
    ("emission", re.compile(r"(emis|glow|_e$)", re.I)),
    ("specular", re.compile(r"(spec|_s$)", re.I)),
    ("mask", re.compile(r"(mask|_mk\b|opacity|alpha)", re.I)),
    ("diffuse", re.compile(r"(diffuse|albedo|basecolor|base_color|_d$|_c$|col)", re.I)),
]

DXGI_ALPHA = {"DXT3", "DXT5", "BC2", "BC3", "BC7"}


def guess_usage(name: str) -> str:
    stem = Path(name).stem
    for usage, pattern in USAGE_PATTERNS:
        if pattern.search(stem):
            return usage
    return "diffuse"


def read_header(path) -> Optional[TextureAsset]:
    """Identify an image file and fill dimensions without decoding pixels."""
    p = Path(path)
    try:
        with open(p, "rb") as f:
            head = f.read(256)
    except OSError:
        return None
    if not head:
        return None
    tex = TextureAsset(name=p.stem, path=str(p), usage=guess_usage(p.name))

    if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) >= 26:
        w, h = struct.unpack_from(">II", head, 16)
        depth, color_type = head[24], head[25]
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type, 3)
        tex.width, tex.height, tex.bit_depth = w, h, depth
        tex.channels, tex.has_alpha = channels, color_type in (4, 6)
        tex.source_format = "PNG"
        return tex

    if head[:3] == b"\xff\xd8\xff":
        tex.source_format = "JPEG"
        tex.channels, tex.has_alpha = 3, False
        dims = _jpeg_dims(p)
        if dims:
            tex.width, tex.height = dims
        return tex

    if head[:2] == b"BM" and len(head) >= 26:
        w, h = struct.unpack_from("<ii", head, 18)
        bpp = struct.unpack_from("<H", head, 28)[0] if len(head) >= 30 else 24
        tex.width, tex.height = abs(w), abs(h)
        tex.channels, tex.has_alpha = (4, True) if bpp == 32 else (3, False)
        tex.source_format = "BMP"
        return tex

    if head[:4] == b"DDS " and len(head) >= 128:
        h, w = struct.unpack_from("<II", head, 12)
        fourcc = head[84:88].decode("ascii", "ignore").strip("\x00")
        mips = struct.unpack_from("<I", head, 28)[0]
        tex.width, tex.height = w, h
        tex.source_format = "DDS"
        tex.has_alpha = fourcc.upper() in DXGI_ALPHA
        tex.channels = 4 if tex.has_alpha else 3
        tex.metadata.update({"fourcc": fourcc, "mipmaps": mips})
        return tex

    if head[:12] == b"\xabKTX 11\xbb\r\n\x1a\n" and len(head) >= 64:
        w, h = struct.unpack_from("<II", head, 36)
        tex.width, tex.height, tex.source_format = w, h, "KTX"
        return tex
    if head[:12] == b"\xabKTX 20\xbb\r\n\x1a\n" and len(head) >= 40:
        w, h = struct.unpack_from("<II", head, 20)
        tex.width, tex.height, tex.source_format = w, h, "KTX2"
        return tex

    if head[:6] in (b"GIF87a", b"GIF89a") and len(head) >= 10:
        w, h = struct.unpack_from("<HH", head, 6)
        tex.width, tex.height, tex.source_format = w, h, "GIF"
        return tex

    # TGA has no magic: validate the 18-byte header instead
    if p.suffix.lower() == ".tga" and len(head) >= 18:
        image_type = head[2]
        if image_type in (0, 1, 2, 3, 9, 10, 11):
            w, h = struct.unpack_from("<HH", head, 12)
            bpp = head[16]
            if 0 < w <= 16384 and 0 < h <= 16384 and bpp in (8, 15, 16, 24, 32):
                tex.width, tex.height = w, h
                tex.bit_depth = 8
                tex.channels = 4 if bpp == 32 else 3
                tex.has_alpha = bpp == 32
                tex.source_format = "TGA"
                return tex
    return None


def _jpeg_dims(path) -> Optional[tuple]:
    try:
        with open(path, "rb") as f:
            f.read(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None
                if marker[1] in (0xD8, 0xD9):
                    continue
                length_raw = f.read(2)
                if len(length_raw) < 2:
                    return None
                length = struct.unpack(">H", length_raw)[0]
                if 0xC0 <= marker[1] <= 0xCF and marker[1] not in (0xC4, 0xC8, 0xCC):
                    body = f.read(5)
                    if len(body) < 5:
                        return None
                    h, w = struct.unpack(">HH", body[1:5])
                    return w, h
                f.seek(length - 2, 1)
    except OSError:
        return None


def convert(src, dest_dir, fmt: str = "png", overwrite: bool = False) -> Optional[Path]:
    """Convert a texture to a modern format. Falls back to a straight copy."""
    src = Path(src)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / (src.stem + "." + fmt.lower())
    if out.exists() and not overwrite:
        return out
    try:
        from PIL import Image
        with Image.open(src) as im:
            if im.mode not in ("RGB", "RGBA", "L"):
                im = im.convert("RGBA" if "A" in im.mode else "RGB")
            im.save(out)
        return out
    except Exception as exc:
        log.debug("pillow conversion failed for %s: %s", src, exc)
    # keep the original bytes rather than losing the asset
    fallback = dest_dir / src.name
    try:
        if not fallback.exists() or overwrite:
            fallback.write_bytes(src.read_bytes())
        return fallback
    except OSError as exc:
        log.warning("texture copy failed for %s: %s", src, exc)
        return None


def write_texture(tex: TextureAsset, dest_dir, fmt: str = "png") -> Optional[Path]:
    """Materialise a TextureAsset (in-memory or on-disk) into dest_dir."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if tex.data:
        ext = (tex.source_format or "png").lower()
        raw = dest_dir / ("%s.%s" % (tex.name or "texture", ext))
        raw.write_bytes(tex.data)
        if ext in ("png", "jpg", "jpeg"):
            return raw
        converted = convert(raw, dest_dir, fmt)
        return converted or raw
    if tex.path and Path(tex.path).exists():
        return convert(tex.path, dest_dir, fmt)
    return None
