"""Unity Texture2D pixel decoding.

Uncompressed formats and the BC1/BC3/BC4/BC5 block formats are decoded here
(vectorised with numpy). Formats we cannot decode are reported by name so the
report says exactly what is missing instead of guessing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Unity TextureFormat enum (the subset that matters here)
TEXTURE_FORMATS = {
    1: "Alpha8", 2: "ARGB4444", 3: "RGB24", 4: "RGBA32", 5: "ARGB32", 6: "ARGBFloat",
    7: "RGB565", 8: "BGR24", 9: "R16", 10: "DXT1", 11: "DXT3", 12: "DXT5", 13: "RGBA4444",
    14: "BGRA32", 15: "RHalf", 16: "RGHalf", 17: "RGBAHalf", 18: "RFloat", 19: "RGFloat",
    20: "RGBAFloat", 21: "YUY2", 22: "RGB9e5Float", 24: "BC6H", 25: "BC7", 26: "BC4",
    27: "BC5", 28: "DXT1Crunched", 29: "DXT5Crunched", 30: "PVRTC_RGB2",
    31: "PVRTC_RGBA2", 32: "PVRTC_RGB4", 33: "PVRTC_RGBA4", 34: "ETC_RGB4",
    41: "EAC_R", 42: "EAC_R_SIGNED", 43: "EAC_RG", 44: "EAC_RG_SIGNED", 45: "ETC2_RGB",
    46: "ETC2_RGBA1", 47: "ETC2_RGBA8", 48: "ASTC_RGB_4x4", 49: "ASTC_RGB_5x5",
    50: "ASTC_RGB_6x6", 51: "ASTC_RGB_8x8", 52: "ASTC_RGB_10x10", 53: "ASTC_RGB_12x12",
    54: "ASTC_RGBA_4x4", 55: "ASTC_RGBA_5x5", 56: "ASTC_RGBA_6x6", 57: "ASTC_RGBA_8x8",
    58: "ASTC_RGBA_10x10", 59: "ASTC_RGBA_12x12", 62: "RG16", 63: "R8",
    72: "ETC_RGB4_3DS", 73: "ETC_RGBA8_3DS",
}


class UnsupportedTextureFormat(Exception):
    pass


def format_name(value: int) -> str:
    return TEXTURE_FORMATS.get(value, "TextureFormat_%d" % value)


# ---------------------------------------------------------------------------
# uncompressed
# ---------------------------------------------------------------------------
def _from_bytes(data: bytes, width: int, height: int, channels: int, order) -> np.ndarray:
    needed = width * height * channels
    if len(data) < needed:
        raise UnsupportedTextureFormat("texture payload too small (%d < %d)" % (len(data), needed))
    arr = np.frombuffer(data[:needed], dtype=np.uint8).reshape(height, width, channels)
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    for dst, src in enumerate(order):
        if src is None:
            continue
        rgba[:, :, dst] = arr[:, :, src]
    return rgba


def _from_16bit(data: bytes, width: int, height: int, shifts, masks, scales) -> np.ndarray:
    needed = width * height * 2
    if len(data) < needed:
        raise UnsupportedTextureFormat("texture payload too small")
    values = np.frombuffer(data[:needed], dtype="<u2").reshape(height, width).astype(np.uint32)
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    for channel, (shift, mask, scale) in enumerate(zip(shifts, masks, scales)):
        if shift is None:
            continue
        rgba[:, :, channel] = (((values >> shift) & mask) * scale).astype(np.uint8)
    return rgba


# ---------------------------------------------------------------------------
# block compressed
# ---------------------------------------------------------------------------
def _blocks(data: bytes, width: int, height: int, block_bytes: int) -> tuple:
    bw = (width + 3) // 4
    bh = (height + 3) // 4
    needed = bw * bh * block_bytes
    if len(data) < needed:
        raise UnsupportedTextureFormat("compressed payload too small (%d < %d)"
                                       % (len(data), needed))
    raw = np.frombuffer(data[:needed], dtype=np.uint8).reshape(bh * bw, block_bytes)
    return raw, bw, bh


def _rgb565(values: np.ndarray) -> np.ndarray:
    r = ((values >> 11) & 0x1F).astype(np.uint16)
    g = ((values >> 5) & 0x3F).astype(np.uint16)
    b = (values & 0x1F).astype(np.uint16)
    out = np.empty(values.shape + (3,), dtype=np.uint8)
    out[..., 0] = (r * 255 + 15) // 31
    out[..., 1] = (g * 255 + 31) // 63
    out[..., 2] = (b * 255 + 15) // 31
    return out


def _decode_bc1_colors(raw: np.ndarray) -> tuple:
    """Returns (colors[n,4,4] rgba, indices[n,16])."""
    c0 = raw[:, 0].astype(np.uint16) | (raw[:, 1].astype(np.uint16) << 8)
    c1 = raw[:, 2].astype(np.uint16) | (raw[:, 3].astype(np.uint16) << 8)
    rgb0 = _rgb565(c0)
    rgb1 = _rgb565(c1)
    n = raw.shape[0]
    colors = np.zeros((n, 4, 4), dtype=np.uint8)
    colors[:, 0, :3] = rgb0
    colors[:, 1, :3] = rgb1
    colors[:, :, 3] = 255

    opaque = c0 > c1
    a = rgb0.astype(np.uint16)
    b = rgb1.astype(np.uint16)
    two_thirds = ((2 * a + b) // 3).astype(np.uint8)
    one_third = ((a + 2 * b) // 3).astype(np.uint8)
    half = ((a + b) // 2).astype(np.uint8)

    colors[opaque, 2, :3] = two_thirds[opaque]
    colors[opaque, 3, :3] = one_third[opaque]
    colors[~opaque, 2, :3] = half[~opaque]
    colors[~opaque, 3, :3] = 0
    colors[~opaque, 3, 3] = 0

    bits = (raw[:, 4].astype(np.uint32) | (raw[:, 5].astype(np.uint32) << 8)
            | (raw[:, 6].astype(np.uint32) << 16) | (raw[:, 7].astype(np.uint32) << 24))
    shifts = np.arange(16, dtype=np.uint32) * 2
    indices = (bits[:, None] >> shifts[None, :]) & 0x3
    return colors, indices


def _decode_alpha_block(raw: np.ndarray) -> np.ndarray:
    """BC3/BC4 alpha block -> [n,16] uint8."""
    a0 = raw[:, 0].astype(np.uint16)
    a1 = raw[:, 1].astype(np.uint16)
    n = raw.shape[0]
    table = np.zeros((n, 8), dtype=np.uint8)
    table[:, 0] = a0
    table[:, 1] = a1
    wide = a0 > a1
    for i in range(1, 7):
        table[wide, i + 1] = ((((7 - i) * a0[wide] + i * a1[wide]) // 7)).astype(np.uint8)
    for i in range(1, 5):
        table[~wide, i + 1] = ((((5 - i) * a0[~wide] + i * a1[~wide]) // 5)).astype(np.uint8)
    table[~wide, 6] = 0
    table[~wide, 7] = 255

    bits = np.zeros(n, dtype=np.uint64)
    for byte in range(6):
        bits |= raw[:, 2 + byte].astype(np.uint64) << np.uint64(8 * byte)
    shifts = (np.arange(16, dtype=np.uint64) * np.uint64(3))
    idx = ((bits[:, None] >> shifts[None, :]) & np.uint64(0x7)).astype(np.int64)
    return np.take_along_axis(table, idx, axis=1)


def _assemble(pixels: np.ndarray, bw: int, bh: int, width: int, height: int) -> np.ndarray:
    """pixels: [nblocks, 16, 4] -> image [height, width, 4]."""
    image = pixels.reshape(bh, bw, 4, 4, 4)          # by, bx, py, px, rgba
    image = image.transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return image[:height, :width, :]


def decode_bc1(data: bytes, width: int, height: int) -> np.ndarray:
    raw, bw, bh = _blocks(data, width, height, 8)
    colors, indices = _decode_bc1_colors(raw)
    pixels = np.take_along_axis(colors, indices[:, :, None].repeat(4, axis=2), axis=1)
    return _assemble(pixels, bw, bh, width, height)


def decode_bc3(data: bytes, width: int, height: int) -> np.ndarray:
    raw, bw, bh = _blocks(data, width, height, 16)
    alpha = _decode_alpha_block(raw[:, :8])
    colors, indices = _decode_bc1_colors(raw[:, 8:])
    colors[:, :, 3] = 255                            # alpha comes from the alpha block
    pixels = np.take_along_axis(colors, indices[:, :, None].repeat(4, axis=2), axis=1)
    pixels[:, :, 3] = alpha
    return _assemble(pixels, bw, bh, width, height)


def decode_bc4(data: bytes, width: int, height: int) -> np.ndarray:
    raw, bw, bh = _blocks(data, width, height, 8)
    red = _decode_alpha_block(raw)
    pixels = np.zeros((raw.shape[0], 16, 4), dtype=np.uint8)
    for channel in range(3):
        pixels[:, :, channel] = red
    pixels[:, :, 3] = 255
    return _assemble(pixels, bw, bh, width, height)


def decode_bc5(data: bytes, width: int, height: int) -> np.ndarray:
    raw, bw, bh = _blocks(data, width, height, 16)
    red = _decode_alpha_block(raw[:, :8])
    green = _decode_alpha_block(raw[:, 8:])
    pixels = np.zeros((raw.shape[0], 16, 4), dtype=np.uint8)
    pixels[:, :, 0] = red
    pixels[:, :, 1] = green
    # reconstruct the normal's Z so the preview looks right
    x = red.astype(np.float32) / 127.5 - 1.0
    y = green.astype(np.float32) / 127.5 - 1.0
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    pixels[:, :, 2] = ((z + 1.0) * 127.5).astype(np.uint8)
    pixels[:, :, 3] = 255
    return _assemble(pixels, bw, bh, width, height)


DECODERS = {
    "DXT1": decode_bc1,
    "DXT5": decode_bc3,
    "BC4": decode_bc4,
    "BC5": decode_bc5,
}


def decode(data: bytes, width: int, height: int, texture_format: int) -> np.ndarray:
    """Decode Unity pixel data to an RGBA image (top-left origin)."""
    if width <= 0 or height <= 0:
        raise UnsupportedTextureFormat("invalid texture size %dx%d" % (width, height))
    name = format_name(texture_format)

    if name in ("RGBA32",):
        image = _from_bytes(data, width, height, 4, (0, 1, 2, 3))
    elif name == "ARGB32":
        image = _from_bytes(data, width, height, 4, (1, 2, 3, 0))
    elif name == "BGRA32":
        image = _from_bytes(data, width, height, 4, (2, 1, 0, 3))
    elif name == "RGB24":
        image = _from_bytes(data, width, height, 3, (0, 1, 2, None))
    elif name == "BGR24":
        image = _from_bytes(data, width, height, 3, (2, 1, 0, None))
    elif name in ("Alpha8", "R8"):
        gray = _from_bytes(data, width, height, 1, (0, 0, 0, None))
        image = gray
        if name == "Alpha8":
            image[:, :, 3] = gray[:, :, 0]
            image[:, :, :3] = 255
    elif name == "RGB565":
        image = _from_16bit(data, width, height, (11, 5, 0, None), (0x1F, 0x3F, 0x1F, 0),
                            (8, 4, 8, 0))
    elif name == "RGBA4444":
        image = _from_16bit(data, width, height, (12, 8, 4, 0), (0xF, 0xF, 0xF, 0xF),
                            (17, 17, 17, 17))
    elif name == "ARGB4444":
        image = _from_16bit(data, width, height, (8, 4, 0, 12), (0xF, 0xF, 0xF, 0xF),
                            (17, 17, 17, 17))
    elif name in DECODERS:
        image = DECODERS[name](data, width, height)
    else:
        raise UnsupportedTextureFormat(
            "Unity texture format %s is not decoded by this build "
            "(add a decoder in plugins/unity)" % name)

    # Unity stores textures bottom-up
    return np.ascontiguousarray(image[::-1])


def unswizzle_normal_map(image: np.ndarray) -> tuple:
    """Undo Unity's DXT5nm packing (X in alpha, R left constant).

    Detected from the content, not from the file name: R is a flat 255 while
    another colour channel varies. Plain white masks (R, G and B all flat) are
    left alone.
    """
    if image.ndim != 3 or image.shape[2] != 4:
        return image, False
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    a = image[:, :, 3]
    r_flat = r.min() >= 250 and r.max() == r.min()
    colour_varies = g.std() > 1.0 or b.std() > 1.0
    alpha_varies = a.std() > 1.0
    if not (r_flat and colour_varies and alpha_varies):
        return image, False
    out = image.copy()
    out[:, :, 0] = a                      # X was parked in alpha
    out[:, :, 1] = g
    out[:, :, 2] = b
    out[:, :, 3] = 255
    return out, True


def to_png(image: np.ndarray) -> Optional[bytes]:
    try:
        import io

        from PIL import Image
        mode = "RGBA"
        img = Image.fromarray(image, mode)
        if not image[:, :, 3].min() < 255:
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None
