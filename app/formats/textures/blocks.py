"""Block-compressed and packed pixel decoding, shared by every plugin.

BC1/BC3/BC4/BC5 (DXT1/DXT5) and the common packed layouts are decoded here with
numpy so that Unity, Unreal and any future plugin use one implementation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class UnsupportedTextureFormat(Exception):
    pass


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


BLOCK_DECODERS = {
    "BC1": decode_bc1, "DXT1": decode_bc1,
    "BC2": None, "DXT3": None,
    "BC3": decode_bc3, "DXT5": decode_bc3,
    "BC4": decode_bc4,
    "BC5": decode_bc5,
}


def decode_packed(data: bytes, width: int, height: int, layout: str) -> np.ndarray:
    """Uncompressed layouts, named by channel order."""
    if layout == "RGBA32":
        return _from_bytes(data, width, height, 4, (0, 1, 2, 3))
    if layout == "ARGB32":
        return _from_bytes(data, width, height, 4, (1, 2, 3, 0))
    if layout == "BGRA32":
        return _from_bytes(data, width, height, 4, (2, 1, 0, 3))
    if layout == "RGB24":
        return _from_bytes(data, width, height, 3, (0, 1, 2, None))
    if layout == "BGR24":
        return _from_bytes(data, width, height, 3, (2, 1, 0, None))
    if layout in ("R8", "G8", "Alpha8"):
        gray = _from_bytes(data, width, height, 1, (0, 0, 0, None))
        if layout == "Alpha8":
            gray[:, :, 3] = gray[:, :, 0]
            gray[:, :, :3] = 255
        return gray
    if layout == "RGB565":
        return _from_16bit(data, width, height, (11, 5, 0, None), (0x1F, 0x3F, 0x1F, 0),
                           (8, 4, 8, 0))
    if layout == "RGBA4444":
        return _from_16bit(data, width, height, (12, 8, 4, 0), (0xF,) * 4, (17,) * 4)
    if layout == "ARGB4444":
        return _from_16bit(data, width, height, (8, 4, 0, 12), (0xF,) * 4, (17,) * 4)
    raise UnsupportedTextureFormat("packed layout %s is not decoded" % layout)


def decode_block(data: bytes, width: int, height: int, name: str) -> np.ndarray:
    decoder = BLOCK_DECODERS.get(name.upper())
    if decoder is None:
        raise UnsupportedTextureFormat("block format %s is not decoded by this build" % name)
    return decoder(data, width, height)


def to_png(image: np.ndarray, flip: bool = False) -> Optional[bytes]:
    """Encode an RGBA array as PNG (dropping a fully opaque alpha channel)."""
    try:
        import io

        from PIL import Image
        if flip:
            image = np.ascontiguousarray(image[::-1])
        img = Image.fromarray(image, "RGBA")
        if image[:, :, 3].min() == 255:
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None
