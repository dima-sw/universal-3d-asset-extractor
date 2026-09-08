"""PlayStation plugin (PS1 / PS2).

Implemented:
  * TIM texture decoding (4/8/16/24 bpp, CLUT) -> PNG
  * detection of PS-EXE, TMD/HMD model containers, TIM2 textures, SYSTEM.CNF
  * a game-profile hook so per-game plugins can slot in under plugins/ps2/<game>

Not implemented (reported, not guessed):
  * TMD/HMD geometry decoding - primitive packet layouts vary per title
  * per-game proprietary archives: those belong in game-specific plugins
"""
from __future__ import annotations

import io
import struct
from pathlib import Path

from app.core.logging_setup import get_logger
from app.plugins.api import (Category, DetectionResult, FileContext, FormatDetector,
                             GameProfile, TextureAsset)

log = get_logger("plugin.playstation")

TIM_ID = 0x00000010
PMODE_BPP = {0: 4, 1: 8, 2: 16, 3: 24}


class TimDetector(FormatDetector):
    name = "ps1_tim"
    priority = 88

    def detect(self, ctx: FileContext) -> DetectionResult:
        h = ctx.header
        if len(h) < 12:
            return self.nope()
        magic, flags = struct.unpack_from("<II", h, 0)
        if magic != TIM_ID:
            return self.nope()
        pmode = flags & 0x07
        if pmode not in PMODE_BPP:
            return self.nope()
        has_clut = bool(flags & 0x08)
        block_size = struct.unpack_from("<I", h, 8)[0]
        if block_size < 12 or block_size > ctx.size:
            return self.nope()
        return self.result("Sony TIM texture", 0.9, Category.TEXTURE, engine="PS1",
                           bpp=PMODE_BPP[pmode], clut=has_clut)


class Tim2Detector(FormatDetector):
    name = "ps2_tim2"
    priority = 88

    def detect(self, ctx: FileContext) -> DetectionResult:
        if ctx.header[:4] != b"TIM2":
            return self.nope()
        return self.result("Sony TIM2 texture", 0.9, Category.TEXTURE, engine="PS2",
                           note="TIM2 decoding not implemented; raw file is preserved")


class TmdDetector(FormatDetector):
    """PS1 TMD / HMD model containers. Structure validated, geometry not decoded."""

    name = "ps1_tmd"
    priority = 86

    def detect(self, ctx: FileContext) -> DetectionResult:
        h = ctx.header
        if len(h) < 12:
            return self.nope()
        magic, flags, n_obj = struct.unpack_from("<III", h, 0)
        if magic == 0x00000041 and 0 < n_obj < 4096 and flags in (0, 1):
            return self.result("Sony TMD model", 0.8, Category.MODEL, engine="PS1",
                               objects=n_obj,
                               note="TMD detected; primitive decoding is game-dependent and "
                                    "not implemented - add a plugin parser for this title")
        if h[:4] == b"\x00\x00\x00\x02" and len(h) >= 20:
            return self.result("Sony HMD model", 0.6, Category.MODEL, engine="PS1",
                               note="HMD detected; geometry decoding not implemented")
        return self.nope()


class PsExeDetector(FormatDetector):
    name = "ps_exe"
    priority = 87

    def detect(self, ctx: FileContext) -> DetectionResult:
        if ctx.header[:8] == b"PS-X EXE":
            return self.result("PlayStation executable", 0.95, Category.EXECUTABLE, engine="PS1",
                               note="game code; scan it with Deep mode to find embedded assets")
        if ctx.name.upper() == "SYSTEM.CNF":
            text = ctx.text_head(2048)
            boot = next((l for l in text.splitlines() if l.upper().startswith("BOOT")), "")
            return self.result("PlayStation SYSTEM.CNF", 0.9, Category.OTHER,
                               engine="PS2" if "BOOT2" in text.upper() else "PS1", boot=boot)
        return self.nope()


class TimTextureParser:
    """Decodes a TIM into a PNG-backed TextureAsset."""

    name = "tim"
    priority = 80
    formats = ("Sony TIM texture",)

    def can_parse(self, ctx: FileContext, detection=None) -> bool:
        if detection is not None and detection.format_name in self.formats:
            return True
        return ctx.header[:4] == struct.pack("<I", TIM_ID)

    def parse(self, ctx: FileContext) -> TextureAsset:
        data = ctx.path.read_bytes()
        flags = struct.unpack_from("<I", data, 4)[0]
        pmode = flags & 0x07
        bpp = PMODE_BPP.get(pmode, 16)
        pos = 8
        palette = None
        if flags & 0x08:
            block_size, = struct.unpack_from("<I", data, pos)
            _dx, _dy, cw, ch = struct.unpack_from("<HHHH", data, pos + 4)
            entries = data[pos + 12: pos + block_size]
            palette = [self._rgba(struct.unpack_from("<H", entries, i)[0])
                       for i in range(0, min(len(entries), cw * ch * 2), 2)]
            pos += block_size

        block_size, = struct.unpack_from("<I", data, pos)
        _dx, _dy, w_words, height = struct.unpack_from("<HHHH", data, pos + 4)
        pixels = data[pos + 12: pos + block_size]
        width = {4: w_words * 4, 8: w_words * 2, 16: w_words, 24: w_words * 2 // 3}[bpp]

        rgba = bytearray()
        if bpp in (4, 8):
            pal = palette or [(i, i, i, 255) for i in range(256)]
            for row in range(height):
                stride = w_words * 2
                line = pixels[row * stride: (row + 1) * stride]
                for byte in line:
                    if bpp == 4:
                        for index in (byte & 0x0F, byte >> 4):
                            rgba += bytes(pal[index] if index < len(pal) else (0, 0, 0, 0))
                    else:
                        rgba += bytes(pal[byte] if byte < len(pal) else (0, 0, 0, 0))
        elif bpp == 16:
            for i in range(0, min(len(pixels), width * height * 2), 2):
                rgba += bytes(self._rgba(struct.unpack_from("<H", pixels, i)[0]))
        else:                                        # 24 bpp
            for i in range(0, min(len(pixels), width * height * 3), 3):
                rgba += bytes((pixels[i], pixels[i + 1], pixels[i + 2], 255))

        tex = TextureAsset(name=ctx.path.stem, width=width, height=height, channels=4,
                           has_alpha=True, source_format="TIM", path=str(ctx.path),
                           metadata={"bpp": bpp, "clut": bool(flags & 0x08)})
        png = self._to_png(rgba, width, height)
        if png:
            tex.data = png
            tex.source_format = "PNG"
            tex.metadata["decoded_from"] = "TIM"
        return tex

    @staticmethod
    def _rgba(value: int) -> tuple:
        r = (value & 0x1F) << 3
        g = ((value >> 5) & 0x1F) << 3
        b = ((value >> 10) & 0x1F) << 3
        stp = (value >> 15) & 1
        alpha = 0 if (value == 0 and not stp) else 255
        return (r, g, b, alpha)

    @staticmethod
    def _to_png(rgba: bytearray, width: int, height: int):
        if width <= 0 or height <= 0:
            return None
        needed = width * height * 4
        if len(rgba) < needed:
            rgba = rgba + bytes(needed - len(rgba))
        try:
            from PIL import Image
            img = Image.frombytes("RGBA", (width, height), bytes(rgba[:needed]))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:
            log.debug("TIM->PNG failed: %s", exc)
            return None


def register(api) -> None:
    api.add_detector(TimDetector())
    api.add_detector(Tim2Detector())
    api.add_detector(TmdDetector())
    api.add_detector(PsExeDetector())
    api.add_texture_parser(TimTextureParser())
    api.add_game_profile(GameProfile(name="PlayStation", platforms=["ps1", "ps2"],
                                     detectors=[TimDetector(), TmdDetector()],
                                     texture_parsers=[TimTextureParser()]))
