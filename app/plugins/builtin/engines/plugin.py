"""Fingerprints for engines whose containers have no reader here yet.

Naming a format precisely is worth as much as extracting it: the report tells
the user which engine they are looking at, why nothing came out, and what a
plugin would have to implement. Guessing "unknown binary" for a 45 GB Anvil
archive helps nobody.

None of these detectors try to defeat encryption or copy protection; where a
container is obfuscated or encrypted that is stated and the file is left alone.
"""
from __future__ import annotations

import struct

from app.plugins.api import Category, DetectionResult, FileContext, FormatDetector

FROSTBITE_DBOBJECT = b"\x00\xd1\xce"          # .toc / .cat / .sb header marker
FROSTBITE_CAS_CHUNK = b"\x00\x00\x11\xe0"     # cas_XX.cas payload chunk


class FrostbiteDetector(FormatDetector):
    """Battlefield / Battlefront / Need for Speed / Dragon Age (EA Frostbite)."""

    name = "frostbite"
    priority = 86

    NOTE = ("Frostbite bundle layer. Reading it needs a plugin that walks "
            "cat/toc/sb catalogues into the cas chunk store; the headers are "
            "obfuscated, which this build does not touch.")

    def detect(self, ctx: FileContext) -> DetectionResult:
        head = ctx.header
        ext = ctx.ext
        if head[:3] == FROSTBITE_DBOBJECT:
            kind = {"toc": "table of contents", "cat": "cas catalogue",
                    "sb": "superbundle"}.get(ext, "container")
            return self.result("Frostbite %s (.%s)" % (kind, ext or "bin"), 0.9,
                               Category.GAME_CONTAINER, engine="Frostbite", note=self.NOTE)
        if ext == "cas" and head[:4] == FROSTBITE_CAS_CHUNK:
            return self.result("Frostbite cas chunk store", 0.9, Category.GAME_CONTAINER,
                               engine="Frostbite", note=self.NOTE)
        if ext == "sb" and b"bundles\x00" in head[:64]:
            return self.result("Frostbite superbundle (.sb)", 0.85, Category.GAME_CONTAINER,
                               engine="Frostbite", note=self.NOTE)
        return self.nope()


class AnvilDetector(FormatDetector):
    """Assassin's Creed / Far Cry-adjacent Ubisoft Anvil archives."""

    name = "anvil_forge"
    priority = 88

    def detect(self, ctx: FileContext) -> DetectionResult:
        if ctx.header[:9] != b"scimitar\x00":
            return self.nope()
        version = struct.unpack_from("<I", ctx.header, 9)[0] if len(ctx.header) >= 13 else 0
        return self.result("Ubisoft Anvil forge archive", 0.95, Category.GAME_CONTAINER,
                           engine="Anvil", forge_version=version,
                           note="Anvil .forge container; a plugin must read the data-file "
                                "index and the per-entry compression (LZO/Oodle) to reach "
                                "the meshes inside")


class WwiseDetector(FormatDetector):
    name = "wwise"
    priority = 85

    def detect(self, ctx: FileContext) -> DetectionResult:
        head = ctx.header
        if head[:4] == b"AKPK":
            return self.result("Wwise audio package (.pck)", 0.95, Category.AUDIO,
                               engine="Wwise", note="audio only: no 3D assets inside")
        if head[:4] == b"BKHD":
            return self.result("Wwise SoundBank (.bnk)", 0.95, Category.AUDIO, engine="Wwise",
                               note="audio only: no 3D assets inside")
        return self.nope()


class MiscContainerDetector(FormatDetector):
    """Other well-known containers, named so the report is precise."""

    name = "misc_containers"
    priority = 83

    SIGNATURES = [
        (b"PSAR", "Sony PlayStation archive (PSARC)", Category.ARCHIVE, "PSARC",
         "needs a PSARC reader plugin"),
        (b"\x89HSP", "Havok packfile", Category.OTHER, "Havok", "physics data"),
        (b"RIFF", None, None, None, None),          # handled by the core detector
        (b"UEFN", "Unreal Editor for Fortnite package", Category.GAME_CONTAINER, "Unreal", ""),
        (b"DDBF", "Decima core file", Category.GAME_CONTAINER, "Decima",
         "Horizon/Death Stranding core archive; needs a Decima plugin"),
        (b"KRAK", "Oodle/Kraken compressed block", Category.COMPRESSED, None,
         "Oodle decompression is not available in this build"),
        (b"\x4b\x41\x52\x4b", "Oodle Kraken container", Category.COMPRESSED, None,
         "Oodle decompression is not available in this build"),
    ]

    def detect(self, ctx: FileContext) -> DetectionResult:
        head = ctx.header
        for magic, name, category, engine, note in self.SIGNATURES:
            if name is None or head[:len(magic)] != magic:
                continue
            meta = {}
            if engine:
                meta["engine"] = engine
            if note:
                meta["note"] = note
            return self.result(name, 0.9, category, **meta)
        return self.nope()


def register(api) -> None:
    api.add_detector(FrostbiteDetector())
    api.add_detector(AnvilDetector())
    api.add_detector(WwiseDetector())
    api.add_detector(MiscContainerDetector())
