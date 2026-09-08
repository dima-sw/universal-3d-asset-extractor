"""Magic-byte signature table, shared by the file detector and the embedded scanner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core.types import Category


@dataclass(frozen=True)
class Signature:
    magic: bytes
    format_name: str
    category: Category
    offset: int = 0
    confidence: float = 0.95
    ext: str = ""
    engine: Optional[str] = None
    embeddable: bool = False      # safe to search for inside arbitrary binaries


S = Signature

SIGNATURES: list = [
    # ---- archives / compression ----------------------------------------
    S(b"PK\x03\x04", "ZIP", Category.ARCHIVE, ext="zip", embeddable=True),
    S(b"PK\x05\x06", "ZIP (empty)", Category.ARCHIVE, ext="zip", confidence=0.7),
    S(b"7z\xbc\xaf\x27\x1c", "7-Zip", Category.ARCHIVE, ext="7z", embeddable=True),
    S(b"Rar!\x1a\x07\x00", "RAR4", Category.ARCHIVE, ext="rar", embeddable=True),
    S(b"Rar!\x1a\x07\x01\x00", "RAR5", Category.ARCHIVE, ext="rar", embeddable=True),
    S(b"\x1f\x8b\x08", "GZIP", Category.COMPRESSED, ext="gz", embeddable=True),
    S(b"BZh", "BZIP2", Category.COMPRESSED, ext="bz2", confidence=0.85),
    S(b"\xfd7zXZ\x00", "XZ", Category.COMPRESSED, ext="xz", embeddable=True),
    S(b"\x04\x22\x4d\x18", "LZ4 frame", Category.COMPRESSED, ext="lz4"),
    S(b"\x28\xb5\x2f\xfd", "Zstandard", Category.COMPRESSED, ext="zst", embeddable=True),
    S(b"MSCF", "Microsoft Cabinet", Category.ARCHIVE, ext="cab", embeddable=True),
    S(b"ustar", "TAR", Category.ARCHIVE, offset=257, ext="tar", confidence=0.9),

    # ---- disc images -----------------------------------------------------
    S(b"CD001", "ISO 9660", Category.DISC_IMAGE, offset=0x8001, ext="iso"),
    S(b"CD001", "ISO 9660 (2352 raw sectors)", Category.DISC_IMAGE, offset=0x9319, ext="bin",
      confidence=0.9),
    S(b"\x01CD001", "ISO 9660 PVD", Category.DISC_IMAGE, offset=0x8000, ext="iso"),
    S(b"MICROSOFT*XBOX*MEDIA", "Xbox XDVDFS", Category.DISC_IMAGE, offset=0x10000, ext="iso"),

    # ---- executables / containers ---------------------------------------
    S(b"MZ", "PE/DOS executable", Category.EXECUTABLE, ext="exe", confidence=0.6),
    S(b"\x7fELF", "ELF executable", Category.EXECUTABLE, confidence=0.9),
    S(b"PS-X EXE", "PlayStation 1 executable", Category.EXECUTABLE, ext="exe", engine="PS1"),

    # ---- textures --------------------------------------------------------
    S(b"\x89PNG\r\n\x1a\n", "PNG", Category.TEXTURE, ext="png", embeddable=True),
    S(b"\xff\xd8\xff", "JPEG", Category.TEXTURE, ext="jpg", embeddable=True),
    S(b"GIF87a", "GIF", Category.TEXTURE, ext="gif", embeddable=True),
    S(b"GIF89a", "GIF", Category.TEXTURE, ext="gif", embeddable=True),
    S(b"BM", "BMP", Category.TEXTURE, ext="bmp", confidence=0.55),
    S(b"DDS ", "DirectDraw Surface", Category.TEXTURE, ext="dds", embeddable=True),
    S(b"\xabKTX 11\xbb\r\n\x1a\n", "KTX", Category.TEXTURE, ext="ktx", embeddable=True),
    S(b"\xabKTX 20\xbb\r\n\x1a\n", "KTX2", Category.TEXTURE, ext="ktx2", embeddable=True),
    S(b"RIFF", "RIFF container (WEBP/WAV/AVI)", Category.OTHER, confidence=0.4),
    S(b"II*\x00", "TIFF (LE)", Category.TEXTURE, ext="tif", confidence=0.7),
    S(b"MM\x00*", "TIFF (BE)", Category.TEXTURE, ext="tif", confidence=0.7),
    S(b"\x00\x00\x00\x0cjP  ", "JPEG 2000", Category.TEXTURE, ext="jp2", confidence=0.8),
    S(b"PVR\x03", "PowerVR texture", Category.TEXTURE, ext="pvr", embeddable=True),
    S(b"Gidx", "Sony GIM texture", Category.TEXTURE, ext="gim", confidence=0.8),
    S(b"\x02\x00\x00\x00TIM", "Sony TIM texture", Category.TEXTURE, ext="tim", confidence=0.6),

    # ---- 3D models -------------------------------------------------------
    S(b"glTF", "GLB (glTF binary)", Category.MODEL, ext="glb", embeddable=True),
    S(b"Kaydara FBX Binary", "FBX (binary)", Category.MODEL, ext="fbx", embeddable=True),
    S(b"ply\n", "PLY", Category.MODEL, ext="ply", confidence=0.9),
    S(b"ply\r\n", "PLY", Category.MODEL, ext="ply", confidence=0.9),
    S(b"solid ", "STL (ascii)", Category.MODEL, ext="stl", confidence=0.6),
    S(b"IDP2", "Quake II MD2", Category.MODEL, ext="md2", engine="id Tech 2"),
    S(b"IDP3", "Quake III MD3", Category.MODEL, ext="md3", engine="id Tech 3"),
    S(b"IDST", "Valve/GoldSrc MDL", Category.MODEL, ext="mdl", engine="Source"),
    S(b"IDSQ", "Valve MDL sequence", Category.ANIMATION, ext="mdl", engine="Source"),
    S(b"VTF\x00", "Valve Texture Format", Category.TEXTURE, ext="vtf", engine="Source"),
    S(b"SMD", "Valve StudioMdl Data", Category.MODEL, ext="smd", engine="Source", confidence=0.5),

    # ---- engine containers ----------------------------------------------
    S(b"UnityFS", "Unity AssetBundle (FS)", Category.GAME_CONTAINER, ext="bundle",
      engine="Unity", embeddable=True),
    S(b"UnityWeb", "Unity AssetBundle (Web)", Category.GAME_CONTAINER, engine="Unity"),
    S(b"UnityRaw", "Unity AssetBundle (Raw)", Category.GAME_CONTAINER, engine="Unity"),
    S(b"UnityArchive", "Unity Archive", Category.GAME_CONTAINER, engine="Unity"),
    S(b"\xc1\x83\x2a\x9e", "Unreal package (UE3/UE4 .pak legacy)", Category.GAME_CONTAINER,
      engine="Unreal"),
    S(b"\x9e\x2a\x83\xc1", "Unreal package (UPK)", Category.GAME_CONTAINER, engine="Unreal"),
    S(b"GDPC", "Godot PCK", Category.GAME_CONTAINER, ext="pck", engine="Godot"),
    S(b"RSCF", "Godot resource", Category.GAME_CONTAINER, engine="Godot", confidence=0.7),
    S(b"CryTek", "CryEngine resource", Category.GAME_CONTAINER, engine="CryEngine",
      confidence=0.7),
    S(b"VPK\x34\x12", "Valve VPK", Category.GAME_CONTAINER, ext="vpk", engine="Source"),
    S(b"WAD3", "GoldSrc WAD", Category.GAME_CONTAINER, ext="wad", engine="GoldSrc"),
    S(b"IWAD", "Doom IWAD", Category.GAME_CONTAINER, ext="wad", engine="id Tech 1"),
    S(b"PWAD", "Doom PWAD", Category.GAME_CONTAINER, ext="wad", engine="id Tech 1"),
    S(b"PACK", "Quake PAK", Category.GAME_CONTAINER, ext="pak", engine="id Tech"),
    S(b"BSA\x00", "Bethesda BSA", Category.GAME_CONTAINER, ext="bsa", engine="Gamebryo"),
    S(b"BTDX", "Bethesda BA2", Category.GAME_CONTAINER, ext="ba2", engine="Creation"),
    S(b"XNB", "XNA content", Category.GAME_CONTAINER, ext="xnb", engine="XNA", confidence=0.7),

    # ---- audio -----------------------------------------------------------
    S(b"OggS", "OGG", Category.AUDIO, ext="ogg", embeddable=True),
    S(b"fLaC", "FLAC", Category.AUDIO, ext="flac", embeddable=True),
    S(b"ID3", "MP3 (ID3)", Category.AUDIO, ext="mp3", confidence=0.7),
    S(b"FSB5", "FMOD sound bank", Category.AUDIO, ext="fsb", confidence=0.9),
    S(b"RIFX", "RIFF (big endian)", Category.OTHER, confidence=0.4),
    S(b"BKHD", "Wwise SoundBank", Category.AUDIO, ext="bnk", confidence=0.9),
    S(b"pBAV", "Sony VAB body", Category.AUDIO, ext="vb", confidence=0.6),
    S(b"VAGp", "Sony VAG audio", Category.AUDIO, ext="vag", embeddable=True),

    # ---- console / retro -------------------------------------------------
    S(b"SCE\x00", "PSP/PS3 signed module", Category.EXECUTABLE, confidence=0.7),
    S(b"~PSP", "PSP PBP", Category.GAME_CONTAINER, ext="pbp", confidence=0.8),
    S(b"\x00PSF", "PSF container", Category.GAME_CONTAINER, confidence=0.7),
    S(b"NPUMDIMG", "PSP UMD image", Category.DISC_IMAGE, confidence=0.9),
]

# Signatures usable for embedded scanning inside unknown binaries.
EMBEDDABLE = [s for s in SIGNATURES if s.embeddable]

MAX_SIGNATURE_OFFSET = max(s.offset + len(s.magic) for s in SIGNATURES)


def match_at_head(header: bytes, read_at=None) -> Optional[Signature]:
    """Best signature match for a file header. `read_at(off, n)` for deep offsets."""
    best: Optional[Signature] = None
    for sig in SIGNATURES:
        end = sig.offset + len(sig.magic)
        if end <= len(header):
            chunk = header[sig.offset:end]
        elif read_at is not None:
            chunk = read_at(sig.offset, len(sig.magic))
        else:
            continue
        if chunk == sig.magic:
            if best is None or sig.confidence > best.confidence:
                best = sig
    return best
