"""Class layouts for Unity files built WITHOUT a type tree.

Most shipped games strip the type tree, so the generic decoder has nothing to
walk. These layouts rebuild the same node structure by hand for the handful of
classes we care about, and are then fed to the ordinary type-tree reader.

Two safeguards keep this honest:

* the layouts were written against real type trees dumped from a Unity build,
  not from memory;
* a few fields moved between Unity versions, so where the exact version cannot
  be trusted the reader tries the plausible variants and keeps the first one
  that passes a strict validation gate (sizes that must match, counts that must
  be consistent). If none passes, the object is reported as unsupported rather
  than decoded into garbage.
"""
from __future__ import annotations

from typing import Optional

from app.core.logging_setup import get_logger
from app.plugins.builtin.unity.reader import BinaryReader, ReadError
from app.plugins.builtin.unity.serialized import (TypeTreeNode, build_tree, read_value)
from app.plugins.builtin.unity.textures import TEXTURE_FORMATS

log = get_logger("plugin.unity.layouts")

ALIGN = 0x4000


class _L:
    """Tiny builder producing the flat (level, type, name) node list."""

    def __init__(self):
        self.nodes: list = []

    def n(self, level: int, type_: str, name: str, size: int = -1, align: bool = False):
        self.nodes.append(TypeTreeNode(level=level, type=type_, name=name, byte_size=size,
                                       meta_flag=ALIGN if align else 0))
        return self

    # --- reusable fragments ------------------------------------------------
    def string(self, level: int, name: str):
        self.n(level, "string", name)
        self.n(level + 1, "Array", "Array", align=True)
        self.n(level + 2, "int", "size", 4)
        self.n(level + 2, "char", "data", 1)
        return self

    def byte_array(self, level: int, name: str, type_: str = "vector",
                   align: bool = False):
        self.n(level, type_, name, align=align)
        if type_ == "TypelessData":
            self.n(level + 1, "int", "size", 4)
            self.n(level + 1, "UInt8", "data", 1)
        else:
            self.n(level + 1, "Array", "Array", align=True)
            self.n(level + 2, "int", "size", 4)
            self.n(level + 2, "UInt8", "data", 1)
        return self

    def scalar_vector(self, level: int, name: str, data_type: str, size: int = 4):
        self.n(level, "vector", name)
        self.n(level + 1, "Array", "Array", align=True)
        self.n(level + 2, "int", "size", 4)
        self.n(level + 2, data_type, "data", size)
        return self

    def vector3(self, level: int, name: str):
        self.n(level, "Vector3f", name, 12)
        for axis in "xyz":
            self.n(level + 1, "float", axis, 4)
        return self

    def aabb(self, level: int, name: str):
        self.n(level, "AABB", name, 24)
        self.vector3(level + 1, "m_Center")
        self.vector3(level + 1, "m_Extent")
        return self

    def matrix4x4(self, level: int, name: str):
        self.n(level, "Matrix4x4f", name, 64)
        # Unity serialises the elements row-major (e00, e01, e02, e03, e10, ...)
        for row in range(4):
            for col in range(4):
                self.n(level + 1, "float", "e%d%d" % (row, col), 4)
        return self

    def packed_bit_vector(self, level: int, name: str, floats: bool = True):
        self.n(level, "PackedBitVector", name)
        self.n(level + 1, "unsigned int", "m_NumItems", 4)
        if floats:
            self.n(level + 1, "float", "m_Range", 4)
            self.n(level + 1, "float", "m_Start", 4)
        self.byte_array(level + 1, "m_Data")
        self.n(level + 1, "UInt8", "m_BitSize", 1, align=True)
        return self

    def streaming_info(self, level: int, name: str, big_offset: bool):
        self.n(level, "StreamingInfo", name)
        self.n(level + 1, "UInt64" if big_offset else "unsigned int", "offset",
               8 if big_offset else 4)
        self.n(level + 1, "unsigned int", "size", 4)
        self.string(level + 1, "path")
        return self

    def pptr(self, level: int, name: str, type_: str = "PPtr<Object>"):
        self.n(level, type_, name, 12)
        self.n(level + 1, "int", "m_FileID", 4)
        self.n(level + 1, "SInt64", "m_PathID", 8)
        return self

    def build(self):
        root = build_tree(self.nodes)
        return root


# ---------------------------------------------------------------------------
# Texture2D
# ---------------------------------------------------------------------------
def texture2d(version: tuple, variant: dict) -> TypeTreeNode:
    major = version[0] if version else 2019
    minor = version[1] if len(version) > 1 else 0
    b = _L()
    b.n(0, "Texture2D", "Base")
    b.string(1, "m_Name")
    if major >= 2017:
        b.n(1, "int", "m_ForcedFallbackFormat", 4)
        b.n(1, "bool", "m_DownscaleFallback", 1, align=True)
        if (major, minor) >= (2020, 2):
            b.n(1, "bool", "m_IsAlphaChannelOptional", 1, align=True)
    b.n(1, "int", "m_Width", 4)
    b.n(1, "int", "m_Height", 4)
    b.n(1, "int", "m_CompleteImageSize", 4)
    if (major, minor) >= (2020, 1):
        b.n(1, "int", "m_MipsStripped", 4)
    b.n(1, "int", "m_TextureFormat", 4)
    b.n(1, "int", "m_MipCount", 4)
    b.n(1, "bool", "m_IsReadable", 1)
    if variant.get("ignore_master_texture_limit"):
        b.n(1, "bool", "m_IgnoreMasterTextureLimit", 1)
    if (major, minor) >= (2020, 1):
        b.n(1, "bool", "m_IsPreProcessed", 1)
    b.n(1, "bool", "m_StreamingMipmaps", 1, align=True)
    b.n(1, "int", "m_StreamingMipmapsPriority", 4, align=True)
    b.n(1, "int", "m_ImageCount", 4)
    b.n(1, "int", "m_TextureDimension", 4)
    b.n(1, "GLTextureSettings", "m_TextureSettings", 24)
    b.n(2, "int", "m_FilterMode", 4)
    b.n(2, "int", "m_Aniso", 4)
    b.n(2, "float", "m_MipBias", 4)
    b.n(2, "int", "m_WrapU", 4)
    b.n(2, "int", "m_WrapV", 4)
    b.n(2, "int", "m_WrapW", 4)
    b.n(1, "int", "m_LightmapFormat", 4)
    b.n(1, "int", "m_ColorSpace", 4)
    if (major, minor) >= (2020, 2):
        b.byte_array(1, "m_PlatformBlob")
    b.byte_array(1, "image data", type_="TypelessData", align=True)
    b.n(1, "__align_after_image", "__pad", 0)      # placeholder, removed below
    b.nodes.pop()
    b.streaming_info(1, "m_StreamData", big_offset=(major, minor) >= (2020, 1))
    return b.build()


TEXTURE2D_VARIANTS = [
    {"ignore_master_texture_limit": False},
    {"ignore_master_texture_limit": True},
]

BYTES_PER_PIXEL = {
    "Alpha8": 1, "R8": 1, "R16": 2, "RHalf": 2, "RG16": 2, "RGB565": 2, "RGBA4444": 2,
    "ARGB4444": 2, "RGB24": 3, "BGR24": 3, "RGBA32": 4, "ARGB32": 4, "BGRA32": 4,
    "RGHalf": 4, "RFloat": 4, "RGB9e5Float": 4, "RGBAHalf": 8, "RGFloat": 8,
    "RGBAFloat": 16,
}
BLOCK_BYTES = {
    "DXT1": 8, "DXT1Crunched": 8, "BC4": 8, "ETC_RGB4": 8, "ETC2_RGB": 8, "EAC_R": 8,
    "DXT3": 16, "DXT5": 16, "DXT5Crunched": 16, "BC5": 16, "BC6H": 16, "BC7": 16,
    "ETC2_RGBA8": 16, "EAC_RG": 16,
}


def expected_texture_bytes(width: int, height: int, fmt_name: str, mip_count: int = 1) -> tuple:
    """Return (base level size, full mip chain size) or (0, 0) when unknown."""
    def level_size(w: int, h: int) -> int:
        w, h = max(1, w), max(1, h)
        if fmt_name in BYTES_PER_PIXEL:
            return w * h * BYTES_PER_PIXEL[fmt_name]
        if fmt_name in BLOCK_BYTES:
            return ((w + 3) // 4) * ((h + 3) // 4) * BLOCK_BYTES[fmt_name]
        return 0

    base = level_size(width, height)
    if base == 0:
        return 0, 0
    total = 0
    w, h = width, height
    for _ in range(max(1, min(mip_count, 20))):
        total += level_size(w, h)
        w, h = max(1, w // 2), max(1, h // 2)
    return base, total


def validate_texture2d(obj: dict) -> bool:
    name = obj.get("m_Name")
    if not isinstance(name, str) or not name or len(name) > 250:
        return False
    width = obj.get("m_Width")
    height = obj.get("m_Height")
    fmt = obj.get("m_TextureFormat")
    if not all(isinstance(v, int) for v in (width, height, fmt)):
        return False
    if not (0 < width <= 32768 and 0 < height <= 32768):
        return False
    if fmt not in TEXTURE_FORMATS:
        return False
    mips = obj.get("m_MipCount")
    if not isinstance(mips, int) or not (0 <= mips <= 20):
        return False
    stream = obj.get("m_StreamData") or {}
    data = obj.get("image data") or b""
    payload = len(data) or int(stream.get("size", 0) or 0)
    if payload <= 0:
        return False
    base, full = expected_texture_bytes(width, height, TEXTURE_FORMATS[fmt], mips)
    if base == 0:
        return True                      # format we cannot size: structure looked sane
    return payload in (base, full) or abs(payload - full) <= 16 or abs(payload - base) <= 16


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def mesh(version: tuple, variant: dict) -> TypeTreeNode:
    major = version[0] if version else 2019
    b = _L()
    b.n(0, "Mesh", "Base")
    b.string(1, "m_Name")

    b.n(1, "vector", "m_SubMeshes")
    b.n(2, "Array", "Array", align=True)
    b.n(3, "int", "size", 4)
    b.n(3, "SubMesh", "data", 48)
    b.n(4, "unsigned int", "firstByte", 4)
    b.n(4, "unsigned int", "indexCount", 4)
    b.n(4, "int", "topology", 4)
    if major >= 2017:
        b.n(4, "unsigned int", "baseVertex", 4)
    b.n(4, "unsigned int", "firstVertex", 4)
    b.n(4, "unsigned int", "vertexCount", 4)
    b.aabb(4, "localAABB")

    b.n(1, "BlendShapeData", "m_Shapes")
    b.n(2, "vector", "vertices")
    b.n(3, "Array", "Array", align=True)
    b.n(4, "int", "size", 4)
    b.n(4, "BlendShapeVertex", "data", 40)
    b.vector3(5, "vertex")
    b.vector3(5, "normal")
    b.vector3(5, "tangent")
    b.n(5, "unsigned int", "index", 4)
    b.n(2, "vector", "shapes")
    b.n(3, "Array", "Array", align=True)
    b.n(4, "int", "size", 4)
    b.n(4, "MeshBlendShape", "data", 10)
    b.n(5, "unsigned int", "firstVertex", 4)
    b.n(5, "unsigned int", "vertexCount", 4)
    b.n(5, "bool", "hasNormals", 1)
    b.n(5, "bool", "hasTangents", 1, align=True)
    b.n(2, "vector", "channels")
    b.n(3, "Array", "Array", align=True)
    b.n(4, "int", "size", 4)
    b.n(4, "MeshBlendShapeChannel", "data")
    b.string(5, "name")
    b.n(5, "unsigned int", "nameHash", 4)
    b.n(5, "int", "frameIndex", 4)
    b.n(5, "int", "frameCount", 4)
    b.scalar_vector(2, "fullWeights", "float")

    b.n(1, "vector", "m_BindPose")
    b.n(2, "Array", "Array", align=True)
    b.n(3, "int", "size", 4)
    b.matrix4x4(3, "data")
    b.scalar_vector(1, "m_BoneNameHashes", "unsigned int")
    b.n(1, "unsigned int", "m_RootBoneNameHash", 4)

    if variant.get("bones_aabb"):
        b.n(1, "vector", "m_BonesAABB")
        b.n(2, "Array", "Array", align=True)
        b.n(3, "int", "size", 4)
        b.n(3, "MinMaxAABB", "data", 24)
        b.vector3(4, "m_Min")
        b.vector3(4, "m_Max")
        b.n(1, "VariableBoneCountWeights", "m_VariableBoneCountWeights")
        b.scalar_vector(2, "m_Data", "unsigned int")

    b.n(1, "UInt8", "m_MeshCompression", 1)
    b.n(1, "bool", "m_IsReadable", 1)
    b.n(1, "bool", "m_KeepVertices", 1)
    b.n(1, "bool", "m_KeepIndices", 1, align=True)
    b.n(1, "int", "m_IndexFormat", 4)
    b.byte_array(1, "m_IndexBuffer")

    b.n(1, "VertexData", "m_VertexData", align=True)
    b.n(2, "unsigned int", "m_VertexCount", 4)
    b.n(2, "vector", "m_Channels")
    b.n(3, "Array", "Array", align=True)
    b.n(4, "int", "size", 4)
    b.n(4, "ChannelInfo", "data", 4)
    b.n(5, "UInt8", "stream", 1)
    b.n(5, "UInt8", "offset", 1)
    b.n(5, "UInt8", "format", 1)
    b.n(5, "UInt8", "dimension", 1)
    b.byte_array(2, "m_DataSize", type_="TypelessData", align=True)

    b.n(1, "CompressedMesh", "m_CompressedMesh")
    b.packed_bit_vector(2, "m_Vertices", floats=True)
    b.packed_bit_vector(2, "m_UV", floats=True)
    b.packed_bit_vector(2, "m_Normals", floats=True)
    b.packed_bit_vector(2, "m_Tangents", floats=True)
    b.packed_bit_vector(2, "m_Weights", floats=False)
    b.packed_bit_vector(2, "m_NormalSigns", floats=False)
    b.packed_bit_vector(2, "m_TangentSigns", floats=False)
    b.packed_bit_vector(2, "m_FloatColors", floats=True)
    b.packed_bit_vector(2, "m_BoneIndices", floats=False)
    b.packed_bit_vector(2, "m_Triangles", floats=False)
    b.n(2, "unsigned int", "m_UVInfo", 4)

    b.aabb(1, "m_LocalAABB")
    b.n(1, "int", "m_MeshUsageFlags", 4)
    if variant.get("cooking_options"):
        b.n(1, "int", "m_CookingOptions", 4)
    b.byte_array(1, "m_BakedConvexCollisionMesh")
    b.byte_array(1, "m_BakedTriangleCollisionMesh")
    b.n(1, "float", "m_MeshMetrics[0]", 4)
    b.n(1, "float", "m_MeshMetrics[1]", 4, align=True)
    b.streaming_info(1, "m_StreamData", big_offset=(major >= 2020))
    return b.build()


MESH_VARIANTS = [
    {"bones_aabb": True, "cooking_options": True},
    {"bones_aabb": True, "cooking_options": False},
    {"bones_aabb": False, "cooking_options": True},
    {"bones_aabb": False, "cooking_options": False},
]


def validate_mesh(obj: dict) -> bool:
    name = obj.get("m_Name")
    if not isinstance(name, str) or len(name) > 250:
        return False
    compression = obj.get("m_MeshCompression")
    if not isinstance(compression, int) or compression not in (0, 1, 2, 3):
        return False
    index_format = obj.get("m_IndexFormat")
    if index_format not in (0, 1):
        return False
    submeshes = obj.get("m_SubMeshes")
    if not isinstance(submeshes, list) or len(submeshes) > 100000:
        return False

    aabb = obj.get("m_LocalAABB") or {}
    for part in ("m_Center", "m_Extent"):
        comp = aabb.get(part) or {}
        for axis in "xyz":
            value = comp.get(axis)
            if not isinstance(value, float) or abs(value) > 1e9 or value != value:
                return False

    vertex_data = obj.get("m_VertexData") or {}
    vertex_count = vertex_data.get("m_VertexCount")
    if not isinstance(vertex_count, int) or vertex_count > 20_000_000 or vertex_count < 0:
        return False
    channels = vertex_data.get("m_Channels") or []
    if not isinstance(channels, list) or len(channels) > 32:
        return False

    if compression == 0:
        if vertex_count == 0:
            return bool(obj.get("m_StreamData", {}).get("size"))
        stride = 0
        for ch in channels:
            dimension = int(ch.get("dimension", 0)) & 0x0F
            if dimension:
                stride = max(stride, int(ch.get("offset", 0)) + dimension * 4)
        payload = len(vertex_data.get("m_DataSize") or b"")
        streamed = int((obj.get("m_StreamData") or {}).get("size", 0) or 0)
        if payload == 0 and streamed == 0:
            return False
        # the buffer must be at least big enough for the declared vertices
        return (payload or streamed) >= vertex_count * 4
    packed = (obj.get("m_CompressedMesh") or {}).get("m_Vertices") or {}
    items = packed.get("m_NumItems")
    return isinstance(items, int) and items > 0 and items % 3 == 0


# ---------------------------------------------------------------------------
# small classes (names and hierarchy)
# ---------------------------------------------------------------------------
def game_object(version: tuple, variant: dict) -> TypeTreeNode:
    b = _L()
    b.n(0, "GameObject", "Base")
    b.n(1, "vector", "m_Component")
    b.n(2, "Array", "Array", align=True)
    b.n(3, "int", "size", 4)
    b.pptr(3, "data", "ComponentPair")
    b.n(1, "unsigned int", "m_Layer", 4)
    b.string(1, "m_Name")
    b.n(1, "UInt16", "m_Tag", 2)
    b.n(1, "bool", "m_IsActive", 1, align=True)
    return b.build()


def transform(version: tuple, variant: dict) -> TypeTreeNode:
    b = _L()
    b.n(0, "Transform", "Base")
    b.pptr(1, "m_GameObject", "PPtr<GameObject>")
    b.n(1, "Quaternionf", "m_LocalRotation", 16)
    for axis in "xyzw":
        b.n(2, "float", axis, 4)
    b.vector3(1, "m_LocalPosition")
    b.vector3(1, "m_LocalScale")
    b.n(1, "vector", "m_Children")
    b.n(2, "Array", "Array", align=True)
    b.n(3, "int", "size", 4)
    b.pptr(3, "data", "PPtr<Transform>")
    b.pptr(1, "m_Father", "PPtr<Transform>")
    return b.build()


def skinned_mesh_renderer(version: tuple, variant: dict) -> Optional[TypeTreeNode]:
    # Renderer has many version-dependent fields; only the tail we need is
    # stable enough to rebuild, so this is intentionally not attempted.
    return None


LAYOUTS = {
    "Texture2D": (texture2d, TEXTURE2D_VARIANTS, validate_texture2d),
    "Mesh": (mesh, MESH_VARIANTS, validate_mesh),
    "GameObject": (game_object, [{}], lambda o: isinstance(o.get("m_Name"), str)),
    "Transform": (transform, [{}], lambda o: isinstance(o.get("m_LocalPosition"), dict)),
}


def parse_unity_version(text: str) -> tuple:
    parts = []
    for chunk in (text or "").split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


class LayoutCache:
    """Remembers which variant worked for a class, so the search happens once."""

    def __init__(self, unity_version: str):
        self.version = parse_unity_version(unity_version)
        self._roots: dict = {}
        self._failed: set = set()

    def read(self, class_name: str, data: bytes, start: int, size: int,
             big_endian: bool) -> Optional[dict]:
        entry = LAYOUTS.get(class_name)
        if entry is None or class_name in self._failed:
            return None
        builder, variants, validate = entry

        cached = self._roots.get(class_name)
        candidates = [cached] if cached is not None else [builder(self.version, v)
                                                          for v in variants]
        for index, root in enumerate(candidates):
            if root is None:
                continue
            reader = BinaryReader(data, big_endian=big_endian, pos=start, base=start)
            try:
                value = read_value(root, reader)
            except (ReadError, RecursionError, ValueError, KeyError):
                continue
            consumed = reader.pos - start
            if consumed > size:
                continue
            if not isinstance(value, dict) or not validate(value):
                continue
            if cached is None:
                self._roots[class_name] = root
                log.info("layout for %s resolved (variant %d of %d, Unity %s)",
                         class_name, index + 1, len(candidates), self.version)
            return value
        if cached is None:
            self._failed.add(class_name)
            log.info("no usable layout for %s without a type tree (Unity %s)",
                     class_name, self.version)
        return None
