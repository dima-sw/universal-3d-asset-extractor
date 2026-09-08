"""Unity SerializedFile reader (.assets, .sharedAssets, levelN, bundle members).

Covers file versions 13-22 (Unity 5.0 through current). Older files and files
built without a type tree are reported as unsupported - the caller must never
present them as corrupt.

Object data is decoded generically by walking the type tree, so no per-class
layout table is needed for the classes we care about.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging_setup import get_logger
from app.plugins.builtin.unity.common_strings import resolve
from app.plugins.builtin.unity.reader import BinaryReader, ReadError

log = get_logger("plugin.unity.serialized")

MIN_VERSION = 13
MAX_VERSION = 23

CLASS_NAMES = {
    1: "GameObject", 2: "Component", 4: "Transform", 20: "Camera", 21: "Material",
    23: "MeshRenderer", 25: "Renderer", 28: "Texture2D", 33: "MeshFilter", 43: "Mesh",
    48: "Shader", 49: "TextAsset", 74: "AnimationClip", 82: "AudioSource", 83: "AudioClip",
    89: "Cubemap", 90: "Avatar", 91: "AnimatorController", 95: "Animator",
    114: "MonoBehaviour", 115: "MonoScript", 128: "Font", 137: "SkinnedMeshRenderer",
    142: "AssetBundle", 143: "CharacterController", 156: "TerrainData",
    187: "SpriteRenderer", 213: "Sprite", 224: "RectTransform", 1001: "PrefabInstance",
}

PRIMITIVES = {
    "SInt8": ("i8", 1), "UInt8": ("u8", 1), "char": ("u8", 1), "bool": ("boolean", 1),
    "SInt16": ("i16", 2), "short": ("i16", 2), "UInt16": ("u16", 2),
    "unsigned short": ("u16", 2), "SInt32": ("i32", 4), "int": ("i32", 4),
    "UInt32": ("u32", 4), "unsigned int": ("u32", 4), "Type*": ("u32", 4),
    "SInt64": ("i64", 8), "long long": ("i64", 8), "UInt64": ("u64", 8),
    "unsigned long long": ("u64", 8), "FileSize": ("u64", 8),
    "float": ("f32", 4), "double": ("f64", 8),
}

ALIGN_FLAG = 0x4000


class UnsupportedSerializedFile(Exception):
    """Recognised as Unity, but this build cannot read it. Never means corrupt."""


@dataclass
class TypeTreeNode:
    version: int = 0
    level: int = 0
    type_flags: int = 0
    type: str = ""
    name: str = ""
    byte_size: int = 0
    index: int = 0
    meta_flag: int = 0
    children: list = field(default_factory=list)

    @property
    def align(self) -> bool:
        return bool(self.meta_flag & ALIGN_FLAG)


@dataclass
class SerializedType:
    class_id: int = 0
    is_stripped: bool = False
    script_type_index: int = -1
    script_id: bytes = b""
    old_type_hash: bytes = b""
    nodes: list = field(default_factory=list)     # flat list
    root: Optional[TypeTreeNode] = None

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, "Class_%d" % self.class_id)


@dataclass
class ObjectInfo:
    path_id: int = 0
    byte_start: int = 0
    byte_size: int = 0
    type_index: int = 0
    class_id: int = 0

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, "Class_%d" % self.class_id)


class SerializedFile:
    def __init__(self, data: bytes, name: str = ""):
        self.data = data
        self.name = name
        self.version = 0
        self.unity_version = ""
        self.target_platform = 0
        self.enable_type_tree = True
        self.big_endian = False
        self.data_offset = 0
        self.types: list = []
        self.objects: list = []
        self.externals: list = []
        self._layouts = None
        self._parse()

    # ------------------------------------------------------------------
    def _parse(self) -> None:
        if len(self.data) < 32:
            raise UnsupportedSerializedFile("file too small to be a SerializedFile")
        head = BinaryReader(self.data, big_endian=True)
        metadata_size = head.u32()
        file_size = head.u32()
        self.version = head.u32()
        data_offset = head.u32()

        if not (MIN_VERSION <= self.version <= MAX_VERSION):
            raise UnsupportedSerializedFile(
                "SerializedFile version %d is outside the supported range %d-%d "
                "(Unity 5.0 .. current). Detected Unity asset, unsupported version / parser."
                % (self.version, MIN_VERSION, MAX_VERSION))

        endianness = head.u8()
        head.skip(3)
        self.big_endian = endianness != 0

        if self.version >= 22:
            metadata_size = head.u32()
            file_size = head.i64()
            data_offset = head.i64()
            head.i64()                                   # unknown / reserved

        self.data_offset = int(data_offset)
        if self.data_offset <= 0 or self.data_offset > len(self.data):
            raise UnsupportedSerializedFile("data offset %d outside the file" % self.data_offset)
        if file_size and abs(int(file_size) - len(self.data)) > 4096:
            log.debug("%s: header file size %d, actual %d", self.name, file_size, len(self.data))

        r = BinaryReader(self.data, big_endian=self.big_endian, pos=head.pos)
        try:
            self.unity_version = r.cstring()
            self.target_platform = r.i32()
            self.enable_type_tree = r.boolean()

            type_count = r.i32()
            if type_count < 0 or type_count > 100000:
                raise UnsupportedSerializedFile("implausible type count %d" % type_count)
            self.types = [self._read_type(r, is_ref_type=False) for _ in range(type_count)]

            object_count = r.i32()
            if object_count < 0 or object_count > 5_000_000:
                raise UnsupportedSerializedFile("implausible object count %d" % object_count)
            for _ in range(object_count):
                self.objects.append(self._read_object_info(r))

            script_count = r.i32()
            for _ in range(script_count):
                r.i32()                                   # file index
                r.align(4)
                r.i64()                                   # local path id

            external_count = r.i32()
            for _ in range(external_count):
                r.cstring()                               # temp empty
                r.read(16)                                # guid
                r.i32()                                   # type
                self.externals.append(r.cstring())        # path name
        except ReadError as exc:
            raise UnsupportedSerializedFile("metadata truncated: %s" % exc)

        if not self.enable_type_tree:
            log.info("%s: built without a type tree (release build)", self.name)

    def _read_type(self, r: BinaryReader, is_ref_type: bool) -> SerializedType:
        t = SerializedType()
        t.class_id = r.i32()
        if self.version >= 16:
            t.is_stripped = r.boolean()
        if self.version >= 17:
            t.script_type_index = r.i16()
        if (is_ref_type and t.script_type_index >= 0) or \
                (self.version < 16 and t.class_id < 0) or \
                (self.version >= 16 and t.class_id == 114):
            t.script_id = r.read(16)
        t.old_type_hash = r.read(16)

        if self.enable_type_tree:
            t.nodes = self._read_type_tree_blob(r)
            t.root = build_tree(t.nodes)
            if self.version >= 21:
                if is_ref_type:
                    r.cstring()
                    r.cstring()
                    r.cstring()
                else:
                    dep_count = r.i32()
                    r.skip(4 * max(0, dep_count))
        return t

    def _read_type_tree_blob(self, r: BinaryReader) -> list:
        node_count = r.i32()
        buffer_size = r.i32()
        if node_count < 0 or node_count > 1_000_000 or buffer_size < 0:
            raise UnsupportedSerializedFile("implausible type tree (%d nodes)" % node_count)
        raw_nodes = []
        for _ in range(node_count):
            version = r.u16()
            level = r.u8()
            type_flags = r.u8()
            type_offset = r.u32()
            name_offset = r.u32()
            byte_size = r.i32()
            index = r.i32()
            meta_flag = r.i32()
            if self.version >= 19:
                r.u64()                                   # ref type hash
            raw_nodes.append((version, level, type_flags, type_offset, name_offset,
                              byte_size, index, meta_flag))
        string_buffer = r.read(buffer_size)
        nodes = []
        for (version, level, type_flags, type_offset, name_offset, byte_size, index,
             meta_flag) in raw_nodes:
            nodes.append(TypeTreeNode(
                version=version, level=level, type_flags=type_flags,
                type=resolve(type_offset, string_buffer),
                name=resolve(name_offset, string_buffer),
                byte_size=byte_size, index=index, meta_flag=meta_flag))
        return nodes

    def _read_object_info(self, r: BinaryReader) -> ObjectInfo:
        info = ObjectInfo()
        r.align(4)
        info.path_id = r.i64()
        info.byte_start = (r.i64() if self.version >= 22 else r.u32()) + self.data_offset
        info.byte_size = r.u32()
        info.type_index = r.i32()
        if 0 <= info.type_index < len(self.types):
            info.class_id = self.types[info.type_index].class_id
        return info

    # ------------------------------------------------------------------
    def objects_of(self, class_name: str) -> list:
        return [o for o in self.objects if o.class_name == class_name]

    def read_object(self, info: ObjectInfo) -> Optional[dict]:
        """Decode one object. Uses the type tree when present, a known class
        layout otherwise. Returns None when neither can decode it."""
        end = info.byte_start + info.byte_size
        if info.byte_start < 0 or end > len(self.data):
            log.debug("%s: object %d out of range", self.name, info.path_id)
            return None

        if not self.enable_type_tree:
            value = self._read_without_type_tree(info)
            if isinstance(value, dict):
                value["__class"] = info.class_name
                value["__path_id"] = info.path_id
            return value

        if not (0 <= info.type_index < len(self.types)):
            return None
        root = self.types[info.type_index].root
        if root is None:
            return None
        r = BinaryReader(self.data, big_endian=self.big_endian, pos=info.byte_start,
                         base=info.byte_start)
        try:
            value = read_value(root, r)
        except (ReadError, RecursionError, struct.error, ValueError) as exc:
            log.debug("%s: object %d (%s) unreadable: %s", self.name, info.path_id,
                      info.class_name, exc)
            return None
        if isinstance(value, dict):
            value["__class"] = info.class_name
            value["__path_id"] = info.path_id
        return value

    def _read_without_type_tree(self, info: ObjectInfo) -> Optional[dict]:
        """Fallback for stripped builds: decode with a known class layout."""
        from app.plugins.builtin.unity.layouts import LayoutCache
        if self._layouts is None:
            self._layouts = LayoutCache(self.unity_version)
        return self._layouts.read(info.class_name, self.data, info.byte_start,
                                  info.byte_size, self.big_endian)

    def summary(self) -> dict:
        counts = {}
        for obj in self.objects:
            counts[obj.class_name] = counts.get(obj.class_name, 0) + 1
        return {"unity_version": self.unity_version, "format_version": self.version,
                "type_tree": self.enable_type_tree, "objects": len(self.objects),
                "classes": counts, "externals": self.externals}


def build_tree(nodes: list) -> Optional[TypeTreeNode]:
    """Turn the flat, level-encoded node list into a tree."""
    if not nodes:
        return None
    root = nodes[0]
    root.children = []
    stack = [root]
    for node in nodes[1:]:
        node.children = []
        while len(stack) > node.level and len(stack) > 1:
            stack.pop()
        if node.level == 0:
            continue
        parent = stack[-1] if stack else root
        parent.children.append(node)
        stack.append(node)
    return root


def read_value(node: TypeTreeNode, r: BinaryReader):
    """Generic type-tree driven decode."""
    type_name = node.type

    if type_name in PRIMITIVES:
        method, _size = PRIMITIVES[type_name]
        value = getattr(r, method)()
        if node.align:
            r.align(4)
        return value

    if type_name == "string":
        value = r.sized_string()
        if node.align:
            r.align(4)
        return value

    if node.children and node.children[0].type == "Array":
        return _read_array(node.children[0], r, node.align)

    # TypelessData is array-shaped itself: [int size, UInt8 data], no Array child
    if type_name == "TypelessData" and len(node.children) >= 2:
        return _read_array(node, r, node.align)

    if type_name == "Array":
        return _read_array(node, r, node.align)

    value = {}
    for child in node.children:
        value[child.name] = read_value(child, r)
    if node.align:
        r.align(4)
    return value


def _read_array(array_node: TypeTreeNode, r: BinaryReader, parent_align: bool):
    if len(array_node.children) < 2:
        raise ReadError("malformed array node %r" % array_node.name)
    size_node, data_node = array_node.children[0], array_node.children[1]
    size = read_value(size_node, r)
    if not isinstance(size, int) or size < 0:
        raise ReadError("bad array size %r" % size)

    if data_node.type in ("UInt8", "SInt8", "char"):
        remaining = r.remaining()
        if size > remaining:
            raise ReadError("byte array of %d exceeds %d remaining" % (size, remaining))
        raw = r.read(size)
        if array_node.align or parent_align:
            r.align(4)
        return raw

    if size > 50_000_000:
        raise ReadError("array too large (%d)" % size)
    items = [read_value(data_node, r) for _ in range(size)]
    if array_node.align or parent_align:
        r.align(4)
    return items
