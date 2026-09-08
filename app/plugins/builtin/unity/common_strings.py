"""Unity's built-in type-tree string buffer.

Type-tree nodes reference strings either by an offset into the file's own string
buffer, or - when bit 31 is set - by an offset into this constant table that
ships inside the Unity runtime. The table is a null-separated blob; offsets are
the cumulative byte positions of each entry.
"""
from __future__ import annotations

COMMON_STRING_LIST = [
    "AABB", "AnimationClip", "AnimationCurve", "AnimationState", "Array", "Base", "BitField",
    "bitset", "bool", "char", "ColorRGBA", "Component", "data", "deque", "double",
    "dynamic_array", "FastPropertyName", "first", "float", "Font", "GameObject",
    "Generic Mono", "GradientNEW", "GUID", "GUIStyle", "int", "list", "long long", "map",
    "Matrix4x4f", "MdFour", "MonoBehaviour", "MonoScript", "m_ByteSize", "m_Curve",
    "m_EditorClassIdentifier", "m_EditorHideFlags", "m_Enabled", "m_ExtensionPtr",
    "m_GameObject", "m_Index", "m_IsArray", "m_IsStatic", "m_MetaFlag", "m_Name",
    "m_ObjectHideFlags", "m_PrefabInternal", "m_PrefabParentObject", "m_Script",
    "m_StaticEditorFlags", "m_Type", "m_Version", "Object", "pair", "PPtr<Component>",
    "PPtr<GameObject>", "PPtr<Material>", "PPtr<MonoBehaviour>", "PPtr<MonoScript>",
    "PPtr<Object>", "PPtr<Prefab>", "PPtr<Sprite>", "PPtr<TextAsset>", "PPtr<Texture>",
    "PPtr<Texture2D>", "PPtr<Transform>", "Prefab", "Quaternionf", "Rectf", "RectInt",
    "RectOffset", "second", "set", "short", "size", "SInt16", "SInt32", "SInt64", "SInt8",
    "staticvector", "string", "TextAsset", "TextMesh", "Texture", "Texture2D", "Transform",
    "TypelessData", "UInt16", "UInt32", "UInt64", "UInt8", "unsigned int",
    "unsigned long long", "unsigned short", "vector", "Vector2f", "Vector3f", "Vector4f",
    "m_ScriptingClassIdentifier", "Gradient", "Type*", "int2_storage", "int3_storage",
    "BoundsInt", "m_CorrespondingSourceObject", "m_PrefabInstance", "m_PrefabAsset",
    "FileSize", "Hash128",
]


def _build() -> dict:
    table = {}
    offset = 0
    for entry in COMMON_STRING_LIST:
        table[offset] = entry
        offset += len(entry) + 1
    return table


COMMON_STRINGS = _build()
COMMON_BLOB_SIZE = sum(len(s) + 1 for s in COMMON_STRING_LIST)


def resolve(offset: int, string_buffer: bytes) -> str:
    """Resolve a type-tree string reference to text."""
    if offset & 0x80000000:
        key = offset & 0x7FFFFFFF
        return COMMON_STRINGS.get(key, "unknown_common_%d" % key)
    end = string_buffer.find(b"\x00", offset)
    if offset < 0 or offset >= len(string_buffer) or end < 0:
        return "unknown_%d" % offset
    return string_buffer[offset:end].decode("utf-8", "replace")
