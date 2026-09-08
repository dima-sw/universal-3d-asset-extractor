"""Unity plugin unit tests.

These use hand-built byte buffers, so they run anywhere - no game files needed.
(The parser was additionally validated against a shipped Unity 2018.4 title:
bundle -> SerializedFile -> 2457 meshes and 268 textures extracted.)
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from app.core.conversion.normalize import _mirror_quat, normalize
from app.core.types import Bone, Mesh, ModelAsset, Skeleton
from app.plugins.builtin.unity import textures as tex_tools
from app.plugins.builtin.unity.compressed import (CompressedMeshError, decode_compressed_mesh,
                                                  unpack_floats, unpack_ints)
from app.plugins.builtin.unity.common_strings import COMMON_STRINGS, resolve
from app.plugins.builtin.unity.reader import BinaryReader
from app.plugins.builtin.unity.serialized import TypeTreeNode, build_tree, read_value


# ---------------------------------------------------------------------------
# type tree
# ---------------------------------------------------------------------------
def node(level, type_, name, size=-1, meta=0):
    return TypeTreeNode(level=level, type=type_, name=name, byte_size=size, meta_flag=meta)


def test_build_tree_nests_by_level():
    nodes = [node(0, "Mesh", "Base"), node(1, "string", "m_Name"),
             node(1, "Vector3f", "m_Center"), node(2, "float", "x"), node(2, "float", "y"),
             node(1, "int", "m_Count")]
    root = build_tree(nodes)
    assert [c.name for c in root.children] == ["m_Name", "m_Center", "m_Count"]
    assert [c.name for c in root.children[1].children] == ["x", "y"]


def test_read_primitives_and_alignment():
    nodes = [node(0, "Base", "Base"), node(1, "UInt8", "flag", 1, 0x4000),
             node(1, "int", "value", 4)]
    root = build_tree(nodes)
    data = b"\x01" + b"\x00\x00\x00" + struct.pack("<i", 1234)
    value = read_value(root, BinaryReader(data))
    assert value == {"flag": 1, "value": 1234}


def test_read_string_and_vector():
    nodes = [node(0, "Base", "Base"),
             node(1, "string", "m_Name"), node(2, "Array", "Array"),
             node(3, "int", "size", 4), node(3, "char", "data", 1),
             node(1, "vector", "values"), node(2, "Array", "Array", meta=0x4000),
             node(3, "int", "size", 4), node(3, "float", "data", 4)]
    root = build_tree(nodes)
    payload = struct.pack("<i", 4) + b"cube" + struct.pack("<i", 2) + struct.pack("<ff", 1.5, 2.5)
    value = read_value(root, BinaryReader(payload))
    assert value["m_Name"] == "cube"
    assert value["values"] == pytest.approx([1.5, 2.5])


def test_typeless_data_is_read_as_a_byte_array():
    """Regression: TypelessData is array-shaped itself, with no Array child."""
    nodes = [node(0, "Base", "Base"),
             node(1, "TypelessData", "m_DataSize", -1, 0x4000),
             node(2, "int", "size", 4), node(2, "UInt8", "data", 1),
             node(1, "int", "trailing", 4)]
    root = build_tree(nodes)
    payload = struct.pack("<i", 3) + b"\xaa\xbb\xcc" + b"\x00" + struct.pack("<i", 7)
    value = read_value(root, BinaryReader(payload))
    assert value["m_DataSize"] == b"\xaa\xbb\xcc"
    assert value["trailing"] == 7


def test_common_string_table_offsets():
    assert COMMON_STRINGS[0] == "AABB"
    assert resolve(0x80000000, b"") == "AABB"
    assert resolve(3, b"ab\x00cd\x00") == "cd"        # file-local string buffer
    assert resolve(0, b"ab\x00cd\x00") == "ab"
    for expected in ("Array", "string", "float", "int", "Texture2D", "vector", "TypelessData"):
        assert expected in COMMON_STRINGS.values()


# ---------------------------------------------------------------------------
# packed bit vectors (compressed meshes)
# ---------------------------------------------------------------------------
def test_unpack_ints_lsb_first():
    vector = {"m_NumItems": 4, "m_BitSize": 4, "m_Data": b"\x21\x43"}
    assert list(unpack_ints(vector)) == [1, 2, 3, 4]


def test_unpack_floats_dequantises():
    vector = {"m_NumItems": 2, "m_BitSize": 4, "m_Data": b"\xf0", "m_Start": -1.0,
              "m_Range": 2.0}
    values = unpack_floats(vector)
    assert values[0] == pytest.approx(-1.0)
    assert values[1] == pytest.approx(1.0)


def test_unpack_rejects_truncated_data():
    with pytest.raises(CompressedMeshError):
        unpack_ints({"m_NumItems": 100, "m_BitSize": 8, "m_Data": b"\x00"})


def test_decode_compressed_mesh_positions_and_triangles():
    # two vertices at the ends of the range, one degenerate triangle
    compressed = {
        "m_Vertices": {"m_NumItems": 6, "m_BitSize": 8, "m_Start": 0.0, "m_Range": 1.0,
                       "m_Data": bytes([0, 0, 0, 255, 255, 255])},
        "m_Triangles": {"m_NumItems": 3, "m_BitSize": 8, "m_Data": bytes([0, 1, 0])},
    }
    out = decode_compressed_mesh(compressed)
    assert out["vertices"].shape == (2, 3)
    assert out["vertices"][0].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert out["vertices"][1].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert list(out["indices"]) == [0, 1, 0]


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------
def _bc1_block(c0: int, c1: int, indices: int) -> bytes:
    return struct.pack("<HHI", c0, c1, indices)


def test_bc1_decode_solid_colors():
    white, black = 0xFFFF, 0x0000
    image = tex_tools.decode(_bc1_block(white, black, 0x00000000), 4, 4, 10)
    assert image.shape == (4, 4, 4)
    assert (image[:, :, :3] == 255).all()
    image = tex_tools.decode(_bc1_block(white, black, 0x55555555), 4, 4, 10)
    assert (image[:, :, :3] == 0).all()


def test_bc3_decode_uses_alpha_block():
    alpha = bytes([255, 0, 0, 0, 0, 0, 0, 0])            # index 0 everywhere -> a0
    data = alpha + _bc1_block(0xFFFF, 0x0000, 0)
    image = tex_tools.decode(data, 4, 4, 12)
    assert (image[:, :, 3] == 255).all()
    assert (image[:, :, 0] == 255).all()


def test_rgba32_roundtrip_and_vertical_flip():
    pixels = bytearray()
    for row in range(2):
        for col in range(2):
            pixels += bytes([row * 100, col * 50, 7, 255])
    image = tex_tools.decode(bytes(pixels), 2, 2, 4)
    # Unity stores bottom-up, so the last source row must come out on top
    assert image[0, 0, 0] == 100
    assert image[1, 0, 0] == 0


def test_unsupported_format_names_itself():
    with pytest.raises(tex_tools.UnsupportedTextureFormat) as exc:
        tex_tools.decode(b"\x00" * 64, 4, 4, 25)          # BC7
    assert "BC7" in str(exc.value)


def test_dxt5nm_unswizzle_is_content_driven():
    normal = np.zeros((4, 4, 4), dtype=np.uint8)
    normal[:, :, 0] = 255                                 # R flat
    normal[:, :, 1] = np.arange(16).reshape(4, 4) * 10     # G varies
    normal[:, :, 2] = 128
    normal[:, :, 3] = np.arange(16).reshape(4, 4) * 5      # X parked in alpha
    fixed, changed = tex_tools.unswizzle_normal_map(normal)
    assert changed
    assert (fixed[:, :, 0] == normal[:, :, 3]).all()
    assert (fixed[:, :, 3] == 255).all()

    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[:, :, :3] = 255                                   # flat white mask
    mask[:, :, 3] = np.arange(16).reshape(4, 4) * 5
    _out, changed = tex_tools.unswizzle_normal_map(mask)
    assert changed is False


# ---------------------------------------------------------------------------
# handedness conversion (shared core code, exercised by the Unity path)
# ---------------------------------------------------------------------------
def test_mirror_quaternion_keeps_the_mirrored_axis():
    assert _mirror_quat((0.1, 0.2, 0.3, 0.9), 2) == pytest.approx((-0.1, -0.2, 0.3, 0.9))


def test_left_to_right_handed_flips_winding_and_z():
    mesh = Mesh(name="m", vertices=[(1, 2, 3), (4, 5, 6), (7, 8, 9)],
                normals=[(0, 0, 1)] * 3, indices=[0, 1, 2])
    skeleton = Skeleton(bones=[Bone("root", -1, translation=(1, 2, 3),
                                    rotation=(0.1, 0.2, 0.3, 0.9))])
    model = ModelAsset(name="m", meshes=[mesh], skeleton=skeleton,
                       coordinate_system="y_up_lh")
    normalize(model, up_axis="y", handedness="right", unit_scale=1.0)
    assert model.meshes[0].vertices[0] == pytest.approx((1, 2, -3))
    assert model.meshes[0].indices == [0, 2, 1]
    assert model.skeleton.bones[0].translation == pytest.approx((1, 2, -3))
    assert model.skeleton.bones[0].rotation == pytest.approx((-0.1, -0.2, 0.3, 0.9))
    assert model.metadata["original_coordinate_system"] == "y_up_lh"


def test_bind_pose_rebind_places_bones_in_mesh_space():
    from app.plugins.builtin.unity.objects import _rebind_from_bind_poses

    def inverse_bind(tx, ty, tz):
        m = np.eye(4)
        m[:3, 3] = (-tx, -ty, -tz)
        return [float(v) for v in m.T.reshape(-1)]        # column-major

    skeleton = Skeleton(bones=[Bone("root", -1), Bone("child", 0)])
    skeleton.bones[0].inverse_bind = inverse_bind(0, 0, 0)
    skeleton.bones[1].inverse_bind = inverse_bind(0, 2, 0)
    assert _rebind_from_bind_poses(skeleton) is True
    assert skeleton.bones[1].translation == pytest.approx((0.0, 2.0, 0.0))
    assert skeleton.bones[0].translation == pytest.approx((0.0, 0.0, 0.0))
