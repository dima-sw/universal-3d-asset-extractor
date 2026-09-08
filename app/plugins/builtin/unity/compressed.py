"""Unity compressed mesh support (m_CompressedMesh / PackedBitVector).

Unity quantises vertex data into bit-packed vectors: values of `m_BitSize` bits
packed LSB-first, plus `m_Start`/`m_Range` for float channels. Normals and
tangents keep only two components; the third is rebuilt from the unit length and
a sign bit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class CompressedMeshError(Exception):
    pass


def _as_bytes(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, list):
        return bytes(bytearray(value))
    return b""


def unpack_ints(vector: dict) -> np.ndarray:
    """PackedBitVector -> uint32 array of m_NumItems values."""
    if not isinstance(vector, dict):
        return np.zeros(0, dtype=np.uint32)
    count = int(vector.get("m_NumItems", 0) or 0)
    bit_size = int(vector.get("m_BitSize", 0) or 0)
    data = _as_bytes(vector.get("m_Data"))
    if count == 0 or bit_size == 0:
        return np.zeros(0, dtype=np.uint32)
    if bit_size > 32:
        raise CompressedMeshError("unsupported bit size %d" % bit_size)
    needed_bits = count * bit_size
    if len(data) * 8 < needed_bits:
        raise CompressedMeshError("packed vector truncated (%d bits, need %d)"
                                  % (len(data) * 8, needed_bits))
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    bits = bits[:needed_bits].reshape(count, bit_size).astype(np.uint64)
    weights = (np.uint64(1) << np.arange(bit_size, dtype=np.uint64))
    return (bits * weights).sum(axis=1).astype(np.uint64)


def unpack_floats(vector: dict) -> np.ndarray:
    """PackedBitVector -> float32 array, de-quantised with start/range."""
    values = unpack_ints(vector)
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    bit_size = int(vector.get("m_BitSize", 0) or 0)
    scale = float((1 << bit_size) - 1) or 1.0
    start = float(vector.get("m_Start", 0.0) or 0.0)
    rng = float(vector.get("m_Range", 0.0) or 0.0)
    return (start + rng * (values.astype(np.float64) / scale)).astype(np.float32)


def _third_component(xy: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Rebuild z for a unit vector stored as (x, y) + sign bit."""
    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)
    z_sqr = 1.0 - x * x - y * y
    z = np.sqrt(np.clip(z_sqr, 0.0, None))
    if signs.size >= z.size:
        z = np.where(signs[:z.size] > 0, z, -z)
    return z.astype(np.float32)


def decode_compressed_mesh(compressed: dict) -> dict:
    """Return {'vertices', 'normals', 'tangents', 'uv0', 'uv1', 'indices',
    'joints', 'weights'} using whatever the mesh actually stores."""
    out: dict = {}

    vertices = unpack_floats(compressed.get("m_Vertices") or {})
    if vertices.size % 3:
        raise CompressedMeshError("packed vertex count %d is not a multiple of 3"
                                  % vertices.size)
    if vertices.size == 0:
        raise CompressedMeshError("compressed mesh has no vertex data")
    positions = vertices.reshape(-1, 3)
    vertex_count = positions.shape[0]
    out["vertices"] = positions

    normal_data = unpack_floats(compressed.get("m_Normals") or {})
    normal_signs = unpack_ints(compressed.get("m_NormalSigns") or {})
    if normal_data.size >= vertex_count * 2:
        xy = normal_data[:vertex_count * 2].reshape(-1, 2)
        z = _third_component(xy, normal_signs)
        out["normals"] = np.column_stack([xy, z]).astype(np.float32)

    tangent_data = unpack_floats(compressed.get("m_Tangents") or {})
    tangent_signs = unpack_ints(compressed.get("m_TangentSigns") or {})
    if tangent_data.size >= vertex_count * 2:
        xy = tangent_data[:vertex_count * 2].reshape(-1, 2)
        z = _third_component(xy, tangent_signs[::2] if tangent_signs.size else tangent_signs)
        out["tangents"] = np.column_stack([xy, z]).astype(np.float32)

    uv = unpack_floats(compressed.get("m_UV") or {})
    if uv.size >= vertex_count * 2:
        out["uv0"] = uv[:vertex_count * 2].reshape(-1, 2)
        remaining = uv[vertex_count * 2:]
        if remaining.size >= vertex_count * 2:
            out["uv1"] = remaining[:vertex_count * 2].reshape(-1, 2)

    colors = unpack_floats(compressed.get("m_FloatColors") or {})
    if colors.size >= vertex_count * 4:
        out["colors"] = colors[:vertex_count * 4].reshape(-1, 4)

    triangles = unpack_ints(compressed.get("m_Triangles") or {})
    if triangles.size:
        indices = triangles.astype(np.int64)
        indices = indices[indices < vertex_count]
        out["indices"] = indices[:len(indices) // 3 * 3]

    weights = unpack_ints(compressed.get("m_Weights") or {})
    bone_indices = unpack_ints(compressed.get("m_BoneIndices") or {})
    if weights.size and bone_indices.size:
        joints, bone_weights = _unpack_skin(weights, bone_indices, vertex_count)
        out["joints"] = joints
        out["weights"] = bone_weights
    return out


def _unpack_skin(weights: np.ndarray, bone_indices: np.ndarray, vertex_count: int) -> tuple:
    """Unity stores weights quantised to 31 with a run-length-ish layout:
    influences accumulate per vertex until they sum to 31 or four are used."""
    joints = np.zeros((vertex_count, 4), dtype=np.int32)
    values = np.zeros((vertex_count, 4), dtype=np.float32)
    vertex = 0
    slot = 0
    total = 0
    bone_pos = 0
    for i in range(weights.size):
        if vertex >= vertex_count:
            break
        w = int(weights[i])
        if bone_pos < bone_indices.size:
            joints[vertex, slot] = int(bone_indices[bone_pos])
        values[vertex, slot] = w / 31.0
        total += w
        bone_pos += 1
        slot += 1
        if slot == 3:
            # the fourth influence is implied by what is left of the budget
            remainder = max(0, 31 - total)
            if remainder and bone_pos < bone_indices.size:
                joints[vertex, 3] = int(bone_indices[bone_pos])
                values[vertex, 3] = remainder / 31.0
                bone_pos += 1
            vertex += 1
            slot = 0
            total = 0
        elif total >= 31:
            vertex += 1
            slot = 0
            total = 0
    sums = values.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    values = values / sums
    return joints, values


def bounds_ok(positions: np.ndarray, aabb: Optional[dict]) -> bool:
    """Cheap correctness gate: decoded points must sit inside the stored AABB."""
    if not isinstance(aabb, dict):
        return True
    center = aabb.get("m_Center") or {}
    extent = aabb.get("m_Extent") or {}
    try:
        c = np.array([center["x"], center["y"], center["z"]], dtype=np.float64)
        e = np.array([extent["x"], extent["y"], extent["z"]], dtype=np.float64)
    except (KeyError, TypeError):
        return True
    if not (np.isfinite(c).all() and np.isfinite(e).all()):
        return True
    tolerance = np.maximum(np.abs(e) * 0.05, 1e-3)
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    return bool(np.all(lo >= c - e - tolerance) and np.all(hi <= c + e + tolerance))
