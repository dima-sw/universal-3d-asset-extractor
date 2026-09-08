"""Coordinate-system and unit normalisation.

Originals are preserved in metadata so nothing is lost by converting.
"""
from __future__ import annotations

from app.core.types import ModelAsset

# source system -> transform needed to reach y-up right-handed
KNOWN_SYSTEMS = {
    "y_up_rh": (0, 1, 2, 1.0, 1.0, 1.0),
    "y_up_lh": (0, 1, 2, 1.0, 1.0, -1.0),
    "z_up_rh": (0, 2, 1, 1.0, 1.0, -1.0),
    "z_up_lh": (0, 2, 1, 1.0, 1.0, 1.0),
}


def _axis_swap(vec, mapping) -> tuple:
    ix, iy, iz, sx, sy, sz = mapping
    v = list(vec) + [0.0] * (3 - len(vec))
    return (v[ix] * sx, v[iy] * sy, v[iz] * sz)


def _mirror_quat(q, axis: int) -> tuple:
    """Mirroring one coordinate axis keeps that axis' quaternion component and
    flips the other two (q and -q are the same rotation, so sign is free)."""
    values = list(q) + [0.0] * (4 - len(q))
    out = [-values[0], -values[1], -values[2], values[3]]
    out[axis] = values[axis]
    return tuple(float(v) for v in out)


def normalize(model: ModelAsset, up_axis: str = "y", handedness: str = "right",
              unit_scale: float = 1.0) -> ModelAsset:
    """Convert a model in place to the requested convention."""
    source = model.coordinate_system or "y_up_rh"
    target = "%s_up_%s" % (up_axis.lower(), "rh" if handedness.lower().startswith("r") else "lh")
    scale = (unit_scale or 1.0) / (model.unit_scale or 1.0)

    model.metadata.setdefault("original_coordinate_system", source)
    model.metadata.setdefault("original_unit_scale", model.unit_scale)

    if source == target and abs(scale - 1.0) < 1e-9:
        return model

    src_map = KNOWN_SYSTEMS.get(source, KNOWN_SYSTEMS["y_up_rh"])
    dst_map = KNOWN_SYSTEMS.get(target, KNOWN_SYSTEMS["y_up_rh"])
    # compose: source -> canonical -> target (both are involutive axis swaps here)
    mapping = src_map if src_map != dst_map else KNOWN_SYSTEMS["y_up_rh"]
    mirrored = (mapping[3] * mapping[4] * mapping[5]) < 0

    for mesh in model.meshes:
        mesh.vertices = [tuple(c * scale for c in _axis_swap(v, mapping)) for v in mesh.vertices]
        mesh.normals = [_axis_swap(n, mapping) for n in mesh.normals]
        for morph in mesh.morph_targets:
            morph.positions = [tuple(c * scale for c in _axis_swap(p, mapping))
                               for p in morph.positions]
            morph.normals = [_axis_swap(n, mapping) for n in morph.normals]
        if mirrored and mesh.indices:
            # a handedness flip mirrors the geometry: restore the facing by
            # reversing each triangle's winding
            idx = mesh.indices
            for i in range(0, len(idx) - 2, 3):
                idx[i + 1], idx[i + 2] = idx[i + 2], idx[i + 1]

    mirror_axis = next((i for i in range(3) if mapping[3 + i] < 0), None) if mirrored else None

    if model.skeleton:
        for bone in model.skeleton.bones:
            bone.translation = tuple(c * scale for c in _axis_swap(bone.translation, mapping))
            if mirror_axis is not None:
                bone.rotation = _mirror_quat(bone.rotation, mirror_axis)
            bone.inverse_bind = None          # recomputed by the exporter after the swap

    for clip in model.animations:
        for track in clip.tracks:
            for key in track.positions:
                key.value = tuple(c * scale for c in _axis_swap(key.value, mapping))
            if mirror_axis is not None:
                for key in track.rotations:
                    key.value = _mirror_quat(key.value, mirror_axis)

    model.coordinate_system = target
    model.unit_scale = unit_scale or 1.0
    model.metadata["normalized"] = True
    return model
