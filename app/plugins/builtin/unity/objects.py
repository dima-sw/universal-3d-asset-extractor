"""Turn decoded Unity objects into the extractor's internal representation.

Texture2D  -> TextureAsset (decoded to PNG when the format is supported)
Mesh       -> Mesh (positions, normals, uvs, colors, skin weights, submeshes)
SkinnedMeshRenderer + Transform graph -> Skeleton with real bone names

Everything that cannot be decoded is reported by name; nothing is guessed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from app.core.logging_setup import get_logger
from app.plugins.builtin.unity import textures as tex_tools
from app.plugins.builtin.unity.compressed import (CompressedMeshError, bounds_ok,
                                                   decode_compressed_mesh)
from app.plugins.builtin.unity.serialized import SerializedFile
from app.plugins.api import Bone, Material, Mesh, ModelAsset, Skeleton, TextureAsset

log = get_logger("plugin.unity.objects")

# vertex channel meaning, by Unity generation
CHANNELS_2018 = {0: "position", 1: "normal", 2: "tangent", 3: "color", 4: "uv0", 5: "uv1",
                 6: "uv2", 7: "uv3", 8: "uv4", 9: "uv5", 10: "uv6", 11: "uv7",
                 12: "blend_weight", 13: "blend_index"}
CHANNELS_5 = {0: "position", 1: "normal", 2: "color", 3: "uv0", 4: "uv1", 5: "uv2",
              6: "uv3", 7: "tangent"}

# VertexFormat enum, 2019+ and the older (pre-2019) numbering
FORMAT_2019 = {0: ("f4", 4), 1: ("f2", 2), 2: ("u1n", 1), 3: ("i1n", 1), 4: ("u2n", 2),
               5: ("i2n", 2), 6: ("u1", 1), 7: ("i1", 1), 8: ("u2", 2), 9: ("i2", 2),
               10: ("u4", 4), 11: ("i4", 4)}
FORMAT_LEGACY = {0: ("f4", 4), 1: ("f2", 2), 2: ("u1n", 1), 3: ("u1n", 1), 4: ("i1n", 1),
                 5: ("u2n", 2), 6: ("i2n", 2), 7: ("u1", 1), 8: ("i1", 1), 9: ("u2", 2),
                 10: ("i2", 2), 11: ("u4", 4), 12: ("i4", 4)}


class UnityDecodeError(Exception):
    pass


@dataclass
class ExtractedTexture:
    name: str
    asset: Optional[TextureAsset] = None
    reason: str = ""
    format_name: str = ""


@dataclass
class ExtractedModel:
    name: str
    model: Optional[ModelAsset] = None
    reason: str = ""
    warnings: list = field(default_factory=list)
    texture_ids: dict = field(default_factory=dict)   # usage -> texture path_id


def _unity_major(version: str) -> int:
    try:
        return int(version.split(".")[0])
    except (ValueError, IndexError):
        return 2019


class UnityAssetReader:
    """Reads one SerializedFile and yields internal assets."""

    def __init__(self, sf: SerializedFile, stream_loader: Optional[Callable] = None):
        self.sf = sf
        self.stream_loader = stream_loader          # (path, offset, size) -> bytes
        self.major = _unity_major(sf.unity_version)
        self._cache: dict = {}

    # ------------------------------------------------------------------
    def object_dict(self, path_id: int) -> Optional[dict]:
        if path_id in self._cache:
            return self._cache[path_id]
        for info in self.sf.objects:
            if info.path_id == path_id:
                value = self.sf.read_object(info)
                self._cache[path_id] = value
                return value
        return None

    def _read_all(self, class_name: str) -> list:
        out = []
        for info in self.sf.objects_of(class_name):
            value = self.sf.read_object(info)
            if value is not None:
                self._cache[info.path_id] = value
                out.append((info, value))
        return out

    # ------------------------------------------------------------------
    # textures
    # ------------------------------------------------------------------
    def textures(self) -> list:
        results = []
        for info, obj in self._read_all("Texture2D"):
            name = obj.get("m_Name") or "Texture_%d" % info.path_id
            try:
                results.append(self._texture(name, obj))
            except Exception as exc:
                results.append(ExtractedTexture(name=name, reason=str(exc)))
        return results

    def _texture(self, name: str, obj: dict) -> ExtractedTexture:
        width = int(obj.get("m_Width", 0))
        height = int(obj.get("m_Height", 0))
        fmt = int(obj.get("m_TextureFormat", 0))
        fmt_name = tex_tools.format_name(fmt)
        data = obj.get("image data")
        if isinstance(data, list):
            data = bytes(bytearray(data))
        if not data:
            stream = obj.get("m_StreamData") or {}
            path = stream.get("path")
            size = int(stream.get("size", 0) or 0)
            offset = int(stream.get("offset", 0) or 0)
            if path and size and self.stream_loader:
                data = self.stream_loader(path, offset, size)
            if not data:
                return ExtractedTexture(name=name, format_name=fmt_name,
                                        reason="pixel data lives in an external stream (%s) "
                                               "that was not available" % (path or "unknown"))
        try:
            image = tex_tools.decode(data, width, height, fmt)
        except tex_tools.UnsupportedTextureFormat as exc:
            return ExtractedTexture(name=name, format_name=fmt_name, reason=str(exc))

        image, unswizzled = tex_tools.unswizzle_normal_map(image)
        png = tex_tools.to_png(image)
        usage = "normal" if unswizzled else _guess_usage(name)
        asset = TextureAsset(name=name, width=width, height=height, channels=4,
                             has_alpha=bool(image[:, :, 3].min() < 255),
                             source_format="PNG" if png else fmt_name, data=png,
                             usage=usage, color_space="linear" if unswizzled else "srgb",
                             metadata={"unity_format": fmt_name,
                                       "dxt5nm_unswizzled": unswizzled})
        if png is None:
            return ExtractedTexture(name=name, format_name=fmt_name,
                                    reason="decoded but PNG encoding failed (Pillow missing?)")
        return ExtractedTexture(name=name, asset=asset, format_name=fmt_name)

    # ------------------------------------------------------------------
    # meshes
    # ------------------------------------------------------------------
    def models(self) -> list:
        skins = self._skin_index()
        results = []
        for info, obj in self._read_all("Mesh"):
            name = obj.get("m_Name") or "Mesh_%d" % info.path_id
            try:
                results.append(self._model(name, info.path_id, obj, skins.get(info.path_id)))
            except UnityDecodeError as exc:
                results.append(ExtractedModel(name=name, reason=str(exc)))
            except Exception as exc:
                results.append(ExtractedModel(name=name, reason="mesh decode failed: %s" % exc))
        return results

    def _skin_index(self) -> dict:
        """mesh path_id -> renderer object (bone list + material list)."""
        out = {}
        for _info, obj in self._read_all("SkinnedMeshRenderer"):
            path_id = (obj.get("m_Mesh") or {}).get("m_PathID")
            if path_id:
                out[path_id] = obj

        # static meshes: MeshFilter holds the mesh, MeshRenderer the materials,
        # both on the same GameObject
        renderers_by_go = {}
        for _info, obj in self._read_all("MeshRenderer"):
            go = (obj.get("m_GameObject") or {}).get("m_PathID")
            if go:
                renderers_by_go[go] = obj
        for _info, obj in self._read_all("MeshFilter"):
            path_id = (obj.get("m_Mesh") or {}).get("m_PathID")
            go = (obj.get("m_GameObject") or {}).get("m_PathID")
            if path_id and go in renderers_by_go and path_id not in out:
                out[path_id] = renderers_by_go[go]
        return out

    TEXTURE_SLOTS = {
        "_MainTex": "diffuse", "_BaseMap": "diffuse", "_BaseColorMap": "diffuse",
        "_BumpMap": "normal", "_NormalMap": "normal", "_DetailNormalMap": "normal",
        "_MetallicGlossMap": "metallic", "_SpecGlossMap": "specular",
        "_OcclusionMap": "ao", "_EmissionMap": "emission", "_MaskMap": "mask",
    }

    def material_textures(self, renderer: Optional[dict]) -> dict:
        """renderer -> {usage: texture path_id}, following Material.m_TexEnvs."""
        out: dict = {}
        if not renderer:
            return out
        for ptr in renderer.get("m_Materials") or []:
            material = self.object_dict((ptr or {}).get("m_PathID") or 0)
            if not isinstance(material, dict):
                continue
            saved = material.get("m_SavedProperties") or {}
            for entry in saved.get("m_TexEnvs") or []:
                if not isinstance(entry, dict):
                    continue
                prop = entry.get("first")
                if isinstance(prop, dict):
                    prop = prop.get("name") or prop.get("Name")
                usage = self.TEXTURE_SLOTS.get(prop)
                if not usage:
                    continue
                tex_ptr = ((entry.get("second") or {}).get("m_Texture") or {})
                path_id = tex_ptr.get("m_PathID")
                if path_id and usage not in out:
                    out[usage] = path_id
        return out

    def _model(self, name: str, path_id: int, obj: dict, renderer: Optional[dict]) -> ExtractedModel:
        warnings = []
        compressed = obj.get("m_CompressedMesh") or {}
        vertex_data = obj.get("m_VertexData") or {}
        vertex_count = int(vertex_data.get("m_VertexCount", 0) or 0)

        if vertex_count == 0 and compressed:
            return self._compressed_model(name, path_id, obj, compressed, renderer, warnings)
        if vertex_count == 0:
            raise UnityDecodeError("mesh has no vertex data in this file")

        raw = vertex_data.get("m_DataSize")
        if isinstance(raw, list):
            raw = bytes(bytearray(raw))
        if not raw:
            stream = obj.get("m_StreamData") or {}
            if stream.get("path") and self.stream_loader:
                raw = self.stream_loader(stream["path"], int(stream.get("offset", 0)),
                                         int(stream.get("size", 0)))
        if not raw:
            raise UnityDecodeError("vertex buffer is stored outside this file and was "
                                   "not available")

        channels = self._channels(vertex_data)
        arrays = self._read_channels(raw, channels, vertex_count)
        if "position" not in arrays:
            raise UnityDecodeError("mesh has no position channel")

        positions = arrays["position"][:, :3].astype(np.float32)
        if not np.isfinite(positions).all():
            raise UnityDecodeError("decoded positions are not finite - vertex layout for this "
                                   "Unity version is not supported")
        self._sanity_check_bounds(positions, obj, warnings)

        mesh = Mesh(name=name)
        mesh.vertices = [tuple(float(c) for c in v) for v in positions]
        if "normal" in arrays:
            mesh.normals = [tuple(float(c) for c in v[:3]) for v in arrays["normal"]]
        for slot in ("uv0", "uv1"):
            if slot in arrays:
                mesh.uv_channels.append([(float(v[0]), float(v[1])) for v in arrays[slot]])
        if "color" in arrays:
            mesh.colors = [tuple(float(c) for c in v) for v in arrays["color"]]

        indices = self._indices(obj, vertex_count)
        mesh.indices = indices

        skeleton = None
        if "blend_index" in arrays and "blend_weight" in arrays:
            mesh.joints = [tuple(int(x) for x in v[:4]) for v in arrays["blend_index"]]
            mesh.weights = [tuple(float(x) for x in v[:4]) for v in arrays["blend_weight"]]
        elif obj.get("m_Skin"):
            joints, weights = self._legacy_skin(obj["m_Skin"], vertex_count)
            mesh.joints, mesh.weights = joints, weights

        if mesh.is_skinned:
            skeleton = self._skeleton(name, obj, renderer, warnings)

        model = ModelAsset(name=name, meshes=[mesh], materials=[Material(name="%s_mat" % name)],
                           skeleton=skeleton, source_format="Unity Mesh")
        mesh.material = 0
        model.metadata.update({"unity_version": self.sf.unity_version,
                               "path_id": path_id,
                               "submeshes": len(obj.get("m_SubMeshes") or [])})
        return ExtractedModel(name=name, model=model, warnings=warnings,
                              texture_ids=self.material_textures(renderer))

    def _compressed_model(self, name: str, path_id: int, obj: dict, compressed: dict,
                          renderer: Optional[dict], warnings: list) -> ExtractedModel:
        """Meshes quantised into m_CompressedMesh (PackedBitVector channels)."""
        try:
            decoded = decode_compressed_mesh(compressed)
        except CompressedMeshError as exc:
            raise UnityDecodeError("compressed mesh: %s" % exc)

        positions = decoded["vertices"]
        if not np.isfinite(positions).all():
            raise UnityDecodeError("compressed mesh produced non-finite positions")
        if not bounds_ok(positions, obj.get("m_LocalAABB")):
            raise UnityDecodeError("compressed mesh decoded outside its stored bounding box; "
                                   "refusing to emit geometry that is probably wrong")

        mesh = Mesh(name=name)
        mesh.vertices = [tuple(float(c) for c in v) for v in positions]
        if "normals" in decoded:
            mesh.normals = [tuple(float(c) for c in v) for v in decoded["normals"]]
        for slot in ("uv0", "uv1"):
            if slot in decoded:
                mesh.uv_channels.append([(float(u), float(v)) for u, v in decoded[slot]])
        if "colors" in decoded:
            mesh.colors = [tuple(float(c) for c in v) for v in decoded["colors"]]
        mesh.indices = [int(i) for i in decoded.get("indices", [])]
        if not mesh.indices:
            mesh.indices = list(range(len(mesh.vertices)))
        if "joints" in decoded and "weights" in decoded:
            mesh.joints = [tuple(int(x) for x in row) for row in decoded["joints"]]
            mesh.weights = [tuple(float(x) for x in row) for row in decoded["weights"]]

        skeleton = self._skeleton(name, obj, renderer, warnings) if mesh.is_skinned else None
        mesh.material = 0
        model = ModelAsset(name=name, meshes=[mesh],
                           materials=[Material(name="%s_mat" % name)], skeleton=skeleton,
                           source_format="Unity Mesh (compressed)")
        model.metadata.update({"unity_version": self.sf.unity_version, "path_id": path_id,
                               "compression": int(obj.get("m_MeshCompression", 1) or 1),
                               "submeshes": len(obj.get("m_SubMeshes") or [])})
        return ExtractedModel(name=name, model=model, warnings=warnings,
                              texture_ids=self.material_textures(renderer))

    def _sanity_check_bounds(self, positions: np.ndarray, obj: dict, warnings: list) -> None:
        aabb = (obj.get("m_LocalAABB") or {})
        center = aabb.get("m_Center")
        extent = aabb.get("m_Extent")
        if not isinstance(center, dict) or not isinstance(extent, dict):
            return
        try:
            c = np.array([center["x"], center["y"], center["z"]], dtype=np.float64)
            e = np.array([extent["x"], extent["y"], extent["z"]], dtype=np.float64)
        except (KeyError, TypeError):
            return
        if not np.isfinite(c).all() or not np.isfinite(e).all():
            return
        lo, hi = positions.min(axis=0), positions.max(axis=0)
        tolerance = np.maximum(np.abs(e) * 4.0 + 1.0, 1.0)
        if np.any(lo < c - e - tolerance) or np.any(hi > c + e + tolerance):
            warnings.append("decoded vertices fall outside the stored bounding box; the vertex "
                            "layout for Unity %s may differ" % self.sf.unity_version)

    def _channels(self, vertex_data: dict) -> list:
        table = CHANNELS_2018 if self.major >= 2018 else CHANNELS_5
        formats = FORMAT_2019 if self.major >= 2019 else FORMAT_LEGACY
        out = []
        for index, channel in enumerate(vertex_data.get("m_Channels") or []):
            dimension = int(channel.get("dimension", 0)) & 0x0F
            if dimension == 0:
                continue
            fmt = int(channel.get("format", 0))
            kind, size = formats.get(fmt, ("f4", 4))
            out.append({"semantic": table.get(index, "channel_%d" % index),
                        "stream": int(channel.get("stream", 0)),
                        "offset": int(channel.get("offset", 0)),
                        "dimension": dimension, "kind": kind, "unit": size})
        return out

    def _read_channels(self, raw: bytes, channels: list, vertex_count: int) -> dict:
        buf = np.frombuffer(raw, dtype=np.uint8)
        streams: dict = {}
        for ch in channels:
            streams.setdefault(ch["stream"], []).append(ch)

        starts = {}
        cursor = 0
        for stream_index in sorted(streams):
            chans = streams[stream_index]
            stride = max(c["offset"] + c["unit"] * c["dimension"] for c in chans)
            stride = (stride + 3) & ~3
            starts[stream_index] = (cursor, stride)
            cursor = (cursor + stride * vertex_count + 15) & ~15

        out = {}
        for stream_index, chans in streams.items():
            start, stride = starts[stream_index]
            for ch in chans:
                width = ch["unit"] * ch["dimension"]
                base = start + ch["offset"]
                idx = base + np.arange(vertex_count, dtype=np.int64) * stride
                span = idx[:, None] + np.arange(width, dtype=np.int64)[None, :]
                if span.size and span.max() >= buf.size:
                    raise UnityDecodeError("vertex buffer is shorter than the channel layout "
                                           "requires (stream %d)" % stream_index)
                rows = buf[span]
                out[ch["semantic"]] = _convert(rows, ch["kind"], ch["dimension"])
        return out

    def _indices(self, obj: dict, vertex_count: int) -> list:
        raw = obj.get("m_IndexBuffer")
        if isinstance(raw, list):
            raw = bytes(bytearray(raw))
        if not raw:
            return list(range(vertex_count))
        wide = int(obj.get("m_IndexFormat", 0) or 0) == 1 or obj.get("m_Use16BitIndices") == 0
        dtype = np.uint32 if wide else np.uint16
        count = len(raw) // np.dtype(dtype).itemsize
        indices = np.frombuffer(raw[:count * np.dtype(dtype).itemsize], dtype=dtype)
        indices = indices[indices < vertex_count]
        indices = indices[:len(indices) // 3 * 3]
        return [int(i) for i in indices]

    def _legacy_skin(self, skin: list, vertex_count: int) -> tuple:
        joints, weights = [], []
        for entry in skin[:vertex_count]:
            w = entry.get("weight") or [entry.get("weight[%d]" % i, 0.0) for i in range(4)]
            b = entry.get("boneIndex") or [entry.get("boneIndex[%d]" % i, 0) for i in range(4)]
            weights.append(tuple(float(x) for x in list(w)[:4]))
            joints.append(tuple(int(x) for x in list(b)[:4]))
        while len(joints) < vertex_count:
            joints.append((0, 0, 0, 0))
            weights.append((1.0, 0.0, 0.0, 0.0))
        return joints, weights

    # ------------------------------------------------------------------
    # skeleton
    # ------------------------------------------------------------------
    def _skeleton(self, mesh_name: str, mesh_obj: dict, renderer: Optional[dict],
                  warnings: list) -> Optional[Skeleton]:
        bind_poses = mesh_obj.get("m_BindPose") or []
        hashes = mesh_obj.get("m_BoneNameHashes") or []
        bone_ptrs = (renderer or {}).get("m_Bones") or []

        transforms = {}
        if bone_ptrs:
            transforms = self._transform_map()

        count = max(len(bind_poses), len(bone_ptrs), len(hashes))
        if count == 0:
            return None

        skeleton = Skeleton(name="%s_Skeleton" % mesh_name)
        path_ids = []
        for i in range(count):
            name = None
            translation = (0.0, 0.0, 0.0)
            rotation = (0.0, 0.0, 0.0, 1.0)
            scale = (1.0, 1.0, 1.0)
            path_id = None
            if i < len(bone_ptrs):
                path_id = (bone_ptrs[i] or {}).get("m_PathID")
                node = transforms.get(path_id)
                if node:
                    name = node["name"]
                    translation = node["translation"]
                    rotation = node["rotation"]
                    scale = node["scale"]
            if name is None:
                if i < len(hashes):
                    name = "bone_%08X" % (int(hashes[i]) & 0xFFFFFFFF)
                else:
                    name = "bone_%d" % i
            path_ids.append(path_id)
            bone = Bone(name=name, parent=-1, translation=translation, rotation=rotation,
                        scale=scale)
            if i < len(bind_poses):
                bone.inverse_bind = _matrix_to_list(bind_poses[i])
            skeleton.bones.append(bone)

        # hierarchy from the Transform graph when we have it
        index_of = {pid: i for i, pid in enumerate(path_ids) if pid}
        linked = 0
        for i, pid in enumerate(path_ids):
            node = transforms.get(pid) if pid else None
            if not node:
                continue
            parent_index = index_of.get(node["father"])
            if parent_index is not None and parent_index != i:
                skeleton.bones[i].parent = parent_index
                linked += 1
        if bone_ptrs and not linked:
            warnings.append("bone hierarchy could not be resolved (Transforms live in another "
                            "file); bones are exported flat")
        if not bone_ptrs:
            warnings.append("no SkinnedMeshRenderer in this file: bone names come from hashes "
                            "and the hierarchy is flat")

        # Transform nodes carry scene-space transforms, but the mesh is in bind
        # space: re-derive each bone's local transform from its bind pose so the
        # rig lines up with the geometry.
        if _rebind_from_bind_poses(skeleton):
            skeleton.metadata["transforms_from"] = "bind pose"
        skeleton.metadata.update({"source": "unity", "resolved_names": bool(linked)})
        return skeleton

    def _transform_map(self) -> dict:
        if "__transforms" in self._cache:
            return self._cache["__transforms"]
        names = {}
        for info, obj in self._read_all("GameObject"):
            names[info.path_id] = obj.get("m_Name") or "GameObject_%d" % info.path_id
        out = {}
        for class_name in ("Transform", "RectTransform"):
            for info, obj in self._read_all(class_name):
                go = (obj.get("m_GameObject") or {}).get("m_PathID")
                father = (obj.get("m_Father") or {}).get("m_PathID")
                pos = obj.get("m_LocalPosition") or {}
                rot = obj.get("m_LocalRotation") or {}
                scale = obj.get("m_LocalScale") or {}
                out[info.path_id] = {
                    "name": names.get(go, "Transform_%d" % info.path_id),
                    "father": father,
                    "translation": (_f(pos, "x"), _f(pos, "y"), _f(pos, "z")),
                    "rotation": (_f(rot, "x"), _f(rot, "y"), _f(rot, "z"), _f(rot, "w", 1.0)),
                    "scale": (_f(scale, "x", 1.0), _f(scale, "y", 1.0), _f(scale, "z", 1.0)),
                }
        self._cache["__transforms"] = out
        return out


def _rebind_from_bind_poses(skeleton) -> bool:
    """Set each bone's local TRS from inv(inverseBindMatrix).

    The bind pose is the only transform that is guaranteed to be in the mesh's
    own space, so a rig rebuilt from it always lines up with the vertices.
    Returns False when the data is not usable and the Transform values stand.
    """
    bones = skeleton.bones
    if not bones or any(b.inverse_bind is None or len(b.inverse_bind) != 16 for b in bones):
        return False
    try:
        globals_ = []
        for bone in bones:
            ibm = np.asarray(bone.inverse_bind, dtype=np.float64).reshape(4, 4).T
            globals_.append(np.linalg.inv(ibm))
    except np.linalg.LinAlgError:
        return False

    for i, bone in enumerate(bones):
        parent = bone.parent
        if 0 <= parent < len(globals_):
            try:
                local = np.linalg.inv(globals_[parent]) @ globals_[i]
            except np.linalg.LinAlgError:
                return False
        else:
            local = globals_[i]
        translation, rotation, scale = _decompose(local)
        if translation is None:
            return False
        bone.translation = translation
        bone.rotation = rotation
        bone.scale = scale
    return True


def _decompose(matrix: np.ndarray) -> tuple:
    translation = tuple(float(v) for v in matrix[:3, 3])
    basis = matrix[:3, :3].copy()
    scale = np.linalg.norm(basis, axis=0)
    if not np.all(np.isfinite(scale)) or np.any(scale < 1e-8):
        return None, None, None
    rot = basis / scale
    if np.linalg.det(rot) < 0:                   # mirrored basis
        scale[0] = -scale[0]
        rot[:, 0] = -rot[:, 0]
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return translation, (float(x), float(y), float(z), float(w)), tuple(float(v) for v in scale)


def _f(d: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(d.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _matrix_to_list(matrix) -> Optional[list]:
    if not isinstance(matrix, dict):
        return None
    keys = ["e%d%d" % (row, col) for col in range(4) for row in range(4)]   # column-major
    try:
        return [float(matrix[k]) for k in keys]
    except (KeyError, TypeError, ValueError):
        return None


def _convert(rows: np.ndarray, kind: str, dimension: int) -> np.ndarray:
    """Bytes -> float/int array of shape (n, dimension)."""
    if kind == "f4":
        values = rows.view(np.float32).reshape(-1, dimension).astype(np.float32)
    elif kind == "f2":
        values = rows.view(np.float16).reshape(-1, dimension).astype(np.float32)
    elif kind in ("u1", "u1n"):
        values = rows.reshape(-1, dimension).astype(np.float32)
        if kind == "u1n":
            values = values / 255.0
    elif kind in ("i1", "i1n"):
        values = rows.view(np.int8).reshape(-1, dimension).astype(np.float32)
        if kind == "i1n":
            values = np.maximum(values / 127.0, -1.0)
    elif kind in ("u2", "u2n"):
        values = rows.view(np.uint16).reshape(-1, dimension).astype(np.float32)
        if kind == "u2n":
            values = values / 65535.0
    elif kind in ("i2", "i2n"):
        values = rows.view(np.int16).reshape(-1, dimension).astype(np.float32)
        if kind == "i2n":
            values = np.maximum(values / 32767.0, -1.0)
    elif kind == "u4":
        values = rows.view(np.uint32).reshape(-1, dimension).astype(np.float32)
    elif kind == "i4":
        values = rows.view(np.int32).reshape(-1, dimension).astype(np.float32)
    else:
        raise UnityDecodeError("unsupported vertex format %r" % kind)
    return values


def _guess_usage(name: str) -> str:
    from app.formats.textures.image import guess_usage
    return guess_usage(name)
