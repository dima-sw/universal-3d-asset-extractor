"""glTF 2.0 / GLB exporter: mesh + materials + textures + skeleton + animations."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.logging_setup import get_logger
from app.core.types import ModelAsset, Skeleton
from app.exporters.base import Exporter, ExportResult

log = get_logger("export.gltf")

FLOAT = 5126
USHORT = 5123
UINT = 5125
UBYTE = 5121
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _trs_matrix(bone) -> np.ndarray:
    m = _quat_to_matrix(bone.rotation)
    s = np.diag([bone.scale[0], bone.scale[1], bone.scale[2], 1.0])
    m = m @ s
    m[0, 3], m[1, 3], m[2, 3] = bone.translation
    return m


def inverse_bind_matrices(skeleton: Skeleton) -> list:
    """Compute inverse bind matrices from the local TRS chain when absent."""
    world: list = []
    for i, bone in enumerate(skeleton.bones):
        local = _trs_matrix(bone)
        if 0 <= bone.parent < i:
            world.append(world[bone.parent] @ local)
        elif 0 <= bone.parent < len(world):
            world.append(world[bone.parent] @ local)
        else:
            world.append(local)
    out = []
    for i, bone in enumerate(skeleton.bones):
        if bone.inverse_bind and len(bone.inverse_bind) == 16:
            out.append([float(v) for v in bone.inverse_bind])
            continue
        try:
            inv = np.linalg.inv(world[i])
        except np.linalg.LinAlgError:
            inv = np.eye(4)
        out.append([float(v) for v in inv.T.reshape(-1)])   # column-major
    return out


class _BufferBuilder:
    def __init__(self):
        self.data = bytearray()
        self.views: list = []
        self.accessors: list = []

    def _align(self, boundary: int = 4) -> None:
        pad = (-len(self.data)) % boundary
        if pad:
            self.data.extend(b"\x00" * pad)

    def add_view(self, payload: bytes, target: Optional[int] = None, stride: Optional[int] = None) -> int:
        self._align()
        offset = len(self.data)
        self.data.extend(payload)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target:
            view["target"] = target
        if stride:
            view["byteStride"] = stride
        self.views.append(view)
        return len(self.views) - 1

    def add_accessor(self, array: np.ndarray, comp_type: int, type_: str,
                     target: Optional[int] = None, normalized: bool = False) -> int:
        payload = array.tobytes()
        view = self.add_view(payload, target)
        acc = {"bufferView": view, "componentType": comp_type,
               "count": int(array.shape[0]), "type": type_}
        if normalized:
            acc["normalized"] = True
        if type_ != "SCALAR" and array.ndim == 2:
            acc["min"] = [float(v) for v in array.min(axis=0)]
            acc["max"] = [float(v) for v in array.max(axis=0)]
        elif type_ == "SCALAR" and array.size:
            acc["min"] = [float(array.min())]
            acc["max"] = [float(array.max())]
        self.accessors.append(acc)
        return len(self.accessors) - 1


class GltfExporter(Exporter):
    """Primary exporter. GLB by default: one file carrying everything."""

    name = "glb"
    format = "glb"
    extension = ".glb"
    supports_skeleton = True
    supports_animation = True
    priority = 100
    binary = True

    def export(self, model: ModelAsset, out_dir: Path, name: Optional[str] = None,
               options: Optional[dict] = None) -> ExportResult:
        options = options or {}
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = name or model.name or "model"
        res = ExportResult(format=self.format)
        try:
            gltf, buffer = self._build(model, res)
        except Exception as exc:
            res.error = "glTF build failed: %s" % exc
            log.warning("%s: %s", stem, exc)
            return res

        if self.binary:
            path = out_dir / (stem + ".glb")
            path.write_bytes(self._pack_glb(gltf, buffer))
        else:
            path = out_dir / (stem + ".gltf")
            bin_path = out_dir / (stem + ".bin")
            bin_path.write_bytes(bytes(buffer))
            gltf["buffers"] = [{"uri": bin_path.name, "byteLength": len(buffer)}]
            path.write_text(json.dumps(gltf, indent=2), encoding="utf-8")
            res.files.append(bin_path)
        res.files.insert(0, path)
        res.ok = True
        return res

    # -- construction ------------------------------------------------------
    def _build(self, model: ModelAsset, res: ExportResult) -> tuple:
        b = _BufferBuilder()
        gltf = {
            "asset": {"version": "2.0", "generator": "Universal 3D Asset Extractor"},
            "scene": 0,
        }
        images: list = []
        samplers_img: list = []
        textures: list = []
        materials: list = []
        image_cache: dict = {}

        for mat in model.materials:
            pbr = {
                "baseColorFactor": [float(v) for v in mat.base_color],
                "metallicFactor": float(mat.metallic),
                "roughnessFactor": float(mat.roughness),
            }
            gmat = {"name": mat.name, "pbrMetallicRoughness": pbr,
                    "doubleSided": bool(mat.double_sided)}
            for usage, tex in mat.textures.items():
                data = self._texture_bytes(tex)
                if not data:
                    res.warnings.append("texture %s missing bytes" % tex.name)
                    continue
                key = id(tex) if tex.data else (tex.path or tex.name)
                if key not in image_cache:
                    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
                    if mime == "image/jpeg" and data[:3] != b"\xff\xd8\xff":
                        converted = self._to_png(tex)
                        if converted is None:
                            res.warnings.append("texture %s: %s not embeddable in glTF"
                                                % (tex.name, tex.source_format))
                            continue
                        data, mime = converted, "image/png"
                    view = b.add_view(data)
                    images.append({"bufferView": view, "mimeType": mime, "name": tex.name})
                    textures.append({"source": len(images) - 1})
                    image_cache[key] = len(textures) - 1
                tex_index = image_cache[key]
                if usage == "diffuse":
                    pbr["baseColorTexture"] = {"index": tex_index}
                elif usage == "normal":
                    gmat["normalTexture"] = {"index": tex_index}
                elif usage in ("roughness", "metallic"):
                    pbr["metallicRoughnessTexture"] = {"index": tex_index}
                elif usage == "ao":
                    gmat["occlusionTexture"] = {"index": tex_index}
                elif usage == "emission":
                    gmat["emissiveTexture"] = {"index": tex_index}
                    gmat["emissiveFactor"] = [1.0, 1.0, 1.0]
            materials.append(gmat)

        meshes: list = []
        for mesh in model.meshes:
            if not mesh.vertices:
                continue
            attrs = {}
            positions = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
            attrs["POSITION"] = b.add_accessor(positions, FLOAT, "VEC3", ARRAY_BUFFER)
            if mesh.normals and len(mesh.normals) == len(mesh.vertices):
                normals = np.asarray(mesh.normals, dtype=np.float32).reshape(-1, 3)
                attrs["NORMAL"] = b.add_accessor(normals, FLOAT, "VEC3", ARRAY_BUFFER)
            for i, channel in enumerate(mesh.uv_channels[:2]):
                if len(channel) != len(mesh.vertices):
                    channel = list(channel) + [(0.0, 0.0)] * (len(mesh.vertices) - len(channel))
                uvs = np.asarray(channel[:len(mesh.vertices)], dtype=np.float32).reshape(-1, 2)
                attrs["TEXCOORD_%d" % i] = b.add_accessor(uvs, FLOAT, "VEC2", ARRAY_BUFFER)
            if mesh.is_skinned and len(mesh.joints) == len(mesh.vertices):
                joints = np.asarray(mesh.joints, dtype=np.uint16).reshape(-1, 4)
                weights = np.asarray(mesh.weights, dtype=np.float32).reshape(-1, 4)
                sums = weights.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1.0
                weights = weights / sums
                attrs["JOINTS_0"] = b.add_accessor(joints, USHORT, "VEC4", ARRAY_BUFFER)
                attrs["WEIGHTS_0"] = b.add_accessor(weights, FLOAT, "VEC4", ARRAY_BUFFER)

            primitive = {"attributes": attrs, "mode": 4}
            if mesh.indices:
                max_index = max(mesh.indices)
                dtype, comp = (np.uint16, USHORT) if max_index < 65535 else (np.uint32, UINT)
                idx = np.asarray(mesh.indices, dtype=dtype)
                primitive["indices"] = b.add_accessor(idx, comp, "SCALAR", ELEMENT_ARRAY_BUFFER)
            if 0 <= mesh.material < len(materials):
                primitive["material"] = mesh.material
            targets = []
            for morph in mesh.morph_targets:
                if len(morph.positions) != len(mesh.vertices):
                    res.warnings.append("morph target %s vertex count mismatch" % morph.name)
                    continue
                pos = np.asarray(morph.positions, dtype=np.float32).reshape(-1, 3)
                targets.append({"POSITION": b.add_accessor(pos, FLOAT, "VEC3", ARRAY_BUFFER)})
            if targets:
                primitive["targets"] = targets
            mesh_def = {"primitives": [primitive], "name": mesh.name}
            if targets:
                mesh_def["extras"] = {"targetNames": [m.name for m in mesh.morph_targets]}
            meshes.append(mesh_def)

        nodes: list = []
        scene_nodes: list = []
        skins: list = []
        joint_node_index: dict = {}

        if model.skeleton and model.skeleton.bones:
            base = len(nodes)
            for i, bone in enumerate(model.skeleton.bones):
                node = {"name": bone.name}
                if tuple(bone.translation) != (0.0, 0.0, 0.0):
                    node["translation"] = [float(v) for v in bone.translation]
                if tuple(bone.rotation) != (0.0, 0.0, 0.0, 1.0):
                    node["rotation"] = [float(v) for v in bone.rotation]
                if tuple(bone.scale) != (1.0, 1.0, 1.0):
                    node["scale"] = [float(v) for v in bone.scale]
                nodes.append(node)
                joint_node_index[i] = base + i
            for i, bone in enumerate(model.skeleton.bones):
                if 0 <= bone.parent < len(model.skeleton.bones):
                    parent = nodes[joint_node_index[bone.parent]]
                    parent.setdefault("children", []).append(joint_node_index[i])
                else:
                    scene_nodes.append(joint_node_index[i])
            ibm = np.asarray(inverse_bind_matrices(model.skeleton), dtype=np.float32)
            skins.append({
                "name": model.skeleton.name,
                "joints": [joint_node_index[i] for i in range(len(model.skeleton.bones))],
                "inverseBindMatrices": b.add_accessor(ibm.reshape(-1, 16), FLOAT, "MAT4"),
            })

        for i, mesh_def in enumerate(meshes):
            node = {"mesh": i, "name": mesh_def.get("name", "mesh_%d" % i)}
            if skins and model.meshes[i].is_skinned:
                node["skin"] = 0
            nodes.append(node)
            scene_nodes.append(len(nodes) - 1)

        animations = self._animations(model, b, joint_node_index, res)

        gltf["scenes"] = [{"nodes": scene_nodes or [0]}]
        if nodes:
            gltf["nodes"] = nodes
        if meshes:
            gltf["meshes"] = meshes
        if materials:
            gltf["materials"] = materials
        if images:
            gltf["images"] = images
            gltf["textures"] = textures
            gltf["samplers"] = samplers_img or [{"magFilter": 9729, "minFilter": 9987,
                                                 "wrapS": 10497, "wrapT": 10497}]
            for t in gltf["textures"]:
                t.setdefault("sampler", 0)
        if skins:
            gltf["skins"] = skins
        if animations:
            gltf["animations"] = animations
        if b.views:
            gltf["bufferViews"] = b.views
        if b.accessors:
            gltf["accessors"] = b.accessors
        gltf["buffers"] = [{"byteLength": len(b.data)}]
        if model.metadata:
            gltf["extras"] = {"source": model.metadata.get("source"),
                              "source_format": model.source_format}
        return gltf, b.data

    def _animations(self, model: ModelAsset, b: _BufferBuilder, joint_node_index: dict,
                    res: ExportResult) -> list:
        if not model.animations or not model.skeleton:
            return []
        name_to_node = {}
        for i, bone in enumerate(model.skeleton.bones):
            if i in joint_node_index:
                name_to_node[bone.name] = joint_node_index[i]
        out = []
        for clip in model.animations:
            channels: list = []
            samplers: list = []
            for track in clip.tracks:
                node = name_to_node.get(track.bone)
                if node is None:
                    continue
                for path, keys, size in (("translation", track.positions, 3),
                                         ("rotation", track.rotations, 4),
                                         ("scale", track.scales, 3)):
                    if not keys:
                        continue
                    times = np.asarray([k.time for k in keys], dtype=np.float32)
                    values = np.asarray([list(k.value)[:size] + [0.0] * (size - len(k.value))
                                         for k in keys], dtype=np.float32).reshape(-1, size)
                    samplers.append({
                        "input": b.add_accessor(times, FLOAT, "SCALAR"),
                        "output": b.add_accessor(values, FLOAT, "VEC%d" % size),
                        "interpolation": "LINEAR",
                    })
                    channels.append({"sampler": len(samplers) - 1,
                                     "target": {"node": node, "path": path}})
            if channels:
                out.append({"name": clip.name, "channels": channels, "samplers": samplers})
            else:
                res.warnings.append("animation %s has no track matching the skeleton" % clip.name)
        return out

    # -- helpers -----------------------------------------------------------
    def _texture_bytes(self, tex) -> Optional[bytes]:
        if tex.data:
            return tex.data
        if tex.path:
            p = Path(tex.path)
            if p.exists():
                try:
                    return p.read_bytes()
                except OSError:
                    return None
        return None

    def _to_png(self, tex) -> Optional[bytes]:
        try:
            import io

            from PIL import Image
            src = io.BytesIO(tex.data) if tex.data else tex.path
            with Image.open(src) as im:
                buf = io.BytesIO()
                im.convert("RGBA" if "A" in im.mode else "RGB").save(buf, format="PNG")
                return buf.getvalue()
        except Exception:
            return None

    def _pack_glb(self, gltf: dict, buffer: bytearray) -> bytes:
        json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        json_pad = (-len(json_bytes)) % 4
        json_bytes += b" " * json_pad
        bin_bytes = bytes(buffer)
        bin_pad = (-len(bin_bytes)) % 4
        bin_bytes += b"\x00" * bin_pad
        total = 12 + 8 + len(json_bytes) + (8 + len(bin_bytes) if bin_bytes else 0)
        out = bytearray()
        out += struct.pack("<III", 0x46546C67, 2, total)
        out += struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
        if bin_bytes:
            out += struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes
        return bytes(out)


class GltfJsonExporter(GltfExporter):
    name = "gltf"
    format = "gltf"
    extension = ".gltf"
    priority = 90
    binary = False
