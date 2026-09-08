"""glTF 2.0 / GLB parser -> internal ModelAsset (mesh, skin, animations, textures)."""
from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

from app.core.detector.base import FileContext
from app.core.types import (AnimationClip, AnimationTrack, Bone, Keyframe, Material, Mesh,
                            ModelAsset, MorphTarget, Skeleton, TextureAsset)
from app.formats.models.base import ModelParser, ParseError

COMPONENT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2),
             5125: ("I", 4), 5126: ("f", 4)}
NUM_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
                  "MAT2": 4, "MAT3": 9, "MAT4": 16}

TEXTURE_SLOTS = {
    "baseColorTexture": "diffuse", "normalTexture": "normal",
    "metallicRoughnessTexture": "roughness", "occlusionTexture": "ao",
    "emissiveTexture": "emission",
}


class GltfParser(ModelParser):
    name = "gltf"
    priority = 90
    formats = ("GLB (glTF binary)", "glTF (JSON)", "GLTF (by extension)", "GLB (by extension)")

    def can_parse(self, ctx: FileContext, detection=None) -> bool:
        if detection is not None and detection.format_name in self.formats:
            return True
        return ctx.ext in ("gltf", "glb")

    # -- container ---------------------------------------------------------
    def _load(self, path: Path) -> tuple:
        raw = path.read_bytes()
        if raw[:4] == b"glTF":
            if len(raw) < 20:
                raise ParseError("truncated GLB")
            _magic, version, _length = struct.unpack_from("<III", raw, 0)
            if version != 2:
                raise ParseError("GLB version %d not supported (only glTF 2.0)" % version)
            offset = 12
            gltf = None
            bin_chunk = b""
            while offset + 8 <= len(raw):
                clen, ctype = struct.unpack_from("<II", raw, offset)
                body = raw[offset + 8: offset + 8 + clen]
                if ctype == 0x4E4F534A:
                    gltf = json.loads(body.decode("utf-8", "replace"))
                elif ctype == 0x004E4942:
                    bin_chunk = body
                offset += 8 + clen + ((4 - (clen % 4)) % 4 if clen % 4 else 0)
            if gltf is None:
                raise ParseError("GLB has no JSON chunk")
            return gltf, bin_chunk
        return json.loads(raw.decode("utf-8", "replace")), b""

    def _buffers(self, gltf: dict, base_dir: Path, glb_bin: bytes) -> list:
        out = []
        for buf in gltf.get("buffers", []):
            uri = buf.get("uri")
            if uri is None:
                out.append(glb_bin)
            elif uri.startswith("data:"):
                out.append(base64.b64decode(uri.split(",", 1)[1]))
            else:
                from urllib.parse import unquote
                p = base_dir / unquote(uri)
                out.append(p.read_bytes() if p.exists() else b"")
        return out

    def _accessor(self, gltf: dict, buffers: list, index: int) -> list:
        acc = gltf["accessors"][index]
        count = acc["count"]
        comps = NUM_COMPONENTS[acc["type"]]
        code, size = COMPONENT[acc["componentType"]]
        if "bufferView" not in acc:
            return [tuple([0] * comps) if comps > 1 else 0 for _ in range(count)]
        view = gltf["bufferViews"][acc["bufferView"]]
        data = buffers[view.get("buffer", 0)]
        base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride") or comps * size
        values = []
        for i in range(count):
            off = base + i * stride
            chunk = data[off: off + comps * size]
            if len(chunk) < comps * size:
                break
            vals = struct.unpack_from("<" + code * comps, chunk, 0)
            values.append(vals[0] if comps == 1 else tuple(vals))
        if acc.get("normalized") and code in ("B", "H", "b", "h"):
            maxv = {"B": 255.0, "H": 65535.0, "b": 127.0, "h": 32767.0}[code]
            values = [v / maxv if comps == 1 else tuple(x / maxv for x in v) for v in values]
        return values

    # -- main --------------------------------------------------------------
    def parse(self, ctx: FileContext) -> ModelAsset:
        gltf, glb_bin = self._load(ctx.path)
        buffers = self._buffers(gltf, ctx.path.parent, glb_bin)
        model = ModelAsset(name=ctx.path.stem, source_format="glTF 2.0")
        model.metadata["generator"] = gltf.get("asset", {}).get("generator")

        images = self._images(gltf, buffers, ctx.path.parent)
        model.materials = self._materials(gltf, images)

        for mesh_def in gltf.get("meshes", []):
            for prim_i, prim in enumerate(mesh_def.get("primitives", [])):
                if prim.get("mode", 4) != 4:
                    continue                       # only triangles for now
                mesh = Mesh(name=mesh_def.get("name") or "mesh_%d" % prim_i)
                attrs = prim.get("attributes", {})
                if "POSITION" not in attrs:
                    continue
                mesh.vertices = [tuple(v) for v in self._accessor(gltf, buffers, attrs["POSITION"])]
                if "NORMAL" in attrs:
                    mesh.normals = [tuple(v) for v in self._accessor(gltf, buffers, attrs["NORMAL"])]
                for uv_slot in ("TEXCOORD_0", "TEXCOORD_1", "TEXCOORD_2"):
                    if uv_slot in attrs:
                        mesh.uv_channels.append([tuple(v) for v in
                                                 self._accessor(gltf, buffers, attrs[uv_slot])])
                if "COLOR_0" in attrs:
                    mesh.colors = [tuple(v) for v in self._accessor(gltf, buffers, attrs["COLOR_0"])]
                if "JOINTS_0" in attrs and "WEIGHTS_0" in attrs:
                    mesh.joints = [tuple(int(x) for x in v)
                                   for v in self._accessor(gltf, buffers, attrs["JOINTS_0"])]
                    mesh.weights = [tuple(float(x) for x in v)
                                    for v in self._accessor(gltf, buffers, attrs["WEIGHTS_0"])]
                if "indices" in prim:
                    mesh.indices = [int(i) for i in self._accessor(gltf, buffers, prim["indices"])]
                else:
                    mesh.indices = list(range(len(mesh.vertices)))
                mesh.material = prim.get("material", -1)
                for t_i, target in enumerate(prim.get("targets", [])):
                    names = mesh_def.get("extras", {}).get("targetNames", [])
                    mt = MorphTarget(name=names[t_i] if t_i < len(names) else "morph_%d" % t_i)
                    if "POSITION" in target:
                        mt.positions = [tuple(v) for v in
                                        self._accessor(gltf, buffers, target["POSITION"])]
                    if "NORMAL" in target:
                        mt.normals = [tuple(v) for v in
                                      self._accessor(gltf, buffers, target["NORMAL"])]
                    mesh.morph_targets.append(mt)
                model.meshes.append(mesh)

        model.skeleton = self._skeleton(gltf, buffers)
        model.animations = self._animations(gltf, buffers)
        return model

    # -- pieces ------------------------------------------------------------
    def _images(self, gltf: dict, buffers: list, base_dir: Path) -> list:
        out = []
        for img in gltf.get("images", []):
            tex = TextureAsset(name=img.get("name") or "image_%d" % len(out))
            mime = img.get("mimeType", "")
            tex.source_format = mime.split("/")[-1].upper() if mime else ""
            uri = img.get("uri")
            if uri and uri.startswith("data:"):
                tex.data = base64.b64decode(uri.split(",", 1)[1])
                tex.source_format = tex.source_format or uri.split(";")[0].split("/")[-1].upper()
            elif uri:
                from urllib.parse import unquote
                p = base_dir / unquote(uri)
                tex.path = str(p) if p.exists() else None
                tex.name = Path(unquote(uri)).stem
                tex.source_format = tex.source_format or p.suffix.lstrip(".").upper()
            elif "bufferView" in img:
                view = gltf["bufferViews"][img["bufferView"]]
                data = buffers[view.get("buffer", 0)]
                off = view.get("byteOffset", 0)
                tex.data = data[off: off + view["byteLength"]]
            out.append(tex)
        return out

    def _materials(self, gltf: dict, images: list) -> list:
        textures = gltf.get("textures", [])
        materials = []
        for i, mat in enumerate(gltf.get("materials", [])):
            m = Material(name=mat.get("name") or "material_%d" % i)
            pbr = mat.get("pbrMetallicRoughness", {})
            m.base_color = tuple(pbr.get("baseColorFactor", [1, 1, 1, 1]))
            m.metallic = float(pbr.get("metallicFactor", 1.0))
            m.roughness = float(pbr.get("roughnessFactor", 1.0))
            m.double_sided = bool(mat.get("doubleSided", False))
            slots = dict(pbr)
            slots.update({k: v for k, v in mat.items() if k in TEXTURE_SLOTS})
            for key, usage in TEXTURE_SLOTS.items():
                ref = slots.get(key)
                if not isinstance(ref, dict):
                    continue
                tex_index = ref.get("index")
                if tex_index is None or tex_index >= len(textures):
                    continue
                src = textures[tex_index].get("source")
                if src is None or src >= len(images):
                    continue
                tex = images[src]
                tex.usage = usage
                if usage in ("normal",):
                    tex.color_space = "linear"
                m.textures[usage] = tex
            materials.append(m)
        return materials

    def _skeleton(self, gltf: dict, buffers: list):
        skins = gltf.get("skins", [])
        nodes = gltf.get("nodes", [])
        if not skins or not nodes:
            return None
        skin = skins[0]
        joints = skin.get("joints", [])
        if not joints:
            return None
        skeleton = Skeleton(name=skin.get("name") or "Skeleton")
        joint_pos = {n: i for i, n in enumerate(joints)}
        inv_binds = []
        if "inverseBindMatrices" in skin:
            inv_binds = self._accessor(gltf, buffers, skin["inverseBindMatrices"])
        parent_of = {}
        for i, node in enumerate(nodes):
            for child in node.get("children", []):
                parent_of[child] = i
        for i, node_index in enumerate(joints):
            node = nodes[node_index] if node_index < len(nodes) else {}
            parent_node = parent_of.get(node_index, -1)
            bone = Bone(
                name=node.get("name") or "bone_%d" % node_index,
                parent=joint_pos.get(parent_node, -1),
                translation=tuple(node.get("translation", (0.0, 0.0, 0.0))),
                rotation=tuple(node.get("rotation", (0.0, 0.0, 0.0, 1.0))),
                scale=tuple(node.get("scale", (1.0, 1.0, 1.0))),
                inverse_bind=list(inv_binds[i]) if i < len(inv_binds) else None,
            )
            skeleton.bones.append(bone)
        skeleton.metadata["source"] = "gltf skin"
        return skeleton

    def _animations(self, gltf: dict, buffers: list) -> list:
        nodes = gltf.get("nodes", [])
        clips = []
        for i, anim in enumerate(gltf.get("animations", [])):
            clip = AnimationClip(name=anim.get("name") or "animation_%d" % i)
            tracks: dict = {}
            for channel in anim.get("channels", []):
                sampler = anim["samplers"][channel["sampler"]]
                target = channel.get("target", {})
                node_index = target.get("node")
                if node_index is None:
                    continue
                bone_name = (nodes[node_index].get("name") if node_index < len(nodes) else None) \
                    or "node_%d" % node_index
                times = self._accessor(gltf, buffers, sampler["input"])
                values = self._accessor(gltf, buffers, sampler["output"])
                track = tracks.setdefault(bone_name, AnimationTrack(bone=bone_name))
                path = target.get("path")
                keys = [Keyframe(float(t), tuple(v) if isinstance(v, tuple) else (float(v),))
                        for t, v in zip(times, values)]
                if path == "translation":
                    track.positions = keys
                elif path == "rotation":
                    track.rotations = keys
                elif path == "scale":
                    track.scales = keys
                if times:
                    clip.duration = max(clip.duration, float(times[-1]))
            clip.tracks = list(tracks.values())
            clips.append(clip)
        return clips
