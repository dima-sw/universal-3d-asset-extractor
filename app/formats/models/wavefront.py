"""Wavefront OBJ (+ MTL) parser."""
from __future__ import annotations

from pathlib import Path

from app.core.detector.base import FileContext
from app.core.types import Material, Mesh, ModelAsset, TextureAsset
from app.formats.models.base import ModelParser

TEX_KEYS = {
    "map_kd": "diffuse", "map_ka": "ao", "map_ks": "specular", "map_bump": "normal",
    "bump": "normal", "norm": "normal", "map_ns": "roughness", "map_pr": "roughness",
    "map_pm": "metallic", "map_ke": "emission", "map_d": "mask",
}


class ObjParser(ModelParser):
    name = "obj"
    priority = 80
    formats = ("Wavefront OBJ", "OBJ (by extension)")

    def can_parse(self, ctx: FileContext, detection=None) -> bool:
        if detection is not None and detection.format_name in self.formats:
            return True
        return ctx.ext == "obj"

    def parse(self, ctx: FileContext) -> ModelAsset:
        model = ModelAsset(name=ctx.path.stem, source_format="OBJ")
        positions: list = []
        normals: list = []
        uvs: list = []
        mesh = Mesh(name=ctx.path.stem)
        vertex_map: dict = {}
        material_index = -1
        mtl_libs: list = []

        def flush(next_name: str = "") -> None:
            nonlocal mesh, vertex_map
            if mesh.vertices:
                model.meshes.append(mesh)
            mesh = Mesh(name=next_name or "mesh_%d" % (len(model.meshes) + 1))
            mesh.material = material_index
            vertex_map = {}

        with open(ctx.path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                tag = parts[0].lower()
                try:
                    if tag == "v":
                        positions.append(tuple(float(x) for x in parts[1:4]))
                    elif tag == "vn":
                        normals.append(tuple(float(x) for x in parts[1:4]))
                    elif tag == "vt":
                        uv = [float(x) for x in parts[1:3]]
                        while len(uv) < 2:
                            uv.append(0.0)
                        uvs.append((uv[0], uv[1]))
                    elif tag == "f":
                        idx = [self._vertex(p, positions, normals, uvs, mesh, vertex_map)
                               for p in parts[1:]]
                        for i in range(1, len(idx) - 1):     # fan-triangulate
                            mesh.indices.extend([idx[0], idx[i], idx[i + 1]])
                    elif tag in ("o", "g"):
                        flush(" ".join(parts[1:]) or "")
                    elif tag == "usemtl":
                        name = " ".join(parts[1:])
                        material_index = self._material_index(model, name)
                        if mesh.vertices:
                            flush()
                        mesh.material = material_index
                    elif tag == "mtllib":
                        mtl_libs.append(" ".join(parts[1:]))
                except (ValueError, IndexError):
                    continue                                  # malformed line: skip it

        flush()
        if not model.meshes and positions:
            m = Mesh(name=ctx.path.stem, vertices=positions)
            model.meshes.append(m)

        if not mtl_libs and model.materials:
            sibling = ctx.path.with_suffix(".mtl")     # usemtl without mtllib is common
            if sibling.exists():
                mtl_libs.append(sibling.name)
        for lib in mtl_libs:
            self._load_mtl(ctx.path.parent / lib, model)
        model.metadata["mtl_libs"] = mtl_libs
        return model

    # -- helpers -----------------------------------------------------------
    def _vertex(self, token: str, positions, normals, uvs, mesh: Mesh, cache: dict) -> int:
        if token in cache:
            return cache[token]
        bits = token.split("/")
        vi = int(bits[0])
        vi = vi - 1 if vi > 0 else len(positions) + vi
        ti = None
        ni = None
        if len(bits) > 1 and bits[1]:
            ti = int(bits[1])
            ti = ti - 1 if ti > 0 else len(uvs) + ti
        if len(bits) > 2 and bits[2]:
            ni = int(bits[2])
            ni = ni - 1 if ni > 0 else len(normals) + ni

        index = len(mesh.vertices)
        mesh.vertices.append(positions[vi] if 0 <= vi < len(positions) else (0.0, 0.0, 0.0))
        if ni is not None and 0 <= ni < len(normals):
            while len(mesh.normals) < index:
                mesh.normals.append((0.0, 0.0, 1.0))
            mesh.normals.append(normals[ni])
        if ti is not None and 0 <= ti < len(uvs):
            if not mesh.uv_channels:
                mesh.uv_channels.append([])
            ch = mesh.uv_channels[0]
            while len(ch) < index:
                ch.append((0.0, 0.0))
            ch.append(uvs[ti])
        cache[token] = index
        return index

    def _material_index(self, model: ModelAsset, name: str) -> int:
        for i, m in enumerate(model.materials):
            if m.name == name:
                return i
        model.materials.append(Material(name=name))
        return len(model.materials) - 1

    def _load_mtl(self, path: Path, model: ModelAsset) -> None:
        if not path.exists():
            return
        current = None
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return
        for line in text.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            key = parts[0].lower()
            if key == "newmtl":
                name = " ".join(parts[1:])
                idx = self._material_index(model, name)
                current = model.materials[idx]
            elif current is None:
                continue
            elif key == "kd" and len(parts) >= 4:
                current.base_color = (float(parts[1]), float(parts[2]), float(parts[3]), 1.0)
            elif key == "ns" and len(parts) >= 2:
                shininess = max(0.0, min(1000.0, float(parts[1])))
                current.roughness = max(0.0, 1.0 - (shininess / 1000.0) ** 0.5)
            elif key in TEX_KEYS:
                tex_path = (path.parent / parts[-1]).resolve()
                usage = TEX_KEYS[key]
                current.textures[usage] = TextureAsset(
                    name=Path(parts[-1]).stem, usage=usage,
                    path=str(tex_path) if tex_path.exists() else None,
                    source_format=Path(parts[-1]).suffix.lstrip(".").upper())
