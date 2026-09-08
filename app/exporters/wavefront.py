"""OBJ exporter. Secondary format: no skeleton, no animation, no morph targets."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.core.types import ModelAsset
from app.exporters.base import Exporter, ExportResult
from app.formats.textures import image as image_tools


class ObjExporter(Exporter):
    name = "obj"
    format = "obj"
    extension = ".obj"
    supports_skeleton = False
    supports_animation = False
    priority = 40

    def export(self, model: ModelAsset, out_dir: Path, name: Optional[str] = None,
               options: Optional[dict] = None) -> ExportResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = name or model.name or "model"
        res = ExportResult(format=self.format)
        if model.skeleton and model.skeleton.bones:
            res.warnings.append("OBJ cannot store the skeleton (%d bones dropped)"
                                % len(model.skeleton.bones))
        if model.animations:
            res.warnings.append("OBJ cannot store %d animation(s)" % len(model.animations))

        obj_path = out_dir / (stem + ".obj")
        mtl_path = out_dir / (stem + ".mtl")
        lines = ["# Universal 3D Asset Extractor", "mtllib %s" % mtl_path.name]
        v_off = vt_off = vn_off = 1
        try:
            for mesh in model.meshes:
                if not mesh.vertices:
                    continue
                lines.append("o %s" % (mesh.name or "mesh"))
                for v in mesh.vertices:
                    lines.append("v %.6f %.6f %.6f" % tuple(v[:3]))
                uvs = mesh.uv_channels[0] if mesh.uv_channels else []
                for uv in uvs:
                    lines.append("vt %.6f %.6f" % (uv[0], uv[1]))
                for n in mesh.normals:
                    lines.append("vn %.6f %.6f %.6f" % tuple(n[:3]))
                if 0 <= mesh.material < len(model.materials):
                    lines.append("usemtl %s" % model.materials[mesh.material].name)
                has_uv, has_n = bool(uvs), bool(mesh.normals)
                idx = mesh.indices or list(range(len(mesh.vertices)))
                for i in range(0, len(idx) - 2, 3):
                    face = []
                    for k in range(3):
                        vi = idx[i + k]
                        parts = [str(vi + v_off)]
                        parts.append(str(vi + vt_off) if has_uv else "")
                        if has_n:
                            parts.append(str(vi + vn_off))
                        face.append("/".join(parts).rstrip("/") if len(parts) > 1 else parts[0])
                    lines.append("f " + " ".join(face))
                v_off += len(mesh.vertices)
                vt_off += len(uvs)
                vn_off += len(mesh.normals)
            obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            tex_dir = out_dir / "textures"
            mtl_lines = []
            for mat in model.materials:
                mtl_lines.append("newmtl %s" % mat.name)
                mtl_lines.append("Kd %.6f %.6f %.6f" % tuple(mat.base_color[:3]))
                mtl_lines.append("Ns %.2f" % max(0.0, (1.0 - mat.roughness) ** 2 * 1000.0))
                for usage, tex in mat.textures.items():
                    written = image_tools.write_texture(tex, tex_dir)
                    if not written:
                        continue
                    key = {"diffuse": "map_Kd", "normal": "map_Bump", "specular": "map_Ks",
                           "ao": "map_Ka", "emission": "map_Ke", "mask": "map_d"}.get(usage)
                    if key:
                        mtl_lines.append("%s textures/%s" % (key, written.name))
                        res.files.append(written)
            mtl_path.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")
            res.files = [obj_path, mtl_path] + res.files
            res.ok = True
        except Exception as exc:
            res.error = "OBJ export failed: %s" % exc
        return res
