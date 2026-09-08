"""COLLADA (.dae) exporter: geometry, materials, skeleton node hierarchy.

Skinning and animation are not written yet - the exporter says so instead of
producing a file that silently lost the rig.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.types import ModelAsset
from app.exporters.base import Exporter, ExportResult
from app.formats.textures import image as image_tools

NS = "http://www.collada.org/2005/11/COLLADASchema"


class ColladaExporter(Exporter):
    name = "dae"
    format = "dae"
    extension = ".dae"
    supports_skeleton = True          # hierarchy only
    supports_animation = False
    priority = 30

    def export(self, model: ModelAsset, out_dir: Path, name: Optional[str] = None,
               options: Optional[dict] = None) -> ExportResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = name or model.name or "model"
        res = ExportResult(format=self.format)
        if model.animations:
            res.warnings.append("DAE exporter does not write animations (%d dropped); "
                                "use GLB to keep them" % len(model.animations))
        if model.has_skinning:
            res.warnings.append("DAE exporter writes the bone hierarchy but not skin weights")

        root = ET.Element("COLLADA", {"xmlns": NS, "version": "1.4.1"})
        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "created").text = datetime.now(timezone.utc).isoformat()
        ET.SubElement(asset, "modified").text = datetime.now(timezone.utc).isoformat()
        ET.SubElement(asset, "up_axis").text = "Y_UP"

        lib_images = ET.SubElement(root, "library_images")
        lib_effects = ET.SubElement(root, "library_effects")
        lib_materials = ET.SubElement(root, "library_materials")
        tex_dir = out_dir / "textures"
        for i, mat in enumerate(model.materials):
            eff_id = "effect_%d" % i
            effect = ET.SubElement(lib_effects, "effect", {"id": eff_id})
            profile = ET.SubElement(effect, "profile_COMMON")
            technique = ET.SubElement(profile, "technique", {"sid": "common"})
            lambert = ET.SubElement(technique, "lambert")
            diffuse = ET.SubElement(lambert, "diffuse")
            tex = mat.textures.get("diffuse")
            written = image_tools.write_texture(tex, tex_dir) if tex else None
            if written:
                img_id = "image_%d" % i
                img = ET.SubElement(lib_images, "image", {"id": img_id})
                ET.SubElement(img, "init_from").text = "textures/" + written.name
                ET.SubElement(diffuse, "texture", {"texture": img_id, "texcoord": "UVMap"})
                res.files.append(written)
            else:
                ET.SubElement(diffuse, "color").text = " ".join(
                    "%.6f" % c for c in mat.base_color)
            material = ET.SubElement(lib_materials, "material",
                                     {"id": "material_%d" % i, "name": mat.name})
            ET.SubElement(material, "instance_effect", {"url": "#" + eff_id})

        lib_geo = ET.SubElement(root, "library_geometries")
        for gi, mesh in enumerate(model.meshes):
            if not mesh.vertices:
                continue
            geo_id = "geometry_%d" % gi
            geo = ET.SubElement(lib_geo, "geometry", {"id": geo_id, "name": mesh.name})
            m = ET.SubElement(geo, "mesh")
            self._source(m, geo_id + "_pos", [c for v in mesh.vertices for c in v[:3]],
                         ("X", "Y", "Z"), len(mesh.vertices))
            if mesh.normals and len(mesh.normals) == len(mesh.vertices):
                self._source(m, geo_id + "_norm", [c for v in mesh.normals for c in v[:3]],
                             ("X", "Y", "Z"), len(mesh.normals))
            vertices = ET.SubElement(m, "vertices", {"id": geo_id + "_vtx"})
            ET.SubElement(vertices, "input", {"semantic": "POSITION",
                                              "source": "#" + geo_id + "_pos"})
            idx = mesh.indices or list(range(len(mesh.vertices)))
            tri_count = len(idx) // 3
            tri_attrs = {"count": str(tri_count)}
            if 0 <= mesh.material < len(model.materials):
                tri_attrs["material"] = "material_%d" % mesh.material
            tris = ET.SubElement(m, "triangles", tri_attrs)
            ET.SubElement(tris, "input", {"semantic": "VERTEX", "source": "#" + geo_id + "_vtx",
                                          "offset": "0"})
            ET.SubElement(tris, "p").text = " ".join(str(i) for i in idx[:tri_count * 3])

        lib_scenes = ET.SubElement(root, "library_visual_scenes")
        scene = ET.SubElement(lib_scenes, "visual_scene", {"id": "scene", "name": "scene"})
        if model.skeleton and model.skeleton.bones:
            for root_index in model.skeleton.roots():
                self._bone_node(scene, model.skeleton, root_index)
        for gi, mesh in enumerate(model.meshes):
            if not mesh.vertices:
                continue
            node = ET.SubElement(scene, "node", {"id": "node_%d" % gi, "name": mesh.name})
            ET.SubElement(node, "instance_geometry", {"url": "#geometry_%d" % gi})
        instance = ET.SubElement(ET.SubElement(root, "scene"), "instance_visual_scene",
                                 {"url": "#scene"})
        instance.tail = "\n"

        path = out_dir / (stem + ".dae")
        try:
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            res.files.insert(0, path)
            res.ok = True
        except Exception as exc:
            res.error = "DAE export failed: %s" % exc
        return res

    def _source(self, parent, sid: str, values: list, params: tuple, count: int) -> None:
        src = ET.SubElement(parent, "source", {"id": sid})
        arr = ET.SubElement(src, "float_array", {"id": sid + "_arr", "count": str(len(values))})
        arr.text = " ".join("%.6f" % v for v in values)
        tech = ET.SubElement(src, "technique_common")
        acc = ET.SubElement(tech, "accessor", {"source": "#" + sid + "_arr",
                                               "count": str(count), "stride": str(len(params))})
        for p in params:
            ET.SubElement(acc, "param", {"name": p, "type": "float"})

    def _bone_node(self, parent, skeleton, index: int):
        bone = skeleton.bones[index]
        node = ET.SubElement(parent, "node", {"id": "bone_%d" % index, "sid": bone.name,
                                              "name": bone.name, "type": "JOINT"})
        t = ET.SubElement(node, "translate")
        t.text = " ".join("%.6f" % v for v in bone.translation)
        for child in skeleton.children(index):
            self._bone_node(node, skeleton, child)
        return node


class FbxExporter(Exporter):
    """FBX is a closed binary format; writing it correctly needs the FBX SDK.

    Reports unsupported rather than emitting a file other tools cannot open.
    """

    name = "fbx"
    format = "fbx"
    extension = ".fbx"
    supports_skeleton = True
    supports_animation = True
    priority = 20

    def export(self, model: ModelAsset, out_dir: Path, name: Optional[str] = None,
               options: Optional[dict] = None) -> ExportResult:
        return self.unsupported(
            "FBX writing is not implemented in this build. Export GLB (lossless for mesh, "
            "skeleton, materials and animation) and convert with Blender or the FBX SDK. "
            "An FBX exporter can be added as a plugin without touching the core.")
