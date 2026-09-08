from __future__ import annotations

import json

import pytest

from app.core.conversion.normalize import normalize
from app.core.detector import FileContext, identify
from app.core.validation import validate_model
from app.exporters import get_exporter
from app.formats.models import parse_model
from app.formats.textures import image as image_tools
from tests.conftest import make_png


def test_obj_parse_materials_and_geometry(game_dir):
    ctx = FileContext(game_dir / "data" / "models" / "cube.obj")
    model = parse_model(ctx, identify(ctx))
    assert model.vertex_count == 8
    assert model.triangle_count == 4                 # two quads fan-triangulated
    assert model.materials[0].name == "body"
    assert "diffuse" in model.materials[0].textures


def test_glb_roundtrip_keeps_mesh_skeleton_and_animation(tmp_path, character_model):
    res = get_exporter("glb").export(character_model, tmp_path, "Character_A")
    assert res.ok, res.error
    out = res.files[0]
    assert out.read_bytes()[:4] == b"glTF"

    ctx = FileContext(out)
    reloaded = parse_model(ctx, identify(ctx))
    assert reloaded.vertex_count == character_model.vertex_count
    assert reloaded.triangle_count == character_model.triangle_count
    assert reloaded.skeleton is not None
    assert len(reloaded.skeleton.bones) == len(character_model.skeleton.bones)
    assert reloaded.skeleton.bones[1].name == "Pelvis"
    assert reloaded.skeleton.bones[1].parent == 0
    assert reloaded.has_skinning
    assert len(reloaded.animations) == 1
    assert reloaded.animations[0].name == "Idle"
    assert reloaded.animations[0].duration == pytest.approx(1.0, abs=1e-3)


def test_gltf_json_exporter_writes_bin_sidecar(tmp_path, character_model):
    res = get_exporter("gltf").export(character_model, tmp_path, "char")
    assert res.ok
    gltf = json.loads((tmp_path / "char.gltf").read_text())
    assert gltf["asset"]["version"] == "2.0"
    assert (tmp_path / "char.bin").exists()


def test_obj_exporter_warns_about_lost_rig(tmp_path, character_model):
    res = get_exporter("obj").export(character_model, tmp_path, "char")
    assert res.ok
    assert any("skeleton" in w for w in res.warnings)
    assert any("animation" in w for w in res.warnings)
    assert (tmp_path / "char.obj").exists()


def test_dae_exporter_writes_xml(tmp_path, character_model):
    res = get_exporter("dae").export(character_model, tmp_path, "char")
    assert res.ok
    text = (tmp_path / "char.dae").read_text()
    assert "COLLADA" in text and "library_geometries" in text


def test_fbx_exporter_reports_unsupported_clearly(tmp_path, character_model):
    res = get_exporter("fbx").export(character_model, tmp_path, "char")
    assert res.ok is False
    assert "not implemented" in res.error
    assert "GLB" in res.error


def test_validation_flags_bad_indices(character_model):
    character_model.meshes[0].indices = [0, 1, 99]
    report = validate_model(character_model)
    assert report.valid is False
    assert any("out-of-range" in i for i in report.issues)


def test_normalize_preserves_original_metadata(character_model):
    character_model.coordinate_system = "z_up_rh"
    normalize(character_model, up_axis="y", handedness="right", unit_scale=1.0)
    assert character_model.coordinate_system == "y_up_rh"
    assert character_model.metadata["original_coordinate_system"] == "z_up_rh"


def test_texture_header_reading(tmp_path):
    p = tmp_path / "hero_n.png"
    p.write_bytes(make_png(32, 16))
    tex = image_tools.read_header(p)
    assert tex.width == 32 and tex.height == 16
    assert tex.source_format == "PNG"
    assert tex.usage == "normal"                    # from the _n suffix


def test_stl_binary_parse(tmp_path):
    import struct
    data = bytearray(b"\x00" * 80 + struct.pack("<I", 1))
    data += struct.pack("<12f", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0) + b"\x00\x00"
    p = tmp_path / "part.stl"
    p.write_bytes(bytes(data))
    ctx = FileContext(p)
    model = parse_model(ctx, identify(ctx))
    assert model.vertex_count == 3
    assert model.triangle_count == 1
