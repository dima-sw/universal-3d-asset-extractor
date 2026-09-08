from __future__ import annotations

import json
import zipfile

from app.core.graph.asset_graph import AssetGraph, name_similarity
from app.core.graph.character import analyse_skeleton, classify
from app.core.pipeline.controller import ExtractorCore
from app.core.types import Asset, AssetType, Classification
from app.exporters import get_exporter
from tests.conftest import CUBE_MTL, CUBE_OBJ, make_png


def test_scan_finds_assets_and_recurses_into_archives(settings, game_dir, tmp_path):
    archive = game_dir / "data" / "pack.pak"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("inner/models/prop.obj", CUBE_OBJ)
        z.writestr("inner/models/prop.mtl", CUBE_MTL)
        z.writestr("inner/models/texture.png", make_png())

    core = ExtractorCore(settings)
    result = core.analyze(game_dir)
    counts = result.counts()
    assert counts["containers"] == 1
    assert counts["models"] == 2                 # cube.obj + prop.obj from inside the pak
    assert counts["textures"] >= 2
    assert counts["failed"] == 0
    names = {a.name for a in result.graph.by_type(AssetType.MODEL)}
    assert {"cube", "prop"} <= names


def test_nesting_depth_limit_is_enforced(settings, tmp_path):
    root = tmp_path / "nested"
    root.mkdir()
    payload = root / "level3.zip"
    with zipfile.ZipFile(payload, "w") as z:
        z.writestr("leaf.obj", CUBE_OBJ)
    for level in (2, 1):
        outer = root / ("level%d.zip" % level)
        with zipfile.ZipFile(outer, "w") as z:
            z.write(payload, payload.name)
        payload.unlink()
        payload = outer
    settings.limits.max_nesting_depth = 1
    core = ExtractorCore(settings)
    result = core.analyze(root)
    assert any("nesting depth" in u.reason for u in result.unsupported)


def test_duplicate_content_is_detected(settings, game_dir):
    copy = game_dir / "data" / "models" / "cube_copy.obj"
    copy.write_text((game_dir / "data" / "models" / "cube.obj").read_text())
    core = ExtractorCore(settings)
    result = core.analyze(game_dir)
    duplicates = result.graph.duplicates()
    assert duplicates, "identical content should be linked as duplicate"


def test_character_pipeline_end_to_end(settings, tmp_path, character_model):
    game = tmp_path / "Game" / "chars"
    game.mkdir(parents=True)
    get_exporter("glb").export(character_model, game, "Character_A")

    core = ExtractorCore(settings)
    result = core.analyze(game.parent)
    assert result.counts()["characters"] == 1
    group = result.characters[0]
    assert group.model_asset.classification == Classification.CHARACTER
    assert group.confidence > 0.5

    report = core.extract(result)
    assert len(report["exported"]) == 1
    out_dir = settings.resolved_output() / "Characters" / "Character_A"
    assert (out_dir / "Character_A.glb").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["asset_type"] == "character"
    assert manifest["bones"] == len(character_model.skeleton.bones)
    assert "Idle" in manifest["animations"]
    assert (settings.resolved_output() / "Reports" / "report.html").exists()
    assert (settings.resolved_output() / "Reports" / "report.json").exists()


def test_cancel_stops_the_scan(settings, game_dir):
    core = ExtractorCore(settings)
    core.cancel()
    result = core.analyze(game_dir)
    assert result.cancelled is True


def test_cache_avoids_second_detection(settings, game_dir):
    core = ExtractorCore(settings)
    core.analyze(game_dir)
    core2 = ExtractorCore(settings, core.db)
    result = core2.analyze(game_dir)
    cached = [f for f in core2.db.connect().execute(
        "SELECT * FROM cache").fetchall()]
    assert cached
    assert result.counts()["models"] == 1


def test_humanoid_rig_detection(character_model):
    rig = analyse_skeleton(character_model.skeleton)
    assert rig.is_humanoid
    assert rig.bone_count == 18
    assert "head" in rig.matched


def test_humanoid_detection_with_bip01_naming():
    from app.core.types import Bone, Skeleton
    names = [("Bip01", -1), ("Bip01_Pelvis", 0), ("Bip01_Spine", 1), ("Bip01_Spine1", 2),
             ("Bip01_Neck", 3), ("Bip01_Head", 4),
             ("Bip01_L_UpperArm", 3), ("Bip01_L_Forearm", 6), ("Bip01_L_Hand", 7),
             ("Bip01_R_UpperArm", 3), ("Bip01_R_Forearm", 9), ("Bip01_R_Hand", 10),
             ("Bip01_L_Thigh", 1), ("Bip01_L_Calf", 12), ("Bip01_L_Foot", 13),
             ("Bip01_R_Thigh", 1), ("Bip01_R_Calf", 15), ("Bip01_R_Foot", 16)]
    skeleton = Skeleton(name="Bip01", bones=[Bone(n, p) for n, p in names])
    rig = analyse_skeleton(skeleton)
    assert rig.is_humanoid
    assert rig.score > 0.5


def test_classification_prefers_structure_over_name(character_model):
    character_model.name = "obj_1234"                # useless name
    classification, confidence, evidence = classify(character_model, "data/obj_1234.glb")
    assert classification == Classification.CHARACTER
    assert confidence > 0.6
    assert evidence["skinned"] is True


def test_graph_links_and_name_similarity():
    graph = AssetGraph()
    model = graph.add(Asset(name="Hero_Body", type=AssetType.MODEL, source="a/Hero_Body.glb"))
    tex = graph.add(Asset(name="Hero_Body_D", type=AssetType.TEXTURE, source="a/Hero_Body_D.png"))
    graph.link(model.id, tex.id, __import__("app.core.types", fromlist=["Relation"]).Relation.CHARACTER_TEXTURE)
    assert graph.neighbours(model.id)[0].id == tex.id
    assert name_similarity("Hero_Body", "Hero_Body_D") > 0.5
    assert name_similarity("Hero", "Barrel") < 0.5
