from __future__ import annotations

from app.core.detector import FileContext, identify
from app.core.pipeline.controller import ExtractorCore
from app.core.registry import ANIMATION_PARSERS
from app.core.types import AssetType, Category

BVH = """HIERARCHY
ROOT Hips
{
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.00 5.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
            OFFSET 0.00 5.00 0.00
        }
    }
}
MOTION
Frames: 3
Frame Time: 0.033333
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 1.0 0.0 0.0 10.0 0.0 0.0 5.0 0.0
0.0 2.0 0.0 0.0 20.0 0.0 0.0 10.0 0.0
"""


def test_bvh_detected_and_parsed(tmp_path):
    p = tmp_path / "walk.bvh"
    p.write_text(BVH)
    ctx = FileContext(p)
    detection = identify(ctx)
    assert detection.format_name == "BVH"
    assert detection.category == Category.ANIMATION

    parser = next(p for p in ANIMATION_PARSERS if p.name == "bvh")
    clip = parser.parse(ctx)
    assert clip.name == "walk"
    assert len(clip.tracks) == 2
    assert clip.metadata["frames"] == 3
    assert clip.duration > 0
    hips = next(t for t in clip.tracks if t.bone == "Hips")
    assert len(hips.positions) == 3
    assert len(hips.rotations) == 3


def test_bvh_lands_in_the_scan_as_an_animation(settings, tmp_path):
    game = tmp_path / "Game"
    game.mkdir()
    (game / "walk.bvh").write_text(BVH)
    core = ExtractorCore(settings)
    result = core.analyze(game)
    assert result.counts()["animations"] == 1
    assert result.graph.by_type(AssetType.ANIMATION)[0].name == "walk"


def test_resume_skips_completed_files(settings, game_dir):
    core = ExtractorCore(settings)
    first = core.analyze(game_dir)
    assert first.counts()["files_scanned"] > 0

    resumed = ExtractorCore(settings, core.db)
    result = resumed.resume_scan(first.scan_id)
    assert result.scan_id == first.scan_id
    assert result.counts()["files_scanned"] == 0        # everything was already done


def test_pause_persists_the_queue(settings, game_dir):
    core = ExtractorCore(settings)
    result = core.analyze(game_dir)
    core.pause()
    row = core.db.get_scan(result.scan_id)
    assert row["status"] == "paused"
    core.resume()
