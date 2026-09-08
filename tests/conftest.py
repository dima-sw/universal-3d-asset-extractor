"""Shared fixtures. No test touches the user's real filesystem outside tmp_path."""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bootstrap import initialize            # noqa: E402
from app.config import Settings                  # noqa: E402
from app.core.types import (AnimationClip, AnimationTrack, Bone, Keyframe, Material, Mesh,
                            ModelAsset, Skeleton)  # noqa: E402

CUBE_OBJ = """# cube
mtllib cube.mtl
v -1 -1 -1
v 1 -1 -1
v 1 1 -1
v -1 1 -1
v -1 -1 1
v 1 -1 1
v 1 1 1
v -1 1 1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 -1
usemtl body
f 1/1/1 2/2/1 3/3/1 4/4/1
f 5/1/1 6/2/1 7/3/1 8/4/1
"""

CUBE_MTL = """newmtl body
Kd 0.8 0.3 0.2
Ns 250
map_Kd texture.png
"""


def make_png(width: int = 8, height: int = 8) -> bytes:
    raw = b"".join(b"\x00" + bytes([200, 60, 40]) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


@pytest.fixture(scope="session", autouse=True)
def _bootstrap(tmp_path_factory):
    initialize(Settings(), load_plugins=True,
               log_dir=str(tmp_path_factory.mktemp("logs")))


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.output_dir = str(tmp_path / "out")
    s.temp_dir = str(tmp_path / "temp")
    s.db_path = str(tmp_path / "test.sqlite3")
    s.workers = 2
    return s


@pytest.fixture
def game_dir(tmp_path) -> Path:
    root = tmp_path / "MyGame"
    (root / "data" / "models").mkdir(parents=True)
    (root / "data" / "models" / "cube.obj").write_text(CUBE_OBJ)
    (root / "data" / "models" / "cube.mtl").write_text(CUBE_MTL)
    (root / "data" / "models" / "texture.png").write_bytes(make_png())
    return root


@pytest.fixture
def character_model() -> ModelAsset:
    hierarchy = [("Root", -1), ("Pelvis", 0), ("Spine", 1), ("Chest", 2), ("Neck", 3),
                 ("Head", 4), ("LeftArm", 3), ("LeftForeArm", 6), ("LeftHand", 7),
                 ("RightArm", 3), ("RightForeArm", 9), ("RightHand", 10),
                 ("LeftUpLeg", 1), ("LeftLeg", 12), ("LeftFoot", 13),
                 ("RightUpLeg", 1), ("RightLeg", 15), ("RightFoot", 16)]
    skeleton = Skeleton(name="Bip01",
                        bones=[Bone(n, p, translation=(0.0, 0.1 * i, 0.0))
                               for i, (n, p) in enumerate(hierarchy)])
    mesh = Mesh(name="body",
                vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                normals=[(0, 0, 1)] * 4,
                uv_channels=[[(0, 0), (1, 0), (1, 1), (0, 1)]],
                indices=[0, 1, 2, 0, 2, 3],
                material=0,
                joints=[(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0)],
                weights=[(1.0, 0.0, 0.0, 0.0)] * 4)
    clip = AnimationClip(name="Idle", duration=1.0, fps=30, tracks=[
        AnimationTrack(bone="Spine",
                       rotations=[Keyframe(0.0, (0, 0, 0, 1)), Keyframe(1.0, (0, 0.1, 0, 0.99))],
                       positions=[Keyframe(0.0, (0, 0, 0)), Keyframe(1.0, (0, 0.2, 0))])])
    return ModelAsset(name="Character_A", meshes=[mesh], materials=[Material(name="skin")],
                      skeleton=skeleton, animations=[clip], source_format="test")
