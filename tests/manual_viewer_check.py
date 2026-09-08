"""Manual GL check: renders a model offscreen-ish and saves a PNG.

Needs a real GL platform (the `offscreen` Qt platform has no GL context), so
this is not part of the automated suite:

    python tests/manual_viewer_check.py path/to/model.glb
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.bootstrap import initialize
from app.config import Settings
from app.core.detector import FileContext, identify
from app.formats.models import parse_model
from app.ui.viewer import ModelViewer


def main(argv) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path("logs/sample/MyGame/data/models/Character_A.glb")
    app = QApplication([argv[0]])
    initialize(Settings(), load_plugins=False)
    ctx = FileContext(path)
    model = parse_model(ctx, identify(ctx))
    viewer = ModelViewer()
    viewer.resize(800, 600)
    viewer.show()
    viewer.load_model(model)
    viewer.show_bones = True

    deadline = time.time() + 3
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    viewer.repaint()
    app.processEvents()
    image = viewer.grabFramebuffer()
    out = Path("logs/viewer_check.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(out))
    print("gl ready:", viewer._ready, "meshes:", len(viewer._meshes),
          "frame:", image.width(), "x", image.height(), "->", out)
    viewer.set_time(0.5)
    viewer.repaint()
    app.processEvents()
    print("skinned frame rendered without error")
    viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
