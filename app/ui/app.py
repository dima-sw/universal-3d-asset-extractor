"""GUI entry point: python -m app.ui.app  (or `python main.py gui`)."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.config import DEFAULT_SETTINGS_PATH, Settings


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv)
    app = QApplication(argv)
    app.setApplicationName("Universal 3D Asset Extractor")
    settings = Settings.load(DEFAULT_SETTINGS_PATH)
    from app.ui.main_window import MainWindow
    window = MainWindow(settings)
    window.show()
    code = app.exec()
    try:
        DEFAULT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        window.settings.save(DEFAULT_SETTINGS_PATH)
    except OSError:
        pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
