"""One entry point that wires the whole core together (no UI involved)."""
from __future__ import annotations

import pathlib
from typing import Optional

from app.config import Settings
from app.core import detector, extraction
from app.core.logging_setup import get_logger, setup_logging
from app.database.db import Database
from app.exporters import bootstrap as bootstrap_exporters
from app.formats.animations import bootstrap as bootstrap_animations
from app.formats.models import bootstrap as bootstrap_models
from app.plugins.loader import load_all

_ready = False
_plugins: list = []


def initialize(settings: Optional[Settings] = None, db: Optional[Database] = None,
               load_plugins: bool = True, plugin_dirs=None, log_dir: str = "logs") -> list:
    """Register built-ins, then plugins. Safe to call more than once."""
    global _ready, _plugins
    settings = settings or Settings()
    setup_logging(log_dir, settings.log_level if not settings.developer_mode else "DEBUG")
    log = get_logger("bootstrap")

    if getattr(settings, "oodle_dll", ""):
        from app.formats.archives.oodle import set_library
        set_library(settings.oodle_dll)

    detector.bootstrap()
    extraction.bootstrap()
    bootstrap_models()
    bootstrap_animations()
    bootstrap_exporters()

    if load_plugins and not _ready:
        dirs = None
        if plugin_dirs:
            from app.plugins.loader import BUILTIN_DIR, USER_DIR
            extra = []
            for entry in plugin_dirs:
                path = pathlib.Path(entry)
                # accept both a folder of plugins and a single plugin folder
                extra.append(path.parent if (path / "plugin.json").exists() else path)
            dirs = [BUILTIN_DIR, USER_DIR] + extra
        _plugins = load_all(dirs, settings, db)
        loaded = [p.name for p in _plugins if p.loaded]
        log.info("plugins loaded: %s", ", ".join(loaded) or "none")
        for p in _plugins:
            if p.loaded:
                continue
            if p.error == "disabled by the user":
                log.info("plugin %s is disabled", p.name)
            else:
                log.warning("plugin %s not loaded: %s", p.name, p.error)
    _ready = True
    return _plugins


def plugins() -> list:
    return list(_plugins)
