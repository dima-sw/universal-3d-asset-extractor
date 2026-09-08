"""Plugin discovery and loading.

Search order:
    1. app/plugins/builtin/*            (shipped)
    2. <user config>/plugins/*          (installed by the user)
    3. paths passed explicitly

A plugin directory must contain plugin.json and the entrypoint it names.
Loading is opt-in code execution from a trusted directory only: files coming
from a scanned game are NEVER loaded, whatever they claim to be.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.logging_setup import get_logger
from app.plugins.api import PLUGIN_API_VERSION, PluginAPI

log = get_logger("plugins")

BUILTIN_DIR = Path(__file__).parent / "builtin"
USER_DIR = Path.home() / ".universal3dextractor" / "plugins"


@dataclass
class PluginInfo:
    name: str
    version: str = "0.0.0"
    author: str = ""
    path: Path = field(default_factory=Path)
    entrypoint: str = "plugin.py"
    supported_extensions: list = field(default_factory=list)
    supported_engines: list = field(default_factory=list)
    api_version: int = PLUGIN_API_VERSION
    loaded: bool = False
    error: Optional[str] = None
    source: str = ""                       # "builtin" | "user"
    enabled: bool = True
    problems: list = field(default_factory=list)
    registered: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version, "author": self.author,
                "path": str(self.path), "engines": self.supported_engines,
                "extensions": self.supported_extensions, "loaded": self.loaded,
                "error": self.error, "source": self.source, "enabled": self.enabled,
                "problems": self.problems, "registered": self.registered}


def discover(dirs=None) -> list:
    roots = [Path(d) for d in (dirs or [])] or [BUILTIN_DIR, USER_DIR]
    found: list = []
    for root in roots:
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            manifest = entry / "plugin.json"
            if not entry.is_dir() or not manifest.exists():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("bad plugin manifest %s: %s", manifest, exc)
                continue
            found.append(PluginInfo(
                name=data.get("name") or entry.name,
                version=str(data.get("version", "0.0.0")),
                author=data.get("author", ""),
                path=entry,
                entrypoint=data.get("entrypoint", "plugin.py"),
                supported_extensions=data.get("supported_extensions", []),
                supported_engines=data.get("supported_engines", []),
                api_version=int(data.get("api_version", PLUGIN_API_VERSION)),
            ))
    return found


def load(info: PluginInfo, settings=None, db=None) -> PluginInfo:
    module_path = info.path / info.entrypoint
    if not module_path.exists():
        info.error = "entrypoint %s missing" % info.entrypoint
        log.warning("plugin %s: %s", info.name, info.error)
        return info
    if info.api_version > PLUGIN_API_VERSION:
        info.error = "plugin needs API v%d, this build provides v%d" % (
            info.api_version, PLUGIN_API_VERSION)
        log.warning("plugin %s: %s", info.name, info.error)
        return info
    mod_name = "u3de_plugin_%s" % info.path.name
    try:
        spec = importlib.util.spec_from_file_location(mod_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError("cannot build import spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        register = getattr(module, "register", None)
        if register is None:
            raise AttributeError("plugin defines no register(api) function")
        register(PluginAPI(info.name, settings))
        info.loaded = True
        log.info("plugin loaded: %s v%s", info.name, info.version)
        if db is not None:
            db.record_plugin(info.name, info.version, str(module_path))
    except Exception as exc:
        info.error = str(exc)
        log.warning("plugin %s failed to load: %s", info.name, exc)
    return info


def load_all(dirs=None, settings=None, db=None, skip_disabled: bool = True) -> list:
    """Load every discovered plugin. Disabled ones are listed but not imported."""
    disabled = set()
    if skip_disabled:
        try:
            from app.plugins.manager import PluginState
            disabled = set(PluginState.load().disabled)
        except Exception:
            disabled = set()
    out = []
    for info in discover(dirs):
        if info.name in disabled:
            info.error = "disabled by the user"
            out.append(info)
            continue
        out.append(load(info, settings, db))
    return out


def reload_all(dirs=None, settings=None, db=None) -> list:
    """Drop every registered extension and load the plugins again.

    Built-in detectors/parsers are re-registered by the bootstrap, so this is
    how a freshly installed plugin becomes active without restarting the app.
    """
    import sys

    from app.core import detector, extraction
    from app.core.registry import reset_all
    from app.exporters import bootstrap as bootstrap_exporters
    from app.formats.animations import bootstrap as bootstrap_animations
    from app.formats.models import bootstrap as bootstrap_models

    for module in [name for name in sys.modules if name.startswith("u3de_plugin_")]:
        sys.modules.pop(module, None)

    reset_all()
    for module in (detector, extraction):
        module._bootstrapped = False          # type: ignore[attr-defined]
    for module_name in ("app.formats.models", "app.formats.animations", "app.exporters"):
        sys.modules[module_name]._bootstrapped = False    # type: ignore[attr-defined]

    detector.bootstrap()
    extraction.bootstrap()
    bootstrap_models()
    bootstrap_animations()
    bootstrap_exporters()
    return load_all(dirs, settings, db)
