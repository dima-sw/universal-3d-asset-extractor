"""Plugin management: install, remove, enable/disable, validate, scaffold.

A user who needs a format nobody has written yet (say Tomb Raider on PS2) can
write a plugin, drop it in, and the app picks it up - no rebuild, no core edit.
This module is what makes that practical:

    extractor plugins new tomb_raider_ps2      # scaffold a working skeleton
    extractor plugins check tomb_raider_ps2    # validate before shipping it
    extractor plugins install ./tomb_raider_ps2.zip
    extractor plugins list / enable / disable / remove

Plugins are Python code and run with the app's permissions. Only two trusted
locations are ever loaded: the built-in directory and the user plugin
directory. Nothing found while scanning a game is ever imported.
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.fs_safety import UnsafePathError, safe_join, sanitize_component
from app.core.logging_setup import get_logger
from app.plugins.loader import BUILTIN_DIR, USER_DIR, PluginInfo, discover, load

log = get_logger("plugins.manager")

STATE_FILE = USER_DIR.parent / "plugins.json"
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_]{1,63}$")
MAX_PLUGIN_BYTES = 64 << 20
REQUIRED_MANIFEST_FIELDS = ("name", "version", "entrypoint")


class PluginError(Exception):
    pass


@dataclass
class PluginState:
    disabled: list = field(default_factory=list)

    @classmethod
    def load(cls) -> "PluginState":
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return cls(disabled=list(data.get("disabled", [])))
        except (OSError, ValueError):
            return cls()

    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({"disabled": sorted(set(self.disabled))}, indent=2),
                              encoding="utf-8")


def user_dir() -> Path:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    return USER_DIR


def slug(name: str) -> str:
    value = re.sub(r"[^a-z0-9_]+", "_", (name or "plugin").lower()).strip("_")
    value = re.sub(r"_+", "_", value) or "plugin"
    return value[:64]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate(path) -> list:
    """Static checks on a plugin directory. Returns a list of problems."""
    path = Path(path)
    problems = []
    manifest_path = path / "plugin.json"
    if not path.is_dir():
        return ["%s is not a directory" % path]
    if not manifest_path.exists():
        return ["plugin.json is missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return ["plugin.json is not valid JSON: %s" % exc]

    for key in REQUIRED_MANIFEST_FIELDS:
        if not manifest.get(key):
            problems.append("plugin.json is missing %r" % key)
    api_version = manifest.get("api_version", 1)
    try:
        from app.plugins.api import PLUGIN_API_VERSION
        if int(api_version) > PLUGIN_API_VERSION:
            problems.append("needs plugin API v%s, this build provides v%d"
                            % (api_version, PLUGIN_API_VERSION))
    except (TypeError, ValueError):
        problems.append("api_version must be a number")

    entry = manifest.get("entrypoint", "plugin.py")
    if "/" in str(entry) or "\\" in str(entry) or str(entry).startswith("."):
        problems.append("entrypoint must be a plain file name inside the plugin folder")
    elif not (path / str(entry)).exists():
        problems.append("entrypoint %r not found" % entry)
    else:
        source = (path / str(entry)).read_text(encoding="utf-8", errors="replace")
        if not re.search(r"^def register\s*\(", source, re.M):
            problems.append("%s defines no top-level register(api) function" % entry)

    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    if total > MAX_PLUGIN_BYTES:
        problems.append("plugin is %.1f MB, over the %d MB limit"
                        % (total / 1e6, MAX_PLUGIN_BYTES // (1 << 20)))
    return problems


def probe(path, settings=None) -> PluginInfo:
    """Import the plugin in isolation and report what it registered.

    This executes the plugin's code, so only call it on a plugin the user has
    chosen to install or check.
    """
    from app.core.registry import summary
    before = summary()
    infos = discover([Path(path).parent])
    info = next((i for i in infos if i.path == Path(path)), None)
    if info is None:
        raise PluginError("no plugin manifest found at %s" % path)
    load(info, settings)
    after = summary()
    registered = {}
    for kind, names in after.items():
        added = [n for n in names if n not in before.get(kind, [])]
        if added:
            registered[kind] = added
    info.registered = registered           # type: ignore[attr-defined]
    return info


# ---------------------------------------------------------------------------
# install / remove
# ---------------------------------------------------------------------------
def install(source, force: bool = False, dest_dir: Optional[Path] = None) -> PluginInfo:
    """Install a plugin from a directory or a .zip into the user plugin folder."""
    source = Path(source)
    dest_root = Path(dest_dir) if dest_dir else user_dir()
    dest_root.mkdir(parents=True, exist_ok=True)

    staging = dest_root / (".staging_%s" % sanitize_component(source.stem))
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)

    try:
        if source.is_dir():
            shutil.copytree(source, staging)
        elif source.suffix.lower() == ".zip":
            _unzip(source, staging)
        else:
            raise PluginError("install expects a plugin folder or a .zip, got %s" % source)

        root = _manifest_root(staging)
        problems = validate(root)
        if problems:
            raise PluginError("plugin rejected:\n  - " + "\n  - ".join(problems))

        manifest = json.loads((root / "plugin.json").read_text(encoding="utf-8"))
        folder = slug(manifest.get("id") or manifest.get("name") or source.stem)
        target = dest_root / folder
        if target.exists():
            if not force:
                raise PluginError("%s is already installed (use force to overwrite)" % folder)
            shutil.rmtree(target)
        shutil.move(str(root), str(target))
        log.info("installed plugin %s -> %s", manifest.get("name"), target)
        infos = discover([dest_root])
        info = next((i for i in infos if i.path == target), None)
        if info is None:
            raise PluginError("installed plugin is not discoverable")
        state = PluginState.load()
        if info.name in state.disabled:
            state.disabled.remove(info.name)
            state.save()
        return info
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _unzip(archive: Path, dest: Path) -> None:
    if archive.stat().st_size > MAX_PLUGIN_BYTES:
        raise PluginError("plugin archive is larger than the allowed size")
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            total += member.file_size
            if total > MAX_PLUGIN_BYTES:
                raise PluginError("plugin archive expands beyond the allowed size")
            try:
                out = safe_join(dest, member.filename)
            except UnsafePathError as exc:
                raise PluginError("unsafe path in archive: %s" % exc)
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _manifest_root(staging: Path) -> Path:
    """Accept both plugin.json at the root and one nested folder deep."""
    if (staging / "plugin.json").exists():
        return staging
    candidates = [p.parent for p in staging.rglob("plugin.json")]
    if not candidates:
        raise PluginError("no plugin.json found in the archive")
    return min(candidates, key=lambda p: len(p.parts))


def remove(name: str) -> bool:
    """Remove a user-installed plugin. Built-ins can be disabled, not deleted."""
    for info in discover([user_dir()]):
        if info.name == name or info.path.name == slug(name):
            shutil.rmtree(info.path, ignore_errors=True)
            state = PluginState.load()
            if info.name in state.disabled:
                state.disabled.remove(info.name)
                state.save()
            log.info("removed plugin %s", info.name)
            return True
    return False


def set_enabled(name: str, enabled: bool) -> bool:
    state = PluginState.load()
    known = {i.name for i in discover()}
    if name not in known:
        return False
    if enabled:
        if name in state.disabled:
            state.disabled.remove(name)
    elif name not in state.disabled:
        state.disabled.append(name)
    state.save()
    return True


def is_enabled(name: str) -> bool:
    return name not in PluginState.load().disabled


def catalogue() -> list:
    """Every discoverable plugin with its source and enabled state."""
    rows = []
    state = PluginState.load()
    for info in discover():
        try:
            info.path.relative_to(BUILTIN_DIR)
            source = "builtin"
        except ValueError:
            source = "user"
        info.source = source                       # type: ignore[attr-defined]
        info.enabled = info.name not in state.disabled   # type: ignore[attr-defined]
        info.problems = validate(info.path)        # type: ignore[attr-defined]
        rows.append(info)
    return rows


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------
TEMPLATE_DIR = Path(__file__).parent / "templates" / "game_plugin"


def scaffold(name: str, dest_dir=None, engine: str = "", extensions=None) -> Path:
    """Create a working plugin skeleton the user can edit and install."""
    if not any(ch.isalnum() for ch in (name or "")):
        raise PluginError("plugin name %r has nothing to make a folder name from" % name)
    folder = slug(name)
    if not SLUG_RE.match(folder):
        raise PluginError("invalid plugin name %r" % name)
    dest_root = Path(dest_dir) if dest_dir else Path.cwd()
    target = dest_root / folder
    if target.exists():
        raise PluginError("%s already exists" % target)
    if not TEMPLATE_DIR.exists():
        raise PluginError("plugin template is missing from this install")

    shutil.copytree(TEMPLATE_DIR, target)
    class_name = "".join(part.capitalize() for part in folder.split("_")) or "MyGame"
    manifest = {
        "name": name,
        "id": folder,
        "version": "0.1.0",
        "author": "",
        "api_version": 1,
        "supported_extensions": list(extensions or []),
        "supported_engines": [engine] if engine else [],
        "entrypoint": "plugin.py",
    }
    (target / "plugin.json").write_text(json.dumps(manifest, indent=4) + "\n", encoding="utf-8")
    for file in (target / "plugin.py", target / "README.md"):
        if file.exists():
            text = file.read_text(encoding="utf-8")
            text = text.replace("{{PLUGIN_NAME}}", name).replace("{{PLUGIN_SLUG}}", folder)
            text = text.replace("{{CLASS_PREFIX}}", class_name)
            file.write_text(text, encoding="utf-8")
    log.info("scaffolded plugin at %s", target)
    return target
