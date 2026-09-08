"""Plugin lifecycle: scaffold, validate, install, enable/disable, remove."""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from app.core.detector import FileContext, identify
from app.plugins import manager
from app.plugins.loader import load_all


@pytest.fixture
def plugin_home(tmp_path, monkeypatch):
    """Redirect the user plugin folder and its state file into tmp_path."""
    home = tmp_path / "plugins"
    home.mkdir()
    monkeypatch.setattr(manager, "USER_DIR", home)
    monkeypatch.setattr(manager, "STATE_FILE", tmp_path / "plugins.json")
    return home


@pytest.fixture
def scaffolded(tmp_path):
    return manager.scaffold("Tomb Raider PS2", tmp_path, engine="PS2", extensions=["tr2"])


def test_scaffold_creates_a_valid_plugin(scaffolded):
    assert (scaffolded / "plugin.json").exists()
    assert (scaffolded / "plugin.py").exists()
    assert (scaffolded / "README.md").exists()
    assert manager.validate(scaffolded) == []

    manifest = json.loads((scaffolded / "plugin.json").read_text())
    assert manifest["name"] == "Tomb Raider PS2"
    assert manifest["id"] == "tomb_raider_ps2"
    assert manifest["supported_engines"] == ["PS2"]

    source = (scaffolded / "plugin.py").read_text()
    assert "{{" not in source and "}}" not in source          # placeholders substituted
    assert "TombRaiderPs2ArchiveDetector" in source


def test_scaffold_rejects_duplicates_and_bad_names(tmp_path, scaffolded):
    with pytest.raises(manager.PluginError):
        manager.scaffold("Tomb Raider PS2", tmp_path)
    with pytest.raises(manager.PluginError):
        manager.scaffold("!!", tmp_path)


def test_validation_reports_specific_problems(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "plugin.json is missing" in manager.validate(empty)[0]

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "plugin.json").write_text("{ not json")
    assert "not valid JSON" in manager.validate(broken)[0]

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "plugin.json").write_text(json.dumps({"name": "x", "version": "1"}))
    problems = manager.validate(incomplete)
    assert any("entrypoint" in p for p in problems)

    future = tmp_path / "future"
    future.mkdir()
    (future / "plugin.json").write_text(json.dumps(
        {"name": "x", "version": "1", "entrypoint": "plugin.py", "api_version": 99}))
    (future / "plugin.py").write_text("def register(api):\n    pass\n")
    assert any("plugin API v99" in p for p in manager.validate(future))

    no_register = tmp_path / "no_register"
    no_register.mkdir()
    (no_register / "plugin.json").write_text(json.dumps(
        {"name": "x", "version": "1", "entrypoint": "plugin.py"}))
    (no_register / "plugin.py").write_text("x = 1\n")
    assert any("register(api)" in p for p in manager.validate(no_register))


def test_install_from_folder_and_use_it(plugin_home, scaffolded, tmp_path):
    info = manager.install(scaffolded, dest_dir=plugin_home)
    assert info.path.parent == plugin_home
    assert info.path.name == "tomb_raider_ps2"

    subprocess.run([sys.executable, str(scaffolded / "make_sample.py")],
                   capture_output=True, check=True)
    loaded = load_all([plugin_home], skip_disabled=False)
    assert loaded and loaded[0].loaded, loaded[0].error

    detection = identify(FileContext(scaffolded / "sample_model.bin"))
    assert detection.format_name == "Tomb Raider PS2 model"
    assert detection.confidence > 0.9

    from app.formats.models import parse_model
    model = parse_model(FileContext(scaffolded / "sample_model.bin"), detection)
    assert model.vertex_count == 4 and model.triangle_count == 2


def test_install_from_zip(plugin_home, scaffolded, tmp_path):
    archive = tmp_path / "tomb_raider_ps2.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for file in scaffolded.rglob("*"):
            if file.is_file():
                zf.write(file, "tomb_raider_ps2/" + file.relative_to(scaffolded).as_posix())
    info = manager.install(archive, dest_dir=plugin_home)
    assert (info.path / "plugin.py").exists()


def test_install_rejects_zip_slip(plugin_home, tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("plugin.json", json.dumps(
            {"name": "evil", "version": "1", "entrypoint": "plugin.py"}))
        zf.writestr("plugin.py", "def register(api):\n    pass\n")
        zf.writestr("../../escaped.py", "print('pwned')")
    with pytest.raises(manager.PluginError):
        manager.install(archive, dest_dir=plugin_home)
    assert not (plugin_home.parent.parent / "escaped.py").exists()


def test_install_refuses_an_invalid_plugin(plugin_home, tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "plugin.json").write_text(json.dumps({"name": "bad", "version": "1",
                                                 "entrypoint": "missing.py"}))
    with pytest.raises(manager.PluginError) as exc:
        manager.install(bad, dest_dir=plugin_home)
    assert "entrypoint" in str(exc.value)


def test_install_twice_needs_force(plugin_home, scaffolded):
    manager.install(scaffolded, dest_dir=plugin_home)
    with pytest.raises(manager.PluginError):
        manager.install(scaffolded, dest_dir=plugin_home)
    assert manager.install(scaffolded, dest_dir=plugin_home, force=True)


def test_enable_disable_and_remove(plugin_home, scaffolded, monkeypatch):
    manager.install(scaffolded, dest_dir=plugin_home)
    monkeypatch.setattr(manager, "discover",
                        lambda dirs=None: _discover_in(plugin_home))

    assert manager.set_enabled("Tomb Raider PS2", False)
    assert manager.is_enabled("Tomb Raider PS2") is False
    loaded = load_all([plugin_home])
    assert loaded[0].loaded is False
    assert loaded[0].error == "disabled by the user"

    assert manager.set_enabled("Tomb Raider PS2", True)
    assert load_all([plugin_home])[0].loaded is True

    assert manager.remove("Tomb Raider PS2") is True
    assert manager.remove("Tomb Raider PS2") is False


def _discover_in(folder):
    from app.plugins.loader import discover as real_discover
    return real_discover([folder])


def test_catalogue_marks_builtin_and_user(plugin_home, scaffolded, monkeypatch):
    manager.install(scaffolded, dest_dir=plugin_home)
    monkeypatch.setattr(manager, "USER_DIR", plugin_home)
    from app.plugins import loader
    monkeypatch.setattr(loader, "USER_DIR", plugin_home)
    rows = {info.name: info for info in manager.catalogue()}
    assert rows["Unity"].source == "builtin"
    assert rows["Tomb Raider PS2"].source == "user"
    assert all(not info.problems for info in rows.values())


def test_slug_normalisation():
    assert manager.slug("Tomb Raider PS2") == "tomb_raider_ps2"
    assert manager.slug("Final Fantasy X-2!!") == "final_fantasy_x_2"
    assert manager.slug("") == "plugin"
