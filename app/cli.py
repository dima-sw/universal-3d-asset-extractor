"""Command line interface. Same core as the GUI, no Qt import anywhere here.

    extractor scan   "D:/Games/Game"
    extractor extract "D:/Games/Game" --output "D:/Extracted" --format glb --deep
    extractor detect  file.bin
    extractor inspect file.bin
    extractor list-plugins
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app.bootstrap import initialize, plugins
from app.config import DuplicatePolicy, OverwritePolicy, ScanMode, Settings
from app.core.detector import FileContext, identify
from app.core.detector.engine import detect_engine
from app.core.detector.heuristics import EmbeddedAssetScanner, entropy_profile
from app.core.pipeline.controller import ExtractorCore
from app.database.db import Database
from app.exporters import available_formats


def build_settings(args) -> Settings:
    s = Settings()
    s.source_dir = str(getattr(args, "source", "") or "")
    s.output_dir = str(getattr(args, "output", "") or "./Extracted")
    if getattr(args, "temp", None):
        s.temp_dir = str(args.temp)
    if getattr(args, "workers", None):
        s.workers = int(args.workers)
    if getattr(args, "format", None):
        s.export_formats = [f.strip().lower() for f in str(args.format).split(",") if f.strip()]
    if getattr(args, "aggressive", False):
        s.scan_mode = ScanMode.AGGRESSIVE
    elif getattr(args, "deep", False):
        s.scan_mode = ScanMode.DEEP
    if getattr(args, "max_depth", None):
        s.limits.max_nesting_depth = int(args.max_depth)
    if getattr(args, "keep_duplicates", False):
        s.duplicate_policy = DuplicatePolicy.EXPORT_ALL
    if getattr(args, "overwrite", False):
        s.overwrite_policy = OverwritePolicy.OVERWRITE
    if getattr(args, "verbose", False):
        s.log_level = "DEBUG"
        s.developer_mode = True
    if getattr(args, "up_axis", None):
        s.coordinates.up_axis = args.up_axis
    if getattr(args, "handedness", None):
        s.coordinates.handedness = args.handedness
    if getattr(args, "unit_scale", None):
        s.coordinates.unit_scale = float(args.unit_scale)
    if getattr(args, "oodle_dll", None):
        s.oodle_dll = str(args.oodle_dll)
    return s


class ConsoleProgress:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.last = 0.0

    def __call__(self, progress) -> None:
        if self.quiet:
            return
        now = time.time()
        if now - self.last < 0.2:
            return
        self.last = now
        line = "  %6d/%-6d  %-11s %s" % (progress.scanned, progress.discovered,
                                         progress.current_operation[:11],
                                         progress.current_file[-60:])
        sys.stdout.write("\r" + line.ljust(100))
        sys.stdout.flush()


def cmd_scan(args) -> int:
    settings = build_settings(args)
    db = Database(settings.resolved_db())
    initialize(settings, db, plugin_dirs=getattr(args, "plugin_dir", None))
    core = ExtractorCore(settings, db, ConsoleProgress(args.quiet))
    result = core.analyze(args.source)
    print()
    print_summary(result)
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2, default=str),
                                   encoding="utf-8")
        print("report written to %s" % args.json)
    core.cleanup_temp()
    return 0


def cmd_extract(args) -> int:
    settings = build_settings(args)
    db = Database(settings.resolved_db())
    initialize(settings, db, plugin_dirs=getattr(args, "plugin_dir", None))
    core = ExtractorCore(settings, db, ConsoleProgress(args.quiet))
    result = core.analyze(args.source)
    print()
    print_summary(result)
    only = None
    if args.characters_only:
        only = [g.model_asset.id for g in result.characters]
        print("exporting %d character(s) only" % len(only))
    report = core.extract(result, selection=only)
    print("exported : %d" % len(report["exported"]))
    print("failed   : %d" % len(report["failed"]))
    print("skipped  : %d" % len(report["skipped"]))
    print("output   : %s" % report["output"])
    print("report   : %s" % report.get("report"))
    core.cleanup_temp()
    return 0


def cmd_detect(args) -> int:
    initialize(Settings())
    ctx = FileContext(args.file)
    print(json.dumps(identify(ctx, deep=True).to_dict(), indent=2, default=str))
    return 0


def cmd_inspect(args) -> int:
    initialize(Settings())
    path = Path(args.file)
    if path.is_dir():
        print(json.dumps([e.to_dict() for e in detect_engine(path)], indent=2))
        return 0
    ctx = FileContext(path)
    detection = identify(ctx, deep=True)
    out = {"file": str(path), "size": ctx.size, "detection": detection.to_dict(),
           "entropy_profile": entropy_profile(path)}
    hits = EmbeddedAssetScanner().scan(path)
    out["embedded"] = [h.to_dict() for h in hits[:100]]
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_list_plugins(args) -> int:
    from app.plugins import manager
    initialize(Settings())
    loaded = {p.name: p for p in plugins()}
    rows = []
    for info in manager.catalogue():
        run = loaded.get(info.name)
        info.loaded = bool(run and run.loaded)
        info.error = (run.error if run else None) or (
            "; ".join(info.problems) if info.problems else None)
        rows.append(info.to_dict())
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
        return 0
    print("%-28s %-8s %-7s %-9s %-12s %s" % ("NAME", "VERSION", "SOURCE", "STATE",
                                             "ENGINES", "NOTE"))
    for p in rows:
        if not p["enabled"]:
            state = "disabled"
        elif p["loaded"]:
            state = "ok"
        else:
            state = "FAILED"
        print("%-28s %-8s %-7s %-9s %-12s %s" % (
            p["name"][:28], p["version"][:8], p["source"], state,
            (",".join(p["engines"]) or "-")[:12], p["error"] or ""))
    print("\nuser plugin folder: %s" % manager.user_dir())
    return 0


def cmd_plugins_install(args) -> int:
    from app.plugins import manager
    initialize(Settings())
    source = Path(args.source)
    if not args.yes:
        print("A plugin is Python code and runs with this application's permissions.")
        print("Install only plugins you trust: %s" % source)
        answer = input("Continue? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted")
            return 1
    try:
        info = manager.install(source, force=args.force)
    except Exception as exc:
        print("install failed: %s" % exc)
        return 2
    print("installed %s %s -> %s" % (info.name, info.version, info.path))
    if args.check:
        info = manager.probe(info.path)
        print("registered: %s" % json.dumps(info.registered))
    return 0


def cmd_plugins_remove(args) -> int:
    from app.plugins import manager
    initialize(Settings())
    if manager.remove(args.name):
        print("removed %s" % args.name)
        return 0
    print("no user plugin named %r (built-ins can be disabled, not removed)" % args.name)
    return 2


def cmd_plugins_toggle(args) -> int:
    from app.plugins import manager
    initialize(Settings())
    enable = args.command_action == "enable"
    if manager.set_enabled(args.name, enable):
        print("%s %s" % ("enabled" if enable else "disabled", args.name))
        return 0
    print("no plugin named %r" % args.name)
    return 2


def cmd_plugins_new(args) -> int:
    from app.plugins import manager
    try:
        target = manager.scaffold(args.name, args.output, engine=args.engine or "",
                                  extensions=(args.extensions or "").split(",")
                                  if args.extensions else None)
    except Exception as exc:
        print("could not create the plugin: %s" % exc)
        return 2
    print("created %s" % target)
    print("next:")
    print("  python %s" % (target / "make_sample.py"))
    print("  extractor scan %s" % target)
    print("  extractor plugins check %s" % target)
    return 0


def cmd_plugins_check(args) -> int:
    from app.plugins import manager
    initialize(Settings(), load_plugins=False)
    path = Path(args.path)
    problems = manager.validate(path)
    if problems:
        print("%s: not valid" % path)
        for problem in problems:
            print("  - %s" % problem)
        return 2
    print("%s: manifest and entry point are valid" % path)
    if args.load:
        try:
            info = manager.probe(path)
        except Exception as exc:
            print("loading failed: %s" % exc)
            return 2
        print("loads cleanly, registers: %s" % json.dumps(info.registered))
    return 0


def cmd_plugins_reload(args) -> int:
    from app.plugins.loader import reload_all
    initialize(Settings(), load_plugins=False)
    infos = reload_all(settings=Settings())
    ok = [i.name for i in infos if i.loaded]
    print("loaded: %s" % (", ".join(ok) or "none"))
    for info in infos:
        if not info.loaded:
            print("  %s: %s" % (info.name, info.error))
    return 0


def cmd_formats(args) -> int:
    initialize(Settings())
    print("export formats: %s" % ", ".join(available_formats()))
    return 0


def print_summary(result) -> None:
    counts = result.counts()
    engines = ", ".join("%s (%.0f%%)" % (e.engine, e.confidence * 100)
                        for e in result.engines[:3]) or "unknown"
    print("source     : %s" % result.source)
    print("engine     : %s" % engines)
    for key in ("files_scanned", "containers", "models", "characters", "animations",
                "textures", "duplicates", "unsupported", "failed"):
        print("%-11s: %s" % (key.replace("_", " "), counts.get(key, 0)))
    print("duration   : %.1fs" % result.duration)
    if result.characters:
        print("\ncharacters:")
        for group in result.characters[:20]:
            print("  %-40s %3.0f%%  %d anim  %d tex" % (
                group.name[:40], group.confidence * 100, len(group.animations),
                len(group.textures)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="extractor",
                                description="Universal 3D game asset extractor")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--output", "-o", help="output directory (default ./Extracted)")
        sp.add_argument("--temp", help="temporary directory")
        sp.add_argument("--workers", type=int, help="worker threads")
        sp.add_argument("--deep", action="store_true", help="analyse unknown files heuristically")
        sp.add_argument("--aggressive", action="store_true",
                        help="deep + expensive binary scanning (slow)")
        sp.add_argument("--max-depth", type=int, help="max archive nesting depth")
        sp.add_argument("--oodle-dll",
                        help="path to an oo2core library you already have; needed for "
                             "Oodle-compressed game data (many Unreal titles)")
        sp.add_argument("--plugin-dir", action="append",
                        help="extra plugin folder to load (repeatable); handy for "
                             "testing a plugin before installing it")
        sp.add_argument("--quiet", "-q", action="store_true")
        sp.add_argument("--verbose", "-v", action="store_true")

    scan = sub.add_parser("scan", help="analyse a folder, no export")
    scan.add_argument("source")
    scan.add_argument("--json", help="write the report to this JSON file")
    common(scan)
    scan.set_defaults(func=cmd_scan)

    ext = sub.add_parser("extract", help="analyse and export assets")
    ext.add_argument("source")
    ext.add_argument("--format", "-f", default="glb",
                     help="comma separated: glb,gltf,obj,dae")
    ext.add_argument("--characters-only", action="store_true")
    ext.add_argument("--keep-duplicates", action="store_true")
    ext.add_argument("--overwrite", action="store_true")
    ext.add_argument("--up-axis", choices=["y", "z"], default="y")
    ext.add_argument("--handedness", choices=["right", "left"], default="right")
    ext.add_argument("--unit-scale", type=float, default=1.0)
    common(ext)
    ext.set_defaults(func=cmd_extract)

    det = sub.add_parser("detect", help="identify one file")
    det.add_argument("file")
    det.set_defaults(func=cmd_detect)

    ins = sub.add_parser("inspect", help="deep inspection of one file or folder")
    ins.add_argument("file")
    ins.set_defaults(func=cmd_inspect)

    lp = sub.add_parser("list-plugins", help="list discovered plugins (alias of 'plugins list')")
    lp.add_argument("--json", action="store_true")
    lp.set_defaults(func=cmd_list_plugins)

    pl = sub.add_parser("plugins", help="manage format plugins")
    pl_sub = pl.add_subparsers(dest="command_action", required=True)

    p_list = pl_sub.add_parser("list", help="show installed plugins and their state")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list_plugins)

    p_new = pl_sub.add_parser("new", help="scaffold a working plugin skeleton")
    p_new.add_argument("name", help='e.g. "Tomb Raider PS2"')
    p_new.add_argument("--output", "-o", help="where to create it (default: here)")
    p_new.add_argument("--engine", help="engine name for the manifest")
    p_new.add_argument("--extensions", help="comma separated list, e.g. tr2,drm")
    p_new.set_defaults(func=cmd_plugins_new)

    p_check = pl_sub.add_parser("check", help="validate a plugin folder")
    p_check.add_argument("path")
    p_check.add_argument("--load", action="store_true",
                         help="also import it and report what it registers")
    p_check.set_defaults(func=cmd_plugins_check)

    p_install = pl_sub.add_parser("install", help="install a plugin folder or .zip")
    p_install.add_argument("source")
    p_install.add_argument("--force", action="store_true", help="overwrite an existing one")
    p_install.add_argument("--yes", "-y", action="store_true", help="skip the trust prompt")
    p_install.add_argument("--check", action="store_true", help="load it after installing")
    p_install.set_defaults(func=cmd_plugins_install)

    p_remove = pl_sub.add_parser("remove", help="remove a user-installed plugin")
    p_remove.add_argument("name")
    p_remove.set_defaults(func=cmd_plugins_remove)

    for action in ("enable", "disable"):
        p_toggle = pl_sub.add_parser(action, help="%s a plugin" % action)
        p_toggle.add_argument("name")
        p_toggle.set_defaults(func=cmd_plugins_toggle)

    p_reload = pl_sub.add_parser("reload", help="reload every plugin without restarting")
    p_reload.set_defaults(func=cmd_plugins_reload)

    fm = sub.add_parser("formats", help="list export formats")
    fm.set_defaults(func=cmd_formats)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except FileNotFoundError as exc:
        print("error: %s" % exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
