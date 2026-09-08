"""Export stage: internal representation -> files on disk + manifests + report."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

from app.config import DuplicatePolicy, OverwritePolicy
from app.core.conversion.normalize import normalize
from app.core.fs_safety import sanitize_component
from app.core.logging_setup import get_logger
from app.core.types import AssetType, Classification, ModelAsset
from app.exporters import get_exporter
from app.formats.textures import image as image_tools

log = get_logger("export")

CLASS_DIR = {
    Classification.CHARACTER: "Characters",
    Classification.NPC: "Characters",
    Classification.CREATURE: "Characters",
    Classification.WEAPON: "Props",
    Classification.PROP: "Props",
    Classification.ENVIRONMENT: "Environment",
    Classification.UNKNOWN: "Models",
}


def unique_path(path: Path, policy: OverwritePolicy) -> Optional[Path]:
    if not path.exists():
        return path
    if policy == OverwritePolicy.OVERWRITE:
        return path
    if policy == OverwritePolicy.SKIP:
        return None
    i = 2
    while True:
        candidate = path.with_name("%s_%02d%s" % (path.stem, i, path.suffix))
        if not candidate.exists():
            return candidate
        i += 1


def export_scan(core, result, formats: list, selection: Optional[list] = None) -> dict:
    """Export every model asset (optionally filtered by `selection` of asset ids)."""
    settings = core.settings
    out_root = settings.resolved_output()
    report = {"exported": [], "failed": [], "skipped": [], "formats": formats,
              "output": str(out_root)}
    selected = set(selection) if selection else None
    coords = settings.coordinates

    grouped = {g.model_asset.id: g for g in result.characters}

    for asset in result.graph.by_type(AssetType.MODEL):
        if selected and asset.id not in selected:
            continue
        if "duplicate_of" in asset.metadata and settings.duplicate_policy == DuplicatePolicy.SKIP:
            report["skipped"].append({"asset": asset.name, "reason": "duplicate"})
            continue
        model: ModelAsset = asset.payload
        if not isinstance(model, ModelAsset):
            continue
        if core.control.cancelled:
            break

        group = grouped.get(asset.id)
        folder_kind = CLASS_DIR.get(asset.classification, "Models")
        safe_name = sanitize_component(asset.name or "model") or "model"
        target_dir = out_root / folder_kind / safe_name
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            normalize(model, coords.up_axis, coords.handedness, coords.unit_scale)
        except Exception as exc:
            log.warning("normalisation failed for %s: %s", asset.name, exc)

        written: list = []
        errors: list = []
        for fmt in formats:
            exporter = get_exporter(fmt)
            if exporter is None:
                errors.append("no exporter for format %r" % fmt)
                continue
            res = exporter.export(model, target_dir, safe_name)
            if res.ok:
                written.extend(res.files)
                for warn in res.warnings:
                    log.info("%s [%s]: %s", asset.name, fmt, warn)
                core.db.record_export(result.scan_id, asset.id, str(res.files[0]), fmt, "ok")
            else:
                errors.append("%s: %s" % (fmt, res.error))
                core.db.record_export(result.scan_id, asset.id, str(target_dir), fmt,
                                      "failed", res.error)

        tex_dir = target_dir / "textures"
        texture_files = []
        for mat in model.materials:
            for tex in mat.textures.values():
                p = image_tools.write_texture(tex, tex_dir)
                if p:
                    texture_files.append(p)
        if group:
            for tex_asset in group.textures:
                payload = tex_asset.payload
                if payload is None:
                    continue
                p = image_tools.write_texture(payload, tex_dir)
                if p:
                    texture_files.append(p)

        anim_files = []
        if group and group.animations:
            anim_dir = target_dir / "animations"
            glb = get_exporter("glb")
            for anim_asset in group.animations:
                clip = anim_asset.payload
                if clip is None or anim_asset.metadata.get("embedded_in") == asset.id:
                    continue          # already inside the character's own GLB
                holder = ModelAsset(name=clip.name, skeleton=model.skeleton,
                                    animations=[clip], source_format=model.source_format)
                res = glb.export(holder, anim_dir, sanitize_component(clip.name))
                if res.ok:
                    anim_files.extend(res.files)

        if settings.keep_raw and asset.metadata.get("path"):
            raw_src = Path(asset.metadata["path"])
            if raw_src.exists():
                raw_dir = out_root / "Raw" / folder_kind
                raw_dir.mkdir(parents=True, exist_ok=True)
                dest = unique_path(raw_dir / raw_src.name, settings.overwrite_policy)
                if dest:
                    try:
                        shutil.copy2(raw_src, dest)
                    except OSError as exc:
                        log.debug("raw copy failed: %s", exc)

        manifest = build_manifest(asset, model, group, written, texture_files, anim_files)
        (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                                  encoding="utf-8")

        entry = {"asset": asset.name, "type": asset.classification.value,
                 "dir": str(target_dir), "files": [str(f) for f in written],
                 "confidence": round(asset.confidence, 3)}
        if errors:
            entry["errors"] = errors
        (report["exported"] if written else report["failed"]).append(entry)

    export_loose_textures(core, result, out_root, report)
    write_reports(core, result, out_root, report)
    return report


def build_manifest(asset, model: ModelAsset, group, written: list, textures: list,
                   animations: list) -> dict:
    embedded = [c.name for c in model.animations]
    external = [Path(a).stem for a in animations]
    return {
        "name": asset.name,
        "source": asset.source,
        "format": asset.format_name,
        "engine": asset.engine or "Unknown",
        "asset_type": asset.classification.value.lower(),
        "confidence": round(asset.confidence, 3),
        "mesh": [Path(f).name for f in written],
        "vertices": model.vertex_count,
        "triangles": model.triangle_count,
        "skinned": model.has_skinning,
        "bones": len(model.skeleton.bones) if model.skeleton else 0,
        "skeleton": model.skeleton.name if model.skeleton else None,
        "coordinate_system": model.coordinate_system,
        "original_coordinate_system": model.metadata.get("original_coordinate_system"),
        "unit_scale": model.unit_scale,
        "animations": embedded + external,
        "textures": [Path(t).name for t in textures],
        "validation": asset.metadata.get("validation"),
        "classification_evidence": asset.metadata.get("classification_evidence"),
        "group_confidence": round(group.confidence, 3) if group else None,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def export_loose_textures(core, result, out_root: Path, report: dict) -> None:
    """Textures not attached to any exported model still reach the user."""
    linked = set()
    for group in result.characters:
        linked.update(t.id for t in group.textures)
    tex_dir = out_root / "Textures"
    count = 0
    for asset in result.graph.by_type(AssetType.TEXTURE):
        if asset.id in linked or "duplicate_of" in asset.metadata:
            continue
        payload = asset.payload
        if payload is None:
            continue
        if image_tools.write_texture(payload, tex_dir):
            count += 1
    report["loose_textures"] = count


def write_reports(core, result, out_root: Path, report: dict) -> None:
    reports_dir = out_root / "Reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    data = result.to_dict()
    data["export"] = {k: v for k, v in report.items() if k != "graph"}
    data["scan_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (reports_dir / "report.json").write_text(json.dumps(data, indent=2, default=str),
                                             encoding="utf-8")
    (reports_dir / "asset_graph.json").write_text(
        json.dumps(result.graph.to_dict(), indent=2, default=str), encoding="utf-8")
    (reports_dir / "report.html").write_text(render_html(data), encoding="utf-8")
    report["report"] = str(reports_dir / "report.html")


def render_html(data: dict) -> str:
    counts = data.get("counts", {})
    rows = "".join("<tr><th>%s</th><td>%s</td></tr>" % (k.replace("_", " ").title(), v)
                   for k, v in counts.items())
    engines = "".join("<li>%s &mdash; %.0f%%</li>" % (e["engine"], e["confidence"] * 100)
                      for e in data.get("engines", []))
    chars = "".join("<li>%s <span class=c>%.0f%%</span> &mdash; %d animations, %d textures</li>"
                    % (c["name"], c["confidence"] * 100, len(c["animations"]),
                       len(c["textures"])) for c in data.get("characters", []))
    unsup = "".join("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                    % (u["file"], u["detected"], u["reason"], u.get("suggested_plugin", ""))
                    for u in data.get("unsupported", [])[:300])
    errors = "".join("<tr><td>%s</td><td>%s</td><td>%s</td></tr>"
                     % (e.get("stage"), e.get("file"), e.get("message"))
                     for e in data.get("errors", [])[:300])
    return """<!doctype html><meta charset="utf-8">
<title>Extraction report</title>
<style>
body{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:1100px;color:#1a1a1a}
h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:2rem}
table{border-collapse:collapse;width:100%%} td,th{border:1px solid #ddd;padding:.35rem .6rem;
text-align:left;vertical-align:top;font-size:13px}
th{background:#f5f5f5;width:14rem} .c{color:#0a7}
code{background:#f5f5f5;padding:.1rem .3rem}
.wrap{overflow-x:auto}
</style>
<h1>Universal 3D Asset Extractor &mdash; report</h1>
<p>Source: <code>%(source)s</code><br>Scan date: %(scan_date)s &middot;
duration %(duration_sec)ss</p>
<h2>Summary</h2><table>%(rows)s</table>
<h2>Engine detection</h2><ul>%(engines)s</ul>
<h2>Characters</h2><ul>%(chars)s</ul>
<h2>Unsupported formats</h2><div class="wrap"><table>
<tr><th>File</th><th>Detected</th><th>Reason</th><th>Suggested plugin</th></tr>%(unsup)s</table></div>
<h2>Errors</h2><div class="wrap"><table>
<tr><th>Stage</th><th>File</th><th>Message</th></tr>%(errors)s</table></div>
""" % {"source": data.get("source", ""), "scan_date": data.get("scan_date", ""),
       "duration_sec": data.get("duration_sec", 0), "rows": rows, "engines": engines or "<li>none</li>",
       "chars": chars or "<li>none</li>", "unsup": unsup, "errors": errors}
