"""The pipeline: scan -> identify -> extract -> parse -> graph -> characters.

This is the only place that knows the whole flow. It never imports the UI.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.config import ScanMode, Settings
from app.core import extraction, validation
from app.core.detector import FileContext, identify
from app.core.detector.engine import detect_engine
from app.core.detector.heuristics import EmbeddedAssetScanner, entropy_profile
from app.core.graph.asset_graph import AssetGraph
from app.core.graph.character import classify
from app.core.hashing import quick_key, sha256_file
from app.core.logging_setup import get_logger, unsupported_logger
from app.core.pipeline.jobs import Control, ExtractionJob, Progress, WorkerPool
from app.core.pipeline.states import State
from app.core.types import (Asset, AssetType, Category, Classification, DetectionResult,
                            Relation, TextureAsset)
from app.database.db import Database
from app.formats.models import parse_model
from app.formats.textures import image as image_tools

log = get_logger("pipeline")

MODEL_CATEGORIES = {Category.MODEL}
TEXTURE_CATEGORIES = {Category.TEXTURE}
ANIMATION_CATEGORIES = {Category.ANIMATION}


@dataclass
class UnsupportedItem:
    path: str
    detected: str
    reason: str
    category: str = ""
    suggested_plugin: str = ""

    def to_dict(self) -> dict:
        return {"file": self.path, "detected": self.detected, "reason": self.reason,
                "category": self.category, "suggested_plugin": self.suggested_plugin}


@dataclass
class ScanResult:
    scan_id: int = -1
    source: str = ""
    graph: AssetGraph = field(default_factory=AssetGraph)
    characters: list = field(default_factory=list)
    engines: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    progress: Progress = field(default_factory=Progress)
    cancelled: bool = False
    duration: float = 0.0

    def counts(self) -> dict:
        c = self.graph.counts()
        return {
            "files_scanned": self.progress.scanned,
            "containers": self.progress.containers,
            "models": c.get("model", 0),
            "animations": c.get("animation", 0),
            "textures": c.get("texture", 0),
            "characters": len(self.characters),
            "failed": self.progress.failed,
            "unsupported": len(self.unsupported),
            "duplicates": self.progress.skipped_duplicates,
        }

    def to_dict(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "source": self.source,
            "engines": [e.to_dict() if hasattr(e, "to_dict") else e for e in self.engines],
            "counts": self.counts(),
            "characters": [g.to_dict() for g in self.characters],
            "unsupported": [u.to_dict() for u in self.unsupported][:500],
            "errors": self.errors[:500],
            "duration_sec": round(self.duration, 2),
            "cancelled": self.cancelled,
        }


class ExtractorCore:
    """Headless core. GUI and CLI are both clients of this class."""

    def __init__(self, settings: Settings, db: Optional[Database] = None,
                 progress_callback: Optional[Callable] = None):
        self.settings = settings
        self.db = db or Database(settings.resolved_db())
        self.control = Control()
        self.progress = Progress()
        self.progress_callback = progress_callback
        self.graph = AssetGraph()
        self.result = ScanResult()
        self._lock = threading.RLock()
        self._seen_container_hashes: set = set()
        self._seen_real_paths: set = set()
        self._asset_hashes: dict = {}
        self._job_counter = 0
        self._pool: Optional[WorkerPool] = None
        self._temp_root: Optional[Path] = None
        self._resume_scan_id: Optional[int] = None
        self._resume_done: set = set()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def analyze(self, source=None) -> ScanResult:
        source = Path(source or self.settings.source_dir).resolve()
        if not source.exists():
            raise FileNotFoundError("source not found: %s" % source)
        started = time.time()

        self.graph = AssetGraph()
        self.progress = Progress()
        self.result = ScanResult(source=str(source), graph=self.graph, progress=self.progress)
        self._temp_root = self.settings.resolved_temp() / ("job_%d" % int(started))
        self._temp_root.mkdir(parents=True, exist_ok=True)

        scan_id = self._resume_scan_id or self.db.create_scan(
            str(source), str(self.settings.resolved_output()), self.settings.scan_mode.value)
        self.result.scan_id = scan_id

        self._emit("engine detection")
        try:
            self.result.engines = detect_engine(source)
        except Exception as exc:
            log.warning("engine detection failed: %s", exc)
        if self.result.engines:
            log.info("engine guess: %s", ", ".join("%s (%.0f%%)" % (e.engine, e.confidence * 100)
                                                   for e in self.result.engines[:3]))

        self._pool = WorkerPool(self.settings.workers, self._handle_job, self.control,
                                on_error=self._on_job_error)
        self._pool.start()
        try:
            self._walk_directory(source, depth=0, parent=None)
            self._pool.wait_idle()
        finally:
            self._pool.shutdown()

        self._emit("building asset graph")
        self.result.characters = self.graph.reconstruct_characters()
        self.progress.characters = len(self.result.characters)
        for group in self.result.characters:
            self.graph.link(group.model_asset.id, group.model_asset.id,
                            Relation.CHARACTER_MESH, group.confidence)

        self.result.cancelled = self.control.cancelled
        self.result.duration = time.time() - started
        engine = self.result.engines[0].engine if self.result.engines else None
        self.db.finish_scan(scan_id, "cancelled" if self.result.cancelled else "completed",
                            self.result.counts(), engine)
        self._emit("done")
        log.info("analyze finished in %.1fs: %s", self.result.duration, self.result.counts())
        return self.result

    def extract(self, result: Optional[ScanResult] = None, selection: Optional[list] = None,
                formats: Optional[list] = None) -> dict:
        """Export assets found by `analyze`. No re-scan."""
        result = result or self.result
        formats = formats or self.settings.export_formats
        from app.core.pipeline.exporting import export_scan
        return export_scan(self, result, formats, selection)

    def resume_scan(self, scan_id: int):
        """Continue an interrupted scan.

        Files already marked terminal in SQLite are skipped, and detection for
        unchanged files comes from the cache, so this fast-forwards rather than
        redoing the work. Note that in-memory asset payloads from the previous
        session are gone: assets are re-parsed as their files are revisited.
        """
        row = self.db.get_scan(scan_id)
        if row is None:
            raise ValueError("no scan with id %d" % scan_id)
        self._resume_scan_id = scan_id
        self._resume_done = {r["virtual_path"] for r in self.db.connect().execute(
            "SELECT virtual_path FROM files WHERE scan_id=? AND status IN "
            "('completed','skipped','unsupported')", (scan_id,)).fetchall()}
        log.info("resuming scan %d: %d file(s) already done", scan_id, len(self._resume_done))
        return self.analyze(row["source"])

    # -- control -----------------------------------------------------------
    def cancel(self) -> None:
        log.info("cancel requested")
        self.control.cancel()

    def pause(self) -> None:
        self.control.pause()
        self._persist_queue("paused")
        self._emit("paused")

    def _persist_queue(self, status: str) -> None:
        """Serialise the pending queue so the scan survives an app restart."""
        if self._pool is None or self.result.scan_id < 0:
            return
        try:
            stats = self.result.counts()
            stats["pending_jobs"] = self._pool.serialize_pending()
            self.db.finish_scan(self.result.scan_id, status, stats,
                                self.result.engines[0].engine if self.result.engines else None)
        except Exception as exc:
            log.debug("queue persistence failed: %s", exc)

    def resume(self) -> None:
        self.control.resume()
        self._emit("running")

    def cleanup_temp(self, force: bool = False) -> None:
        if not self._temp_root or not self._temp_root.exists():
            return
        if self.progress.failed and self.settings.keep_temp_on_error and not force:
            log.info("keeping temp dir for debugging: %s", self._temp_root)
            return
        shutil.rmtree(self._temp_root, ignore_errors=True)

    # ------------------------------------------------------------------
    # scanning
    # ------------------------------------------------------------------
    def _walk_directory(self, root: Path, depth: int, parent: Optional[int],
                        virtual_prefix: str = "") -> None:
        """Streaming traversal. Directory recursion is depth-limited and loop-safe."""
        limits = self.settings.limits
        stack = [(root, 0)]
        while stack:
            if self.control.cancelled:
                return
            current, dir_depth = stack.pop()
            if dir_depth > limits.max_scan_depth:
                continue
            try:
                entries = list(os.scandir(current))
            except OSError as exc:
                self._record_error("scan", current, str(exc))
                continue
            for entry in entries:
                if self.control.cancelled:
                    return
                try:
                    if entry.is_symlink():
                        continue                       # never follow symlinks
                    if entry.is_dir(follow_symlinks=False):
                        real = os.path.realpath(entry.path)
                        if real in self._seen_real_paths:
                            continue
                        self._seen_real_paths.add(real)
                        stack.append((Path(entry.path), dir_depth + 1))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    size = entry.stat(follow_symlinks=False).st_size
                    if size > limits.max_single_file_size:
                        self._note_unsupported(entry.path, "oversized",
                                               "file larger than the configured limit")
                        continue
                    rel = os.path.relpath(entry.path, root)
                    vpath = (virtual_prefix + "/" + rel.replace("\\", "/")) if virtual_prefix \
                        else rel.replace("\\", "/")
                    if vpath in self._resume_done:
                        continue                   # finished in a previous session
                    job = ExtractionJob(input_file=Path(entry.path), virtual_path=vpath,
                                        depth=depth, parent_id=parent,
                                        priority=self._priority(entry.name))
                    with self._lock:
                        self.progress.discovered += 1
                    assert self._pool is not None
                    self._pool.submit(job)
                except OSError as exc:
                    self._record_error("scan", entry.path, str(exc))

    @staticmethod
    def _priority(name: str) -> int:
        low = name.lower()
        if low.endswith((".glb", ".gltf", ".fbx", ".obj", ".dae", ".dff")):
            return 80
        if low.endswith((".pak", ".zip", ".iso", ".bin", ".assets", ".bundle", ".pck")):
            return 70
        return 50

    # ------------------------------------------------------------------
    # per-file work
    # ------------------------------------------------------------------
    def _handle_job(self, job: ExtractionJob) -> None:
        if self.control.cancelled:
            return
        ctx = FileContext(job.input_file, header_size=self.settings.limits.header_read_size,
                          depth=job.depth, container=job.virtual_path)
        self.progress.current_file = job.virtual_path
        self.progress.current_operation = "identifying"

        detection = self._detect_cached(ctx)
        job.detected_format = detection.format_name
        job.status = State.IDENTIFIED

        file_id = self._persist_file(job, ctx, detection)
        job.file_id = file_id

        try:
            if extraction.is_container(detection) and detection.category != Category.TEXTURE:
                self._process_container(job, ctx, detection)
            elif detection.category in MODEL_CATEGORIES:
                self._process_model(job, ctx, detection)
            elif detection.category in TEXTURE_CATEGORIES:
                self._process_texture(job, ctx, detection)
            elif detection.category in ANIMATION_CATEGORIES:
                self._process_animation(job, ctx, detection)
            else:
                self._process_unknown(job, ctx, detection)
            if job.status not in (State.FAILED, State.UNSUPPORTED):
                job.status = State.COMPLETED
        except Exception as exc:
            job.status = State.FAILED
            job.error = str(exc)
            with self._lock:
                self.progress.failed += 1
            self._record_error("process", job.input_file, str(exc), file_id)
            log.warning("failed on %s: %s", job.virtual_path, exc)
        finally:
            with self._lock:
                self.progress.scanned += 1
            if file_id > 0:
                self.db.set_file_status(file_id, job.status.value)
            self._emit(job.status.value)

    def _detect_cached(self, ctx: FileContext) -> DetectionResult:
        try:
            key = quick_key(ctx.path)
        except OSError:
            key = ""
        if key:
            row = self.db.cache_get(str(ctx.path), key)
            if row is not None:
                return DetectionResult(True, row["format"], row["confidence"] or 0.0,
                                       Category(row["category"] or "unknown"),
                                       {"engine": row["engine"], "cached": True},
                                       row["detector"] or "cache")
        detection = identify(ctx, deep=self.settings.scan_mode != ScanMode.STANDARD)
        if key:
            try:
                self.db.cache_put(str(ctx.path), key, detection)
            except Exception as exc:
                log.debug("cache write failed: %s", exc)
        return detection

    # -- containers --------------------------------------------------------
    def _process_container(self, job: ExtractionJob, ctx: FileContext,
                           detection: DetectionResult) -> None:
        with self._lock:
            self.progress.containers += 1
        if job.depth >= self.settings.limits.max_nesting_depth:
            self._note_unsupported(job.virtual_path, detection.format_name,
                                   "max nesting depth %d reached"
                                   % self.settings.limits.max_nesting_depth)
            job.status = State.SKIPPED
            return

        digest = self._content_hash(ctx.path)
        with self._lock:
            if digest in self._seen_container_hashes:
                self.progress.skipped_duplicates += 1
                job.status = State.SKIPPED
                return
            self._seen_container_hashes.add(digest)

        job.status = State.EXTRACTING
        self.progress.current_operation = "extracting"
        with self._lock:
            self._job_counter += 1
            job_dir = self._temp_root / ("c%05d_%s" % (self._job_counter, ctx.path.stem[:40]))
        res = extraction.extract(ctx, detection, job_dir, self.settings.limits)
        job.output_files = res.files

        if not res.ok:
            reason = res.unsupported_reason or "; ".join(res.errors[:2]) or "extraction failed"
            self._note_unsupported(job.virtual_path, detection.format_name, reason,
                                   detection.category.value,
                                   self._suggest_plugin(detection))
            job.status = State.UNSUPPORTED if res.unsupported_reason else State.FAILED
            if job.status == State.FAILED:
                with self._lock:
                    self.progress.failed += 1
            shutil.rmtree(job_dir, ignore_errors=True)
            return

        for err in res.errors[:20]:
            self._record_error("extract", job.virtual_path, err, job.file_id)

        container_asset = Asset(name=ctx.path.name, type=AssetType.CONTAINER,
                                source=job.virtual_path, confidence=detection.confidence,
                                format_name=detection.format_name, engine=detection.engine,
                                content_hash=digest,
                                metadata={"entries": res.entries, "extractor": res.extractor})
        self.graph.add(container_asset)
        self.db.insert_asset(self.result.scan_id, container_asset, job.file_id)

        # re-scan what came out; files already on disk (cue) keep their own depth
        if job_dir.exists():
            self._walk_directory(job_dir, depth=job.depth + 1, parent=job.file_id,
                                 virtual_prefix=job.virtual_path + "!")

    # -- models ------------------------------------------------------------
    def _process_model(self, job: ExtractionJob, ctx: FileContext,
                       detection: DetectionResult) -> None:
        job.status = State.PARSING
        self.progress.current_operation = "parsing model"
        try:
            model = parse_model(ctx, detection)
        except Exception as exc:
            self._note_unsupported(job.virtual_path, detection.format_name,
                                   "parser error: %s" % exc, detection.category.value,
                                   self._suggest_plugin(detection))
            job.status = State.UNSUPPORTED
            return
        if model is None:
            self._note_unsupported(job.virtual_path, detection.format_name,
                                   "no parser available for this format",
                                   detection.category.value, self._suggest_plugin(detection))
            job.status = State.UNSUPPORTED
            return

        job.status = State.VALIDATING
        report = validation.validate_model(model)
        conf = validation.confidence(detection, report)
        classification, class_conf, evidence = classify(model, job.virtual_path)

        digest = self._content_hash(ctx.path)
        asset = Asset(name=model.name or ctx.path.stem, type=AssetType.MODEL,
                      source=job.virtual_path, confidence=conf,
                      classification=classification, format_name=detection.format_name,
                      engine=detection.engine, payload=model, content_hash=digest,
                      metadata={"validation": report.to_dict(),
                                "classification_confidence": round(class_conf, 3),
                                "classification_evidence": evidence,
                                "vertices": model.vertex_count,
                                "triangles": model.triangle_count,
                                "skinned": model.has_skinning,
                                "bones": len(model.skeleton.bones) if model.skeleton else 0,
                                "path": str(ctx.path),
                                "skeleton_name": model.skeleton.name if model.skeleton else None})
        self.graph.add(asset)
        self.db.insert_asset(self.result.scan_id, asset, job.file_id)
        with self._lock:
            self.progress.models += 1
            if "duplicate_of" in asset.metadata:
                self.progress.skipped_duplicates += 1

        if model.skeleton and model.skeleton.bones:
            skel = Asset(name=model.skeleton.name, type=AssetType.SKELETON,
                         source=job.virtual_path, confidence=conf, payload=model.skeleton,
                         format_name=detection.format_name,
                         metadata={"bones": len(model.skeleton.bones)})
            self.graph.add(skel)
            self.graph.link(asset.id, skel.id, Relation.MESH_SKELETON)
            self.db.insert_asset(self.result.scan_id, skel, job.file_id)

        for clip in model.animations:
            anim = Asset(name=clip.name, type=AssetType.ANIMATION, source=job.virtual_path,
                         confidence=conf, payload=clip, format_name=detection.format_name,
                         metadata={"duration": clip.duration, "tracks": len(clip.tracks),
                                   "embedded_in": asset.id})
            self.graph.add(anim)
            self.graph.link(asset.id, anim.id, Relation.CHARACTER_ANIMATION)
            self.db.insert_asset(self.result.scan_id, anim, job.file_id)
            with self._lock:
                self.progress.animations += 1

        for mat in model.materials:
            mat_asset = Asset(name=mat.name, type=AssetType.MATERIAL, source=job.virtual_path,
                              confidence=conf, payload=mat, format_name=detection.format_name)
            self.graph.add(mat_asset)
            self.graph.link(asset.id, mat_asset.id, Relation.MESH_MATERIAL)
            for usage, tex in mat.textures.items():
                tex_asset = Asset(name=tex.name, type=AssetType.TEXTURE,
                                  source=tex.path or job.virtual_path, confidence=conf,
                                  payload=tex, format_name=tex.source_format,
                                  metadata={"usage": usage, "path": tex.path,
                                            "embedded": bool(tex.data)})
                self.graph.add(tex_asset)
                self.graph.link(mat_asset.id, tex_asset.id, Relation.MATERIAL_TEXTURE)
                with self._lock:
                    self.progress.textures += 1

    # -- textures ----------------------------------------------------------
    def _process_texture(self, job: ExtractionJob, ctx: FileContext,
                         detection: DetectionResult) -> None:
        job.status = State.PARSING
        tex = self._parse_texture_with_plugins(ctx, detection)
        if tex is None:
            tex = image_tools.read_header(ctx.path)
        if tex is None:
            tex = TextureAsset(name=ctx.path.stem, path=str(ctx.path),
                               source_format=detection.format_name,
                               usage=image_tools.guess_usage(ctx.path.name))
        digest = self._content_hash(ctx.path)
        asset = Asset(name=tex.name, type=AssetType.TEXTURE, source=job.virtual_path,
                      confidence=max(detection.confidence, 0.5) if tex.width else
                      detection.confidence * 0.8,
                      format_name=tex.source_format or detection.format_name,
                      engine=detection.engine, payload=tex, content_hash=digest,
                      metadata={"width": tex.width, "height": tex.height,
                                "channels": tex.channels, "alpha": tex.has_alpha,
                                "usage": tex.usage, "path": str(ctx.path)})
        self.graph.add(asset)
        self.db.insert_asset(self.result.scan_id, asset, job.file_id)
        with self._lock:
            self.progress.textures += 1
            if "duplicate_of" in asset.metadata:
                self.progress.skipped_duplicates += 1

    def _parse_texture_with_plugins(self, ctx: FileContext, detection: DetectionResult):
        """Plugin texture parsers get first refusal (TIM, TIM2, console formats)."""
        from app.core.registry import TEXTURE_PARSERS
        for parser in TEXTURE_PARSERS:
            try:
                if parser.can_parse(ctx, detection):
                    return parser.parse(ctx)
            except Exception as exc:
                log.debug("texture parser %s failed on %s: %s", parser.name, ctx.path, exc)
        return None

    def _process_animation(self, job: ExtractionJob, ctx: FileContext,
                           detection: DetectionResult) -> None:
        from app.core.registry import ANIMATION_PARSERS
        for parser in ANIMATION_PARSERS:
            try:
                if parser.can_parse(ctx, detection):
                    clip = parser.parse(ctx)
                    asset = Asset(name=clip.name, type=AssetType.ANIMATION,
                                  source=job.virtual_path, confidence=detection.confidence,
                                  payload=clip, format_name=detection.format_name,
                                  metadata={"tracks": len(clip.tracks)})
                    self.graph.add(asset)
                    self.db.insert_asset(self.result.scan_id, asset, job.file_id)
                    with self._lock:
                        self.progress.animations += 1
                    return
            except Exception as exc:
                log.debug("animation parser %s failed: %s", parser.name, exc)
        self._note_unsupported(job.virtual_path, detection.format_name,
                               "animation format has no parser yet",
                               detection.category.value, self._suggest_plugin(detection))
        job.status = State.UNSUPPORTED

    # -- unknown -----------------------------------------------------------
    def _process_unknown(self, job: ExtractionJob, ctx: FileContext,
                         detection: DetectionResult) -> None:
        mode = self.settings.scan_mode
        if mode == ScanMode.STANDARD:
            if detection.category == Category.UNKNOWN:
                self._note_unsupported(job.virtual_path, detection.format_name,
                                       "unknown format (run a Deep scan to analyse it)",
                                       detection.category.value)
            return

        self.progress.current_operation = "deep analysis"
        limits = self.settings.limits
        max_bytes = limits.embedded_scan_max_bytes
        if mode == ScanMode.AGGRESSIVE:
            max_bytes = max(max_bytes, min(ctx.size, 2 << 30))
        scanner = EmbeddedAssetScanner(max_bytes=max_bytes)
        try:
            hits = scanner.scan(ctx.path)
        except OSError as exc:
            self._record_error("deep_scan", ctx.path, str(exc), job.file_id)
            return

        if not hits:
            profile = entropy_profile(ctx.path) if mode == ScanMode.AGGRESSIVE else []
            self._note_unsupported(job.virtual_path, detection.format_name,
                                   "no known signature found inside the blob",
                                   detection.category.value)
            if profile:
                log.debug("entropy profile %s: %s", ctx.path.name, profile)
            return

        for hit in hits[:256]:
            log.info("embedded asset in %s at %s: %s", ctx.path.name, hex(hit.offset),
                     hit.format_name)
            asset = Asset(name="%s@%s" % (ctx.path.stem, hex(hit.offset)),
                          type=AssetType.TEXTURE if hit.category == Category.TEXTURE
                          else AssetType.UNKNOWN,
                          source="%s#%s" % (job.virtual_path, hex(hit.offset)),
                          confidence=0.5, format_name=hit.format_name,
                          metadata={"embedded": hit.to_dict(), "container": str(ctx.path)})
            self.graph.add(asset)
            self.db.insert_asset(self.result.scan_id, asset, job.file_id)
            if hit.category == Category.TEXTURE:
                with self._lock:
                    self.progress.textures += 1

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _content_hash(self, path) -> Optional[str]:
        try:
            size = Path(path).stat().st_size
        except OSError:
            return None
        limit = None if size <= (64 << 20) else (16 << 20)
        try:
            digest = sha256_file(path, limit=limit)
        except OSError:
            return None
        return digest if limit is None else "%s:%d" % (digest, size)

    def _persist_file(self, job: ExtractionJob, ctx: FileContext,
                      detection: DetectionResult) -> int:
        try:
            stat = ctx.path.stat()
            record = {
                "path": str(ctx.path), "virtual_path": job.virtual_path,
                "parent_file_id": job.parent_id, "depth": job.depth,
                "size": stat.st_size, "mtime": stat.st_mtime, "sha256": None,
                "format": detection.format_name, "category": detection.category.value,
                "confidence": detection.confidence, "engine": detection.engine,
                "status": State.IDENTIFIED.value, "detector": detection.detector,
                "metadata_json": None,
            }
            return self.db.upsert_file(self.result.scan_id, record)
        except Exception as exc:
            log.debug("persist file failed: %s", exc)
            return -1

    def _note_unsupported(self, path, detected: str, reason: str, category: str = "",
                          suggested: str = "") -> None:
        item = UnsupportedItem(str(path), detected, reason, category, suggested)
        with self._lock:
            self.result.unsupported.append(item)
            self.progress.unsupported += 1
        unsupported_logger().info("%s | %s | %s", path, detected, reason)

    @staticmethod
    def _suggest_plugin(detection: DetectionResult) -> str:
        engine = (detection.engine or "").lower()
        if engine:
            return "plugins/%s" % engine.replace(" ", "_")
        if detection.category == Category.GAME_CONTAINER:
            return "plugins/<game_name>"
        return ""

    def _record_error(self, stage: str, path, message: str, file_id: Optional[int] = None) -> None:
        entry = {"stage": stage, "file": str(path), "message": str(message)[:500]}
        with self._lock:
            self.result.errors.append(entry)
        try:
            self.db.record_error(self.result.scan_id, stage, str(path), message, file_id)
        except Exception:
            pass

    def _on_job_error(self, job: ExtractionJob, exc: Exception) -> None:
        with self._lock:
            self.progress.failed += 1
        self._record_error("worker", job.input_file, str(exc), job.file_id)

    def _emit(self, operation: str) -> None:
        self.progress.current_operation = operation
        if self.progress_callback:
            try:
                self.progress_callback(self.progress)
            except Exception:
                pass
