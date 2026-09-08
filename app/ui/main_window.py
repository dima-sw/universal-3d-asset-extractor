"""Main window. A thin client of ExtractorCore: it starts jobs and shows results."""
from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QSlider,
                               QSpinBox, QSplitter, QTabWidget, QTextEdit, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from app.bootstrap import initialize, plugins
from app.config import ScanMode, Settings
from app.core.pipeline.controller import ExtractorCore
from app.core.types import AssetType
from app.database.db import Database
from app.exporters import available_formats
from app.ui.viewer import ModelViewer
from app.ui.worker import ExportWorker, ScanWorker, run_in_thread

STAT_KEYS = [("Files scanned", "scanned"), ("Files remaining", "remaining"),
             ("Containers found", "containers"), ("Models found", "models"),
             ("Animations found", "animations"), ("Textures found", "textures"),
             ("Characters", "characters"), ("Duplicates skipped", "duplicates"),
             ("Unsupported", "unsupported"), ("Failed", "failed")]


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings = None):
        super().__init__()
        self.settings = settings or Settings()
        self.db = Database(self.settings.resolved_db())
        initialize(self.settings, self.db)
        self.core = ExtractorCore(self.settings, self.db)
        self.result = None
        self._thread: QThread = None
        self._worker = None

        self.setWindowTitle("Universal 3D Asset Extractor")
        self.resize(1280, 860)
        self._build_ui()
        self._build_menu()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        # --- source / output --------------------------------------------
        paths = QGroupBox("Source and output")
        form = QFormLayout(paths)
        self.source_edit = QLineEdit(self.settings.source_dir)
        self.output_edit = QLineEdit(self.settings.output_dir or str(Path.cwd() / "Extracted"))
        form.addRow("Source", self._with_browse(self.source_edit, directory=True))
        form.addRow("Output", self._with_browse(self.output_edit, directory=True))

        controls = QHBoxLayout()
        self.mode_box = QComboBox()
        self.mode_box.addItems(["Standard scan", "Deep scan", "Aggressive scan"])
        self.format_box = QComboBox()
        self.format_box.addItems(["glb", "gltf", "obj", "dae", "glb,obj"])
        self.workers_box = QSpinBox()
        self.workers_box.setRange(1, 64)
        self.workers_box.setValue(self.settings.workers)
        self.analyze_btn = QPushButton("Analyze")
        self.extract_btn = QPushButton("Extract")
        self.extract_chars_btn = QPushButton("Extract characters")
        self.pause_btn = QPushButton("Pause")
        self.cancel_btn = QPushButton("Cancel")
        self.extract_btn.setEnabled(False)
        self.extract_chars_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        for w in (QLabel("Mode"), self.mode_box, QLabel("Format"), self.format_box,
                  QLabel("Workers"), self.workers_box):
            controls.addWidget(w)
        controls.addStretch(1)
        for b in (self.analyze_btn, self.extract_btn, self.extract_chars_btn,
                  self.pause_btn, self.cancel_btn):
            controls.addWidget(b)
        form.addRow(controls)
        layout.addWidget(paths)

        # --- scan stats ---------------------------------------------------
        scan_box = QGroupBox("Scan")
        scan_layout = QVBoxLayout(scan_box)
        stats_row = QHBoxLayout()
        self.stat_labels = {}
        for title, key in STAT_KEYS:
            holder = QVBoxLayout()
            value = QLabel("0")
            f = QFont()
            f.setPointSize(15)
            f.setBold(True)
            value.setFont(f)
            caption = QLabel(title)
            caption.setStyleSheet("color: #888;")
            holder.addWidget(value)
            holder.addWidget(caption)
            stats_row.addLayout(holder)
            self.stat_labels[key] = value
        scan_layout.addLayout(stats_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.current_label = QLabel("idle")
        self.current_label.setStyleSheet("color: #888;")
        scan_layout.addWidget(self.progress)
        scan_layout.addWidget(self.current_label)
        layout.addWidget(scan_box)

        # --- results + viewer --------------------------------------------
        splitter = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Asset", "Type", "Confidence", "Details"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.itemSelectionChanged.connect(self._on_selection)
        splitter.addWidget(self.tree)

        right = QTabWidget()
        viewer_panel = QWidget()
        vlayout = QVBoxLayout(viewer_panel)
        self.viewer = ModelViewer()
        self.viewer.status_changed.connect(lambda s: self.viewer_status.setText(s))
        vlayout.addWidget(self.viewer, 1)

        view_controls = QHBoxLayout()
        self.render_box = QComboBox()
        self.render_box.addItems(["textured", "solid", "wireframe"])
        self.render_box.currentTextChanged.connect(self.viewer.set_render_mode)
        self.bones_check = QCheckBox("Bones")
        self.bones_check.toggled.connect(self._toggle_bones)
        self.anim_box = QComboBox()
        self.anim_box.currentIndexChanged.connect(self._on_animation_selected)
        self.play_btn = QPushButton("Play")
        self.pause_anim_btn = QPushButton("Pause")
        self.reset_btn = QPushButton("Reset")
        self.play_btn.clicked.connect(self.viewer.play)
        self.pause_anim_btn.clicked.connect(self.viewer.pause)
        self.reset_btn.clicked.connect(self._reset_animation)
        for w in (QLabel("Render"), self.render_box, self.bones_check, QLabel("Animation"),
                  self.anim_box, self.play_btn, self.pause_anim_btn, self.reset_btn):
            view_controls.addWidget(w)
        view_controls.addStretch(1)
        vlayout.addLayout(view_controls)

        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.sliderMoved.connect(self._scrub)
        self.viewer.frame_changed.connect(self._on_frame)
        self.time_label = QLabel("0.00 s")
        timeline_row = QHBoxLayout()
        timeline_row.addWidget(self.timeline, 1)
        timeline_row.addWidget(self.time_label)
        vlayout.addLayout(timeline_row)

        self.viewer_status = QLabel("no model loaded")
        self.viewer_status.setStyleSheet("color: #888;")
        vlayout.addWidget(self.viewer_status)
        right.addTab(viewer_panel, "3D preview")

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        right.addTab(self.details, "Details")

        self.unsupported_view = QTreeWidget()
        self.unsupported_view.setHeaderLabels(["File", "Detected", "Reason", "Suggested plugin"])
        right.addTab(self.unsupported_view, "Unsupported")

        self.errors_view = QTreeWidget()
        self.errors_view.setHeaderLabels(["Stage", "File", "Message"])
        right.addTab(self.errors_view, "Errors")

        from app.ui.plugins_panel import PluginsPanel
        self.plugins_panel = PluginsPanel(self.settings, self.db)
        right.addTab(self.plugins_panel, "Plugins")

        self.settings_panel = self._build_settings_panel()
        right.addTab(self.settings_panel, "Settings")
        splitter.addWidget(right)
        splitter.setSizes([460, 820])
        layout.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self.analyze_btn.clicked.connect(self.start_analyze)
        self.extract_btn.clicked.connect(lambda: self.start_extract(False))
        self.extract_chars_btn.clicked.connect(lambda: self.start_extract(True))
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.cancel_btn.clicked.connect(self.cancel)

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        self.up_axis_box = QComboBox()
        self.up_axis_box.addItems(["y", "z"])
        self.hand_box = QComboBox()
        self.hand_box.addItems(["right", "left"])
        self.unit_box = QDoubleSpinBox()
        self.unit_box.setRange(0.0001, 10000.0)
        self.unit_box.setDecimals(4)
        self.unit_box.setValue(1.0)
        self.nesting_box = QSpinBox()
        self.nesting_box.setRange(1, 64)
        self.nesting_box.setValue(self.settings.limits.max_nesting_depth)
        self.keep_raw_check = QCheckBox("Keep raw extracted assets")
        self.keep_raw_check.setChecked(self.settings.keep_raw)
        self.dedup_check = QCheckBox("Skip duplicate content")
        self.dedup_check.setChecked(True)
        self.dev_check = QCheckBox("Developer mode (debug logging, detector details)")
        form.addRow("Up axis", self.up_axis_box)
        form.addRow("Handedness", self.hand_box)
        form.addRow("Unit scale (1 unit = n metres)", self.unit_box)
        form.addRow("Max archive nesting", self.nesting_box)
        form.addRow(self.keep_raw_check)
        form.addRow(self.dedup_check)
        form.addRow(self.dev_check)
        form.addRow(QLabel("Export formats available: %s" % ", ".join(available_formats())))
        plugin_rows = "\n".join("%s %s%s" % (p.name, p.version, "" if p.loaded else
                                             "  [FAILED: %s]" % p.error) for p in plugins())
        box = QTextEdit(plugin_rows or "no plugins found")
        box.setReadOnly(True)
        box.setMaximumHeight(140)
        form.addRow("Plugins", box)
        return panel

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        open_report = QAction("Open last HTML report", self)
        open_report.triggered.connect(self._open_report)
        file_menu.addAction(open_report)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _with_browse(self, edit: QLineEdit, directory: bool = True) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._pick(edit, directory))
        row.addWidget(btn)
        return holder

    def _pick(self, edit: QLineEdit, directory: bool) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder", edit.text() or str(Path.home()))
        if path:
            edit.setText(path)

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    def _apply_settings(self) -> None:
        self.settings.source_dir = self.source_edit.text()
        self.settings.output_dir = self.output_edit.text()
        self.settings.workers = self.workers_box.value()
        self.settings.scan_mode = [ScanMode.STANDARD, ScanMode.DEEP,
                                   ScanMode.AGGRESSIVE][self.mode_box.currentIndex()]
        self.settings.export_formats = [f.strip() for f in self.format_box.currentText().split(",")]
        self.settings.coordinates.up_axis = self.up_axis_box.currentText()
        self.settings.coordinates.handedness = self.hand_box.currentText()
        self.settings.coordinates.unit_scale = self.unit_box.value()
        self.settings.limits.max_nesting_depth = self.nesting_box.value()
        self.settings.keep_raw = self.keep_raw_check.isChecked()
        self.settings.developer_mode = self.dev_check.isChecked()

    def start_analyze(self) -> None:
        source = self.source_edit.text().strip()
        if not source or not Path(source).exists():
            QMessageBox.warning(self, "Source", "Pick an existing source folder first.")
            return
        self._apply_settings()
        self.core = ExtractorCore(self.settings, self.db)
        self.tree.clear()
        self.unsupported_view.clear()
        self.errors_view.clear()
        self._set_running(True)
        self._worker = ScanWorker(self.core, source)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_scan_done)
        self._worker.failed.connect(self._on_failed)
        self._thread = run_in_thread(self._worker)

    def start_extract(self, characters_only: bool) -> None:
        if self.result is None:
            QMessageBox.information(self, "Extract", "Run Analyze first.")
            return
        self._apply_settings()
        selection = [g.model_asset.id for g in self.result.characters] if characters_only else None
        self._set_running(True)
        self._worker = ExportWorker(self.core, self.result, self.settings.export_formats, selection)
        self._worker.finished.connect(self._on_export_done)
        self._worker.failed.connect(self._on_failed)
        self._thread = run_in_thread(self._worker)

    def toggle_pause(self) -> None:
        if self.core.control.paused:
            self.core.resume()
            self.pause_btn.setText("Pause")
        else:
            self.core.pause()
            self.pause_btn.setText("Resume")

    def cancel(self) -> None:
        self.core.cancel()
        self.current_label.setText("cancelling, finishing running jobs...")

    def _set_running(self, running: bool) -> None:
        self.analyze_btn.setEnabled(not running)
        self.extract_btn.setEnabled(not running and self.result is not None)
        self.extract_chars_btn.setEnabled(not running and bool(
            self.result.characters if self.result else False))
        self.pause_btn.setEnabled(running)
        self.cancel_btn.setEnabled(running)

    # ------------------------------------------------------------------
    # signals
    # ------------------------------------------------------------------
    def _on_progress(self, snapshot: dict) -> None:
        for key, label in self.stat_labels.items():
            label.setText(str(snapshot.get(key, 0)))
        self.progress.setValue(int(snapshot.get("fraction", 0) * 100))
        self.current_label.setText("%s  |  %s" % (snapshot.get("operation", ""),
                                                  snapshot.get("current_file", "")))

    def _on_scan_done(self, result) -> None:
        self.result = result
        self._set_running(False)
        self.populate_results(result)
        engines = ", ".join("%s (%.0f%%)" % (e.engine, e.confidence * 100)
                            for e in result.engines[:3]) or "unknown engine"
        self.current_label.setText("analyze finished in %.1fs - %s" % (result.duration, engines))

    def _on_export_done(self, report: dict) -> None:
        self._set_running(False)
        self.current_label.setText("exported %d asset(s) to %s"
                                   % (len(report.get("exported", [])), report.get("output")))
        QMessageBox.information(self, "Extraction complete",
                                "Exported: %d\nFailed: %d\nSkipped: %d\n\nReport: %s"
                                % (len(report.get("exported", [])), len(report.get("failed", [])),
                                   len(report.get("skipped", [])), report.get("report")))

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        QMessageBox.critical(self, "Error", message)

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------
    def populate_results(self, result) -> None:
        self.tree.clear()
        groups = {}
        for title in ("Characters", "Models", "Animations", "Textures", "Containers"):
            item = QTreeWidgetItem([title])
            item.setExpanded(True)
            self.tree.addTopLevelItem(item)
            groups[title] = item

        for group in result.characters:
            node = QTreeWidgetItem([group.name, "Character",
                                    "%.0f%%" % (group.confidence * 100),
                                    "%d animations, %d textures" % (len(group.animations),
                                                                    len(group.textures))])
            node.setData(0, Qt.UserRole, group.model_asset)
            groups["Characters"].addChild(node)
            for anim in group.animations:
                child = QTreeWidgetItem([anim.name, "Animation", "", anim.source])
                child.setData(0, Qt.UserRole, anim)
                node.addChild(child)

        character_ids = {g.model_asset.id for g in result.characters}
        for asset in result.graph.by_type(AssetType.MODEL):
            if asset.id in character_ids:
                continue
            node = QTreeWidgetItem([asset.name, asset.classification.value,
                                    "%.0f%%" % (asset.confidence * 100),
                                    "%s | %s" % (asset.format_name, asset.source)])
            node.setData(0, Qt.UserRole, asset)
            groups["Models"].addChild(node)

        for asset in result.graph.by_type(AssetType.ANIMATION):
            node = QTreeWidgetItem([asset.name, "Animation",
                                    "%.0f%%" % (asset.confidence * 100), asset.source])
            node.setData(0, Qt.UserRole, asset)
            groups["Animations"].addChild(node)

        for asset in result.graph.by_type(AssetType.TEXTURE)[:5000]:
            meta = asset.metadata
            node = QTreeWidgetItem([asset.name, "Texture",
                                    "%.0f%%" % (asset.confidence * 100),
                                    "%sx%s %s" % (meta.get("width", "?"), meta.get("height", "?"),
                                                  meta.get("usage", ""))])
            node.setData(0, Qt.UserRole, asset)
            groups["Textures"].addChild(node)

        for asset in result.graph.by_type(AssetType.CONTAINER):
            node = QTreeWidgetItem([asset.name, "Container", "", asset.format_name])
            node.setData(0, Qt.UserRole, asset)
            groups["Containers"].addChild(node)

        for title, item in groups.items():
            item.setText(0, "%s (%d)" % (title, item.childCount()))

        self.unsupported_view.clear()
        for u in result.unsupported[:2000]:
            QTreeWidgetItem(self.unsupported_view,
                            [u.path, u.detected, u.reason, u.suggested_plugin])
        self.errors_view.clear()
        for e in result.errors[:2000]:
            QTreeWidgetItem(self.errors_view, [e.get("stage", ""), e.get("file", ""),
                                               e.get("message", "")])

    def _on_selection(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        asset = items[0].data(0, Qt.UserRole)
        if asset is None:
            return
        payload = {"asset": asset.to_dict()}
        if self.settings.developer_mode:
            payload["metadata"] = asset.metadata
        self.details.setText(json.dumps(payload, indent=2, default=str))
        from app.core.types import ModelAsset
        if isinstance(asset.payload, ModelAsset):
            self.viewer.load_model(asset.payload)
            self._fill_animations(asset.payload)
        elif asset.metadata.get("path"):
            self.viewer.load_file(asset.metadata["path"])

    def _fill_animations(self, model) -> None:
        self.anim_box.blockSignals(True)
        self.anim_box.clear()
        for clip in model.animations:
            self.anim_box.addItem("%s (%.2fs)" % (clip.name, clip.duration))
        self.anim_box.blockSignals(False)
        if model.animations:
            self.viewer.load_animation(model.animations[0])

    def _on_animation_selected(self, index: int) -> None:
        if self.viewer.model and 0 <= index < len(self.viewer.model.animations):
            self.viewer.load_animation(self.viewer.model.animations[index])

    def _toggle_bones(self, on: bool) -> None:
        self.viewer.show_bones = on
        self.viewer.update()

    def _reset_animation(self) -> None:
        self.viewer.reset_animation()
        self.timeline.setValue(0)

    def _scrub(self, value: int) -> None:
        clip = self.viewer.clip
        if not clip or clip.duration <= 0:
            return
        self.viewer.set_time(clip.duration * value / 1000.0)
        self.time_label.setText("%.2f s" % self.viewer.time)

    def _on_frame(self, t: float) -> None:
        clip = self.viewer.clip
        if clip and clip.duration > 0:
            self.timeline.setValue(int(1000 * t / clip.duration))
        self.time_label.setText("%.2f s" % t)

    def _open_report(self) -> None:
        report = Path(self.output_edit.text() or ".") / "Reports" / "report.html"
        if report.exists():
            webbrowser.open(report.as_uri())
        else:
            QMessageBox.information(self, "Report", "No report yet - run Extract first.")

    def closeEvent(self, event) -> None:
        try:
            self.core.cancel()
            self.core.cleanup_temp()
        finally:
            super().closeEvent(event)
