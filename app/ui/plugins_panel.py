"""Plugins tab: install, enable/disable, remove and scaffold, without the CLI."""
from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QMessageBox, QPushButton, QTextEdit, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from app.plugins import manager
from app.plugins.loader import reload_all

TRUST_WARNING = ("A plugin is Python code and runs with this application's permissions.\n"
                 "Install only plugins you trust.\n\nInstall:\n%s")


class PluginsPanel(QWidget):
    def __init__(self, settings, db=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.db = db
        layout = QVBoxLayout(self)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Plugin", "Version", "Source", "State", "Engines", "Note"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.itemSelectionChanged.connect(self._on_select)
        layout.addWidget(self.tree, 1)

        buttons = QHBoxLayout()
        self.install_btn = QPushButton("Install folder...")
        self.install_zip_btn = QPushButton("Install .zip...")
        self.new_btn = QPushButton("New plugin...")
        self.toggle_btn = QPushButton("Disable")
        self.remove_btn = QPushButton("Remove")
        self.reload_btn = QPushButton("Reload all")
        self.folder_btn = QPushButton("Open plugin folder")
        for b in (self.install_btn, self.install_zip_btn, self.new_btn, self.toggle_btn,
                  self.remove_btn, self.reload_btn, self.folder_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(170)
        layout.addWidget(self.details)

        hint = QLabel("A plugin adds formats without touching the core. "
                      "New plugin... creates a working skeleton you can edit, "
                      "then Install folder... puts it in place.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888;")
        layout.addWidget(hint)

        self.install_btn.clicked.connect(self.install_folder)
        self.install_zip_btn.clicked.connect(self.install_zip)
        self.new_btn.clicked.connect(self.create_plugin)
        self.toggle_btn.clicked.connect(self.toggle_selected)
        self.remove_btn.clicked.connect(self.remove_selected)
        self.reload_btn.clicked.connect(self.reload_plugins)
        self.folder_btn.clicked.connect(
            lambda: webbrowser.open(manager.user_dir().as_uri()))
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self.tree.clear()
        for info in manager.catalogue():
            if not info.enabled:
                state = "disabled"
            elif info.problems:
                state = "invalid"
            else:
                state = "ok"
            note = "; ".join(info.problems) if info.problems else (info.error or "")
            item = QTreeWidgetItem([info.name, info.version, info.source, state,
                                    ",".join(info.supported_engines) or "-", note])
            item.setData(0, Qt.UserRole, info)
            self.tree.addTopLevelItem(item)
        self._on_select()

    def _selected(self):
        items = self.tree.selectedItems()
        return items[0].data(0, Qt.UserRole) if items else None

    def _on_select(self) -> None:
        info = self._selected()
        if info is None:
            self.details.setPlainText("")
            self.toggle_btn.setEnabled(False)
            self.remove_btn.setEnabled(False)
            return
        self.toggle_btn.setEnabled(True)
        self.toggle_btn.setText("Enable" if not info.enabled else "Disable")
        self.remove_btn.setEnabled(info.source == "user")
        self.details.setPlainText(json.dumps(info.to_dict(), indent=2))

    # ------------------------------------------------------------------
    def install_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select the plugin folder")
        if path:
            self._install(Path(path))

    def install_zip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select the plugin archive", "",
                                              "Plugin archive (*.zip)")
        if path:
            self._install(Path(path))

    def _install(self, source: Path) -> None:
        problems = manager.validate(source) if source.is_dir() else []
        if problems:
            QMessageBox.warning(self, "Plugin rejected",
                                "That folder is not a valid plugin:\n\n- "
                                + "\n- ".join(problems))
            return
        if QMessageBox.question(self, "Install plugin", TRUST_WARNING % source,
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            info = manager.install(source, force=True)
        except Exception as exc:
            QMessageBox.critical(self, "Install failed", str(exc))
            return
        self.reload_plugins()
        QMessageBox.information(self, "Plugin installed",
                                "%s %s is installed and active." % (info.name, info.version))

    def create_plugin(self) -> None:
        name, ok = QInputDialog.getText(self, "New plugin",
                                        "Plugin name (e.g. Tomb Raider PS2):")
        if not ok or not name.strip():
            return
        target_dir = QFileDialog.getExistingDirectory(self, "Where should it be created?")
        if not target_dir:
            return
        try:
            path = manager.scaffold(name.strip(), target_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Could not create the plugin", str(exc))
            return
        QMessageBox.information(
            self, "Plugin created",
            "Created:\n%s\n\nEdit plugin.py, then use Install folder... on it.\n"
            "README.md in that folder explains the format hooks." % path)
        webbrowser.open(Path(path).as_uri())

    def toggle_selected(self) -> None:
        info = self._selected()
        if info is None:
            return
        manager.set_enabled(info.name, not info.enabled)
        self.reload_plugins()

    def remove_selected(self) -> None:
        info = self._selected()
        if info is None or info.source != "user":
            return
        if QMessageBox.question(self, "Remove plugin",
                                "Delete %s from the plugin folder?" % info.name,
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        manager.remove(info.name)
        self.reload_plugins()

    def reload_plugins(self) -> None:
        try:
            reload_all(settings=self.settings, db=self.db)
        except Exception as exc:
            QMessageBox.critical(self, "Reload failed", str(exc))
        self.refresh()
