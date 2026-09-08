"""Qt thread wrapper around the headless core. The core never imports Qt."""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

from app.core.pipeline.controller import ExtractorCore


class ScanWorker(QObject):
    progress = Signal(dict)
    finished = Signal(object)          # ScanResult
    failed = Signal(str)

    def __init__(self, core: ExtractorCore, source: str):
        super().__init__()
        self.core = core
        self.source = source

    def run(self) -> None:
        try:
            self.core.progress_callback = lambda p: self.progress.emit(p.snapshot())
            result = self.core.analyze(self.source)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class ExportWorker(QObject):
    progress = Signal(dict)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, core: ExtractorCore, result, formats: list, selection=None):
        super().__init__()
        self.core = core
        self.result = result
        self.formats = formats
        self.selection = selection

    def run(self) -> None:
        try:
            report = self.core.extract(self.result, selection=self.selection,
                                       formats=self.formats)
            self.finished.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))


def run_in_thread(worker: QObject) -> QThread:
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    for signal_name in ("finished", "failed"):
        getattr(worker, signal_name).connect(thread.quit)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
