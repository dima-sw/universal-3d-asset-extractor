"""Job model + worker pool with cooperative cancel / pause / resume.

The UI thread never runs work; it only reads Progress snapshots.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.core.logging_setup import get_logger
from app.core.pipeline.states import State

log = get_logger("jobs")


@dataclass
class ExtractionJob:
    input_file: Path
    virtual_path: str
    depth: int = 0
    priority: int = 50
    parent_id: Optional[int] = None
    detected_format: Optional[str] = None
    status: State = State.DISCOVERED
    progress: float = 0.0
    error: Optional[str] = None
    output_files: list = field(default_factory=list)
    file_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"input": str(self.input_file), "virtual_path": self.virtual_path,
                "depth": self.depth, "priority": self.priority,
                "format": self.detected_format, "status": self.status.value,
                "error": self.error, "outputs": [str(p) for p in self.output_files]}

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionJob":
        return cls(input_file=Path(d["input"]), virtual_path=d["virtual_path"],
                   depth=int(d.get("depth", 0)), priority=int(d.get("priority", 50)),
                   detected_format=d.get("format"),
                   status=State(d.get("status", "discovered")), error=d.get("error"),
                   output_files=[Path(p) for p in d.get("outputs", [])])


@dataclass
class Progress:
    discovered: int = 0
    scanned: int = 0
    containers: int = 0
    models: int = 0
    animations: int = 0
    textures: int = 0
    characters: int = 0
    failed: int = 0
    unsupported: int = 0
    skipped_duplicates: int = 0
    current_file: str = ""
    current_operation: str = "idle"
    started_at: float = field(default_factory=time.time)

    @property
    def remaining(self) -> int:
        return max(0, self.discovered - self.scanned)

    @property
    def fraction(self) -> float:
        return (self.scanned / self.discovered) if self.discovered else 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def snapshot(self) -> dict:
        return {"discovered": self.discovered, "scanned": self.scanned,
                "remaining": self.remaining, "containers": self.containers,
                "models": self.models, "animations": self.animations,
                "textures": self.textures, "characters": self.characters,
                "failed": self.failed, "unsupported": self.unsupported,
                "duplicates": self.skipped_duplicates,
                "current_file": self.current_file, "operation": self.current_operation,
                "fraction": round(self.fraction, 4), "elapsed": round(self.elapsed, 2)}


class Control:
    """Cancel / pause tokens shared by every worker."""

    def __init__(self):
        self._cancel = threading.Event()
        self._resume = threading.Event()
        self._resume.set()

    def cancel(self) -> None:
        self._cancel.set()
        self._resume.set()          # release anyone parked in pause

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    def checkpoint(self, timeout: float = 0.25) -> bool:
        """Block while paused. Returns False when the caller should stop."""
        while not self._resume.wait(timeout):
            if self._cancel.is_set():
                return False
        return not self._cancel.is_set()


class WorkerPool:
    """Priority job queue + threads. Work items are (priority, seq, job)."""

    def __init__(self, workers: int, handler: Callable, control: Control,
                 on_error: Optional[Callable] = None):
        self.handler = handler
        self.control = control
        self.on_error = on_error
        self.queue: queue.PriorityQueue = queue.PriorityQueue()
        self._threads: list = []
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._active = 0
        self._active_lock = threading.Condition()
        self.worker_count = max(1, workers)
        self._stop = threading.Event()

    # -- queue -------------------------------------------------------------
    def submit(self, job: ExtractionJob) -> None:
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        self.queue.put((-job.priority, seq, job))

    def pending(self) -> int:
        return self.queue.qsize()

    def serialize_pending(self) -> list:
        """Snapshot of queued jobs (for pause/resume persistence)."""
        items = []
        drained = []
        while True:
            try:
                drained.append(self.queue.get_nowait())
            except queue.Empty:
                break
        for entry in drained:
            items.append(entry[2].to_dict())
            self.queue.put(entry)
        return items

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        for i in range(self.worker_count):
            t = threading.Thread(target=self._run, name="worker-%d" % i, daemon=True)
            t.start()
            self._threads.append(t)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                _, _, job = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self.control.checkpoint():
                self.queue.task_done()
                continue
            with self._active_lock:
                self._active += 1
            try:
                self.handler(job)
            except Exception as exc:
                job.status = State.FAILED
                job.error = str(exc)
                log.debug("job crashed: %s\n%s", exc, traceback.format_exc())
                if self.on_error:
                    try:
                        self.on_error(job, exc)
                    except Exception:
                        pass
            finally:
                with self._active_lock:
                    self._active -= 1
                    self._active_lock.notify_all()
                self.queue.task_done()

    def wait_idle(self, poll: float = 0.05) -> None:
        """Wait until the queue is empty and no worker is busy."""
        while True:
            if self.control.cancelled:
                return
            with self._active_lock:
                idle = self._active == 0
            if idle and self.queue.empty():
                # settle: workers may have just enqueued children
                time.sleep(poll)
                with self._active_lock:
                    idle = self._active == 0
                if idle and self.queue.empty():
                    return
            time.sleep(poll)

    def shutdown(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
