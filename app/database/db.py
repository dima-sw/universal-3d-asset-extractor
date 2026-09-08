"""SQLite cache + scan persistence (resume, dedup, incremental rescan)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from app.core.logging_setup import get_logger

log = get_logger("db")

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    output TEXT,
    mode TEXT,
    started_at REAL,
    finished_at REAL,
    status TEXT DEFAULT 'running',
    engine TEXT,
    stats_json TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    virtual_path TEXT,
    parent_file_id INTEGER,
    depth INTEGER DEFAULT 0,
    size INTEGER,
    mtime REAL,
    sha256 TEXT,
    format TEXT,
    category TEXT,
    confidence REAL,
    engine TEXT,
    status TEXT DEFAULT 'discovered',
    detector TEXT,
    metadata_json TEXT,
    UNIQUE(scan_id, virtual_path)
);
CREATE INDEX IF NOT EXISTS idx_files_scan ON files(scan_id);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(sha256);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(scan_id, status);

CREATE TABLE IF NOT EXISTS containers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    extractor TEXT,
    entries INTEGER,
    bytes INTEGER,
    temp_dir TEXT,
    status TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    scan_id INTEGER NOT NULL,
    file_id INTEGER,
    type TEXT NOT NULL,
    name TEXT,
    classification TEXT,
    format TEXT,
    engine TEXT,
    confidence REAL,
    content_hash TEXT,
    source TEXT,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_assets_scan ON assets(scan_id, type);
CREATE INDEX IF NOT EXISTS idx_assets_hash ON assets(content_hash);

CREATE TABLE IF NOT EXISTS asset_references (
    source_asset_id TEXT NOT NULL,
    target_asset_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    PRIMARY KEY (source_asset_id, target_asset_id, relation_type)
);

CREATE TABLE IF NOT EXISTS exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER,
    asset_id TEXT,
    output_path TEXT,
    format TEXT,
    status TEXT,
    error TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER,
    file_id INTEGER,
    stage TEXT,
    path TEXT,
    message TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS plugins (
    name TEXT PRIMARY KEY,
    version TEXT,
    entrypoint TEXT,
    enabled INTEGER DEFAULT 1,
    loaded_at REAL
);

CREATE TABLE IF NOT EXISTS cache (
    quick_key TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    format TEXT,
    category TEXT,
    confidence REAL,
    engine TEXT,
    detector TEXT,
    parser_version TEXT,
    status TEXT,
    results_json TEXT,
    updated_at REAL,
    PRIMARY KEY (path, quick_key)
);
"""

PARSER_VERSION = "1"


class Database:
    """Thread-safe-enough wrapper: one connection per thread."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ---- scans -----------------------------------------------------------
    def create_scan(self, source: str, output: str, mode: str) -> int:
        with self._write_lock:
            conn = self.connect()
            cur = conn.execute(
                "INSERT INTO scans(source, output, mode, started_at, status) VALUES (?,?,?,?,?)",
                (str(source), str(output), str(mode), time.time(), "running"))
            conn.commit()
            return int(cur.lastrowid)

    def finish_scan(self, scan_id: int, status: str, stats: dict, engine: Optional[str] = None) -> None:
        with self._write_lock:
            conn = self.connect()
            conn.execute("UPDATE scans SET finished_at=?, status=?, stats_json=?, engine=? WHERE id=?",
                         (time.time(), status, json.dumps(stats), engine, scan_id))
            conn.commit()

    def get_scan(self, scan_id: int) -> Optional[sqlite3.Row]:
        return self.connect().execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()

    def latest_scan(self, source: Optional[str] = None) -> Optional[sqlite3.Row]:
        if source:
            return self.connect().execute(
                "SELECT * FROM scans WHERE source=? ORDER BY id DESC LIMIT 1", (str(source),)).fetchone()
        return self.connect().execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()

    def resumable_scans(self) -> list:
        return self.connect().execute(
            "SELECT * FROM scans WHERE status IN ('running','paused') ORDER BY id DESC").fetchall()

    # ---- files -----------------------------------------------------------
    def upsert_file(self, scan_id: int, record: dict) -> int:
        cols = ("path", "virtual_path", "parent_file_id", "depth", "size", "mtime", "sha256",
                "format", "category", "confidence", "engine", "status", "detector", "metadata_json")
        values = [record.get(c) for c in cols]
        with self._write_lock:
            conn = self.connect()
            cur = conn.execute(
                "INSERT INTO files(scan_id,%s) VALUES (?,%s) "
                "ON CONFLICT(scan_id, virtual_path) DO UPDATE SET %s"
                % (",".join(cols), ",".join("?" * len(cols)),
                   ",".join("%s=excluded.%s" % (c, c) for c in cols)),
                [scan_id] + values)
            conn.commit()
            if cur.lastrowid:
                return int(cur.lastrowid)
        row = self.connect().execute(
            "SELECT id FROM files WHERE scan_id=? AND virtual_path=?",
            (scan_id, record.get("virtual_path"))).fetchone()
        return int(row["id"]) if row else -1

    def set_file_status(self, file_id: int, status: str) -> None:
        with self._write_lock:
            conn = self.connect()
            conn.execute("UPDATE files SET status=? WHERE id=?", (status, file_id))
            conn.commit()

    def pending_files(self, scan_id: int) -> list:
        return self.connect().execute(
            "SELECT * FROM files WHERE scan_id=? AND status NOT IN "
            "('completed','failed','skipped') ORDER BY id", (scan_id,)).fetchall()

    def files_by_hash(self, scan_id: int, sha256: str) -> list:
        return self.connect().execute(
            "SELECT * FROM files WHERE scan_id=? AND sha256=?", (scan_id, sha256)).fetchall()

    def count_files(self, scan_id: int) -> int:
        row = self.connect().execute("SELECT COUNT(*) c FROM files WHERE scan_id=?", (scan_id,)).fetchone()
        return int(row["c"]) if row else 0

    # ---- assets ----------------------------------------------------------
    def insert_asset(self, scan_id: int, asset, file_id: Optional[int] = None) -> None:
        with self._write_lock:
            conn = self.connect()
            conn.execute(
                "INSERT OR REPLACE INTO assets(id,scan_id,file_id,type,name,classification,"
                "format,engine,confidence,content_hash,source,metadata_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (asset.id, scan_id, file_id, asset.type.value, asset.name,
                 asset.classification.value, asset.format_name, asset.engine,
                 asset.confidence, asset.content_hash, asset.source,
                 json.dumps(asset.metadata, default=str)))
            conn.commit()

    def insert_reference(self, ref) -> None:
        with self._write_lock:
            conn = self.connect()
            conn.execute(
                "INSERT OR IGNORE INTO asset_references(source_asset_id,target_asset_id,"
                "relation_type,confidence) VALUES (?,?,?,?)",
                (ref.source_asset, ref.target_asset, ref.relation.value, ref.confidence))
            conn.commit()

    def assets(self, scan_id: int, type_: Optional[str] = None) -> list:
        if type_:
            return self.connect().execute(
                "SELECT * FROM assets WHERE scan_id=? AND type=? ORDER BY name",
                (scan_id, type_)).fetchall()
        return self.connect().execute(
            "SELECT * FROM assets WHERE scan_id=? ORDER BY type, name", (scan_id,)).fetchall()

    def asset_counts(self, scan_id: int) -> dict:
        rows = self.connect().execute(
            "SELECT type, COUNT(*) c FROM assets WHERE scan_id=? GROUP BY type", (scan_id,)).fetchall()
        return {r["type"]: r["c"] for r in rows}

    # ---- exports / errors -------------------------------------------------
    def record_export(self, scan_id: int, asset_id: str, output_path: str, fmt: str,
                      status: str, error: Optional[str] = None) -> None:
        with self._write_lock:
            conn = self.connect()
            conn.execute("INSERT INTO exports(scan_id,asset_id,output_path,format,status,error,"
                         "created_at) VALUES (?,?,?,?,?,?,?)",
                         (scan_id, asset_id, str(output_path), fmt, status, error, time.time()))
            conn.commit()

    def record_error(self, scan_id: int, stage: str, path: str, message: str,
                     file_id: Optional[int] = None) -> None:
        with self._write_lock:
            conn = self.connect()
            conn.execute("INSERT INTO errors(scan_id,file_id,stage,path,message,created_at) "
                         "VALUES (?,?,?,?,?,?)",
                         (scan_id, file_id, stage, str(path), str(message)[:2000], time.time()))
            conn.commit()

    def errors(self, scan_id: int) -> list:
        return self.connect().execute(
            "SELECT * FROM errors WHERE scan_id=? ORDER BY id DESC", (scan_id,)).fetchall()

    # ---- cache -----------------------------------------------------------
    def cache_get(self, path: str, quick_key: str) -> Optional[sqlite3.Row]:
        return self.connect().execute(
            "SELECT * FROM cache WHERE path=? AND quick_key=? AND parser_version=?",
            (str(path), quick_key, PARSER_VERSION)).fetchone()

    def cache_put(self, path: str, quick_key: str, detection, sha256: Optional[str] = None,
                  status: str = "identified", results: Optional[dict] = None) -> None:
        with self._write_lock:
            conn = self.connect()
            conn.execute(
                "INSERT OR REPLACE INTO cache(quick_key,path,sha256,format,category,confidence,"
                "engine,detector,parser_version,status,results_json,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (quick_key, str(path), sha256, detection.format_name, detection.category.value,
                 detection.confidence, detection.engine, detection.detector, PARSER_VERSION,
                 status, json.dumps(results or {}, default=str), time.time()))
            conn.commit()

    # ---- plugins ---------------------------------------------------------
    def record_plugin(self, name: str, version: str, entrypoint: str) -> None:
        with self._write_lock:
            conn = self.connect()
            conn.execute("INSERT OR REPLACE INTO plugins(name,version,entrypoint,enabled,loaded_at)"
                         " VALUES (?,?,?,1,?)", (name, version, entrypoint, time.time()))
            conn.commit()

    def executemany(self, sql: str, rows: Iterable) -> None:
        with self._write_lock:
            conn = self.connect()
            conn.executemany(sql, rows)
            conn.commit()
