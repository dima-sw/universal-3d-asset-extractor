"""Central configuration. Core reads this; UI only edits it."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class ScanMode(str, Enum):
    STANDARD = "standard"      # known formats only
    DEEP = "deep"              # + heuristics on unknown files
    AGGRESSIVE = "aggressive"  # + expensive embedded/compression brute scan


class DuplicatePolicy(str, Enum):
    SKIP = "skip"
    EXPORT_ALL = "export_all"


class OverwritePolicy(str, Enum):
    SKIP = "skip"
    OVERWRITE = "overwrite"
    RENAME = "rename"


@dataclass
class Limits:
    max_nesting_depth: int = 10             # archive-inside-archive depth
    max_scan_depth: int = 64                # directory depth
    max_extracted_size: int = 8 * 1024 ** 3  # per container, decompression bomb guard
    max_single_file_size: int = 2 * 1024 ** 3
    max_entries_per_archive: int = 200_000
    max_compression_ratio: float = 500.0    # bomb heuristic
    header_read_size: int = 64 * 1024       # bytes read for detection
    embedded_scan_max_bytes: int = 256 * 1024 * 1024


@dataclass
class CoordinateConfig:
    up_axis: str = "y"          # "y" | "z"
    handedness: str = "right"   # "right" | "left"
    unit_scale: float = 1.0     # 1 unit = 1 meter


@dataclass
class Settings:
    source_dir: str = ""
    output_dir: str = ""
    temp_dir: str = ""
    workers: int = max(2, (os.cpu_count() or 4) - 1)
    scan_mode: ScanMode = ScanMode.STANDARD
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.SKIP
    overwrite_policy: OverwritePolicy = OverwritePolicy.RENAME
    export_formats: list = field(default_factory=lambda: ["glb"])
    keep_temp_on_error: bool = True
    keep_raw: bool = True
    developer_mode: bool = False
    log_level: str = "INFO"
    oodle_dll: str = ""          # path to an oo2core runtime the user owns
    limits: Limits = field(default_factory=Limits)
    coordinates: CoordinateConfig = field(default_factory=CoordinateConfig)
    db_path: str = ""

    # ---- paths -----------------------------------------------------------
    def resolved_temp(self) -> Path:
        p = Path(self.temp_dir) if self.temp_dir else Path(self.output_dir or ".") / "temp"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolved_output(self) -> Path:
        p = Path(self.output_dir or "./Extracted")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolved_db(self) -> Path:
        if self.db_path:
            return Path(self.db_path)
        return self.resolved_output() / "extractor.sqlite3"

    # ---- (de)serialisation ----------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["scan_mode"] = self.scan_mode.value
        d["duplicate_policy"] = self.duplicate_policy.value
        d["overwrite_policy"] = self.overwrite_policy.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        data = dict(data or {})
        limits = Limits(**data.pop("limits", {}) or {})
        coords = CoordinateConfig(**data.pop("coordinates", {}) or {})
        for key, enum_cls in (("scan_mode", ScanMode),
                              ("duplicate_policy", DuplicatePolicy),
                              ("overwrite_policy", OverwritePolicy)):
            if key in data and data[key] is not None:
                data[key] = enum_cls(data[key])
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in data.items() if k in known}
        return cls(limits=limits, coordinates=coords, **data)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Settings":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


DEFAULT_SETTINGS_PATH = Path.home() / ".universal3dextractor" / "settings.json"
