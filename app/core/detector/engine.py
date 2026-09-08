"""Engine detection from directory structure (a whole-tree signal, not per file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EngineEvidence:
    engine: str
    confidence: float
    evidence: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"engine": self.engine, "confidence": round(self.confidence, 3),
                "evidence": self.evidence[:12]}


# rule: (engine, needle, kind, weight)  kind: "file" | "dir" | "suffix"
RULES = [
    ("Unity", "globalgamemanagers", "file", 0.5),
    ("Unity", "resources.assets", "file", 0.35),
    ("Unity", "unityplayer.dll", "file", 0.4),
    ("Unity", "il2cpp_data", "dir", 0.3),
    ("Unity", "level0", "file", 0.2),
    ("Unity", ".assets", "suffix", 0.2),
    ("Unity", ".bundle", "suffix", 0.1),
    ("Unity", ".unity3d", "suffix", 0.25),

    ("Unreal", ".uasset", "suffix", 0.4),
    ("Unreal", ".umap", "suffix", 0.3),
    ("Unreal", ".pak", "suffix", 0.1),
    ("Unreal", ".utoc", "suffix", 0.35),
    ("Unreal", "engine", "dir", 0.15),
    ("Unreal", "paks", "dir", 0.2),

    ("Source", "gameinfo.txt", "file", 0.5),
    ("Source", ".vpk", "suffix", 0.3),
    ("Source", ".bsp", "suffix", 0.2),
    ("Source", ".vtf", "suffix", 0.25),
    ("Source", ".mdl", "suffix", 0.15),

    ("Godot", "project.godot", "file", 0.6),
    ("Godot", ".pck", "suffix", 0.3),
    ("Godot", ".tscn", "suffix", 0.3),

    ("CryEngine", "system.cfg", "file", 0.3),
    ("CryEngine", ".pak", "suffix", 0.05),
    ("CryEngine", ".cgf", "suffix", 0.4),
    ("CryEngine", ".caf", "suffix", 0.3),

    ("RenderWare", ".dff", "suffix", 0.45),
    ("RenderWare", ".txd", "suffix", 0.4),
    ("RenderWare", ".ifp", "suffix", 0.2),

    ("id Tech", ".pk3", "suffix", 0.35),
    ("id Tech", ".pk4", "suffix", 0.35),
    ("id Tech", ".bsp", "suffix", 0.1),

    ("GameMaker", "data.win", "file", 0.6),
    ("GameMaker", "game.unx", "file", 0.5),

    ("Frostbite", ".cas", "suffix", 0.5),
    ("Frostbite", ".sb", "suffix", 0.4),
    ("Frostbite", ".toc", "suffix", 0.3),
    ("Frostbite", "cas.cat", "file", 0.6),

    ("Anvil", ".forge", "suffix", 0.6),
    ("Anvil", "datapc.forge", "file", 0.5),

    ("CRI Middleware", ".afs", "suffix", 0.5),
    ("CRI Middleware", ".adx", "suffix", 0.3),

    ("PS2", "system.cnf", "file", 0.35),
    ("PS1", "psx.exe", "file", 0.4),
    ("PS1", "slus_", "prefix", 0.35),
    ("PS1", "sces_", "prefix", 0.35),
    ("PS2", "slus_", "prefix", 0.2),
]


def detect_engine(root, max_files: int = 40000) -> list:
    """Walk (bounded) and score engines. Returns ranked EngineEvidence list."""
    scores: dict = {}
    evidence: dict = {}
    hits: dict = {}
    seen = 0
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).lower()
        dir_tokens = set(rel_dir.replace("\\", "/").split("/"))
        for engine, needle, kind, weight in RULES:
            if kind == "dir" and needle in dir_tokens:
                _add(scores, evidence, engine, weight, "dir:%s" % needle, hits)
        for fn in filenames:
            seen += 1
            low = fn.lower()
            for engine, needle, kind, weight in RULES:
                if kind == "file" and low == needle:
                    _add(scores, evidence, engine, weight, "file:%s" % fn, hits)
                elif kind == "suffix" and low.endswith(needle):
                    _add(scores, evidence, engine, weight * 0.6, "ext:%s" % needle, hits)
                elif kind == "prefix" and low.startswith(needle):
                    _add(scores, evidence, engine, weight, "name:%s" % fn, hits)
            if seen >= max_files:
                break
        # Unity's tell-tale "<Game>_Data" folder
        for d in dirnames:
            if d.lower().endswith("_data"):
                _add(scores, evidence, "Unity", 0.35, "dir:%s" % d, hits)
        if seen >= max_files:
            break

    out = [EngineEvidence(e, min(0.99, s), evidence.get(e, [])) for e, s in scores.items()]
    out.sort(key=lambda x: -x.confidence)
    return [e for e in out if e.confidence >= 0.2]


_MAX_HITS_PER_RULE = 3


def _add(scores: dict, evidence: dict, engine: str, weight: float, note: str,
         hits: Optional[dict] = None) -> None:
    """Repeated weak evidence must not saturate: 200 files with the same
    extension are barely stronger proof than three of them."""
    if hits is not None:
        key = (engine, note)
        seen = hits.get(key, 0)
        if seen >= _MAX_HITS_PER_RULE:
            return
        hits[key] = seen + 1
        weight = weight / (seen + 1)
    scores[engine] = min(0.99, scores.get(engine, 0.0) + weight)
    lst = evidence.setdefault(engine, [])
    if note not in lst and len(lst) < 32:
        lst.append(note)
