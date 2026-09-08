"""Exporter facade."""
from __future__ import annotations

from typing import Optional

from app.core.registry import EXPORTERS
from app.exporters.base import Exporter, ExportResult

_bootstrapped = False


def bootstrap() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    from app.exporters.collada import ColladaExporter, FbxExporter
    from app.exporters.gltf import GltfExporter, GltfJsonExporter
    from app.exporters.wavefront import ObjExporter
    for cls in (GltfExporter, GltfJsonExporter, ObjExporter, ColladaExporter, FbxExporter):
        EXPORTERS.register(cls())
    _bootstrapped = True


def get_exporter(fmt: str) -> Optional[Exporter]:
    bootstrap()
    fmt = (fmt or "").lower().lstrip(".")
    for exp in EXPORTERS:
        if exp.format == fmt or exp.name == fmt:
            return exp
    return None


def available_formats() -> list:
    bootstrap()
    return [e.format for e in EXPORTERS]


__all__ = ["Exporter", "ExportResult", "get_exporter", "available_formats", "bootstrap"]
