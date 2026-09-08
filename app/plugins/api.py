"""Public plugin API. A plugin touches only what is exposed here.

    plugins/my_game/
        plugin.json     manifest
        plugin.py       defines register(api)

The core never imports a plugin module by name; the loader discovers them.
Plugins are code: only files under a trusted plugin directory are ever loaded,
never files found while scanning user content.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.core.detector.base import FileContext, FormatDetector      # re-export
from app.core.extraction.base import ContainerExtractor, ExtractionResult
from app.core.registry import (ANIMATION_PARSERS, COMPRESSION_CODECS, DETECTORS, EXPORTERS,
                               EXTRACTORS, GAME_PROFILES, MODEL_PARSERS, TEXTURE_PARSERS)
from app.core.types import (AnimationClip, Asset, AssetType, Bone, Category, DetectionResult,
                            Material, Mesh, ModelAsset, Skeleton, TextureAsset)
from app.exporters.base import Exporter
from app.formats.models.base import ModelParser

PLUGIN_API_VERSION = 1


@dataclass
class GameProfile:
    """Ties detectors/parsers to one game or engine family."""

    name: str
    detectors: list = field(default_factory=list)
    extractors: list = field(default_factory=list)
    model_parsers: list = field(default_factory=list)
    animation_parsers: list = field(default_factory=list)
    texture_parsers: list = field(default_factory=list)
    platforms: list = field(default_factory=list)
    priority: int = 50

    def matches(self, evidence: dict) -> bool:
        return False


class PluginAPI:
    """Handed to `register(api)`; the only supported extension surface."""

    version = PLUGIN_API_VERSION

    def __init__(self, plugin_name: str, settings=None):
        self.plugin_name = plugin_name
        self.settings = settings

    def add_detector(self, detector: FormatDetector, priority: Optional[int] = None):
        return DETECTORS.register(detector, priority)

    def add_extractor(self, extractor: ContainerExtractor, priority: Optional[int] = None):
        return EXTRACTORS.register(extractor, priority)

    def add_model_parser(self, parser: ModelParser, priority: Optional[int] = None):
        return MODEL_PARSERS.register(parser, priority)

    def add_texture_parser(self, parser, priority: Optional[int] = None):
        return TEXTURE_PARSERS.register(parser, priority)

    def add_animation_parser(self, parser, priority: Optional[int] = None):
        return ANIMATION_PARSERS.register(parser, priority)

    def add_exporter(self, exporter: Exporter, priority: Optional[int] = None):
        return EXPORTERS.register(exporter, priority)

    def add_compression_codec(self, codec, priority: Optional[int] = None):
        return COMPRESSION_CODECS.register(codec, priority)

    def add_game_profile(self, profile: GameProfile):
        return GAME_PROFILES.register(profile, profile.priority)


__all__ = [
    "PluginAPI", "GameProfile", "PLUGIN_API_VERSION",
    "FormatDetector", "FileContext", "DetectionResult", "Category",
    "ContainerExtractor", "ExtractionResult", "ModelParser", "Exporter",
    "ModelAsset", "Mesh", "Material", "Skeleton", "Bone", "AnimationClip",
    "TextureAsset", "Asset", "AssetType",
]
