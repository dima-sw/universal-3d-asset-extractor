"""Global extension points. The core knows these registries and nothing else.

A plugin adds support by registering objects here; no core file changes.
"""
from __future__ import annotations

from typing import Callable, Optional

from app.core.logging_setup import get_logger

log = get_logger("registry")


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: list = []

    def register(self, item, priority: Optional[int] = None):
        prio = priority if priority is not None else getattr(item, "priority", 50)
        self._items.append((prio, len(self._items), item))
        self._items.sort(key=lambda t: (-t[0], t[1]))
        log.debug("registered %s: %s (priority %s)", self.kind,
                  getattr(item, "name", type(item).__name__), prio)
        return item

    def all(self) -> list:
        return [i for _, _, i in self._items]

    def clear(self) -> None:
        self._items.clear()

    def __iter__(self):
        return iter(self.all())

    def __len__(self) -> int:
        return len(self._items)


DETECTORS = Registry("detector")
EXTRACTORS = Registry("extractor")        # containers -> files
MODEL_PARSERS = Registry("model_parser")
TEXTURE_PARSERS = Registry("texture_parser")
ANIMATION_PARSERS = Registry("animation_parser")
EXPORTERS = Registry("exporter")
GAME_PROFILES = Registry("game_profile")
COMPRESSION_CODECS = Registry("compression_codec")

_ALL = {
    "detectors": DETECTORS,
    "extractors": EXTRACTORS,
    "model_parsers": MODEL_PARSERS,
    "texture_parsers": TEXTURE_PARSERS,
    "animation_parsers": ANIMATION_PARSERS,
    "exporters": EXPORTERS,
    "game_profiles": GAME_PROFILES,
    "compression_codecs": COMPRESSION_CODECS,
}


def registry(name: str) -> Registry:
    return _ALL[name]


def summary() -> dict:
    return {k: [getattr(i, "name", type(i).__name__) for i in r.all()] for k, r in _ALL.items()}


def reset_all() -> None:
    for r in _ALL.values():
        r.clear()


def detector(cls=None, *, priority: Optional[int] = None) -> Callable:
    """Class decorator: @detector registers an instance of the detector."""
    def wrap(c):
        DETECTORS.register(c(), priority)
        return c
    return wrap(cls) if cls is not None else wrap


def exporter(cls=None, *, priority: Optional[int] = None) -> Callable:
    def wrap(c):
        EXPORTERS.register(c(), priority)
        return c
    return wrap(cls) if cls is not None else wrap


def model_parser(cls=None, *, priority: Optional[int] = None) -> Callable:
    def wrap(c):
        MODEL_PARSERS.register(c(), priority)
        return c
    return wrap(cls) if cls is not None else wrap


def extractor(cls=None, *, priority: Optional[int] = None) -> Callable:
    def wrap(c):
        EXTRACTORS.register(c(), priority)
        return c
    return wrap(cls) if cls is not None else wrap
