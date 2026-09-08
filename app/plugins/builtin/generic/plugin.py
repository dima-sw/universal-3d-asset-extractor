"""Template plugin: copy this directory to add support for one game or format.

    plugins/my_game/
        plugin.json      name, version, api_version, entrypoint
        plugin.py        register(api)

Nothing in app/core changes when you add a plugin. The four hooks are:

    api.add_detector(...)          bytes  -> DetectionResult
    api.add_extractor(...)         file   -> many files (re-scanned automatically)
    api.add_model_parser(...)      file   -> ModelAsset (internal representation)
    api.add_texture_parser(...)    file   -> TextureAsset
    api.add_animation_parser(...)  file   -> AnimationClip
    api.add_exporter(...)          ModelAsset -> files
    api.add_game_profile(...)      groups the above under one game/engine

The example below detects a made-up container and shows the required shapes.
It is intentionally inert on real data: the magic never occurs in the wild.
"""
from __future__ import annotations

import struct

from app.plugins.api import (Category, ContainerExtractor, DetectionResult, ExtractionResult,
                             FileContext, FormatDetector, GameProfile, Mesh, ModelAsset,
                             ModelParser)

EXAMPLE_MAGIC = b"XYZ0"


class ExampleDetector(FormatDetector):
    name = "example_container"
    priority = 50

    def detect(self, ctx: FileContext) -> DetectionResult:
        if ctx.header[:4] != EXAMPLE_MAGIC:
            return self.nope()
        count = struct.unpack_from("<I", ctx.header, 4)[0] if len(ctx.header) >= 8 else 0
        return self.result("Example Game Container", 0.95, Category.GAME_CONTAINER,
                           entries=count, engine="ExampleEngine")


class ExampleExtractor(ContainerExtractor):
    name = "example_container"
    priority = 50
    formats = ("Example Game Container",)

    def extract(self, ctx: FileContext, dest, budget, limits) -> ExtractionResult:
        # Real plugins stream entries out with safe_join() and budget.account().
        return self.unsupported("template plugin: nothing to extract")


class ExampleModelParser(ModelParser):
    name = "example_model"
    priority = 50
    formats = ("Example Game Model",)

    def parse(self, ctx: FileContext) -> ModelAsset:
        model = ModelAsset(name=ctx.path.stem, source_format="Example")
        model.meshes.append(Mesh(name="mesh", vertices=[], indices=[]))
        return model


def register(api) -> None:
    api.add_detector(ExampleDetector())
    api.add_extractor(ExampleExtractor())
    api.add_game_profile(GameProfile(name="Example Game", detectors=[ExampleDetector()],
                                     extractors=[ExampleExtractor()], priority=10))
