"""{{PLUGIN_NAME}} - asset support plugin.

This file is a working starting point, not a stub: as shipped it recognises a
made-up container ("MYG0") and parses a made-up model ("MYM0"), so you can run
the pipeline end to end before you change anything, then replace the parsing
with your game's real structure.

Everything you may use lives in app.plugins.api. The core never imports this
file by name - the loader discovers it through plugin.json.

Workflow
--------
    extractor plugins check .              # validate this folder
    extractor detect  path/to/file.bin     # what does the app see?
    extractor inspect path/to/file.bin     # entropy, embedded assets, offsets
    extractor plugins install .            # copy into the user plugin folder
    extractor scan path/to/game            # run it for real

The four things a plugin can add
--------------------------------
    api.add_detector(...)          bytes  -> DetectionResult      (what is it?)
    api.add_extractor(...)         file   -> many files           (open it up)
    api.add_model_parser(...)      file   -> ModelAsset           (geometry)
    api.add_texture_parser(...)    file   -> TextureAsset         (pixels)
    api.add_animation_parser(...)  file   -> AnimationClip        (motion)

Produce the internal representation and you get GLB/glTF/OBJ/DAE export, the
3D viewer, character grouping and the reports for free.
"""
from __future__ import annotations

import struct
from pathlib import Path

from app.plugins.api import (Bone, Category, ContainerExtractor, DetectionResult,
                             ExtractionResult, FileContext, FormatDetector, GameProfile,
                             Material, Mesh, ModelAsset, ModelParser, Skeleton, TextureAsset)

# Replace these with your game's real signatures.
ARCHIVE_MAGIC = b"MYG0"
MODEL_MAGIC = b"MYM0"


# ---------------------------------------------------------------------------
# 1. Detection - decide from the bytes, never from the extension alone
# ---------------------------------------------------------------------------
class {{CLASS_PREFIX}}ArchiveDetector(FormatDetector):
    name = "{{PLUGIN_SLUG}}_archive"
    priority = 60            # 100 = magic bytes, 20 = filename hints

    def detect(self, ctx: FileContext) -> DetectionResult:
        if ctx.header[:4] != ARCHIVE_MAGIC or ctx.size < 12:
            return self.nope()
        count = struct.unpack_from("<I", ctx.header, 4)[0]
        if not (0 < count < 100000):
            return self.nope()          # structure has to make sense, or it is not ours
        return self.result("{{PLUGIN_NAME}} archive", 0.95, Category.GAME_CONTAINER,
                           entries=count)


class {{CLASS_PREFIX}}ModelDetector(FormatDetector):
    name = "{{PLUGIN_SLUG}}_model"
    priority = 60

    def detect(self, ctx: FileContext) -> DetectionResult:
        if ctx.header[:4] != MODEL_MAGIC:
            return self.nope()
        return self.result("{{PLUGIN_NAME}} model", 0.95, Category.MODEL)


# ---------------------------------------------------------------------------
# 2. Extraction - one file becomes many, which the pipeline re-scans by itself
# ---------------------------------------------------------------------------
class {{CLASS_PREFIX}}Extractor(ContainerExtractor):
    """Layout of the example container:

        magic  "MYG0"      4 bytes
        count  uint32      number of entries
        table  count x (uint32 offset, uint32 size)
        payloads...
    """

    name = "{{PLUGIN_SLUG}}_archive"
    priority = 60
    formats = ("{{PLUGIN_NAME}} archive",)

    def extract(self, ctx: FileContext, dest: Path, budget, limits) -> ExtractionResult:
        from app.core.fs_safety import safe_join           # always write through this
        res = ExtractionResult(extractor=self.name)
        header = ctx.read_at(0, 8)
        count = struct.unpack_from("<I", header, 4)[0]
        table = ctx.read_at(8, count * 8)
        if len(table) < count * 8:
            return self.unsupported("entry table is truncated")

        with open(ctx.path, "rb") as fh:
            for index in range(count):
                offset, size = struct.unpack_from("<II", table, index * 8)
                if size == 0 or offset + size > ctx.size:
                    res.errors.append("entry %d is out of range" % index)
                    continue
                if size > limits.max_single_file_size:
                    res.skipped.append("entry %d is too large" % index)
                    continue
                fh.seek(offset)
                payload = fh.read(size)
                out = safe_join(dest, "%05d.bin" % index)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(payload)
                budget.account(size)          # keeps archive bombs in check
                res.files.append(out)
                res.entries += 1
        res.ok = bool(res.files)
        return res


# ---------------------------------------------------------------------------
# 3. Parsing - produce the internal model, nothing format-specific escapes here
# ---------------------------------------------------------------------------
class {{CLASS_PREFIX}}ModelParser(ModelParser):
    """Layout of the example model:

        magic       "MYM0"    4 bytes
        vertexCount uint32
        indexCount  uint32
        vertices    vertexCount x (float x, y, z)
        indices     indexCount  x uint16
    """

    name = "{{PLUGIN_SLUG}}_model"
    priority = 60
    formats = ("{{PLUGIN_NAME}} model",)

    def parse(self, ctx: FileContext) -> ModelAsset:
        data = ctx.path.read_bytes()
        vertex_count, index_count = struct.unpack_from("<II", data, 4)
        mesh = Mesh(name=ctx.path.stem)

        offset = 12
        for _ in range(vertex_count):
            x, y, z = struct.unpack_from("<3f", data, offset)
            mesh.vertices.append((x, y, z))
            offset += 12
        for _ in range(index_count):
            mesh.indices.append(struct.unpack_from("<H", data, offset)[0])
            offset += 2

        # Optional pieces, fill them in when your format has them:
        #   mesh.normals      = [(x, y, z), ...]            one per vertex
        #   mesh.uv_channels  = [[(u, v), ...]]             one list per UV set
        #   mesh.joints       = [(j0, j1, j2, j3), ...]     skinning
        #   mesh.weights      = [(w0, w1, w2, w3), ...]
        #   skeleton = Skeleton(bones=[Bone("Root", -1), Bone("Spine", 0), ...])
        #   material = Material(name="body")
        #   material.textures["diffuse"] = TextureAsset(name="body_d", data=png_bytes)

        model = ModelAsset(name=ctx.path.stem, meshes=[mesh],
                           materials=[Material(name="%s_mat" % ctx.path.stem)],
                           source_format="{{PLUGIN_NAME}}")
        mesh.material = 0

        # If your game is left-handed (most console games are), say so and the
        # exporter converts winding, positions and rotations for you:
        #   model.coordinate_system = "y_up_lh"      # or "z_up_rh", "z_up_lh"
        return model


# ---------------------------------------------------------------------------
# 4. Registration - the only entry point the loader calls
# ---------------------------------------------------------------------------
def register(api) -> None:
    api.add_detector({{CLASS_PREFIX}}ArchiveDetector())
    api.add_detector({{CLASS_PREFIX}}ModelDetector())
    api.add_extractor({{CLASS_PREFIX}}Extractor())
    api.add_model_parser({{CLASS_PREFIX}}ModelParser())
    api.add_game_profile(GameProfile(
        name="{{PLUGIN_NAME}}",
        platforms=[],                     # e.g. ["ps2"], ["pc"]
        detectors=[{{CLASS_PREFIX}}ArchiveDetector()],
        extractors=[{{CLASS_PREFIX}}Extractor()],
        model_parsers=[{{CLASS_PREFIX}}ModelParser()],
    ))
