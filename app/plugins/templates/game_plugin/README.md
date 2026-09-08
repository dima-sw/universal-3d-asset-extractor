# {{PLUGIN_NAME}}

Asset support plugin for the Universal 3D Asset Extractor.

## Try it before you change anything

The template ships with a made-up format so the whole pipeline works out of the
box. Build a sample file:

```bash
python make_sample.py
```

```bash
extractor detect sample_model.bin
```

```bash
extractor scan .
```

You should see a model found and, after `extractor extract . -o out`, a
`sample_model.glb` you can open in the viewer.

## Then make it yours

1. Replace `ARCHIVE_MAGIC` / `MODEL_MAGIC` in `plugin.py` with your game's real
   signatures. If the format has no magic, prove the structure instead: check
   that counts, offsets and sizes are consistent, and return a lower confidence.
2. Rewrite `{{CLASS_PREFIX}}Extractor.extract` for the real container layout.
   Anything you write out is re-scanned automatically, so a nested archive
   only needs one layer per plugin.
3. Rewrite `{{CLASS_PREFIX}}ModelParser.parse` to fill `Mesh` (positions,
   indices, normals, UVs, joints, weights) and, when the format has them,
   `Skeleton`, `Material` and `TextureAsset`.
4. Update `plugin.json`: name, version, supported extensions and engines.

## Useful tools while reverse engineering

```bash
extractor inspect path/to/file.bin
```

prints entropy per block, embedded assets found by signature, and the detection
result - usually enough to tell a container from a compressed blob.

```bash
extractor plugins check .
```

validates the manifest and the entry point before you install.

## Install

```bash
extractor plugins install .
```

Copies the folder into the user plugin directory. `extractor plugins list`
shows what is installed, `enable` / `disable` / `remove` manage it.

## Rules the loader enforces

* `plugin.json` must declare `name`, `version` and `entrypoint`
* the entry point must define `register(api)` at the top level
* write files only through `safe_join()` and account bytes with `budget`
* never execute anything found inside the game data
