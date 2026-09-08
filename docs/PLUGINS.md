# Writing a plugin

A plugin teaches the app one game or one format. It is a folder with a manifest
and a Python file, and it never requires a change to the core. Ship it, and
whoever installs it gets that game supported.

Worked example throughout: *Tomb Raider on PS2*, a game nothing here supports.

---

## 1. Create the skeleton

```bash
extractor plugins new "Tomb Raider PS2" --engine PS2
```

```
tomb_raider_ps2/
├── plugin.json      manifest
├── plugin.py        detectors, extractors, parsers, register(api)
├── make_sample.py   builds sample files in the template's own format
└── README.md
```

The skeleton already works on its own made-up format, so you can watch the
whole pipeline run before touching anything:

```bash
python tomb_raider_ps2/make_sample.py
```

```bash
extractor extract tomb_raider_ps2 -o out
```

You get `out/Props/sample_model/sample_model.glb`. Now replace the made-up parts
with the real ones.

---

## 2. Look at the real files first

```bash
extractor inspect "F:/games/tomb_raider/DATA/CHAR.BIN"
```

prints the detection result, entropy per block (flat 7.9 means compressed or
encrypted, 4-6 means structured data), any embedded assets found by signature,
and the file size. That is usually enough to tell a container from a model.

Useful questions:

* does it start with 4 readable letters? that is your magic
* is there a count followed by rising offsets? that is a container
* do the same 32 or 48 bytes repeat? that is a table of records

---

## 3. Detection

A detector proves what a file is. Bytes decide, never the extension alone.

```python
class TombRaiderPs2ArchiveDetector(FormatDetector):
    name = "tomb_raider_ps2_archive"
    priority = 60          # 100 = magic table, 20 = filename hints

    def detect(self, ctx: FileContext) -> DetectionResult:
        if ctx.header[:4] != b"TR2\x00":
            return self.nope()
        count = struct.unpack_from("<I", ctx.header, 4)[0]
        if not (0 < count < 100000):
            return self.nope()        # implausible: not ours after all
        return self.result("Tomb Raider PS2 archive", 0.95,
                           Category.GAME_CONTAINER, entries=count)
```

`ctx` gives you `header` (first 64 KB), `tail`, `read_at(offset, length)`,
`size`, `ext`, `name`, `siblings`, `looks_textual()`. Reads are lazy, so
inspecting a 4 GB file costs nothing.

**No magic at all?** Prove the structure and return a lower confidence, the way
`plugins/retro_containers` does: offsets must be aligned, strictly increasing,
inside the file, and must tile it with little slack. A wrong guess costs the
user a broken extraction, so make the check strict.

Categories that matter: `GAME_CONTAINER` and `ARCHIVE` get an extractor called,
`MODEL` gets a model parser, `TEXTURE` a texture parser, `ANIMATION` an
animation parser.

---

## 4. Extraction

An extractor turns one file into many. Whatever you write out is re-scanned
automatically, so nested archives need one plugin layer each, not a recursive
monster.

```python
class TombRaiderPs2Extractor(ContainerExtractor):
    name = "tomb_raider_ps2_archive"
    formats = ("Tomb Raider PS2 archive",)

    def extract(self, ctx, dest, budget, limits) -> ExtractionResult:
        res = ExtractionResult(extractor=self.name)
        count = struct.unpack_from("<I", ctx.read_at(4, 4), 0)[0]
        table = ctx.read_at(8, count * 8)
        with open(ctx.path, "rb") as fh:
            for i in range(count):
                offset, size = struct.unpack_from("<II", table, i * 8)
                if offset + size > ctx.size:
                    res.errors.append("entry %d out of range" % i)
                    continue
                fh.seek(offset)
                out = safe_join(dest, "%05d.dat" % i)   # never build paths by hand
                out.write_bytes(fh.read(size))
                budget.account(size)                    # bomb protection
                res.files.append(out)
                res.entries += 1
        res.ok = bool(res.files)
        return res
```

Rules:

* write only through `safe_join(dest, name)`
* call `budget.account(size)` for every byte
* honour `limits.max_single_file_size` and `limits.max_entries_per_archive`
* never raise for bad content: collect `res.errors` / `res.skipped`
* cannot handle a variant? `return self.unsupported("why, and what is missing")`

---

## 5. Parsing into the internal model

This is the part that pays off: fill the internal representation and you get
GLB/glTF/OBJ/DAE export, the 3D viewer, character detection, dedup, manifests
and reports for free.

```python
class TombRaiderPs2ModelParser(ModelParser):
    name = "tomb_raider_ps2_model"
    formats = ("Tomb Raider PS2 model",)

    def parse(self, ctx: FileContext) -> ModelAsset:
        mesh = Mesh(name=ctx.path.stem)
        mesh.vertices = [(x, y, z), ...]
        mesh.indices = [0, 1, 2, ...]          # triangle list
        mesh.normals = [...]                    # optional, one per vertex
        mesh.uv_channels = [[(u, v), ...]]      # optional, one list per set
        mesh.joints = [(j0, j1, j2, j3), ...]   # optional, skinning
        mesh.weights = [(w0, w1, w2, w3), ...]

        skeleton = Skeleton(name="rig", bones=[
            Bone("Root", parent=-1),
            Bone("Spine", parent=0, translation=(0, 0.2, 0)),
        ])

        model = ModelAsset(name=ctx.path.stem, meshes=[mesh], skeleton=skeleton,
                           source_format="Tomb Raider PS2")
        model.coordinate_system = "y_up_lh"     # most console games are left-handed
        return model
```

Set `coordinate_system` correctly (`y_up_rh`, `y_up_lh`, `z_up_rh`, `z_up_lh`)
and the normaliser fixes axes, winding and rotations on export instead of you
doing it by hand.

Textures work the same way:

```python
tex = TextureAsset(name="lara_body", width=256, height=256,
                   data=png_bytes,          # already encoded, or
                   path="/tmp/lara.png")    # a file on disk
material.textures["diffuse"] = tex          # diffuse|normal|roughness|metallic|ao|emission|mask
```

Animations:

```python
clip = AnimationClip(name="walk", duration=1.2, fps=30, tracks=[
    AnimationTrack(bone="Spine",
                   rotations=[Keyframe(0.0, (0, 0, 0, 1)),
                              Keyframe(1.2, (0, 0.1, 0, 0.99))]),
])
```

---

## 6. Register

```python
def register(api) -> None:
    api.add_detector(TombRaiderPs2ArchiveDetector())
    api.add_extractor(TombRaiderPs2Extractor())
    api.add_model_parser(TombRaiderPs2ModelParser())
    api.add_texture_parser(...)      # optional
    api.add_animation_parser(...)    # optional
    api.add_exporter(...)            # optional, a new output format
    api.add_game_profile(GameProfile(name="Tomb Raider PS2", platforms=["ps2"]))
```

`register(api)` is the only thing the loader calls.

---

## 7. Validate, install, share

```bash
extractor plugins check tomb_raider_ps2 --load
```

checks the manifest and entry point, then imports it and prints exactly what it
registered.

```bash
extractor plugins install tomb_raider_ps2
```

copies it into the user plugin folder (`~/.universal3dextractor/plugins`). Zip
that folder and anyone can install it the same way, or through the GUI's
**Plugins** tab → *Install .zip...*.

```bash
extractor plugins list       # what is installed, and its state
extractor plugins disable "Tomb Raider PS2"
extractor plugins remove  "Tomb Raider PS2"
extractor plugins reload     # pick up edits without restarting
```

---

## Manifest reference

```json
{
    "name": "Tomb Raider PS2",
    "id": "tomb_raider_ps2",
    "version": "0.1.0",
    "author": "you",
    "api_version": 1,
    "supported_extensions": ["tr2", "drm"],
    "supported_engines": ["PS2"],
    "entrypoint": "plugin.py"
}
```

`name`, `version` and `entrypoint` are required. `api_version` higher than the
app's is refused with a clear message rather than crashing.

---

## Safety contract

Plugins are Python and run with the app's permissions, so:

* only two locations are ever imported: the built-in folder and the user plugin
  folder. Nothing found while scanning a game is ever loaded, whatever it claims
  to be
* the GUI and CLI both warn before installing
* a plugin must not execute files it finds, spawn processes from scanned data,
  or write outside `dest`
* the source tree stays read-only: work in `dest` and the temp folder

## Debugging

```bash
extractor detect file.bin        # what detector wins, and why
extractor inspect file.bin       # entropy, embedded assets, offsets
extractor scan folder -v         # DEBUG logging, per-file decisions
```

Enable **Developer mode** in the GUI settings to see detector confidence, the
chosen parser and the asset graph per selected item.

## When a format beats you

Report it instead of guessing. A precise "detected, unsupported, here is why"
is a useful result: it tells the next person exactly what to implement, and it
keeps a wrong mesh out of the user's output.

```python
return self.unsupported("Tomb Raider PS2 v2 archives use LZSS blocks; "
                        "the decompressor is not implemented")
```
