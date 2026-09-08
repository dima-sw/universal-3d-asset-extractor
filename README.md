# Universal 3D Asset Extractor

Desktop application and CLI that scans a folder recursively — through archives,
disc images and game containers — identifies what is inside by **content, not by
file extension**, rebuilds usable 3D assets in an engine-neutral internal
representation, and exports them as GLB/glTF/OBJ/DAE with textures, skeletons
and animations.

The point of the project is the **platform**, not any single game: a new format
becomes a plugin, and the core never changes.

```
folder → recursive scan → detection → extraction/mount → re-scan → parsing
       → internal asset model → asset graph → character reconstruction
       → normalisation → export → viewer / report
```

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

```bash
python main.py scan "D:/Games/MyGame"
```

```bash
python main.py extract "D:/Games/MyGame" --output "D:/Extracted" --format glb --deep
```

Other commands:

```bash
python main.py detect file.bin
```

```bash
python main.py inspect file.bin
```

```bash
python main.py list-plugins
```

The GUI (`python main.py`) has Analyze / Extract as separate steps, live scan
stats, pause / resume / cancel, a results tree grouped by Characters / Models /
Animations / Textures, an Unsupported tab, an Errors tab, and a 3D preview with
orbit-zoom-pan, solid/wireframe/textured modes, a bone overlay, and animation
playback with a timeline.

## Architecture

```
app/
├── config.py               settings, limits, coordinate config
├── bootstrap.py            wires built-ins + plugins (no UI import)
├── cli.py                  CLI (same core as the GUI)
├── core/
│   ├── types.py            internal asset model (Mesh/Skeleton/AnimationClip/...)
│   ├── registry.py         extension points: detectors, extractors, parsers, exporters
│   ├── fs_safety.py        path traversal, symlinks, size/entry/ratio budgets
│   ├── hashing.py          quick key + SHA-256 (content identity, dedup)
│   ├── validation.py       mesh/skeleton validation + confidence scoring
│   ├── detector/           magic bytes, structure, engine signatures, heuristics
│   ├── extraction/         archives, ISO 9660 / BIN-CUE reader
│   ├── conversion/         normalisation: up-axis, handedness, unit scale
│   ├── graph/              asset relationship graph + character detection
│   └── pipeline/           states, jobs/workers, controller, exporting
├── formats/                models (OBJ/PLY/STL/glTF), textures, animations (BVH)
├── exporters/              glb, gltf, obj, dae (fbx reports unsupported)
├── plugins/                api.py, loader.py, builtin/{unity,unreal,playstation,generic}
├── database/               SQLite schema + cache/persistence
└── ui/                     PySide6 window, worker threads, OpenGL viewer
```

Rules the code follows:

* the core never imports the UI; the CLI and GUI are both clients of
  `ExtractorCore`
* parsers never talk to exporters — everything goes through the internal
  representation, so a new exporter works with every existing parser and vice
  versa
* one file failing never stops a scan; failures land in the report
* the source tree is read-only, always

## Detection

Order of evidence: magic bytes → header structure → file structure → filename →
directory structure → engine signatures → heuristics. A `.bin` is analysed, not
skipped; a `.png` that isn't a PNG is not believed.

Unknown blobs (Deep / Aggressive modes) go through entropy profiling, embedded
signature scanning (`EmbeddedAssetScanner` reports e.g. `PNG @ 0x13A400,
1024x1024`) and compression detection (zlib/gzip/lzma/zstd/lz4).

## Safety

Every scanned byte is treated as hostile:

* archive members are sanitised and confined (zip-slip, absolute paths, drive
  letters, `..`), symlinks are skipped
* extraction budgets cap total size, entry count and compression ratio
  (archive bombs)
* nesting depth and directory depth are capped; already-seen container hashes
  and real paths break cycles
* nothing found while scanning is ever executed, imported or loaded as a plugin
* temp data lives in a per-job directory and is cleaned up (kept on error for
  debugging when enabled)

The tool is for content you are allowed to extract. It contains no DRM or
copy-protection circumvention, and none should be added.

## What works today

| Area | Status |
| --- | --- |
| Recursive scan through nested containers | working |
| ZIP / TAR / GZIP / BZIP2 / XZ | working |
| 7z, RAR | working when `py7zr` / `rarfile` are installed, otherwise reported |
| ISO 9660 (2048 and 2352-byte sectors, Joliet), BIN/CUE | working, streamed, no full-image dump |
| Unity `UnityFS` bundles | header + block table + member split (LZ4 pure-python, LZMA, none) |
| Unity SerializedFile (v13-22) | TypeTree-driven object decoding; **release builds with the type tree stripped** fall back to class layouts validated against real type trees |
| Unity `Mesh` → GLB | vertex streams **and** compressed (PackedBitVector) meshes, submeshes, skin weights, bone names/hierarchy from SkinnedMeshRenderer + Transform, bone transforms re-derived from the bind pose, left-handed → glTF conversion |
| Unity `Texture2D` → PNG | uncompressed formats, DXT1/DXT5/BC4/BC5, `.resS` streams, DXT5nm un-swizzling |
| Unreal `.pak` versions 1-11 | classic **and** UE5 path-hash/full-directory index, Zlib/Gzip/LZ4/Zstd, plus Oodle when an `oo2core` runtime you already own is available |
| Unreal package (`.uasset` + `.uexp`/`.ubulk`) | summary, name table, imports, exports, tagged properties - layout is derived from the tables themselves, so unversioned and licensee builds read without a version database |
| Unreal `Texture2D` → PNG (UE4) | DXT1/DXT5/BC4/BC5 and packed formats, mips from `.uexp` or `.ubulk` |
| glTF 2.0 / GLB | full parse and export: mesh, UVs, materials, textures, skin, animations, morph targets |
| OBJ/MTL, PLY (ascii+binary), STL (ascii+binary) | parse |
| BVH | parse to `AnimationClip` + skeleton |
| PS1 TIM textures | decoded to PNG |
| CRI `AFS` archives (PS2/Dreamcast) | extracted with their real file names |
| Generic offset-table containers | structure proven by validation, then peeled open (the workhorse layout of PS1/PS2 games) |
| Frostbite / Anvil / Wwise / PSARC / Decima | fingerprinted and named precisely (no reader) |
| PNG/JPEG/BMP/GIF/DDS/KTX/KTX2/TGA headers | identified, dimensions preserved, converted via Pillow |
| Character detection, humanoid rig heuristics, asset graph | working |
| GLB / glTF / OBJ / DAE export, manifests, HTML+JSON reports | working |
| SQLite cache, dedup by SHA-256, resume | working |
| 3D viewer with CPU skinning and animation playback | working |

## What is detected but not yet parsed

These report precisely what they are and what is missing — never "corrupt":

* Unity `Material` / `SkinnedMeshRenderer` in stripped release builds: textures
  are extracted and matched to models by name and path, but not through the
  material's texture slots
* Unity `AnimationClip` curves, `Avatar` rigs, `MonoBehaviour` scripts;
  crunched, BC6H/BC7, ETC/ASTC/PVRTC textures
* Frostbite (`cas`/`cat`/`toc`/`sb`), Ubisoft Anvil (`.forge`), Decima, PSARC:
  identified by engine, no container reader
* Per-game PS2 formats: the pipeline reaches them (ISO → AFS → offset-table
  container → payloads) but the payloads themselves are title-specific
* Unreal `StaticMesh` / `SkeletalMesh` geometry: packages are read and the
  meshes are listed with their material slots, but the render-data blob is not
  decoded yet - that is the next step
* Unreal UE5 texture mip records (the header is found, the mip layout differs),
  IoStore `.utoc/.ucas`, AES-encrypted paks
* Oodle-compressed data when no `oo2core` library is present: point the app at
  one with `--oodle-dll` or the `oodle_dll` setting (games that ship Oodle data
  usually ship the library too; this app never bundles it)
* PS1 `TMD`/`HMD` geometry (primitive packet layouts are title-dependent),
  TIM2, and per-game proprietary archives
* Source, RenderWare, Godot, CryEngine: detected by signature/engine
  fingerprint, no asset parsers yet
* UDF / XDVDFS / GameCube-Wii disc filesystems
* FBX **writing** — the exporter says so and points at GLB rather than emitting
  a file other tools cannot open

## Adding a format (plugin)

Nothing here supports Tomb Raider on PS2. A user can add it themselves, and
everyone who installs their plugin gets that game:

```bash
extractor plugins new "Tomb Raider PS2" --engine PS2
```

That scaffolds a **working** plugin (it recognises and parses a made-up format
out of the box, so the pipeline can be run before any reverse engineering),
with `plugin.json`, `plugin.py`, `make_sample.py` and a README. Edit the
detector, the extractor and the parser, then:

```bash
extractor plugins check tomb_raider_ps2 --load
```

```bash
extractor plugins install tomb_raider_ps2
```

```bash
extractor plugins list
```

| Command | Does |
| --- | --- |
| `plugins new NAME` | scaffold a working skeleton |
| `plugins check PATH [--load]` | validate the manifest and entry point, optionally import it and print what it registered |
| `plugins install PATH` | install a folder or a `.zip` (asks for confirmation first) |
| `plugins list` | name, version, source, state, engines, errors |
| `plugins enable/disable NAME` | turn one off without deleting it |
| `plugins remove NAME` | delete a user-installed plugin |
| `plugins reload` | pick up edits without restarting |

The GUI has the same functions under the **Plugins** tab, including
*Install .zip...* for sharing plugins with other people.

A plugin fills the internal representation and gets GLB/glTF/OBJ/DAE export,
the 3D viewer, character grouping, dedup, manifests and reports for free. The
five hooks:

```python
def register(api):
    api.add_detector(...)           # bytes  -> DetectionResult
    api.add_extractor(...)          # file   -> many files (re-scanned automatically)
    api.add_model_parser(...)       # file   -> ModelAsset
    api.add_texture_parser(...)     # file   -> TextureAsset
    api.add_animation_parser(...)   # file   -> AnimationClip
```

**[docs/PLUGINS.md](docs/PLUGINS.md)** is the full guide, written around the
Tomb Raider PS2 example: how to read an unknown file, how to prove a format
without a magic number, the safety contract, and how to report a format you
could not crack instead of guessing.

Plugins are Python and run with the app's permissions, so only two locations
are ever imported: the built-in folder and the user plugin folder
(`~/.universal3dextractor/plugins`). Nothing found while scanning a game is
ever loaded, whatever it claims to be.

## Output layout

```
Extracted/
├── Characters/Character_A/{Character_A.glb, textures/, animations/, manifest.json}
├── Models/          Props/          Environment/
├── Textures/        loose textures not attached to a model
├── Raw/             untouched copies of the source assets (optional)
└── Reports/{report.html, report.json, asset_graph.json}
```

## Settings that matter

Scan mode (Standard / Deep / Aggressive — Aggressive is off by default because
it is slow), worker count, max archive nesting (default 10), max extracted size,
max single file size, duplicate policy, overwrite policy, temp directory,
coordinate system (default Y-up, right-handed, 1 unit = 1 m; originals are kept
in metadata).

## Tests

```bash
python -m pytest tests -q
```

67 tests cover detection, archives, ISO, models, exporters, animation, the
pipeline, plugins, resume, the Unity reader (type tree, packed bit vectors, BC
decoding, handedness conversion) and security (zip-slip, symlink members,
archive bombs, entry limits, corrupted and truncated files, nesting, source
immutability). They use hand-built byte buffers only - no game files needed.

The Unity layouts are checked against type trees dumped from a real Unity build
(`tests/data/ref_*.txt`): 36/36 and 221/221 nodes must match exactly, including
alignment flags and matrix element order.

Validated end to end against installed games:

| Game | Engine | Result |
| --- | --- | --- |
| Moving Out | Unity 2018.4, bundles with type tree | 2,457 GLB models, 25 characters, 0 failures, 74 s |
| (same, texture bundle) | Unity 2018.4 | 268 of 273 textures; the 5 misses are BC7/crunched, named |
| Boomerang Fu | Unity 2019.2, **type tree stripped** | 888 models, 364 textures, 22 characters, 0 failures |
| Battlefield 1 | Frostbite | identified as Frostbite (99%), every container named; no reader |
| Assassin's Creed Odyssey | Anvil | identified as Anvil (99%); `.forge` named; no reader |
| Dragon Ball BT3 (PS2 ISOs) | proprietary | ISO → AFS (real names) → offset-table containers → title-specific payloads |
| It Takes Two | Unreal 4, pak v11 Zlib | pak index, 150/150 packages parsed, 129 textures decoded to PNG |
| Chained Together | Unreal 5, pak v11 Oodle | pak index and packages parsed 150/150 (Oodle via the user's own runtime); texture mips pending |
| Days Gone | Unreal 4 licensee, pak v3 Oodle | pak index and packages parsed 150/150 |
| Split Fiction | Unreal 5, pak v11 Oodle | pak index read (54,016 entries) |

The OpenGL viewer needs a real GL context, so it has a manual check instead:

```bash
python tests/manual_viewer_check.py path/to/model.glb
```

## Roadmap

* **MVP 1 (done)** — scanner, detection, archives, ISO, cache, plugin
  architecture, texture/model detection, progress + error handling
* **MVP 2 (done for Unity)** — glTF/GLB export, viewer, skeleton, animation,
  character grouping, Unity mesh/texture extraction; Unreal asset-level parsing
  (`.uasset` object graph) is the remaining piece
* **MVP 3 (in progress)** — the format roadmap, in this order:
  1. Unreal: pak (done), package reader (done), textures (UE4 done, UE5 mips
     pending), **mesh geometry next**
  2. RenderWare DFF/TXD (~30-40% of PS2 titles, plus the GTA-era PC games)
  3. Generic PS2 VIF/GIF geometry ripper (fallback for the rest: mesh and UVs,
     rarely rigs)
  4. PS1 TMD/PMD geometry and TIM2 textures
  5. Source (MDL/VTX/VVD + VPK), Godot, CryEngine, id Tech
  6. FBX reading, then FBX writing
  7. BC7/BC6H, crunched, ASTC/ETC/PVRTC textures, Oodle
  8. Re-injection: putting a modified model back into the original container
  9. Heuristic geometry sniffer for formats nobody has documented

  Everything a plugin cannot cover stays a plugin: the roadmap fills the
  shared layers, users fill the per-game ones.
