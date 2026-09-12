# Development notes

Architecture, commands, and conventions for working in this repository.

## What this is

Train action-recognition models on a user's own video dataset. A Django web
app covers labeling, detect+track extraction (for multi-person raw footage),
training, and inference; equivalent CLI tools (`ar-*` entry points) do the
same without the browser. Both read/write the same `data/`, `configs/`, and
`runs/` — the web app is an interface onto the library in `src/`, not a
separate implementation.

## Install

```bash
pip install -e ".[webapp,dev]"
```

Python 3.9+. `pretrained: true` (the default for model configs) needs
network access once to download ImageNet/Kinetics weights.

## Commands

```bash
pytest                                  # full suite (library + webapp)
pytest tests/test_models.py             # one file
pytest tests/test_models.py::test_name  # one test
pytest tests/webapp                     # webapp-only tests

python manage.py migrate && python manage.py runserver
```

`pytest` is configured via `[tool.pytest.ini_options]` in `pyproject.toml`:
`testpaths = ["tests"]`, and `pythonpath = ["."]` puts the repo root on
`sys.path` before `DJANGO_SETTINGS_MODULE = "webapp.settings"` is resolved
(pytest-django needs this earlier than any `conftest.py` runs, so it's an ini
option, not fixture setup). Run pytest from the repo root.

There's no configured lint/format command — match existing style rather than
introducing a new tool.

Every change to `src/` or `core/` should land with matching test coverage in
the same layout: library tests as `tests/test_<module>.py`, webapp/Django
tests under `tests/webapp/test_<module>.py`. Run the relevant file (or the
full suite) before considering a change done.

### CLI entry points (`pyproject.toml` `[project.scripts]`)

| command | module |
|---|---|
| `ar-train` | `action_recognition.training.train:main` |
| `ar-predict` | `action_recognition.inference.predict:main` |
| `ar-predict-scene` | `action_recognition.inference.scene_predict:main` |
| `ar-build-manifest` | `action_recognition.scripts.build_manifest:main` |
| `ar-split-dataset` | `action_recognition.scripts.split_dataset:main` |
| `ar-extract-tracks` | `action_recognition.scripts.extract_tracks:main` |
| `ar-check-setup` | `action_recognition.scripts.check_setup:main` |

## Pipeline (in order)

1. **(Optional) `ar-extract-tracks`** — for raw footage with multiple people
   in frame. Detects people (`tracking/detectors`), tracks them across frames
   (`tracking/trackers`), and crops one clip per tracked person per time
   window into `data/tracks/<video_stem>/`, plus a `tracks_index.csv`. From
   here on that folder is just an ordinary folder of single-person clips.
2. **Manifest** — `ar-build-manifest` (for `<root>/<class_name>/<video>`
   layouts) or the web app's Label page (for unlabeled clips) produces
   `data/manifest.csv`: columns `video_path, label`. This CSV is the single
   source of truth mapping videos to labels.
3. **Split** — `ar-split-dataset` adds a stratified `split` column
   (`train`/`val`/`test`, one label group at a time so every class appears in
   every split when it has enough examples) and writes `data/label_map.json`.
4. **Train** — `ar-train --config configs/<arch>.yaml`. Checkpoints
   (`best.pt`/`last.pt`) go to `train.output_dir` and bundle model weights +
   label map + the training config together, so inference needs nothing else.
5. **Inference** — `ar-predict` for a single pre-trimmed clip, or
   `ar-predict-scene` (detect+track, then classify each track) for raw
   multi-person video.

## Architecture

### Config loading (`training/train.py`)

Configs are YAML, deep-merged over the hardcoded `DEFAULTS` dict in
`train.py` (also documented, field-for-field, in `configs/default.yaml`), then
CLI flags (`--epochs`, `--manifest`, etc.) override the merged result. A
config file only needs to list what it changes. The same layered pattern
(defaults dict/YAML → deep merge → CLI override) applies to
`configs/tracking/default.yaml` for the detect/track/extract pipeline.

### Registries (three independent instances of the same pattern)

- `action_recognition.models.registry` — `@register_model("name")` on an
  `nn.Module`; `build_model(name, num_classes, **kwargs)`;
  `available_models()`.
- `action_recognition.tracking.detectors.registry` — `@register_detector`;
  detector classes implement `detect(frame) -> list[Detection]`.
- `action_recognition.tracking.trackers.registry` — `@register_tracker`.

Adding an implementation to any of these means: write the class, decorate it,
then **import the module in that package's `__init__.py`** (registration is
an import-time side effect — nothing is auto-discovered). Config files
reference implementations by their registered string name
(`model.name`, `tracking.detector.name`, `tracking.tracker.name`).

Model contract: `forward()` takes `(B, T, C, H, W)` and returns
`(B, num_classes)` logits — that's the only requirement to plug in a new
architecture.

### `src/action_recognition/` layout

```
data/        manifest I/O (manifest.py), video decoding, Dataset, transforms
models/      registry + cnn_lstm (2D CNN per-frame -> LSTM) + r3d (3D CNN, Kinetics-pretrained)
tracking/    detectors/ + trackers/ registries, extract.py (detect+track+crop pipeline), types.py
training/    config loading + CLI (train.py), train/eval loop (engine.py)
inference/   predict.py (single clip), scene_predict.py (multi-person: detect+track then classify each)
scripts/     build_manifest, split_dataset, extract_tracks — the standalone CLI wrappers
utils/       checkpoint save/load, logging, seeding
```

Checkpoints are self-contained: `save_checkpoint` writes
`{model_state, label_map, config, epoch, metric}` together, so any code
loading a `.pt` (inference, resuming, the webapp) doesn't need the manifest
or config file that produced it.

### Django project: `webapp/` settings package, `core/` app, root-level `manage.py`

`manage.py`, the `core` app, and the `webapp` package (settings/urls/wsgi/asgi
only — there is no separate `config/` package) all live directly at the repo
root, alongside `src/`, `tests/`, `configs/`, `data/`, `runs/`.

- `core/services/` holds the actual logic (`training.py`, `extraction.py`,
  `manifest.py`, `inference.py`); `core/views/` are thin wrappers around it.
- Training and extraction runs are launched via `subprocess` as **detached
  background OS processes** (`training.py`/`extraction.py` docstrings), not
  Celery/threads — so the Django dev server can stop and the run keeps going.
  Liveness/exit status is *not* trusted from the launching process's own
  handle (dev-server autoreload or multiple workers means a later request may
  be served by a different process). Instead each run's shell wrapper appends
  an explicit exit-code marker line to its log file, and `refresh_status()`
  reads that back plus an `os.kill(pid, 0)` check against the OS — apply the
  same pattern if you add another kind of background run.
  `TrainingRun`/`TrackExtractionRun` (`core/models.py`) store `status`/`pid`
  as last-known values refreshed on read, not as live state.
  `settings.CONFIGS_DIR` / `RUNS_DIR` / `DATA_DIR` resolve to the repo-root
  `configs/`, `runs/`, `data/` (`REPO_ROOT = BASE_DIR` in `webapp/settings.py`,
  since `manage.py` and `webapp/` are both at the repo root now) — the webapp
  and CLI share these directories, nothing is duplicated elsewhere.
- Webapp-owned state (`db.sqlite3`, uploaded inference videos under `media/`)
  is gitignored and lives at the repo root, next to `manage.py`.

## Data conventions

- `data/` is gitignored entirely except `README.md`/`.gitkeep` — it's where a
  user's own videos and generated artifacts (`manifest.csv`, `label_map.json`,
  `tracks/`) live, never committed.
- Manifest CSV columns: `video_path, label`, plus `split` once
  `stratified_split`/`ar-split-dataset` has run.
- Video extensions recognized throughout: `.mp4`, `.avi`, `.mov`, `.mkv`,
  `.webm` (`data/manifest.py::VIDEO_EXTENSIONS`).
- `runs/`, `checkpoints/`, `*.pt`/`*.pth` are gitignored — checkpoints are
  build artifacts of a training run, not source.

## Known limitations (don't silently "fix" without discussion — these are documented trade-offs)

- Frame sampling is uniform across a clip; no optical flow or dedicated
  motion features beyond what `r3d18`'s 3D convs learn implicitly.
- Per-frame augmentation (crop/flip) is independent per sampled frame, not
  consistent across a clip.
- One clip per video — long videos aren't split into multiple training clips.
- Default tracker (`iou`) is a greedy IoU matcher with no motion model; can
  lose identity through occlusion or fast/crowded scenes. Default detector
  only distinguishes "person" (COCO), no re-identification. Both are
  swappable via their registries without touching the extraction or
  scene-inference pipeline.
- Labels are single-label/softmax — no multi-label per-person tagging.

## Scaling direction (long-term target — not the current architecture)

The long-term goal is a multi-tenant SaaS shape (many users, each with their
own datasets, labeling, training runs, and inference). **None of this is
built.** The current design documented above — SQLite, subprocess-launched
background jobs (not a real queue), local filesystem storage under `data/`
and `media/`, Django's own dev server — is intentionally single-user
and should keep working as-is; don't start migrating pieces of it toward
this opportunistically. Getting to the multi-tenant target touches nearly
every layer at once (data model needs per-tenant isolation, background jobs
need a real broker + horizontally scalable GPU worker fleet in place of
`subprocess`, storage needs to move off the local filesystem, the web tier
needs to run behind a load balancer with per-tenant quotas on training
concurrency) — treat it as a deliberate, separately-scoped initiative to
design and sequence explicitly, not a background refactor.

## Attribution

Keep all committed artifacts (code, code comments, commit messages, branch
names, PR titles/descriptions) free of AI-tool attribution.
