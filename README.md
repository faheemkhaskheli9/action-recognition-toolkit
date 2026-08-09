# Action Recognition

Train action recognition models on **your own** video dataset: label clips in
a Django web app, then train and compare different model architectures on
them — no need to hand-organize files or write training code per experiment.
Raw footage with several people in frame (a restaurant, warehouse, retail
floor, anything) isn't a special case either — a detect + track step turns it
into single-person clips first, so the taxonomy you train on is entirely
yours to define.

## Install

```bash
pip install -e ".[webapp,dev]"
```

Requires Python 3.9+. PyTorch/torchvision wheels are large; if you're offline
afterward, note that architectures using `pretrained: true` (the default) need
network access once to download ImageNet/Kinetics weights.

## Web app

`webapp/` is a Django app (HTML/CSS/JS front end, Python/Django back end)
covering the whole workflow in a browser: label videos, start training runs
(as background processes, no terminal needed), watch their live log, browse
past runs and their checkpoints, and run inference on an uploaded video.

```bash
cd webapp
python manage.py migrate
python manage.py createsuperuser  # first run only — every page requires login
python manage.py runserver
```

Open http://127.0.0.1:8000/ and log in with the account you just created.
For additional logins later, run `createsuperuser` again or use
`manage.py shell` + `User.objects.create_user(...)` for a non-admin login.
Don't mark extra accounts `is_staff` unless they're meant to have
unmediated `/admin/` access to the run database — `/admin/` gates itself
separately from the rest of the app (see `core/middleware.py`) and isn't
scoped per-page.

Pages:

- **Label** — drop video files (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
  anywhere under `data/raw/` (flat or nested, layout doesn't matter), point
  the form at that folder, and label each clip (pick an existing label or
  type a new one). Labels are written incrementally to `data/manifest.csv`,
  so you can stop and resume anytime.
- **Extract tracks** — got raw footage with multiple people in frame instead
  of pre-trimmed single-person clips? Point this at a folder of those videos;
  it detects + tracks every person (as a detached background process) and
  writes one short cropped clip per tracked person under `data/tracks/`.
  That folder is then just another folder of clips — hand it to the Label
  page below like any other.
- **Extraction runs** — list of past/running extraction runs and their logs.
- **Review manifest** — edit or delete previously labeled rows.
- **Train** — pick a config, optionally override manifest/model/epochs/batch
  size, and launch `ar-train` as a detached background process.
- **Runs** — list of past/running runs; a run's detail page tails its log
  live and lists any `best.pt`/`last.pt` it produced.
- **Inference** — upload a video (or pick one already in your dataset), pick
  a checkpoint from a finished run, get top-k predictions. Check "scene
  mode" if the video has multiple people — it detects+tracks each one and
  returns a per-person timeline instead of a single whole-clip prediction.

It reads/writes the same `data/`, `configs/`, and `runs/` the CLI tools below
use — nothing is duplicated, the web app is just an interface onto them. Its
own state (`db.sqlite3` tracking runs, uploaded inference videos in `media/`)
lives under `webapp/` and is gitignored.

### Environment variables

Unset, these default to `runserver`-on-localhost behavior — nothing below
is required for local development. Set all three before deploying anywhere
reachable by someone other than the operator at their own keyboard: this
app handles uploaded video and per-person track data.

| Variable | Default | Purpose |
|---|---|---|
| `DJANGO_SECRET_KEY` | a fixed insecure dev key | Django's cryptographic signing key. Generate a real one (`django.core.management.utils.get_random_secret_key()`) per deployment. |
| `DJANGO_DEBUG` | `true` | Set `false` for any non-localhost deployment. Also turns on `SECURE_SSL_REDIRECT`/`SESSION_COOKIE_SECURE`/`CSRF_COOKIE_SECURE`/HSTS (`webapp/config/settings.py`), so only set this once the app is actually served over HTTPS behind a real web server — `runserver` itself is plain HTTP and would lock you out. |
| `DJANGO_ALLOWED_HOSTS` | empty (only `runserver`'s implicit localhost) | Comma-separated hostnames, e.g. `example.com,www.example.com`. |

Not configurable via environment variable, deliberate for v1.0: session/
CSRF cookie lifetime (Django's 2-week default) and login rate-limiting/
lockout — reasonable for a single-operator deployment with no public
signup; revisit if that changes. See `docs/plans/v1-release-plan.md` §A4.

## CLI workflow

Everything the web app does is also available from the command line.

### 0. Multi-person scenes: detect + track people (optional)

Skip this step if your videos are already trimmed to one subject each. If
instead you have raw footage with several people in frame — a restaurant,
warehouse, retail floor, anything — run detection + tracking first to turn
it into the single-person clips the rest of the pipeline expects:

```bash
ar-extract-tracks data/raw_scenes data/tracks --config configs/tracking/default.yaml
```

This detects people (COCO-pretrained torchvision Faster R-CNN by default),
tracks them across frames (a dependency-free greedy IoU tracker by default —
both are swappable, see `configs/tracking/default.yaml`), and crops+writes a
short clip per tracked person per time window under
`data/tracks/<video_stem>/`, plus a `tracks_index.csv` for provenance. Point
`ar-build-manifest`/the Label page at `data/tracks` from here on — it's an
ordinary flat folder of clips, so whatever action taxonomy you label them
with (staff tasks, customer behavior, anything) trains the same way as any
hand-trimmed dataset.

### 1. Build a manifest

**Already have videos organized as `<root>/<class_name>/<video file>`?**
Build the manifest directly instead of using the Label page:

```bash
python -m action_recognition.scripts.build_manifest data/raw data/manifest.csv
```

### 2. Split into train/val/test

```bash
python -m action_recognition.scripts.split_dataset data/manifest.csv
```

This adds a `split` column (stratified per class) and writes
`data/label_map.json`. Tune with `--val-frac`, `--test-frac`, `--seed`.

### 3. Train

```bash
ar-train --config configs/cnn_lstm.yaml
# or
ar-train --config configs/r3d18.yaml
```

Common overrides don't need a new config file:

```bash
ar-train --config configs/cnn_lstm.yaml --manifest data/manifest.csv --epochs 30 --batch-size 16
```

Checkpoints (`best.pt`, `last.pt`) land in `train.output_dir` from the config.
Each checkpoint bundles the model weights, the label map, and the config used
to train it, so inference doesn't need any of that repeated.

### 4. Run inference

```bash
ar-predict runs/cnn_lstm/best.pt path/to/new_video.mp4
```

### 5. Multi-person scene inference (optional)

To classify every person in a new raw multi-person video with an already
trained checkpoint — not just a single pre-trimmed clip — detect+track runs
again, this time feeding each track straight into the model instead of
writing clip files:

```bash
ar-predict-scene runs/cnn_lstm/best.pt path/to/scene_video.mp4 --annotate out.mp4
```

Prints a JSON timeline (one entry per tracked person, per time window: box,
label, confidence); `--annotate` also writes a copy of the video with boxes
and labels burned in.

## Architectures

Two are included, both selectable via `model.name` in a config:

| name       | approach                                            | config              |
|------------|------------------------------------------------------|---------------------|
| `cnn_lstm` | Per-frame 2D CNN (ResNet-18) features fed into an LSTM | `configs/cnn_lstm.yaml` |
| `r3d18`    | 3D CNN (ResNet-style spatiotemporal convs, Kinetics-pretrained) | `configs/r3d18.yaml`   |

### Adding your own architecture

Every model is a `nn.Module` whose `forward()` takes a `(B, T, C, H, W)` clip
tensor and returns `(B, num_classes)` logits — that's the only contract. To
add one:

```python
# src/action_recognition/models/my_model.py
from .registry import register_model
import torch.nn as nn

@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        ...
    def forward(self, x):  # x: (B, T, C, H, W)
        ...
```

Then import it in `src/action_recognition/models/__init__.py` next to the
existing `cnn_lstm, r3d` import, and reference `name: my_model` from a config.
`ar-train --help` doesn't currently print the registry — check
`action_recognition.models.available_models()` if you forget a name.

## Config reference

See `configs/default.yaml` for every available field with comments. A config
file only needs to override what it changes from those defaults.

## Project layout

```
src/action_recognition/
  data/        manifest I/O, video decoding, dataset, transforms
  models/      architecture registry + implementations
  tracking/    person detector + tracker registries, detect/track/crop pipeline
  training/    config loading, train/eval loop, CLI
  inference/   predict CLI (single clip) + scene_predict CLI (multi-person)
  scripts/     build_manifest, split_dataset, extract_tracks CLIs
webapp/        Django app: track extraction + labeling + training job
               management + run dashboard + inference (single or scene mode)
configs/       YAML configs, one per architecture + the defaults;
               configs/tracking/ for detector/tracker/extraction settings
tests/         pytest suite (manifest, dataset, model registry, tracking)
```

## Known limitations

- Frame sampling is uniform across the clip duration; there's no optical-flow
  or motion-specific feature extraction beyond what `r3d18`'s 3D convolutions
  learn implicitly.
- Data augmentation (random crop/flip) is applied independently per sampled
  frame rather than consistently across a clip's frames.
- Single-clip-per-video sampling — long videos aren't split into multiple
  training clips.
- The default tracker (`iou`) is a simple greedy matcher with no motion
  model — it can lose an identity through a long occlusion or a
  fast-moving/crowded scene. A motion-model alternative (`sort`: Kalman
  filter + Hungarian assignment) is available via
  `tracking.tracker.name: sort` and survives short occlusion gaps `iou`
  can't (see `tests/test_trackers.py`), but isn't the default yet — no MOT-
  metric (IDF1/ID-switch) evaluation against a labeled clip exists to
  confirm it's a strict improvement across the board. The default detector
  only distinguishes "person" (COCO), with no re-identification across
  cameras or across a gap where a track was lost. Both detector and tracker
  are swappable via the `tracking.detectors`/`tracking.trackers` registries
  (same pattern as `action_recognition.models`) without touching the
  extraction or scene-inference pipeline.
- A clip file's label is still single-label (softmax) — a person doing two
  things at once isn't represented. Multi-label per-person tagging would
  need a different loss/config, not currently implemented.
- The webapp gates every page behind login (single shared user table, no
  per-role/per-site permissions yet) but has no login rate-limiting/lockout
  and no password-reset flow — acceptable for a single-operator deployment,
  not for a multi-user one. See `docs/plans/app-ux-permissions-security-scalability.md`
  for the planned per-site RBAC direction.

## Roadmap

This README documents the app as it is today. Longer-term direction lives
under [`docs/plans/`](docs/plans/):

- [`v1-release-plan.md`](docs/plans/v1-release-plan.md) — hardening this
  existing single-user tool into a tagged v1.0 (versioning, CI, security/
  UX follow-ups) before anything below starts.
- [`multi-camera-person-monitoring.md`](docs/plans/multi-camera-person-monitoring.md)
  — phased roadmap for extending this into multi-camera monitoring with
  cross-camera person re-identification.
- [`app-ux-permissions-security-scalability.md`](docs/plans/app-ux-permissions-security-scalability.md)
  — UI/UX, permissions/RBAC, security, and scaling detail for those phases.

None of that changes how the tool works today — see "Known limitations"
above for the current, honest state of things.

## Tests

```bash
pytest
```
