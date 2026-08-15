# Extract + Track: Step-by-Step Guide

*How raw, multi-person footage becomes labeled single-person training clips
— every processing stage explained, plus how to make sure none of the
clips a run produces get left behind.*

Use this when your source footage has **more than one person in frame**
(a restaurant floor, a warehouse aisle, a retail store, a security camera).
The Label page and the rest of the pipeline expect one person per clip —
this step gets you there without manually trimming anything by hand.

---

## 0. Before you start

You need a folder of raw video files (`.mp4`, `.avi`, `.mov`, `.mkv`,
`.webm`). Layout doesn't matter — flat or nested, searched recursively. In
the web app this is a **Dataset**'s `video_dir`; on the CLI it's just a
path you pass in.

If you don't have a Dataset yet, create one first (web app → Datasets →
New) and point it at the folder holding your raw footage.

---

## 1. The processing pipeline, stage by stage

Everything below happens once per source video, driven by
[`extract_tracks_to_clips`](../src/action_recognition/tracking/extract.py).

### Stage 1 — Detect

The detector (default: `fasterrcnn`, COCO "person" class only) runs over
every `frame_stride`-th frame (default `2`, i.e. every other frame) and
returns person bounding boxes above `score_thresh` (default `0.6`). Lower
confidence detections are discarded — they never reach tracking.

*Code:* `track_video()` in [`extract.py`](../src/action_recognition/tracking/extract.py).

### Stage 2 — Track

The tracker (default: `iou`, a greedy IoU matcher) links detections across
frames into per-person tracks. A track survives up to `max_age` frames
(default `15`, in detector-stride units) with no matching detection before
it's closed — so brief occlusion is tolerated, longer gaps split into a new
track. There's no re-identification: if a person leaves and re-enters
frame, they become a *new* track, not a continuation of the old one.

### Stage 3 — Window

Each finished track is sliced into overlapping fixed-length windows —
`windows_for_track()`:

- `window_frames` (default `16`) — frames per output clip
- `stride_frames` (default `8`) — hop between windows, so consecutive
  windows overlap by half
- `min_track_frames` (default `16`) — a track shorter than this produces
  **zero** clips; the last partial window of a longer track is kept only
  if it's still at least this long, otherwise it's dropped rather than
  zero-padded

A single track of, say, 40 frames yields windows `[0:16]`, `[8:24]`,
`[16:32]`, `[24:40]` — 4 clips from one person, not 1.

### Stage 4 — Crop + write

For each window, `write_window_clip()` re-decodes just those frames from
the source video, pads the track's box by `crop_padding` (default `0.2` =
20% on each side), crops, resizes to `output_size × output_size` (default
`224×224`), and writes an mp4 at `fps` (default `10`). The result is
saved to:

```
<output_dir>/<video_stem>/track<track_id>_win<window_index>.mp4
```

### Stage 5 — Index

After each source video finishes, one row per clip is appended to
`<output_dir>/tracks_index.csv`:

```
source_video, track_id, window_index, start_frame, end_frame, num_frames, clip_path
```

This file is rewritten after *every* video (not just at the end), so an
interrupted run still leaves a complete, accurate record of everything
finished so far — this is what makes `--resume` (CLI) / **Resume** (web
app) safe.

---

## 2. Running it

### Option A — Web app (recommended if you're already using it for labeling)

1. **Extract tracks** page → pick the Dataset (or a single video within
   it), give the run a name, optionally pick a non-default tracking
   config → **Start**.
2. It launches as a detached background process — you can navigate away or
   close the browser; it keeps running.
3. The run's detail page auto-refreshes: live log tail, clip count so far,
   and a per-video/per-track breakdown once each video's rows land in
   `tracks_index.csv`.
4. If it fails or you stop it partway through (host restart, etc.), open
   the run and click **Resume** — it re-launches with `--resume`, which
   skips every video already recorded in `tracks_index.csv` and discards
   any partial `<video_stem>/` folder left by a video that didn't finish,
   so you never get a mix of stale and fresh windows for the same video.
5. Once the run shows **Succeeded**, click **Import clips into dataset**
   (see [§3](#3-using-all-the-extracted-data) — this is the step that
   actually makes the clips usable).

### Option B — CLI

```bash
ar-extract-tracks data/raw/multi_person_footage data/tracks/run1
```

- First argument: a folder (searched recursively) or a single video file.
- Second argument: output folder — created if it doesn't exist.
- `--config configs/tracking/<file>.yaml` to override detector/tracker/
  window settings (see [`configs/tracking/default.yaml`](../configs/tracking/default.yaml)
  for every field and its default).
- `--device cpu|cuda|mps` to force a device (autodetected otherwise).
- `--resume` to continue an interrupted run against the same output
  folder, same semantics as the web app's Resume button.

```bash
# resume after a crash / Ctrl-C
ar-extract-tracks data/raw/multi_person_footage data/tracks/run1 --resume
```

When it finishes:

```
Wrote 214 clip(s) across 12 video(s) to data/tracks/run1
Index: data/tracks/run1/tracks_index.csv
```

---

## 3. Using all the extracted data

A run's output folder is just clips on disk — nothing downstream sees them
until they're labeled. Two ways this silently loses clips if you're not
careful, and how to avoid each:

### Don't hand-pick a subset — import everything

**Web app:** click **Import clips into dataset** on the finished run's
detail page. This copies *every* clip the run produced — not a sample —
into the dataset's `video_dir`
(`services.extraction.import_clips`, [`extraction.py`](../webapp/core/services/extraction.py)),
flattening `output_dir/<video_stem>/trackN_winM.mp4` into
`video_dir/<run_name>__<video_stem>__trackN_winM.mp4` (prefixed with the
run name so two runs sharing a video stem don't collide). Clicking it
again after a later/resumed run only copies what's new — it skips any
destination file that already exists with the same size, so it's always
safe to re-click rather than track manually what's already been pulled in.

**CLI:** there's no separate import step — point `ar-build-manifest` or
the Label page directly at the tracks output folder. Just be aware of the
layout mismatch: `ar-build-manifest` expects `<root>/<class_name>/<video>`
folders, but tracks output is `<root>/<video_stem>/<clip>` — that's *not*
class-folder layout, so `ar-build-manifest` won't infer labels from it.
Use the Label page against the tracks folder instead (or move/symlink
clips into class folders yourself first if you already know the labels).

### Don't leave imported clips unlabeled

Importing (or pointing Label at the folder) makes clips *visible* to
labeling — it doesn't label them. Check progress on the **Label** page,
which shows labeled/total against everything under the video folder
(`services.manifest.progress`); anything short of the total is still
sitting there unused by training, since `ar-train` only ever sees rows
that made it into `data/manifest.csv`. Work the Label page until that
counter reads fully labeled before moving on to split/train.

### Sanity-check nothing got dropped upstream, too

- Compare the run's reported clip count against `tracks_index.csv`'s row
  count — they should match exactly once the run shows Succeeded.
- A source video contributing *zero* clips isn't necessarily a bug: no
  detections above `score_thresh`, or every track shorter than
  `min_track_frames`, both legitimately produce nothing. If that's
  happening more than expected, lower `score_thresh` or `min_track_frames`
  in a custom tracking config and re-run rather than assuming the clips
  are ""missing"" from disk.
- `tracks_index.csv`'s `track_id`/`window_index` columns tell you exactly
  how many windows came from each person in each video — useful for
  spotting a track that got fragmented by occlusion (many short tracks
  from what should be one person) versus one that's genuinely a different
  person.

Once every clip a run produced is imported and every imported clip is
labeled, proceed exactly like any other labeled folder: `ar-split-dataset`
→ `ar-train`. See the [root README](../README.md) and
[`data/README.md`](../data/README.md) for that part of the pipeline.
