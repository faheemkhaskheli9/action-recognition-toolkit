# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Training and extraction runs can now be cancelled from the webapp: a
  "Cancel run" button on the run's detail page (shown only while it's
  `RUNNING`) terminates its OS process tree and marks it `CANCELLED` — a
  status distinct from `FAILED` so the run list/detail page can tell "the
  user stopped this" from "this crashed." A cancelled extraction run can be
  resumed afterward the same way a failed one can
  (`services.extraction.can_resume`/`resume_run` already keyed off "not
  RUNNING," so no change needed there). `services._background` grows a
  matching `terminate()` alongside its existing `launch_detached`/
  `is_pid_alive` — it kills the whole process tree the run's wrapper
  started (`os.killpg` on POSIX, `taskkill /T /F` on Windows), not just the
  tracked pid, since the wrapper runs the real command as its own child
  rather than exec'ing into it.

## [1.1.0] — 2026-08-14

### Added
- Datasets: the webapp now has a first-class `Dataset` (its own video folder
  + manifest CSV, created/managed on a new "Manage datasets" page) instead
  of one fixed `data/raw` folder shared by everything. The Dataset, Label,
  and Review manifest pages all operate within whichever dataset is
  currently selected (a dataset switcher replaces the old manifest-path
  picker on those pages), and Extract tracks runs detect+track against a
  chosen dataset's videos — either every video in it, or just one you pick.
  Existing installs get a "Default" dataset pointing at their existing
  `data/raw`/`data/manifest.csv` automatically on upgrade (migration
  `0003_dataset`), so nothing already labeled needs to move.
- `ar-extract-tracks` (and `action_recognition.scripts.extract_tracks`
  generally) now accepts a single video file as `video_dir`, not just a
  folder — runs detect+track on just that one video instead of requiring it
  to be staged into its own folder first.
- `ar-extract-tracks --resume` picks an interrupted run back up instead of
  restarting from scratch: it skips every video already recorded in
  `output_dir/tracks_index.csv` and discards any partial clips left under
  `output_dir/<video_stem>/` for whichever video was mid-extraction when the
  run stopped, so that one video redoes cleanly instead of mixing stale and
  fresh windows. `tracks_index.csv` is now rewritten after every video (not
  just once at the end) so a killed/crashed run always leaves a resumable
  index of whatever finished. The webapp's extraction detail page grows a
  matching "Resume extraction" button once a run has stopped (failed or the
  process died) — it relaunches `ar-extract-tracks --resume` against the
  same `output_dir`, appending to the existing log instead of replacing it.
- The extraction detail page gained a "Results" section: a per-video,
  per-track breakdown of every clip a run produced (frame range, frame
  count, and a playable thumbnail), read straight from
  `output_dir/tracks_index.csv` (`services.extraction.results`). Since that
  index is rewritten after every source video finishes, results appear
  progressively for a still-running run too — reload the page to pick up
  newly finished videos.
- Label page redesign: the start/end span fields are now backed by a visual
  drag-to-mark timeline (existing spans shown as blocks, a playhead, span
  preview playback) instead of plain number inputs — the numbers stay as
  the underlying fields the `add_span` POST contract already expected.
  Clips from a multi-person Extract-tracks import are matched back to their
  source video/track (via `tracks_index.csv`, nothing new persisted) and
  shown grouped by tracked person instead of as unrelated videos; labeling
  a track applies one label to every window clip in it, with a per-window
  fallback for a track whose action changes partway through. Labeling is
  also keyboard-driven now: quick-pick label chips plus digit keys 1-9 on
  the primary single-video form, space to play/pause, I/O to mark the span
  in/out point, and arrow keys to nudge the playhead — all scoped to not
  fire while typing in a text field.

### Changed
- `services.manifest.known_videos` takes an optional `repo_root` for
  callers whose `data_dir` is nested deeper than `settings.DATA_DIR` (e.g.
  one dataset's own video folder) — `path` in its results is still
  repo-root-relative either way.

- Manifest rows can now label just a span (`start_time`/`end_time`, in
  seconds) of a longer source video instead of always the whole file. The
  Label page gained a scrubber-driven "label just part of this video" tool
  — mark a start/end while playing the video, add a label, repeat for as
  many spans as the video has, then move to the next video explicitly
  ("Done with this video") — alongside the existing whole-file labeling
  flow, which is unchanged. Training (`VideoClipDataset`/`read_clip_frames`)
  samples within a row's span instead of across the whole file when one is
  set; a manifest that never labels a span keeps its previous
  `video_path,label[,split]` shape. Dataset and Review-manifest pages
  display labeled spans read-only; deleting one only removes that row.
  Single-clip inference and `ar-build-manifest` are unchanged — spans are a
  training/labeling concept, not extended to those in this change.
- The video, video-folder, and manifest-file pickers (Inference, Extract
  tracks, Train, Dataset, Label, Review manifest) are now a searchable,
  paginated dropdown instead of a plain `<datalist>` — focusing the field
  browses everything already on disk eight at a time, typing filters by
  name, and free text for a new path still works exactly as before. Backed
  by a small vanilla-JS component (`app.js`, `[data-picker]`) reading the
  same `known_videos`/`known_video_dirs`/`known_manifest_paths` data the
  view already computed, embedded via Django's `json_script` filter — no
  new endpoints or view changes.

### Fixed
- Extraction runs' per-video progress bar (`extraction_detail`/
  `extraction_list`) never rendered against a real run: the parsing regex
  required `Tracking`/`Skipping` at the very start of the log line, but the
  subprocess's actual log lines are prefixed with a timestamp/level/logger
  name first.
- Switching datasets mid-label (via the dataset picker, without finishing
  or skipping the current video first) could save the previous dataset's
  video into the newly-selected dataset's manifest — the Label page's
  "current video"/"skipped" session state wasn't scoped per dataset.
- `ar-predict` always ran on CPU regardless of an available GPU; it now
  autodetects like training, scene inference, and the detector already did.
- Creating a dataset whose name collided with an existing one (including a
  double form submit) 500'd instead of showing a form error.
- Deleting a dataset-page manifest row with a non-numeric span time 500'd
  instead of showing a form error.
- The extraction detail page's "Import clips" help text always said
  `data/raw` and linked to Label/Dataset with no dataset selected, both
  stale since the Dataset model replaced the single fixed `data/raw`
  folder — it now names and links to the run's actual target dataset.

## [1.0.0] — 2026-08-09

First tagged release. Hardens the app that already existed (label →
extract-tracks → train → infer, CLI + Django webapp) into a stable,
tested, documented baseline — see
[`docs/plans/v1-release-plan.md`](docs/plans/v1-release-plan.md) for the
full scope and what's deliberately deferred to later releases.

### Added
- `sort` tracker (`tracking.tracker.name: sort`): Kalman-filter motion
  model + Hungarian assignment, alongside the existing `iou` tracker.
  Survives short occlusion gaps that split `iou` into two track IDs.
- Site-wide login gate (`core.middleware.LoginRequiredMiddleware`) — every
  page now requires an authenticated session by default, with a small
  allowlist (login page, static assets, `/admin/`, which gates itself).
- Production-security settings read from the environment
  (`DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`), with
  SSL redirect / secure cookies / HSTS turned on automatically once
  `DJANGO_DEBUG=false`. Existing `python manage.py runserver` on
  localhost with no env vars set keeps working unchanged.
- `docs/` — a beginner-oriented walkthrough of the existing pipeline, plus
  the multi-camera monitoring roadmap and its UX/permissions/security/
  scalability companion doc, for planned future work.
- The Dataset page now explains the full pipeline in-app (including the
  multi-person raw-footage path: Extract tracks → Label → Train →
  Inference with scene mode) instead of only in `README.md`; Inference's
  "scene mode" help text — previously defined but never rendered — now
  shows up on the page.
- Video-folder, manifest-file, and video fields across Extract tracks,
  Train, Dataset, Label, Review manifest, and Inference are now searchable
  dropdowns of values already seen on disk instead of blank text boxes —
  free text still works for a new path. Inference can also run directly
  against an already-uploaded dataset video instead of requiring a fresh
  upload every time.
- Inference video uploads now save to a fixed, name-derived path under
  `webapp/media/uploads/` (`get_valid_filename(uploaded_file.name)`)
  instead of a random `uuid4`-prefixed one, and are no longer deleted
  after prediction — re-uploading the same filename overwrites the same
  path rather than piling up a new file per request.
- A finished extraction run's detail page now has an "Import clips into
  dataset" button (`services.extraction.import_clips`) that copies its
  output into `data/raw`, flattening `<video_stem>/trackN_winM.mp4` into
  unique filenames and skipping clips already imported — closes the gap
  left by fixing the Dataset/Label video folder (below): multi-person
  clips from Extract tracks are reachable from the browser again without a
  manual filesystem move.

### Fixed
- `services/manifest.py` was missing the `VIDEO_EXTENSIONS` import that
  `views/dataset.py` already referenced — latent `AttributeError` on that
  code path.
- `services/_background.py::launch_detached` now creates the log file
  synchronously before returning, so a run's detail page can link to a log
  that's guaranteed to exist rather than racing the detached process's own
  first write.
- `tests/webapp/test_training_service.py` / `test_views.py` were mocking
  `subprocess.Popen` directly even though `training.py` already called
  `services._background.launch_detached` — the mocks were never actually
  exercised. Updated to mock the real call site.

### Changed
- The Dataset and Label pages' video folder is now a fixed backend value
  (`data/raw`, `services.manifest.DEFAULT_VIDEO_DIR`) instead of a
  user-editable, per-session text field — every upload and labeling session
  reads/writes the same folder. Extract tracks output
  (`data/tracks/<name>/`) is no longer reachable from these pages directly;
  use the new "Import clips into dataset" button (above) to copy clips into
  `data/raw`, or run `ar-build-manifest` against the tracks folder from the
  CLI instead. The manifest-file picker is unaffected and still switches
  freely.
- Package version now tracked in one place conceptually (`__init__.py` and
  `pyproject.toml` both bumped together, with a test asserting they stay
  in sync going forward).
- Removed `requirements.txt` — it duplicated `pyproject.toml`'s
  `dependencies` by hand and had already drifted (missing `scipy`). Use
  `pip install -e ".[webapp,dev]"` as documented in `README.md`.
