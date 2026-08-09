# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
- Package version now tracked in one place conceptually (`__init__.py` and
  `pyproject.toml` both bumped together, with a test asserting they stay
  in sync going forward).
- Removed `requirements.txt` — it duplicated `pyproject.toml`'s
  `dependencies` by hand and had already drifted (missing `scipy`). Use
  `pip install -e ".[webapp,dev]"` as documented in `README.md`.
