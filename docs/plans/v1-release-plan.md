# Path to v1.0 — hardening the current app, then handing off to the roadmap

Companion to the two existing planning docs:
[`multi-camera-person-monitoring.md`](multi-camera-person-monitoring.md)
(phased architecture roadmap, phases 0–5) and
[`app-ux-permissions-security-scalability.md`](app-ux-permissions-security-scalability.md)
(UI/UX, permissions, security, scalability detail on top of those phases).
Neither of those documents ever ships a tagged release — they describe a
long-horizon build-out. This document is the missing piece: it closes out
the app that exists **today** (single-user label → extract → train → infer
tool, CLI + webapp) into a stable, tagged **v1.0**, and only then hands off
into the roadmap's Phase 0+ as **v1.1+**. Two tracks, run in this order —
Track A is the near-term, self-contained piece; Track B is "continue what
the other two docs already scoped," restated here only as a sequencing
note, not re-planned.

---

## 0. Where things stand right now

The working tree already has uncommitted work that *is* the start of both
tracks at once — worth naming explicitly since it changes what "start
here" means:

- **Track B, Phase 0 (roadmap doc §Phase 0):** `sort_tracker.py` — a
  Kalman-filter + Hungarian-assignment tracker registered as `sort`,
  alongside `iou`, plus `tests/test_trackers.py` including a head-to-head
  test proving it survives an occlusion gap that splits `iou` into two
  tracks. This is functionally complete for Phase 0's stated scope.
- **Track A prerequisite (permissions doc §5, §2.1):** site-wide login
  gate — `core/middleware.py` (`LoginRequiredMiddleware`), `core/login.html`,
  `tests/webapp/test_auth.py`, and a hardened `webapp/config/settings.py`
  (env-sourced `SECRET_KEY`/`DEBUG`/`ALLOWED_HOSTS`, HSTS/secure-cookie
  settings gated on `DEBUG=False`). Also functionally complete for what the
  permissions doc asked for as a first step.
- `DEVELOPMENT.md` picked up an **Attribution** section (no AI-tool
  attribution in committed code/comments/git) — already in effect, nothing
  to do.
- All 146 tests pass with this work in the tree (verified before writing
  this plan). None of it is committed yet — still sitting as local changes
  on `master`.

So step one of Track A is *land what's already here*, not invent new work.

---

## Track A — v1.0: harden the app that exists today

Scope: the single-user ML tool as documented in `DEVELOPMENT.md` and `README.md`
— label, extract-tracks, train, infer, both CLI and webapp. Explicitly
**not** in scope: `Site`/`Camera`/re-ID/monitoring (that's Track B). This
matches DEVELOPMENT.md's existing stance that multi-tenant/scaling work is a
"deliberate, separately-scoped initiative," extended here to mean the
whole monitoring build-out too — v1.0 is "the tool works well and is safe
to hand to someone else," not "the roadmap is further along."

### A1. Land the in-flight work

1. Split the current uncommitted diff into two commits (tracker, auth+
   security-settings) — they're unrelated changes and reviewing/reverting
   either independently should be possible.
2. Before committing the auth change, close two gaps it leaves:
   - `core/admin.py` registers `TrainingRun`/`TrackExtractionRun` with no
     extra gate beyond Django's own staff-user flag — matches the
     permissions doc's §2.5 observation almost verbatim ("nothing stops
     someone from becoming one"). For v1.0 (still single-operator), it's
     enough to note in `README.md`/`DEVELOPMENT.md` that no user should be
     marked `is_staff` unless they're meant to have unmediated DB access —
     but add a one-line webapp test asserting `/admin/` still redirects an
     anonymous session to login, so the exemption in the middleware
     (`_EXEMPT_PREFIXES`) doesn't silently regress into "admin reachable by
     anyone."
   - No first-run story for creating the first login. `python manage.py
     createsuperuser` works today but isn't documented — add it to
     `README.md`'s "Web app" section right after `migrate` so a fresh clone
     doesn't dead-end at the new login page.
3. Run the full suite once more after landing (`pytest`) and commit.

### A2. Versioning & packaging

- Bump `src/action_recognition/__init__.py::__version__` and
  `pyproject.toml`'s `[project].version` from `0.1.0` to `1.0.0` together —
  today they'd be free to drift since nothing checks them against each
  other; add a one-line test (`tests/test_utils.py` or a new
  `tests/test_version.py`) asserting they match, so future releases can't
  forget one.
- `requirements.txt` and `pyproject.toml`'s `dependencies` list the same
  packages by hand in two places (already slightly different: `requirements.txt`
  is missing `scipy`, which `sort_tracker.py` now imports directly). Pick
  one source of truth — drop `requirements.txt` in favor of
  `pip install -e ".[webapp,dev]"` (already the documented install command)
  and keep the file only if something outside pip (e.g. a Dockerfile) needs
  a flat requirements list; otherwise remove it to stop the drift outright.
- Add `CHANGELOG.md` (Keep a Changelog style is fine) seeded with a `1.0.0`
  entry summarizing what shipped: two model architectures, detect+track
  extraction with swappable detector/tracker backends (now including
  `sort`), webapp covering the full pipeline, login-gated access. Future
  changes get an entry each — cheap now, valuable the first time someone
  asks "what changed since v1."

### A3. CI

There is no `.github/workflows/` (or equivalent) today — every test run so
far has been manual. For a tagged v1.0 this is the highest-leverage single
addition:

- One workflow: checkout → `pip install -e ".[webapp,dev]"` → `pytest`, on
  push and PR against `master`. CPU-only torch is fine (no GPU needed for
  the test suite — confirm none of the 146 tests require CUDA; a quick
  `grep -r cuda tests/` before wiring this up is worth doing to avoid
  a false-green CI on a runner with no GPU vs. a real requirement).
  Matches DEVELOPMENT.md's existing "run the relevant file (or the full suite)
  before considering a change done" rule — CI is that rule enforced
  automatically instead of trusted by convention.
- Cache pip/torch downloads (`actions/cache` keyed on `pyproject.toml`) —
  torch wheels are large and DEVELOPMENT.md already flags the install as
  network-heavy; an uncached CI run would be slow on every push.
- Not in scope for v1.0: coverage thresholds, multi-OS/multi-Python-version
  matrix, publishing to PyPI. One green pipeline on one platform is the
  bar; expand later if it earns its keep.

### A4. Security follow-ups beyond what's already staged

The permissions doc's §3.1 "fix first" list (`SECRET_KEY`, `DEBUG`,
`ALLOWED_HOSTS`, HTTPS/cookie flags) is the in-flight settings.py change —
already covered by A1. What's left for a v1.0 bar specifically:

- `.env.example` (or a short "Environment variables" table in
  `README.md`) documenting `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`,
  `DJANGO_ALLOWED_HOSTS` — they exist in `settings.py` now but aren't
  documented anywhere a new deployer would find them.
- Session/CSRF cookie lifetime and `SESSION_COOKIE_AGE` aren't set
  explicitly (Django's default is 2 weeks) — reasonable for a single
  operator, but worth a one-line comment in `settings.py` saying that's a
  deliberate choice, not an oversight, so a future security pass doesn't
  flag it as unreviewed.
- Rate-limiting/lockout on the login view is explicitly **out of scope**
  for v1.0 — single-operator, no public signup, low attack surface; note
  it as a known limitation (same treatment as the existing "Known
  limitations" section) rather than silently adding it or silently
  skipping it undocumented.

### A5. UX / error-state pass over the existing pages

A lightweight audit, not a redesign — walk each of the 8 existing pages
(Dataset, Extract tracks, Extraction runs, Label, Review manifest, Train,
Runs, Inference) against one checklist, matching the status-badge/empty-
state conventions the permissions doc's §1.5 already establishes as the
target style for *new* pages:

- Every list page has an explicit empty state (not just an empty table).
- Every background-run page (Extraction runs, Runs) makes a `FAILED`
  status as visually distinct as `RUNNING`/`SUCCEEDED` — confirm against
  `core/models.py`'s `Status` choices and the templates that render them.
- Form validation errors render inline next to the field, matching the new
  `login.html`'s `p.error` pattern (`webapp/core/static/core/style.css`),
  not just a generic message banner.
- Long-running actions (start training, start extraction) give immediate
  feedback (redirect to the run's log-tail page) rather than a silent
  POST — spot-check this already works via `tests/webapp/test_views.py`
  and extend if any start-action lacks the same redirect-to-detail
  pattern.

This should land as a handful of small, independently-reviewable PRs (one
per page or per finding), each with the matching webapp test per
DEVELOPMENT.md's coverage rule — not one large sweep.

### A6. Documentation

- `README.md` is already thorough for *how to use* the app; it doesn't
  currently link to `docs/plans/` at all. Add a short "Roadmap" section
  near the bottom pointing at this file and the two it builds on, so
  someone reading the README finds the longer-term direction instead of
  assuming the "Known limitations" section is the final word.
- Add `CONTRIBUTING.md` covering: install command, `pytest` invocation
  (and the `pythonpath`/`DJANGO_SETTINGS_MODULE` note from DEVELOPMENT.md,
  since it's the one non-obvious setup step), the registry pattern for
  adding a model/detector/tracker, and the Attribution rule from
  `DEVELOPMENT.md` restated for external contributors who won't necessarily
  read `DEVELOPMENT.md` itself.

### A7. Release checklist (do once, in order)

1. A1–A6 merged, `pytest` green, CI green on `master`.
2. Tag `v1.0.0`, `CHANGELOG.md`'s `1.0.0` section dated.
3. Confirm `pip install -e ".[webapp,dev]"` on a clean checkout still
   works end-to-end (install → migrate → runserver → login) — the one
   thing none of the automated tests actually exercise (they run against
   an already-configured `pytest-django` environment, not a fresh
   `manage.py migrate`).

---

## Track B — v1.1+: resume the roadmap where Phase 0 already started

No new planning here — `multi-camera-person-monitoring.md` already lays
out Phase 0–5 and `app-ux-permissions-security-scalability.md` already
reshapes the phase order (its §5: auth no later than Phase 2, security
hardening before any non-localhost deployment). What this section adds is
just the connective tissue now that Track A exists as a named release:

- **Phase 0 (tracker) is already substantially done** by the in-flight
  `sort_tracker.py` — confirm it's fully closed out per the roadmap doc's
  Phase 0 description ("add an evaluation harness... using MOT metrics
  (IDF1, ID switches)") before calling it finished. `tests/test_trackers.py`
  today proves *qualitative* behavior (bridges an occlusion gap) with
  synthetic detections; a small labeled clip + IDF1/ID-switch numbers is
  the one piece the roadmap doc asked for that isn't in yet.
- **Auth is already substantially done** ahead of its roadmap trigger
  (permissions doc said "no later than Phase 2"; it's landing before v1.0
  instead) — Track B's Phase 2 can build `SiteMembership`/per-site RBAC
  (permissions doc §2.3) directly on top of the `LoginRequiredMiddleware`
  foundation rather than wiring auth on from scratch.
- Everything else — Phase 1 (re-ID embeddings), Phase 2's `Site`/`Camera`/
  `Zone` data model and cross-camera linking, Phase 3 (monitoring/alerts),
  Phase 4 (real-time streaming), Phase 5 (hardening) — proceeds exactly as
  scoped in the roadmap doc, starting only after v1.0 is tagged so the
  roadmap build-out has a stable base to branch from and a clean rollback
  point if needed.

---

## 4. Suggested next step

Track A, in order: A1 (land + close the two admin/first-run gaps) is small
and unblocks everything else in Track A, since A2–A6 all assume the
in-flight diff is committed. Say the word and it can be scoped into
concrete file changes.
