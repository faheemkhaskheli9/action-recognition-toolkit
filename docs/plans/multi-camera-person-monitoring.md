# Multi-camera person monitoring — research + roadmap

**Goal:** extend the current single-video detect→track→classify pipeline into
a monitoring system for a fixed site (e.g. a mall or store): multiple
cameras, staff vs. customer distinction, per-person action timelines, and
re-identification of a person who leaves one camera's view and reappears in
another covering the same physical space. Real-time (live camera) operation
is the priority target; batch/offline processing on recorded footage is
built first because it's where tracking and re-ID correctness can actually
be measured and debugged.

Decisions already made (2026-08-08):
- **Both real-time and batch, real-time is the priority** — but sequenced so
  real-time is a later phase built on a correctness foundation, not the
  first thing shipped.
- **Cross-camera matching = appearance embedding + zone/topology
  calibration.** A per-site camera adjacency graph (which camera's exit
  zone feeds which camera's entrance zone, and expected transit time)
  disambiguates appearance matches instead of relying on visual similarity
  alone.
- **Staff vs. customer = both enrolled gallery and zone heuristic,
  configurable per deployment.**

---

## 1. What similar systems do (research)

**Multi-target multi-camera tracking (MTMC)** is a well-studied pipeline
shape, consistently three stages: *detect → track per camera → re-identify
across cameras*. That's a direct extension of what this repo already has
(detect + track per video) — the new work is stage three plus a
zone/topology layer that recent industrial systems add specifically to
handle open-set re-ID (people the system has never enrolled) at
production accuracy:

- **MICRO-TRACK** (arXiv:2409.03879) — a real-time, modular industrial MTMC
  system built specifically for the *open-set* case (no known gallery of
  everyone up front), which is exactly this project's situation (a mall
  doesn't have a pre-enrolled photo of every customer). Confirms the
  detect → embed → associate-across-cameras shape and that open-set
  handling (an unmatched embedding becomes a *new* global identity, not a
  forced match) needs to be designed in from the start, not bolted on.
- Zone/topology-assisted association (camera adjacency + transit-time
  windows narrowing candidate matches before appearance comparison) shows
  up repeatedly in recent MTMC work (e.g. GRAP-MOT, arXiv:2510.21482) —
  validates the calibration approach chosen above over pure appearance
  matching, which degrades badly when people wear similar clothing
  (uniforms, school groups).
- **Person re-ID benchmarks/models**: Market-1501 / DukeMTMC-reID are the
  standard benchmarks; OSNet-family embedding networks (small, fast,
  omni-scale features) are the common lightweight choice for production
  MTMC, not a full ReID transformer — relevant because this repo has no
  GPU-cluster assumption and needs to run one embedding model per detected
  person per frame alongside the existing detector and action model.

**Tracker choice within a single camera**: this repo's own tracker
(`iou_tracker.py`) is explicitly documented as "SORT without the Kalman
motion model" and known to lose identity through occlusion — exactly the
failure mode current tracking-by-detection literature addresses by adding
(a) a motion model (Kalman filter, as in SORT/ByteTrack/BoT-SORT) and (b) an
appearance embedding for re-matching after a track is lost (DeepSORT,
StrongSORT, BoT-SORT with `with_reid` on). Practical guidance from current
comparisons: ByteTrack for speed on the common case, BoT-SORT (motion +
appearance + camera-motion compensation) for crowded/occluded scenes like a
mall, StrongSORT when identity switches are the most expensive failure mode.
Because this project already needs a person-appearance embedding for
*cross-camera* re-ID, reusing that same embedding for *within-camera*
re-matching after an occlusion (which BoT-SORT/StrongSORT do) is close to
free — one embedding model serves both problems.

**Retail-specific monitoring** (staff engagement, dwell time, restricted-zone
alerts, theft-pattern detection) is consistently built as a rules/analytics
layer *on top of* tracked identities and per-person action classification —
not a different core pipeline. That matches this repo's existing shape:
`scene_predict.py` already produces a per-track, per-time-window action
timeline for one video; monitoring features are downstream consumers of
that timeline plus a cross-camera identity, not a new detection/tracking
stack.

Sources: [Multi-Camera Industrial Open-Set Person Re-Identification and
Tracking (arXiv:2409.03879)](https://arxiv.org/abs/2409.03879) ·
[GRAP-MOT (arXiv:2510.21482)](https://arxiv.org/pdf/2510.21482) ·
[Multi-Camera-Person-Re-Identification (GitHub, Market-1501/DukeMTMC-reID
benchmarks)](https://github.com/SurajDonthi/Multi-Camera-Person-Re-Identification)
· [BoT-SORT (arXiv:2206.14651)](https://arxiv.org/pdf/2206.14651) ·
[Ultralytics tracker comparison](https://www.rizwanai.com/blog/yolo-object-tracker-comparison-botsort-bytetrack-ocsort-deepocsort-fasttrack-tracktrack)
· [viso.ai — computer vision in retail](https://viso.ai/applications/computer-vision-in-retail/)

---

## 2. Gap analysis against this codebase

| Need | Exists today | Gap |
|---|---|---|
| Detect people in a frame | `tracking/detectors` (Faster R-CNN) | fine as-is for phase 0 |
| Track a person within one video | `tracking/trackers` (`iou` tracker) | no motion model, no re-match after occlusion — loses ID easily in crowds |
| Classify a tracked person's action | `inference/scene_predict.py` | per-video only; no continuous/streaming mode |
| Recognize the same person across videos/cameras | **nothing** | new: appearance embedding + cross-camera linking |
| Know which cameras cover physically adjacent space | **nothing** | new: site/camera/zone data model + calibration UI |
| Tell staff from customers | **nothing** | new: staff gallery enrollment and/or zone heuristic |
| Live camera input | **nothing** — everything is a video *file* run as a subprocess that exits | new: long-lived stream workers, different lifecycle than `TrainingRun`/`TrackExtractionRun` |
| Dwell time / alerts / dashboards | **nothing** | new: monitoring/analytics layer over the per-person timeline |
| Privacy/retention controls | **nothing** | new — required before processing identifiable customer video, see §6 |

---

## 3. Proposed architecture additions

New `src/action_recognition/` package, following the existing registry
pattern (`@register_x`, `build_x`, own `__init__.py` import-time
registration):

```
reid/            embedding model registry (contract: person crop -> fixed-size
                 float vector). Start with a small OSNet-style backbone.
                 registry.py, osnet.py, gallery.py (cosine-sim nearest-neighbor
                 match against enrolled/observed embeddings)
tracking/
  trackers/      add a `bytetrack` or `botsort`-style tracker (motion model;
                 optionally consumes reid embeddings for re-match after an
                 occlusion) alongside the existing `iou` one — same registry,
                 no changes needed to extract.py's call sites
  linking.py     cross-camera identity linking: given tracks + embeddings +
                 timestamps from multiple cameras, plus a site topology graph,
                 produce global Person IDs (open-set: no topology/appearance
                 match => new identity, never a forced match)
  topology.py    camera/zone adjacency graph + transit-time windows, loaded
                 from a per-site config (mirrors configs/tracking/default.yaml
                 pattern: YAML, deep-merged over defaults)
streaming/       (phase 4) frame source abstraction — same `detect+track`
                 inner loop as extract.py, fed by cv2.VideoCapture(rtsp_url)
                 or a file, running as a long-lived worker instead of a
                 one-shot subprocess
monitoring/      (phase 3) dwell time, zone occupancy, staff/customer
                 interaction, alert rule evaluation over a Person's
                 aggregated cross-camera timeline
```

Model contract additions (documented alongside the existing "Model
contract" note in CLAUDE.md once built): a ReID model takes an `(C, H, W)`
person crop and returns a fixed-length embedding — that's the only
requirement to plug in a new backbone via the registry.

### Data model (Django, `webapp/core/models.py`)

New models, additive only — nothing here changes `TrainingRun` /
`TrackExtractionRun`:

- `Site` — one physical location (a mall, a store). Holds the retention
  policy (§6).
- `Camera` — belongs to a `Site`; `source` (file glob for batch, RTSP URL
  for live), `is_live` flag.
- `Zone` — belongs to a `Camera`; a named polygon region (`entrance`,
  `checkout`, `staff_only`, ...) used both for the staff/customer zone
  heuristic and for topology (`Zone.adjacent_to = [Zone, ...]`, `transit_time_range`).
- `Identity` — a global person: `kind` (`staff`/`customer`/`unknown`),
  optional `name`, one or more enrollment embeddings for staff.
- `IdentityObservation` — links a `Track` (existing per-video track) to an
  `Identity`, with confidence and the matching method used (appearance-only
  vs. appearance+topology).
- `Alert` — rule name, `Identity`/`Zone`/`Camera` reference, timestamp,
  severity — output of the monitoring rule layer.
- `StreamSession` (phase 4) — replaces the "one subprocess that exits" model
  for live cameras: a long-lived worker with heartbeat + restart policy,
  since there's no natural "exit code" for a feed that's supposed to run
  forever. Needs its own liveness pattern, analogous to but distinct from
  the existing `os.kill(pid, 0)` + exit-code-marker approach documented in
  CLAUDE.md for finite runs.

---

## 4. Phased roadmap

Each phase is shippable and testable on its own; later phases build on
earlier ones. Batch/file-based work comes first even though real-time is
the priority, because tracking and re-ID quality has to be measurable
(ground truth, repeatable runs) before it's worth running live.

### Phase 0 — Stronger single-camera tracking (foundation)
Add a motion-model tracker (ByteTrack/BoT-SORT-style: Kalman filter +
Hungarian assignment) to `tracking/trackers` alongside `iou`, selectable via
the existing `tracking.tracker.name` config key — no call-site changes
needed elsewhere. Add an evaluation harness (`tests/test_trackers.py` plus a
small labeled clip) using MOT metrics (IDF1, ID switches) so tracker
quality regressions are caught, not just "it still runs."

### Phase 1 — Person re-ID embeddings
New `reid/` registry with one small backbone (OSNet-style). Extract an
embedding per track by sampling a few crops (reuse
`tracking/extract.py::read_window_frames`) and averaging. This is also what
lets Phase 0's tracker re-match an identity after a brief occlusion, so it
pays for itself within a single camera before cross-camera linking exists.

### Phase 2 — Cross-camera identity linking (batch)
`tracking/linking.py` + `tracking/topology.py`: given tracks from multiple
video files (recorded from different cameras covering the same site) plus a
site topology config, produce global `Identity` records. New CLI
(`ar-link-identities`) and a webapp "Site & Cameras" page to configure zones
and adjacency. Ship with the open-set rule from the research above: no
confident match (appearance *and* topology/timing) creates a new identity
rather than forcing a wrong one.

### Phase 3 — Monitoring layer (batch)
Dwell time, zone occupancy, staff/customer interaction detection, and
configurable alert rules (e.g. "customer in staff-only zone",
"loitering > N minutes"), built over the per-`Identity` aggregated action
timeline (which is Phase 2's linked tracks + the existing
`scene_predict.py`-style per-window classification). Webapp "Monitoring"
dashboard: per-site timeline, per-identity history, alert log. This is the
first phase that delivers the user-facing "monitor staff or customers"
outcome end-to-end, still on recorded footage.

### Phase 4 — Real-time streaming
The big architectural shift, scoped deliberately (same spirit as the
"Scaling direction" section in CLAUDE.md — not a background refactor):
- `streaming/` workers consuming RTSP/webcam sources continuously, running
  the same detect→track→embed→classify loop as `extract.py` but as a
  persistent process instead of a one-shot subprocess over a finite file.
- Replace the finite-run liveness pattern (`os.kill` + exit-code marker,
  meant for a process that's expected to terminate) with a heartbeat +
  auto-restart pattern for `StreamSession`, since a live camera worker is
  meant to run indefinitely and "it exited" is a failure, not completion.
- Cross-camera hand-off needs an event stream between per-camera workers
  and the linking logic (Phase 2's `linking.py`, now called continuously
  instead of once per batch) — a lightweight pub/sub (e.g. Redis Streams)
  rather than Django ORM polling, to keep latency low without introducing a
  full task-queue framework prematurely.
- Live alerts pushed to the webapp (WebSockets/SSE) instead of the current
  request/refresh-on-read pattern used for training/extraction status.
- GPU throughput: N simultaneous camera streams sharing one GPU needs
  batched inference across streams and adaptive frame-skipping, not N
  independent model instances — this is the main scaling risk and should
  get a throughput benchmark (streams-per-GPU at target latency) before
  committing to a camera count in any real deployment.

### Phase 5 — Hardening
Staff gallery management UI (enroll/re-enroll photos), camera/zone
calibration UI (draw zones on a frame, define adjacency visually rather
than hand-written YAML), alert delivery integrations (email/webhook),
retention/purge job for stored video and embeddings, and a permanent MOT +
re-ID accuracy regression suite.

---

## 5. Non-goals (explicitly out of scope here, matching CLAUDE.md's existing stance)

- Multi-tenant SaaS (many independent customers/sites with isolation,
  billing, per-tenant quotas) — this plan is for one operator monitoring
  one or more of *their own* sites, same single-user posture the project
  already commits to. Don't let "multiple cameras/sites" get conflated with
  "multiple tenants."
- A full task-queue framework (Celery, etc.) — Phase 4 introduces a
  lightweight pub/sub only where the subprocess model genuinely can't work
  (continuous streams), not a wholesale infra migration.

## 6. Privacy & legal — needs a decision before any real customer footage is processed

Re-identifying customers across cameras using appearance/face-adjacent
embeddings is biometric processing in a lot of jurisdictions (GDPR in the
EU, BIPA in Illinois, similar laws elsewhere) and carries real legal
exposure if built without consent/notice/retention controls. This plan
includes the technical hooks (`Site.retention_policy`, a purge job in
Phase 5) but **the actual policy — consent signage, retention period,
opt-out handling, whether customer (vs. staff-only) re-ID is enabled at
all — is a legal/business decision for the deployer, not something to
default silently.** Recommend surfacing this explicitly in the webapp setup
flow for a new `Site` (e.g. a required acknowledgment) once Phase 2 ships,
rather than treating it as a later add-on.

## 7. Suggested next step

Phase 0 is small, self-contained, and immediately improves the existing
single-video pipeline (better tracking through occlusion) independent of
everything else — good candidate to start with. Say the word and I'll scope
it into concrete file changes + tests.
