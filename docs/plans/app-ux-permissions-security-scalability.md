# Multi-camera monitoring app — UI/UX, permissions, security, scalability

Companion to
[`multi-camera-person-monitoring.md`](multi-camera-person-monitoring.md)
(research + phased architecture roadmap, phases 0–5). That document answers
*what gets built and in what order*. This one answers *how it becomes a
real, usable, safe-to-deploy application*: the screens someone actually
clicks through, who's allowed to do what, how camera credentials and
biometric embeddings are protected, and how the system scales from "one
operator, a few recorded clips" to "one operator, many live cameras across
sites" without over-building infra it doesn't need yet.

Nothing here changes the phase order in the roadmap doc. It adds detail
*inside* phases 2–5 and calls out a small amount of work that should land
**earlier than its phase would suggest** because it's a prerequisite for
handling real camera/biometric data at all (see §5).

---

## 1. UI/UX plan

### 1.1 Design approach

Stay with what's already here rather than introducing a second stack:
server-rendered Django templates (`core/templates/core/*.html`) + one shared
`style.css`, the same pattern as the existing Dataset/Extraction/Training/
Inference pages. Two additions, both scoped to where they're actually
needed rather than applied wholesale:

- **Partial-refresh polling** for anything that updates on its own (camera
  health, alert feed, occupancy counts) — same idea as the existing run
  status pages, just on a shorter interval. No SPA framework.
- **WebSockets (Django Channels)** only for Phase 4's live camera view and
  live alert feed, where polling-latency is the actual product (an alert
  that's 5 seconds stale is a worse alert). Everything else keeps using
  request/response + polling, matching the project's existing bias toward
  the simplest thing that works.

### 1.2 Information architecture

Extend the existing top nav (`base.html`) with one new group, kept visually
separate from the existing ML-pipeline pages since it's a different
audience (a site manager watching a dashboard vs. someone training a model):

```
Dataset · Extraction · Training · Inference        <- existing, unchanged
─────────────────────────────────────────────────
Sites          Monitoring        Identities        <- new
```

- **Sites** — configuration surface: list of sites, cameras per site, zone
  drawing, topology (adjacency) editing, alert rule config, retention
  policy, site membership (who has access). This is the "setup" area,
  Site Manager/Admin territory.
- **Monitoring** — the operational surface: live/recent camera grid,
  occupancy per zone, alert feed, drill-down into one person's cross-camera
  timeline. This is what an Operator lives in day to day.
- **Identities** — staff gallery (enroll/re-enroll/deactivate), and the
  identity-correction queue (linking mistakes a human flagged, see 1.4).

### 1.3 New screens (one per roadmap phase they belong to)

**Phase 2 — Site & Cameras**
- *Site list* → *Site detail*: cameras table (name, source type
  file/RTSP, status badge, last-seen), "+ Add camera" form.
- *Camera detail*: latest snapshot with a canvas overlay to draw named zone
  polygons (`entrance`, `checkout`, `staff_only`, ...) directly on the
  frame — this replaces hand-written zone YAML with something a non-engineer
  site manager can actually do.
- *Topology editor*: simple two-column picker ("Zone A exits into Zone B",
  expected transit time range) rather than a free-form graph canvas —
  matches how the underlying `topology.py` config is actually structured
  (adjacency + transit windows) and is far less error-prone to build/test
  than a drag-and-drop graph UI.
- *Site setup wizard* ties these together in order: add site → add cameras
  → draw zones → define topology → **acknowledge retention/privacy policy**
  (blocking step, see §4.6) → activate. Activation is what turns on
  cross-camera linking for that site; a site with an unacknowledged policy
  stays batch/single-camera-only.

**Phase 3 — Monitoring**
- *Dashboard*: per-site camera grid (thumbnail + status), zone occupancy
  counts, live alert feed panel. This is the primary "monitor staff/
  customers" screen the whole project is for.
- *Person timeline*: one `Identity`'s cross-camera path over a time range —
  chronological strip of (camera, zone, time-in/out, action label,
  match-confidence) with evidence thumbnails per hop. This is where "did
  this person really walk from camera 2 to camera 5" gets verified visually.
- *Alerts list*: filterable by site/zone/severity/status
  (open/acknowledged/dismissed), each linking into the person timeline that
  triggered it.
- *Alert rule config*: per-site rules built from the same zones/identities
  data (e.g. "customer identity in `staff_only` zone", "any identity
  present in a zone > N minutes") — simple rule-builder form, not a DSL.

**Phase 5 — Identities & correction loop**
- *Staff gallery*: enroll a staff `Identity` from one or more photos/crops,
  deactivate on offboarding.
- *Correction queue*: low-confidence or user-flagged `IdentityObservation`
  links surfaced for a human to confirm/reject/merge/split. This closes the
  loop the roadmap's open-set design implies — appearance+topology matching
  will sometimes be wrong, and the UI needs a first-class place to fix it
  rather than only exposing raw confidence numbers on the timeline. Confirms
  feed back as (weak) supervision for future re-ID tuning.

**Phase 4 — Live**
- *Live camera view*: video tile(s) with live detection/track boxes drawn
  over a WebSocket-pushed stream of positions (not raw video over WS —
  push metadata, render boxes over an existing low-latency video element,
  e.g. MJPEG/HLS snapshot refresh, to avoid building a video-over-WS
  pipeline from scratch).

### 1.4 Key flows

- **Onboarding a site** (wizard above) — the one flow that must be
  impossible to skip past the privacy acknowledgment.
- **Investigating an alert**: Alerts list → click alert → Person timeline
  (see the evidence) → confirm it's correct (closes alert) or flag the
  identity link as wrong (routes to Correction queue) — one click either
  way, no dead-end screens.
- **Enrolling staff**: Identities → + Add → upload 1–3 photos → preview
  detected face/person crop → save. Should also be reachable directly from
  a Person timeline ("this unknown identity is actually staff member X") so
  enrollment doesn't require leaving the investigation flow.

### 1.5 States, empty states, degraded states

- Camera status badges: `OK` / `DEGRADED` (frames arriving late/dropped) /
  `DOWN` (no frames) — mirrors the existing `TrainingRun`/
  `TrackExtractionRun` status-badge convention already in the templates, so
  it reads as consistent with the rest of the app rather than a bolted-on
  style.
- Empty states written explicitly for: no cameras yet, no alerts yet
  (good state, phrase it as such — "no alerts" isn't an error), a person
  timeline with only one camera hop (not yet linked anywhere, not a bug).
- A `DOWN` camera must visibly demote confidence on any alert that depended
  on it, rather than silently keeping the last-known occupancy count.

### 1.6 Audience/ergonomics

Monitoring/Sites screens are desktop/control-room-oriented (wide camera
grids, dense tables) — no need to optimize these for mobile. The Alerts
list is the one screen worth keeping usable on a phone, since an on-call
person reacting to an alert away from a desk is a real scenario; everything
else can assume a desktop browser, consistent with the rest of this
internal tool.

---

## 2. Permissions plan

### 2.1 Where this stands today

`django.contrib.auth` is installed (`INSTALLED_APPS`, `AuthenticationMiddleware`,
password validators all present) but **nothing in `core/views/` currently
checks it** — every existing page is reachable by anyone who can reach the
dev server, and `django.contrib.admin` is registered with no additional
restriction. That's an acceptable posture for a single-user local ML tool.
It stops being acceptable the moment the app holds camera credentials,
biometric embeddings, and live footage of real people — so turning auth
*on*, not just having it installed, is a prerequisite for Phase 2 (see §5),
not a nice-to-have.

### 2.2 Roles

Matches the "configurable per deployment" decision already made for staff
vs. customer distinction — roles are assignable per site, not global-only:

| Role | Can |
|---|---|
| **Admin** | Everything: manage users/roles, all sites, retention policy, Django admin access |
| **Site Manager** | Configure cameras/zones/topology/alert rules for *their assigned* site(s); enroll/deactivate staff identities; review the correction queue |
| **Operator** | View Monitoring dashboard, alerts, person timelines for *their assigned* site(s); acknowledge/dismiss alerts; cannot change config |
| **Auditor** | Read-only: audit log, alert history, retention/policy records — no live video, no config. For compliance/legal review without granting operational access |

### 2.3 Scoping — per-site, not global

Django's built-in permission system is global (a user either can or can't
edit `Camera` objects, full stop) — insufficient here, since one operator
is expected to manage *multiple sites* and a Site Manager for Site A must
not see Site B. Add a `SiteMembership(user, site, role)` model and enforce
it in a shared decorator/mixin (`@site_permission_required(role)`) used by
every Sites/Monitoring/Identities view — same "small shared helper, applied
consistently" shape as the existing `_background.py` liveness pattern, just
for authorization instead of process status. Every new view in this plan
gets a webapp test asserting a user outside the site's membership gets a
403, per the existing CLAUDE.md coverage rule.

### 2.4 Sensitive actions — always Admin/Site Manager + audited

Enrolling or deleting a staff identity, exporting video/embeddings,
changing a site's retention policy, confirming/overriding an identity match
in the correction queue. These write to the audit log (§4.5) regardless of
who performs them.

### 2.5 Django admin

`/admin` currently has no extra gate beyond Django's own staff-user check
(and no users are marked staff yet, so it's effectively unused — but
nothing stops someone from becoming one). Before this ships past localhost:
restrict to Admin-role users only, and treat it as a break-glass tool, not
a normal workflow surface — it gives unmediated read/write to `Identity`
and embedding tables, bypassing the per-site scoping in §2.3 entirely.

### 2.6 Non-human access

RTSP ingestion workers (Phase 4) and any external alert-webhook receiver
need credentials distinct from human logins — a `ServiceAccount`/API-key
model, scoped to one site, individually revocable, never sharing a login
with a person (so revoking a departed integration doesn't touch human
accounts and vice versa).

---

## 3. Security plan

### 3.1 Fix first, regardless of new features

`webapp/config/settings.py` today: `DEBUG = True`, a `SECRET_KEY` hardcoded
in source, `ALLOWED_HOSTS = []`, no auth enforced anywhere. This is
reasonable for `runserver` on localhost during ML development and wrong
the moment real camera feeds or another person's data touch the app. None
of this is new work invented by this plan — it's existing debt that becomes
a real vulnerability once the app stops being single-operator-localhost-only,
so it belongs in the same PR that turns auth on (§5), not deferred further.

Concretely, before any non-localhost deployment: `SECRET_KEY` from an
environment variable, `DEBUG = False`, explicit `ALLOWED_HOSTS`,
`SECURE_SSL_REDIRECT`/`SESSION_COOKIE_SECURE`/`CSRF_COOKIE_SECURE`/HSTS,
and a real web server (nginx/Caddy) terminating TLS in front of Django
rather than `runserver`.

### 3.2 AuthN / AuthZ

Django session auth for humans (already available, just needs wiring up
per §2), with the password validators already configured. Given the
sensitivity of what Admin/Site Manager accounts can see (live camera feeds,
biometric embeddings), require 2FA for those two roles once the user
system exists — Operator/Auditor can stay password-only for v1. SSO/OIDC
is a reasonable later addition if deployed inside an org with an existing
identity provider, but not a v1 requirement — don't build it speculatively.

### 3.3 Secrets

RTSP camera credentials and any webhook/API keys must not sit in the DB in
plaintext — encrypt these fields at rest (e.g. Fernet via
`django-cryptography` or an equivalent already-audited library, not a
hand-rolled cipher) even though the DB itself is access-controlled, since a
DB backup or a SQL-injection-class bug shouldn't hand over live camera
access. Never commit real secrets to `configs/` — mirrors the project's
existing gitignore stance on `data/`, extended to credentials.

### 3.4 Data protection

The sensitive payload in this app is video + appearance embeddings — the
embeddings *are* the biometric artifact, more sensitive than the video
frames they're derived from because they're small, portable, and directly
comparable across a whole `Identity` table. Concretely:

- **At rest**: normal filesystem/DB access control on `data/`, `media/`,
  and wherever `reid` embeddings land is the floor; disk encryption at the
  host level covers the video files; consider field-level encryption for
  the embedding vectors themselves given they're the actual re-identifying
  artifact, not just another blob.
- **In transit**: HTTPS for the webapp (§3.1), and camera feeds should
  reach ingestion workers over a private network/VPN or RTSP-over-TLS
  rather than exposing camera IPs to the public internet — camera feeds are
  frequently the weakest link in real deployments (default credentials,
  unencrypted RTSP), and that risk exists independent of anything this app
  does with the footage afterward.

### 3.5 Audit logging

Every identity enrollment/deactivation, every confirm/override in the
correction queue, every retention-policy change, and every permission
change gets an immutable audit record (actor, timestamp, before/after).
This is both a security-review need and the practical answer to the
access-log expectations in GDPR/BIPA-type regimes flagged in the roadmap
doc's privacy section — build it as a first-class model from the start
rather than reconstructing it from scattered logs later.

### 3.6 Retention & deletion

The roadmap already flags a Phase 5 purge job; make it a *tested* guarantee
("purge deletes video + embeddings + observations older than the site's
policy," verified in `tests/webapp/`), not just a documented promise —
this is the one place where "we said we'd delete it" needs to actually be
true.

### 3.7 Threat model highlights specific to this app

1. **Compromised Operator account** → can watch live feeds / browse person
   timelines for their assigned site. Mitigated by least-privilege scoping
   (§2.3) and audit trail (§3.5), not preventable outright — accept and
   contain, matching normal practice for an app that legitimately needs to
   show video to authorized humans.
2. **Unrestricted Django admin** → direct biometric-DB access, bypassing
   all of the above. Mitigated by §2.5.
3. **Leaked RTSP credentials** → attacker watches or injects a camera
   feed. Mitigated by §3.3 + §3.4's network isolation.
4. **False positive re-ID match** → a customer gets mislabeled as a
   flagged/staff identity, potentially triggering a consequential action
   (e.g. a dispatched security response). Mitigate by never letting a
   below-threshold match drive an automated consequential action — route it
   through the correction-queue/human-confirm flow (§1.3) instead. This is
   a correctness-meets-safety concern as much as a security one.

---

## 4. Scalability plan

### 4.1 Where today's architecture caps out

SQLite + Django dev server + one detached subprocess per run is exactly
right for one operator running occasional batch jobs, which is what the
app is today. Real-time multi-camera monitoring breaks three of those
assumptions at once (already flagged in the roadmap's Phase 4 section):
SQLite's single-writer lock, `runserver` as a production web tier, and
"one subprocess that's expected to exit" as the run model. Scale in
stages, matched to what's actually being built at each roadmap phase —
don't jump straight to heavy infra the current camera count doesn't need.

### 4.2 Stage A — still batch (lands with roadmap Phase 2/3)

- **SQLite → PostgreSQL.** Low-risk swap (Django's ORM already abstracts
  it) but necessary once Phase 2/3 introduce concurrent writes from
  multiple background jobs (linking runs, identity edits, alert writes)
  hitting the DB at the same time — SQLite's single-writer lock turns into
  real contention here, not a theoretical one.
- Everything else (subprocess-per-run, local filesystem storage) stays as
  is — batch linking runs are still finite processes, same shape as
  `TrainingRun`/`TrackExtractionRun`.

### 4.3 Stage B — live, single site / single GPU host (roadmap Phase 4)

- **Long-lived stream workers**, one per camera or batched across a small
  group per process, replacing the finite-subprocess model for anything
  marked `is_live` — needs the heartbeat/restart liveness pattern the
  roadmap doc already calls out as distinct from the existing exit-code-
  marker convention.
- **Redis** for pub/sub (cross-camera identity hand-off between workers and
  `linking.py`, now called continuously) and as the heartbeat/cache store —
  deliberately the smallest broker that satisfies this, not Kafka.
- **Django Channels** for pushing live alerts and box overlays to the
  Monitoring dashboard over WebSockets — natural fit since it's already a
  Django app, no separate real-time service to stand up.
- **GPU throughput is the real ceiling**, not the web/DB tier: benchmark
  streams-per-GPU at target latency before committing to any camera count
  in a real deployment (roadmap already flags this) — pick a frame-skip/
  resolution policy from that number, don't guess.

### 4.4 Stage C — many cameras / multiple sites, still one operator (roadmap Phase 5+)

- Horizontal scaling of stream workers across multiple machines, with a
  simple camera→worker-host assignment table — a process supervisor
  (systemd/Supervisor) per host is enough at this scale; no need for
  Kubernetes-grade orchestration until camera count and host count actually
  justify it.
- **Object storage** (S3-compatible) for recorded video/clips once it
  outgrows one disk — local filesystem storage (today's model) is fine
  until then.
- Redis Streams is very likely sufficient as the event backbone at this
  scale too; only reach for a heavier broker (Kafka) if measured event
  volume actually demands it.

### 4.5 Explicitly not built at any stage

Multi-tenant isolation/billing, cloud-agnostic orchestration, Celery — same
non-goals the roadmap doc already states in its §5, restated here because
scalability work is exactly where "let's just use Kubernetes/Celery/Kafka
since we're scaling anyway" scope-creep tends to sneak in. Stay on the
smallest infra that satisfies the camera count actually in front of you.

### 4.6 Observability needed to know when to move stages

Per-camera FPS/latency, GPU utilization, queue depth (Stage B+), DB
connection/lock contention. A simple metrics page is enough at Stage A/B;
Prometheus/Grafana is reasonable once Stage B is running for real. Without
these numbers, "should we move to Stage C" is a guess — tie a per-site
camera-count admission check to them so adding camera #N doesn't silently
degrade every other camera already running.

---

## 5. How this reshapes the roadmap's phase order

Two things from this document should land **earlier** than the roadmap
doc's phase numbering alone would imply, because they're prerequisites for
handling real camera/biometric data safely rather than features of any one
phase:

1. **Auth + per-site RBAC (§2) no later than Phase 2.** Phase 2 is the
   first phase that stores anything sensitive (site/camera config, and
   soon after, `Identity`/embedding data) — it should not ship with every
   page open to anyone who can reach the server.
2. **Production security hardening (§3.1) before any non-localhost
   deployment**, independent of phase number — this is fixing existing
   defaults, not new functionality, so it isn't really "Phase N work" at
   all; treat it as a standing requirement checked before any real
   deployment regardless of which roadmap phase is in progress at the time.

Everything else in this document (UI screens, Stage A/B/C scaling) maps
onto the existing phases as annotated inline in §1 and §4 — no other
reordering.

---

## 6. Suggested next step

The roadmap doc's own suggested starting point (Phase 0 — stronger
single-camera tracking) is still the right first PR; it's independent of
everything in this document. The first piece of *this* document worth
scoping alongside it is small and self-contained too: wiring up
`django.contrib.auth` (login required on existing views, a `SiteMembership`
model ready for Phase 2 to build on) — cheap now, and it means Phase 2
never ships even briefly without access control. Say which one (or both)
to scope into concrete file changes + tests.
