from __future__ import annotations

import mimetypes
from pathlib import Path

import pandas as pd
from action_recognition.data.manifest import discover_videos
from django.contrib import messages
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.urls import reverse

from .. import services
from ..forms import ManifestFormSet
from ..models import Dataset
from ..paginate import paginate
from ..paths import is_within_repo, resolve_repo_path


def _session_keys(dataset: Dataset) -> tuple[str, str]:
    """Session keys for this dataset's sticky "current unit" (a plain video
    or a multi-person track group, see _resolve_current_unit) and "skipped
    this session" state -- namespaced per dataset so switching datasets via
    the picker mid-label can't leak one dataset's in-progress unit into
    another's (a flat, unscoped key used to let a save after switching
    datasets write the old dataset's video into the new dataset's
    manifest)."""
    return f"current_unit:{dataset.slug}", f"skipped:{dataset.slug}"


def _grouped_paths(groups: list[dict]) -> set[str]:
    return {clip["path"] for group in groups for track in group["tracks"] for clip in track["clips"]}


def _group_key(group: dict) -> str:
    return f"{group['run_id']}:{group['source_video']}"


def _find_group(groups: list[dict], run_id: int, source_video: str) -> dict | None:
    for group in groups:
        if group["run_id"] == run_id and group["source_video"] == source_video:
            return group
    return None


def _group_labeled(group: dict, labeled_paths: set[str]) -> bool:
    return all(clip["path"] in labeled_paths for track in group["tracks"] for clip in track["clips"])


def _resolve_current_unit(sticky: dict | None, video_dir: Path, manifest: pd.DataFrame, skipped: set, groups: list[dict]) -> dict | None:
    """Picks the video or multi-person group the labeler sees next, sticky
    across requests (`sticky` is the session's last unit) so that adding one
    span or labeling one track of a group -- which writes manifest rows
    without being "done" -- doesn't jump the page elsewhere on the next
    load. Plain videos are offered before groups, in discovery order; the
    first group with any unlabeled clip comes after."""
    labeled_paths = set(manifest["video_path"]) if not manifest.empty else set()
    grouped_paths = _grouped_paths(groups)

    if sticky:
        if sticky.get("kind") == "video":
            candidate = resolve_repo_path(sticky["path"])
            if candidate.exists() and str(candidate.resolve()) not in grouped_paths:
                return sticky
        elif sticky.get("kind") == "group":
            group = _find_group(groups, sticky["run_id"], sticky["source_video"])
            if group and not _group_labeled(group, labeled_paths):
                return sticky

    for video in discover_videos(video_dir):
        resolved = str(video.resolve())
        if resolved in grouped_paths or resolved in labeled_paths or resolved in skipped:
            continue
        return {"kind": "video", "path": str(video)}

    for group in groups:
        if _group_key(group) in skipped or _group_labeled(group, labeled_paths):
            continue
        return {"kind": "group", "run_id": group["run_id"], "source_video": group["source_video"]}

    return None


def _group_context(unit: dict, groups: list[dict], manifest: pd.DataFrame) -> dict | None:
    group = _find_group(groups, unit["run_id"], unit["source_video"])
    if group is None:
        return None
    labeled_paths = set(manifest["video_path"]) if not manifest.empty else set()
    tracks = []
    for track in group["tracks"]:
        clips = [
            {**clip, "name": Path(clip["path"]).name, "labeled": clip["path"] in labeled_paths}
            for clip in track["clips"]
        ]
        tracks.append({"track_id": track["track_id"], "clips": clips, "labeled": all(c["labeled"] for c in clips)})
    return {
        "run_id": group["run_id"],
        "run_name": group["run_name"],
        "source_video": group["source_video"],
        "tracks": tracks,
        "track_count": len(tracks),
        "clip_count": sum(len(t["clips"]) for t in tracks),
    }


def label_videos(request):
    dataset = services.datasets.resolve(request)
    context = {"dataset": dataset, "datasets": Dataset.objects.all()}

    if dataset is None:
        return render(request, "core/labeling.html", context)

    context["video_dir"] = dataset.video_dir
    video_dir = resolve_repo_path(dataset.video_dir)
    manifest_path = resolve_repo_path(dataset.manifest_path)

    if not video_dir.exists():
        context["missing_dir"] = True
        return render(request, "core/labeling.html", context)

    manifest = services.manifest.load_manifest(manifest_path)
    current_key, skipped_key = _session_keys(dataset)
    skipped = set(request.session.get(skipped_key, []))

    # Clips imported from a track-extraction run (services.extraction.
    # import_clips) get grouped by source video/track here instead of being
    # shown as unrelated videos -- see _resolve_current_unit.
    provenance = services.extraction.provenance_for_dataset(dataset)
    groups = services.extraction.group_clips(provenance)

    # The current unit is sticky across requests (rather than re-picked
    # every time) so that adding a labeled span, or labeling one track of a
    # group -- which write manifest rows without being "done" -- don't
    # cause the very next page load to jump elsewhere.
    unit = _resolve_current_unit(request.session.get(current_key), video_dir, manifest, skipped, groups)
    request.session[current_key] = unit

    labeled_count, total = services.manifest.progress(video_dir, manifest)

    context.update(
        {
            "current_unit": unit,
            "known_labels": services.manifest.known_labels(manifest),
            "labeled_count": labeled_count,
            "total": total,
        }
    )

    if unit and unit["kind"] == "video":
        current = resolve_repo_path(unit["path"])
        context.update(
            {
                "current": unit["path"],
                "current_name": current.name,
                "current_spans": services.manifest.spans_for(manifest, current),
            }
        )
    elif unit and unit["kind"] == "group":
        context["current_group"] = _group_context(unit, groups, manifest)

    return render(request, "core/labeling.html", context)


def save_label(request):
    if request.method != "POST":
        return redirect("core:label_videos")

    dataset = services.datasets.resolve_from_post(request)
    if dataset is None:
        messages.error(request, "Create a dataset first.")
        return redirect("core:manage_datasets")

    label = request.POST.get("label", "").strip()
    if not label:
        messages.error(request, "Label can't be empty.")
        return redirect("core:label_videos")

    video_path = resolve_repo_path(request.POST["video_path"])
    manifest_path = resolve_repo_path(dataset.manifest_path)
    manifest = services.manifest.load_manifest(manifest_path)
    services.manifest.save_label(manifest_path, manifest, video_path, label)
    current_key, _ = _session_keys(dataset)
    # "keep_current" is set by the per-window label form inside a track
    # group (_label_track_group.html) -- labeling one window clip there
    # doesn't mean the whole group is done, unlike the plain single-video
    # form, which always advances.
    if not request.POST.get("keep_current"):
        request.session.pop(current_key, None)
    return redirect("core:label_videos")


def _parse_span(request):
    """Shared start_time/end_time validation for add_span/delete_span.
    Returns (start_time, end_time) as floats, or None + an error message."""
    try:
        start_time = float(request.POST["start_time"])
        end_time = float(request.POST["end_time"])
    except (KeyError, ValueError):
        return None, None, "Start/end time must be numbers."
    if start_time < 0 or end_time <= start_time:
        return None, None, "End time must be after start time, and both must be non-negative."
    return start_time, end_time, None


def add_span(request):
    if request.method != "POST":
        return redirect("core:label_videos")

    dataset = services.datasets.resolve_from_post(request)
    if dataset is None:
        messages.error(request, "Create a dataset first.")
        return redirect("core:manage_datasets")

    label = request.POST.get("label", "").strip()
    if not label:
        messages.error(request, "Label can't be empty.")
        return redirect("core:label_videos")

    start_time, end_time, error = _parse_span(request)
    if error:
        messages.error(request, error)
        return redirect("core:label_videos")

    video_path = resolve_repo_path(request.POST["video_path"])
    manifest_path = resolve_repo_path(dataset.manifest_path)
    manifest = services.manifest.load_manifest(manifest_path)
    services.manifest.save_label(
        manifest_path, manifest, video_path, label, start_time=start_time, end_time=end_time
    )
    # current_video stays set on purpose -- stay on this video so more spans
    # can be added; "Done with this video" (finish_video) advances instead.
    return redirect("core:label_videos")


def delete_span(request):
    if request.method != "POST":
        return redirect("core:label_videos")

    dataset = services.datasets.resolve_from_post(request)
    if dataset is None:
        messages.error(request, "Create a dataset first.")
        return redirect("core:manage_datasets")

    start_time, end_time, error = _parse_span(request)
    if error:
        messages.error(request, error)
        return redirect("core:label_videos")

    video_path = resolve_repo_path(request.POST["video_path"])
    manifest_path = resolve_repo_path(dataset.manifest_path)
    manifest = services.manifest.load_manifest(manifest_path)
    services.manifest.delete_entry(manifest_path, manifest, video_path, start_time=start_time, end_time=end_time)
    return redirect("core:label_videos")


def finish_video(request):
    if request.method == "POST":
        dataset = services.datasets.resolve_from_post(request)
        if dataset is not None:
            current_key, _ = _session_keys(dataset)
            request.session.pop(current_key, None)
    return redirect("core:label_videos")


def skip_video(request):
    if request.method == "POST":
        dataset = services.datasets.resolve_from_post(request)
        if dataset is None:
            messages.error(request, "Create a dataset first.")
            return redirect("core:manage_datasets")

        current_key, skipped_key = _session_keys(dataset)
        skipped = set(request.session.get(skipped_key, []))
        skipped.add(request.POST.get("video_path", ""))
        request.session[skipped_key] = list(skipped)
        request.session.pop(current_key, None)
    return redirect("core:label_videos")


def skip_group(request):
    """Same as skip_video, but for a whole multi-person track group -- keyed
    by run/source-video (_group_key) instead of a single video_path, since a
    group isn't any one file."""
    if request.method == "POST":
        dataset = services.datasets.resolve_from_post(request)
        if dataset is None:
            messages.error(request, "Create a dataset first.")
            return redirect("core:manage_datasets")

        try:
            run_id = int(request.POST["run_id"])
        except (KeyError, ValueError):
            return redirect("core:label_videos")
        source_video = request.POST.get("source_video", "")

        current_key, skipped_key = _session_keys(dataset)
        skipped = set(request.session.get(skipped_key, []))
        skipped.add(f"{run_id}:{source_video}")
        request.session[skipped_key] = list(skipped)
        request.session.pop(current_key, None)
    return redirect("core:label_videos")


def label_track(request):
    """Labels every window clip of one track (one tracked person, across
    however many time-windows extraction split them into) in a single
    write -- the common case for a multi-person import, where the whole
    track is one continuous action. For the less-common case where the
    action changes mid-track, the group template's "label windows
    separately" toggle falls back to save_label per clip instead."""
    if request.method != "POST":
        return redirect("core:label_videos")

    dataset = services.datasets.resolve_from_post(request)
    if dataset is None:
        messages.error(request, "Create a dataset first.")
        return redirect("core:manage_datasets")

    label = request.POST.get("label", "").strip()
    if not label:
        messages.error(request, "Label can't be empty.")
        return redirect("core:label_videos")

    try:
        run_id = int(request.POST["run_id"])
        track_id = int(request.POST["track_id"])
    except (KeyError, ValueError):
        messages.error(request, "Missing track to label.")
        return redirect("core:label_videos")
    source_video = request.POST.get("source_video", "")

    provenance = services.extraction.provenance_for_dataset(dataset)
    clips = sorted(
        (
            path
            for path, info in provenance.items()
            if info["run_id"] == run_id and info["source_video"] == source_video and info["track_id"] == track_id
        ),
        key=lambda path: provenance[path]["window_index"],
    )
    if not clips:
        messages.error(request, "That track has no clips to label.")
        return redirect("core:label_videos")

    manifest_path = resolve_repo_path(dataset.manifest_path)
    manifest = services.manifest.load_manifest(manifest_path)
    services.manifest.save_labels(manifest_path, manifest, [(Path(p), label) for p in clips])
    # current_unit stays sticky on purpose -- same reasoning as add_span:
    # labeling one track doesn't mean the whole source video's group is done.
    return redirect("core:label_videos")


def serve_video(request):
    raw = request.GET.get("path", "")
    path = resolve_repo_path(raw)
    if not is_within_repo(path) or not path.is_file():
        raise Http404
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(open(path, "rb"), content_type=content_type)


def review_manifest(request):
    dataset = services.datasets.resolve(request)
    if dataset is None:
        return render(request, "core/review_manifest.html", {"dataset": None, "datasets": Dataset.objects.all()})

    manifest_path = resolve_repo_path(dataset.manifest_path)
    manifest = services.manifest.load_manifest(manifest_path)
    has_split = "split" in manifest.columns
    has_spans = "start_time" in manifest.columns

    # Paginated over row *position*, not row data -- GET builds the
    # formset's initial data from manifest.iloc[start:end]; POST needs that
    # same [start:end) to know which slice of the full manifest this page's
    # (possibly edited/deleted) formset replaces, so the same page() call
    # against the same-length manifest gives both sides identical bounds
    # without POST having to re-derive anything from submitted data. The
    # form has no explicit action, so it resubmits to the current URL --
    # request.GET (built from the URL's query string regardless of method)
    # still has ?page= on POST, no hidden field needed.
    page = paginate(request, range(len(manifest)))
    start, end = max(page.start_index() - 1, 0), page.end_index()

    if request.method == "POST":
        formset = ManifestFormSet(request.POST)
        if formset.is_valid():
            rows = []
            for form in formset:
                if form in formset.deleted_forms:
                    continue
                data = form.cleaned_data
                row = {"video_path": data["video_path"], "label": data["label"]}
                if has_split:
                    row["split"] = data.get("split") or ""
                if has_spans:
                    row["start_time"] = float(data["start_time"]) if data.get("start_time") else None
                    row["end_time"] = float(data["end_time"]) if data.get("end_time") else None
                rows.append(row)
            columns = (
                ["video_path", "label"]
                + (["split"] if has_split else [])
                + (["start_time", "end_time"] if has_spans else [])
            )
            page_df = pd.DataFrame(rows, columns=columns)
            # Splice this page's rows back in at the position they came
            # from -- every row outside [start:end), on any other page,
            # wasn't part of this page's formset and so is carried through
            # untouched no matter how many pages the manifest spans.
            new_df = pd.concat([manifest.iloc[:start], page_df, manifest.iloc[end:]], ignore_index=True)
            services.manifest.write_manifest(new_df, manifest_path)
            messages.success(request, f"Saved {len(page_df)} row(s) to {manifest_path}")
            return redirect(f"{reverse('core:review_manifest')}?dataset={dataset.slug}&page={page.number}")
    else:
        initial = [
            {
                "video_path": row["video_path"],
                "label": row["label"],
                "split": row.get("split", "") if has_split else "",
                "start_time": row.get("start_time", "") if has_spans else "",
                "end_time": row.get("end_time", "") if has_spans else "",
                "span_display": (
                    f"{services.manifest.format_seconds(row['start_time'])}"
                    f"–{services.manifest.format_seconds(row['end_time'])}"
                    if has_spans and pd.notna(row.get("start_time")) and pd.notna(row.get("end_time"))
                    else ""
                ),
            }
            for _, row in manifest.iloc[start:end].iterrows()
        ]
        formset = ManifestFormSet(initial=initial)

    return render(
        request,
        "core/review_manifest.html",
        {
            "formset": formset,
            "dataset": dataset,
            "datasets": Dataset.objects.all(),
            "has_split": has_split,
            "has_spans": has_spans,
            "page_obj": page,
        },
    )
