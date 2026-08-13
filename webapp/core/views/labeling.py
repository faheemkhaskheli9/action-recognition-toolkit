from __future__ import annotations

import mimetypes

import pandas as pd
from django.contrib import messages
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.urls import reverse

from .. import services
from ..forms import ManifestFormSet
from ..models import Dataset
from ..paths import is_within_repo, resolve_repo_path


def _dataset_from_post(request) -> Dataset | None:
    slug = request.POST.get("dataset")
    return Dataset.objects.filter(slug=slug).first() if slug else None


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
    skipped = set(request.session.get("skipped", []))

    # The current video is sticky across requests (rather than re-picked from
    # next_unlabeled() every time) so that adding a labeled span -- which
    # writes a manifest row for this video without being "done" with it --
    # doesn't cause the very next page load to jump to a different video.
    current_str = request.session.get("current_video")
    current = None
    if current_str:
        candidate = resolve_repo_path(current_str)
        if candidate.exists():
            current = candidate
    if current is None:
        current = services.manifest.next_unlabeled(video_dir, manifest, skipped)
        request.session["current_video"] = str(current) if current else None

    labeled_count, total = services.manifest.progress(video_dir, manifest)

    context.update(
        {
            "current": str(current) if current else None,
            "current_name": current.name if current else None,
            "current_spans": services.manifest.spans_for(manifest, current) if current else [],
            "known_labels": services.manifest.known_labels(manifest),
            "labeled_count": labeled_count,
            "total": total,
        }
    )
    return render(request, "core/labeling.html", context)


def save_label(request):
    if request.method != "POST":
        return redirect("core:label_videos")

    dataset = _dataset_from_post(request)
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
    request.session.pop("current_video", None)
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

    dataset = _dataset_from_post(request)
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

    dataset = _dataset_from_post(request)
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
        request.session.pop("current_video", None)
    return redirect("core:label_videos")


def skip_video(request):
    if request.method == "POST":
        skipped = set(request.session.get("skipped", []))
        skipped.add(request.POST.get("video_path", ""))
        request.session["skipped"] = list(skipped)
        request.session.pop("current_video", None)
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
            new_df = pd.DataFrame(rows, columns=columns)
            services.manifest.write_manifest(new_df, manifest_path)
            messages.success(request, f"Saved {len(new_df)} rows to {manifest_path}")
            return redirect(f"{reverse('core:review_manifest')}?dataset={dataset.slug}")
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
            for _, row in manifest.iterrows()
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
        },
    )
