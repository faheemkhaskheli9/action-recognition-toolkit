"""The dataset CRUD page: upload videos, see every video and its label, edit
labels inline, delete entries -- all scoped to the current Dataset (see
services.datasets). `label_videos` (labeling.py) is the focused one-at-a-time
labeling stepper; this page is the "see the whole dataset" and "get videos
in in the first place" counterpart."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from django.contrib import messages
from django.shortcuts import redirect, render
from django.urls import reverse

from .. import services
from ..models import Dataset
from ..paginate import paginate
from ..paths import is_within_repo, resolve_repo_path

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    name = Path(name).name  # drop any directory components the browser sent
    name = _UNSAFE_NAME_CHARS.sub("_", name).lstrip(".")
    return name or "video"


def _redirect_to_dataset(dataset_slug: str):
    url = f"{reverse('core:dataset_list')}?dataset={dataset_slug}"
    return redirect(url)


def dataset_list(request):
    dataset = services.datasets.resolve(request)
    context = {"dataset": dataset, "datasets": Dataset.objects.all()}

    if dataset is None:
        context["known_labels"] = []
        return render(request, "core/dataset_list.html", context)

    context["video_dir"] = dataset.video_dir
    video_dir = resolve_repo_path(dataset.video_dir)
    manifest_path = resolve_repo_path(dataset.manifest_path)

    if not video_dir.exists():
        context["missing_dir"] = True
        context["known_labels"] = []
        return render(request, "core/dataset_list.html", context)

    manifest = services.manifest.load_manifest(manifest_path)
    rows = services.manifest.dataset_rows(video_dir, manifest)
    labeled_count = sum(1 for row in rows if row["labeled"])

    label_counts: dict[str, int] = {}
    for row in rows:
        if row["labeled"]:
            label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    # total/labeled_count/unlabeled_count/label_counts are dataset-wide
    # stats -- computed from every row before paginating, since only the
    # table itself (`rows`) should shrink to one page's worth.
    page = paginate(request, rows)
    context.update(
        {
            "rows": page.object_list,
            "page_obj": page,
            "total": len(rows),
            "labeled_count": labeled_count,
            "unlabeled_count": len(rows) - labeled_count,
            "label_counts": sorted(label_counts.items()),
            "known_labels": services.manifest.known_labels(manifest),
        }
    )
    return render(request, "core/dataset_list.html", context)


def upload_videos(request):
    if request.method != "POST":
        return redirect("core:dataset_list")

    dataset = services.datasets.resolve_from_post(request)
    if dataset is None:
        messages.error(request, "Create a dataset first.")
        return redirect("core:manage_datasets")

    video_dir = resolve_repo_path(dataset.video_dir)
    manifest_path = resolve_repo_path(dataset.manifest_path)
    label = request.POST.get("label", "").strip()

    files = request.FILES.getlist("videos")
    if not files:
        messages.error(request, "Choose at least one video file to upload.")
        return _redirect_to_dataset(dataset.slug)

    video_dir.mkdir(parents=True, exist_ok=True)
    manifest = services.manifest.load_manifest(manifest_path)

    saved = 0
    for uploaded in files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in services.manifest.VIDEO_EXTENSIONS:
            messages.error(request, f"Skipped {uploaded.name}: not a supported video type.")
            continue

        dest = video_dir / _safe_filename(uploaded.name)
        if dest.exists():
            dest = video_dir / f"{dest.stem}_{uuid.uuid4().hex[:8]}{dest.suffix}"

        with open(dest, "wb") as f:
            for chunk in uploaded.chunks():
                f.write(chunk)
        saved += 1

        if label:
            manifest = services.manifest.set_label(manifest_path, manifest, dest, label)

    if saved:
        suffix = f", labeled {label!r}" if label else " — label them on this page or on Label"
        messages.success(request, f"Uploaded {saved} video(s){suffix}.")

    return _redirect_to_dataset(dataset.slug)


def update_label(request):
    if request.method != "POST":
        return redirect("core:dataset_list")

    dataset = services.datasets.resolve_from_post(request)
    if dataset is None:
        messages.error(request, "Create a dataset first.")
        return redirect("core:manage_datasets")

    label = request.POST.get("label", "").strip()
    if not label:
        messages.error(request, "Label can't be empty.")
        return _redirect_to_dataset(dataset.slug)

    manifest_path = resolve_repo_path(dataset.manifest_path)
    video_path = resolve_repo_path(request.POST["video_path"])
    manifest = services.manifest.load_manifest(manifest_path)
    services.manifest.set_label(manifest_path, manifest, video_path, label)
    messages.success(request, f"Labeled {video_path.name} as {label!r}.")

    return _redirect_to_dataset(dataset.slug)


def _optional_float(request, key):
    raw = request.POST.get(key)
    return float(raw) if raw not in (None, "") else None


def delete_entry(request):
    if request.method != "POST":
        return redirect("core:dataset_list")

    dataset = services.datasets.resolve_from_post(request)
    if dataset is None:
        messages.error(request, "Create a dataset first.")
        return redirect("core:manage_datasets")

    delete_file = request.POST.get("delete_file") == "on"

    manifest_path = resolve_repo_path(dataset.manifest_path)
    video_path = resolve_repo_path(request.POST["video_path"])

    if delete_file and not is_within_repo(video_path):
        messages.error(request, "Refusing to delete a file outside the repo.")
        return _redirect_to_dataset(dataset.slug)

    # Absent when deleting the whole-video row (the default, from the inline
    # edit table); present when deleting one labeled span from the list
    # below it, so only that span's row is removed.
    try:
        start_time = _optional_float(request, "start_time")
        end_time = _optional_float(request, "end_time")
    except ValueError:
        messages.error(request, "Start/end time must be numbers.")
        return _redirect_to_dataset(dataset.slug)

    manifest = services.manifest.load_manifest(manifest_path)
    services.manifest.delete_entry(
        manifest_path, manifest, video_path, start_time=start_time, end_time=end_time, delete_file=delete_file
    )
    suffix = " and deleted the file" if delete_file else ""
    messages.success(request, f"Removed {video_path.name} from the dataset{suffix}.")

    return _redirect_to_dataset(dataset.slug)
