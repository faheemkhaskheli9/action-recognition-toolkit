from __future__ import annotations

import mimetypes

import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.urls import reverse

from .. import services
from ..forms import ManifestFormSet
from ..paths import is_within_repo, resolve_repo_path


def label_videos(request):
    video_dir_str = request.GET.get("video_dir") or request.session.get("video_dir") or "data/raw"
    manifest_path_str = (
        request.GET.get("manifest_path") or request.session.get("manifest_path") or "data/manifest.csv"
    )
    request.session["video_dir"] = video_dir_str
    request.session["manifest_path"] = manifest_path_str

    video_dir = resolve_repo_path(video_dir_str)
    manifest_path = resolve_repo_path(manifest_path_str)

    context = {
        "video_dir": video_dir_str,
        "manifest_path": manifest_path_str,
        "known_video_dirs": services.manifest.known_video_dirs(settings.DATA_DIR),
        "known_manifest_paths": services.manifest.known_manifest_paths(settings.DATA_DIR),
    }

    if not video_dir.exists():
        context["missing_dir"] = True
        return render(request, "core/labeling.html", context)

    manifest = services.manifest.load_manifest(manifest_path)
    skipped = set(request.session.get("skipped", []))
    current = services.manifest.next_unlabeled(video_dir, manifest, skipped)
    labeled_count, total = services.manifest.progress(video_dir, manifest)

    context.update(
        {
            "current": str(current) if current else None,
            "current_name": current.name if current else None,
            "known_labels": services.manifest.known_labels(manifest),
            "labeled_count": labeled_count,
            "total": total,
        }
    )
    return render(request, "core/labeling.html", context)


def save_label(request):
    if request.method != "POST":
        return redirect("core:label_videos")

    label = request.POST.get("label", "").strip()
    if not label:
        messages.error(request, "Label can't be empty.")
        return redirect("core:label_videos")

    video_path = resolve_repo_path(request.POST["video_path"])
    manifest_path = resolve_repo_path(request.POST["manifest_path"])
    manifest = services.manifest.load_manifest(manifest_path)
    services.manifest.save_label(manifest_path, manifest, video_path, label)
    return redirect("core:label_videos")


def skip_video(request):
    if request.method == "POST":
        skipped = set(request.session.get("skipped", []))
        skipped.add(request.POST.get("video_path", ""))
        request.session["skipped"] = list(skipped)
    return redirect("core:label_videos")


def serve_video(request):
    raw = request.GET.get("path", "")
    path = resolve_repo_path(raw)
    if not is_within_repo(path) or not path.is_file():
        raise Http404
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(open(path, "rb"), content_type=content_type)


def review_manifest(request):
    manifest_path_str = (
        request.GET.get("manifest_path") or request.POST.get("manifest_path") or "data/manifest.csv"
    )
    manifest_path = resolve_repo_path(manifest_path_str)
    manifest = services.manifest.load_manifest(manifest_path)
    has_split = "split" in manifest.columns

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
                rows.append(row)
            columns = ["video_path", "label"] + (["split"] if has_split else [])
            new_df = pd.DataFrame(rows, columns=columns)
            services.manifest.write_manifest(new_df, manifest_path)
            messages.success(request, f"Saved {len(new_df)} rows to {manifest_path}")
            return redirect(f"{reverse('core:review_manifest')}?manifest_path={manifest_path_str}")
    else:
        initial = [
            {
                "video_path": row["video_path"],
                "label": row["label"],
                "split": row.get("split", "") if has_split else "",
            }
            for _, row in manifest.iterrows()
        ]
        formset = ManifestFormSet(initial=initial)

    return render(
        request,
        "core/review_manifest.html",
        {
            "formset": formset,
            "manifest_path": manifest_path_str,
            "has_split": has_split,
            "known_manifest_paths": services.manifest.known_manifest_paths(settings.DATA_DIR),
        },
    )
