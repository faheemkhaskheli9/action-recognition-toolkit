from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from .. import services
from ..forms import StartExtractionForm
from ..models import TrackExtractionRun
from ..paginate import paginate
from ..paths import resolve_repo_path


def start_extraction(request):
    dataset = services.datasets.resolve(request)

    if request.method == "POST":
        form = StartExtractionForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data["name"]
            if services.extraction.run_dir(name).exists():
                form.add_error("name", "An extraction run with this name already exists.")
            else:
                run = services.extraction.start_run(
                    name=name,
                    dataset=form.cleaned_data["dataset"],
                    video=form.cleaned_data.get("video") or None,
                    config_path=Path(form.cleaned_data["config"]) if form.cleaned_data.get("config") else None,
                )
                messages.success(request, f"Started track extraction '{run.name}' (pid {run.pid}).")
                return redirect("core:extraction_detail", pk=run.pk)
    else:
        form = StartExtractionForm(initial={"dataset": dataset} if dataset else None)

    known_videos = (
        services.manifest.known_videos(resolve_repo_path(dataset.video_dir), repo_root=settings.REPO_ROOT)
        if dataset
        else []
    )
    return render(
        request,
        "core/extraction_form.html",
        {"form": form, "dataset": dataset, "known_videos": known_videos},
    )


def extraction_list(request):
    # refresh_status/clip_count/progress only for the runs actually shown --
    # each reads that run's log file (and clip_count may scan its output
    # dir), no reason to pay that cost for the whole history on every load.
    page = paginate(request, TrackExtractionRun.objects.all())
    runs = list(page.object_list)
    for run in runs:
        services.extraction.refresh_status(run)
        run.clip_count = services.extraction.clip_count(run)
        run.progress = services.extraction.progress(run) if run.status == TrackExtractionRun.Status.RUNNING else None
    return render(request, "core/extraction_list.html", {"runs": runs, "page_obj": page})


def extraction_detail(request, pk):
    run = get_object_or_404(TrackExtractionRun, pk=pk)
    services.extraction.refresh_status(run)
    # Same target `import_extraction_clips` actually copies into -- shown so
    # the "Import clips" help text and its Label/Dataset links describe
    # where clips really land, not the pre-Dataset-model fixed `data/raw`.
    target_dataset = run.dataset or services.datasets.get_current(request)
    context = {
        "run": run,
        "log_tail": services.extraction.tail_log(run),
        "clip_count": services.extraction.clip_count(run),
        "progress": services.extraction.progress(run),
        "results": services.extraction.results(run),
        "target_dataset": target_dataset,
    }
    return render(request, "core/extraction_detail.html", context)


def resume_extraction(request, pk):
    if request.method != "POST":
        return redirect("core:extraction_detail", pk=pk)

    run = get_object_or_404(TrackExtractionRun, pk=pk)
    services.extraction.refresh_status(run)

    if not services.extraction.can_resume(run):
        messages.error(request, "This run is still going — wait for it to stop before resuming.")
        return redirect("core:extraction_detail", pk=pk)

    run = services.extraction.resume_run(run)
    messages.success(request, f"Resumed track extraction '{run.name}' (pid {run.pid}).")
    return redirect("core:extraction_detail", pk=pk)


def cancel_extraction(request, pk):
    if request.method != "POST":
        return redirect("core:extraction_detail", pk=pk)

    run = get_object_or_404(TrackExtractionRun, pk=pk)
    services.extraction.refresh_status(run)

    if run.status != TrackExtractionRun.Status.RUNNING:
        messages.error(request, "This run has already stopped.")
        return redirect("core:extraction_detail", pk=pk)

    services.extraction.cancel_run(run)
    messages.success(request, f"Cancelled track extraction '{run.name}'.")
    return redirect("core:extraction_detail", pk=pk)


def import_extraction_clips(request, pk):
    if request.method != "POST":
        return redirect("core:extraction_detail", pk=pk)

    run = get_object_or_404(TrackExtractionRun, pk=pk)
    services.extraction.refresh_status(run)

    if run.status != TrackExtractionRun.Status.SUCCEEDED:
        messages.error(request, "Wait for the extraction run to finish before importing its clips.")
        return redirect("core:extraction_detail", pk=pk)

    # Import back into the run's own source dataset by default -- it's still
    # the natural home for its clips. Falls back to whichever dataset is
    # otherwise current if that dataset has since been deleted.
    target = run.dataset or services.datasets.get_current(request)
    if target is None:
        messages.error(request, "Create a dataset to import these clips into first.")
        return redirect("core:extraction_detail", pk=pk)

    dest_dir = resolve_repo_path(target.video_dir)
    imported = services.extraction.import_clips(run, dest_dir)
    if imported:
        messages.success(
            request,
            f"Imported {imported} clip(s) into {target.name!r}. Head to Label to tag them.",
        )
    else:
        messages.info(request, "Nothing new to import — every clip from this run is already in the dataset.")
    return redirect("core:extraction_detail", pk=pk)


def extraction_log_partial(request, pk):
    run = get_object_or_404(TrackExtractionRun, pk=pk)
    services.extraction.refresh_status(run)
    return JsonResponse(
        {
            "status": run.status,
            "log_tail": services.extraction.tail_log(run),
            "clip_count": services.extraction.clip_count(run),
            "progress": services.extraction.progress(run),
        }
    )
