from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from .. import services
from ..forms import StartExtractionForm
from ..models import TrackExtractionRun


def start_extraction(request):
    if request.method == "POST":
        form = StartExtractionForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data["name"]
            if services.extraction.run_dir(name).exists():
                form.add_error("name", "An extraction run with this name already exists.")
            else:
                run = services.extraction.start_run(
                    name=name,
                    video_dir=form.cleaned_data["video_dir"],
                    config_path=Path(form.cleaned_data["config"]) if form.cleaned_data.get("config") else None,
                )
                messages.success(request, f"Started track extraction '{run.name}' (pid {run.pid}).")
                return redirect("core:extraction_detail", pk=run.pk)
    else:
        form = StartExtractionForm()
    known_video_dirs = services.manifest.known_video_dirs(settings.DATA_DIR)
    return render(request, "core/extraction_form.html", {"form": form, "known_video_dirs": known_video_dirs})


def extraction_list(request):
    runs = list(TrackExtractionRun.objects.all())
    for run in runs:
        services.extraction.refresh_status(run)
    return render(request, "core/extraction_list.html", {"runs": runs})


def extraction_detail(request, pk):
    run = get_object_or_404(TrackExtractionRun, pk=pk)
    services.extraction.refresh_status(run)
    context = {
        "run": run,
        "log_tail": services.extraction.tail_log(run),
        "clip_count": services.extraction.clip_count(run),
    }
    return render(request, "core/extraction_detail.html", context)


def extraction_log_partial(request, pk):
    run = get_object_or_404(TrackExtractionRun, pk=pk)
    services.extraction.refresh_status(run)
    return JsonResponse(
        {
            "status": run.status,
            "log_tail": services.extraction.tail_log(run),
            "clip_count": services.extraction.clip_count(run),
        }
    )
