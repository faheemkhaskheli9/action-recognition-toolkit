"""Pipeline-overview landing page (`/`). Read-only: shows what datasets
exist, and the most recent extraction/training runs with live status, so a
returning user has somewhere to orient before picking a pipeline step from
the header nav. Deliberately does *not* recompute per-dataset video/labeled
counts here (that's what the Dataset page's `dataset_list` view already
does, scoped to one dataset) -- doing that per-dataset walk of every video
file for *every* dataset on every Home hit doesn't scale as datasets grow,
so this page only shows cheap, indexed model-query data."""
from __future__ import annotations

from django.shortcuts import render

from .. import services
from ..models import Dataset, TrackExtractionRun, TrainingRun
from ..paths import resolve_repo_path

RECENT_RUNS_LIMIT = 5


def home(request):
    # video_dir existence is a single stat() per dataset -- cheap. Video/label
    # *counts* are deliberately not computed here; see module docstring.
    datasets = list(Dataset.objects.all())
    for dataset in datasets:
        dataset.video_dir_exists = resolve_repo_path(dataset.video_dir).exists()

    extraction_runs = list(TrackExtractionRun.objects.all()[:RECENT_RUNS_LIMIT])
    for run in extraction_runs:
        services.extraction.refresh_status(run)

    training_runs = list(TrainingRun.objects.all()[:RECENT_RUNS_LIMIT])
    for run in training_runs:
        services.training.refresh_status(run)

    running_extractions = TrackExtractionRun.objects.filter(status=TrackExtractionRun.Status.RUNNING).count()
    running_trainings = TrainingRun.objects.filter(status=TrainingRun.Status.RUNNING).count()

    context = {
        "datasets": datasets,
        "recent_extraction_runs": extraction_runs,
        "recent_training_runs": training_runs,
        "running_extractions": running_extractions,
        "running_trainings": running_trainings,
    }
    return render(request, "core/home.html", context)
