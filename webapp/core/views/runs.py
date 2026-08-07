from __future__ import annotations

from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render

from .. import services
from ..models import TrainingRun


def run_list(request):
    runs = list(TrainingRun.objects.all())
    for run in runs:
        services.training.refresh_status(run)
    return render(request, "core/training_list.html", {"runs": runs})


def run_detail(request, pk):
    run = get_object_or_404(TrainingRun, pk=pk)
    services.training.refresh_status(run)
    context = {
        "run": run,
        "log_tail": services.training.tail_log(run),
        "checkpoints": services.training.checkpoints(run),
    }
    return render(request, "core/training_detail.html", context)


def run_log_partial(request, pk):
    run = get_object_or_404(TrainingRun, pk=pk)
    services.training.refresh_status(run)
    return JsonResponse(
        {
            "status": run.status,
            "log_tail": services.training.tail_log(run),
            "return_code": run.return_code,
        }
    )
