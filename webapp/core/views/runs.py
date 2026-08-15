from __future__ import annotations

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from .. import services
from ..models import TrainingRun
from ..paginate import paginate


def run_list(request):
    # refresh_status only for the runs actually shown -- each check reads
    # that run's log file, no reason to pay that cost for every run in the
    # history when just one page's worth is ever rendered.
    page = paginate(request, TrainingRun.objects.all())
    for run in page.object_list:
        services.training.refresh_status(run)
    return render(request, "core/training_list.html", {"runs": page.object_list, "page_obj": page})


def run_detail(request, pk):
    run = get_object_or_404(TrainingRun, pk=pk)
    services.training.refresh_status(run)
    context = {
        "run": run,
        "log_tail": services.training.tail_log(run),
        "checkpoints": services.training.checkpoints(run),
    }
    return render(request, "core/training_detail.html", context)


def cancel_run(request, pk):
    if request.method != "POST":
        return redirect("core:run_detail", pk=pk)

    run = get_object_or_404(TrainingRun, pk=pk)
    services.training.refresh_status(run)

    if run.status != TrainingRun.Status.RUNNING:
        messages.error(request, "This run has already stopped.")
        return redirect("core:run_detail", pk=pk)

    services.training.cancel_run(run)
    messages.success(request, f"Cancelled training run '{run.name}'.")
    return redirect("core:run_detail", pk=pk)


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
