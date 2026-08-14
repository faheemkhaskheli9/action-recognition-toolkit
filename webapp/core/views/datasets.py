"""Dataset management: create/list/delete Dataset rows themselves (as
opposed to dataset.py, which manages the videos inside whichever dataset is
currently selected)."""
from __future__ import annotations

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render

from .. import services
from ..models import Dataset
from ..paths import resolve_repo_path


def manage_datasets(request):
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        if not name:
            messages.error(request, "Dataset name can't be empty.")
        else:
            try:
                dataset = services.datasets.create_dataset(name)
            except ValueError as exc:
                messages.error(request, str(exc))
            else:
                services.datasets.set_current(request, dataset)
                messages.success(request, f"Created dataset {dataset.name!r}.")
                return redirect("core:dataset_list")

    rows = []
    for dataset in Dataset.objects.all():
        video_dir = resolve_repo_path(dataset.video_dir)
        video_count = len(services.manifest.discover_videos(video_dir)) if video_dir.exists() else 0
        rows.append({"dataset": dataset, "video_count": video_count})

    return render(request, "core/datasets_manage.html", {"rows": rows})


def delete_dataset(request, pk):
    if request.method != "POST":
        return redirect("core:manage_datasets")

    dataset = get_object_or_404(Dataset, pk=pk)
    delete_files = request.POST.get("delete_files") == "on"
    name = dataset.name

    current = services.datasets.get_current(request)
    services.datasets.delete_dataset(dataset, delete_files=delete_files)
    if current is not None and current.pk == dataset.pk:
        request.session.pop(services.datasets.SESSION_KEY, None)

    suffix = " and deleted its files" if delete_files else ""
    messages.success(request, f"Deleted dataset {name!r}{suffix}.")
    return redirect("core:manage_datasets")
