from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect, render

from .. import services
from ..forms import StartTrainingForm


def start_training(request):
    if request.method == "POST":
        form = StartTrainingForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data["name"]
            if services.training.run_dir(name).exists():
                form.add_error("name", "A run with this name already exists.")
            else:
                run = services.training.start_run(
                    name=name,
                    config_path=Path(form.cleaned_data["config"]),
                    manifest_path=form.cleaned_data.get("manifest_path") or None,
                    model_name=form.cleaned_data.get("model_name") or None,
                    epochs=form.cleaned_data.get("epochs"),
                    batch_size=form.cleaned_data.get("batch_size"),
                )
                messages.success(request, f"Started training run '{run.name}' (pid {run.pid}).")
                return redirect("core:run_detail", pk=run.pk)
    else:
        form = StartTrainingForm()
    known_manifest_paths = services.manifest.known_manifest_paths(settings.DATA_DIR)
    return render(request, "core/training_form.html", {"form": form, "known_manifest_paths": known_manifest_paths})
