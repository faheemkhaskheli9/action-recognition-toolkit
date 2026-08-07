from __future__ import annotations

import uuid
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.shortcuts import render

from .. import services
from ..forms import InferenceForm


def _save_upload(uploaded_file) -> Path:
    upload_dir = settings.MEDIA_ROOT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{uuid.uuid4().hex}_{uploaded_file.name}"
    with open(dest, "wb") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    return dest


def inference(request):
    checkpoint_choices = services.inference.checkpoint_choices()
    if not checkpoint_choices:
        return render(request, "core/inference.html", {"no_checkpoints": True})

    result = None
    scene_result = None
    if request.method == "POST":
        form = InferenceForm(request.POST, request.FILES, checkpoint_choices=checkpoint_choices)
        if form.is_valid():
            uploaded = form.cleaned_data["video"]
            checkpoint_path = Path(form.cleaned_data["checkpoint"])
            tmp_path = _save_upload(uploaded)
            try:
                if form.cleaned_data["scene_mode"]:
                    scene_result = services.inference.predict_scene(checkpoint_path, tmp_path)
                else:
                    result = services.inference.predict_video(
                        checkpoint_path, tmp_path, top_k=form.cleaned_data["top_k"]
                    )
            except Exception as exc:
                messages.error(request, f"Prediction failed: {exc}")
            finally:
                tmp_path.unlink(missing_ok=True)
    else:
        form = InferenceForm(checkpoint_choices=checkpoint_choices)

    return render(request, "core/inference.html", {"form": form, "result": result, "scene_result": scene_result})
