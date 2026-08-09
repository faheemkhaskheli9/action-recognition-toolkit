from __future__ import annotations

import uuid
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.shortcuts import render

from .. import services
from ..forms import InferenceForm
from ..paths import is_within_repo, resolve_repo_path


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
            checkpoint_path = Path(form.cleaned_data["checkpoint"])
            uploaded = form.cleaned_data["video"]

            tmp_path = None
            if uploaded:
                video_path = tmp_path = _save_upload(uploaded)
            else:
                video_path = resolve_repo_path(form.cleaned_data["existing_video"])
                if not is_within_repo(video_path) or not video_path.is_file():
                    form.add_error("existing_video", "Not a valid video in this dataset.")
                    video_path = None

            if video_path is not None:
                try:
                    if form.cleaned_data["scene_mode"]:
                        scene_result = services.inference.predict_scene(checkpoint_path, video_path)
                    else:
                        result = services.inference.predict_video(
                            checkpoint_path, video_path, top_k=form.cleaned_data["top_k"]
                        )
                except Exception as exc:
                    messages.error(request, f"Prediction failed: {exc}")
                finally:
                    if tmp_path is not None:
                        tmp_path.unlink(missing_ok=True)
    else:
        form = InferenceForm(checkpoint_choices=checkpoint_choices)

    context = {
        "form": form,
        "result": result,
        "scene_result": scene_result,
        "known_videos": services.manifest.known_videos(settings.DATA_DIR),
    }
    return render(request, "core/inference.html", context)
