"""CRUD for Dataset -- the named collection of videos (its own folder +
manifest CSV) that the Dataset/Label/Review pages operate within, and that
extraction runs against. See models.Dataset for the fields."""
from __future__ import annotations

import shutil

from django.utils.text import slugify

from ..models import Dataset
from ..paths import is_within_repo, resolve_repo_path

# Session key holding the slug of the dataset the Dataset/Label/Review/
# Extraction pages are currently scoped to -- same "sticky selection" shape
# the app used for manifest_path before datasets existed.
SESSION_KEY = "dataset_slug"


def _unique_slug(name: str) -> str:
    base = slugify(name) or "dataset"
    slug = base
    n = 2
    while Dataset.objects.filter(slug=slug).exists():
        slug = f"{base}-{n}"
        n += 1
    return slug


def create_dataset(name: str) -> Dataset:
    slug = _unique_slug(name)
    return Dataset.objects.create(
        name=name,
        slug=slug,
        video_dir=f"data/datasets/{slug}/raw",
        manifest_path=f"data/datasets/{slug}/manifest.csv",
    )


def get_current(request) -> Dataset | None:
    """The dataset currently in scope -- resolved from the session, falling
    back to the alphabetically-first dataset if the session's slug is
    missing or no longer exists, and to None if there are no datasets at all
    yet (the empty-state every dataset-scoped view has to handle)."""
    slug = request.session.get(SESSION_KEY)
    if slug:
        dataset = Dataset.objects.filter(slug=slug).first()
        if dataset is not None:
            return dataset
    return Dataset.objects.order_by("name").first()


def set_current(request, dataset: Dataset) -> None:
    request.session[SESSION_KEY] = dataset.slug


def resolve(request) -> Dataset | None:
    """The dataset this request should operate on: an explicit `dataset`
    slug (`?dataset=` on GET, or a same-named POST field, e.g. a formset
    submission) takes priority and becomes the new sticky selection;
    otherwise falls back to get_current(). Used by every dataset-scoped view
    (dataset_list, label_videos, review_manifest, start_extraction) so
    picking a different dataset from a page's dataset switcher is just a
    normal link/GET-form reload, exactly like the old manifest_path picker."""
    slug = request.GET.get("dataset") or request.POST.get("dataset")
    if slug:
        dataset = Dataset.objects.filter(slug=slug).first()
        if dataset is not None:
            set_current(request, dataset)
            return dataset
    return get_current(request)


def delete_dataset(dataset: Dataset, delete_files: bool = False) -> None:
    """Delete a dataset's row, and optionally its video folder + manifest
    file too -- gated through is_within_repo the same way
    views.dataset.delete_entry gates single-file deletes, so this can never
    reach outside the repo."""
    if delete_files:
        video_dir = resolve_repo_path(dataset.video_dir)
        if is_within_repo(video_dir) and video_dir.is_dir():
            shutil.rmtree(video_dir)
        manifest_path = resolve_repo_path(dataset.manifest_path)
        if is_within_repo(manifest_path):
            manifest_path.unlink(missing_ok=True)
    dataset.delete()
