from __future__ import annotations

from pathlib import Path

from django.conf import settings


def resolve_repo_path(path_str: str) -> Path:
    """Interpret a path relative to the repo root, same as the CLI tools do
    when run from there — mirrors how ar-train/ar-predict resolve --manifest etc."""
    path = Path(path_str)
    return path if path.is_absolute() else settings.REPO_ROOT / path


def is_within_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(settings.REPO_ROOT.resolve())
    except ValueError:
        return False
    return True
