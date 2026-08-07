from pathlib import Path

from core.paths import is_within_repo, resolve_repo_path


def test_resolve_repo_path_leaves_absolute_paths_alone(settings, tmp_path):
    settings.REPO_ROOT = tmp_path
    absolute = tmp_path.parent / "elsewhere" / "video.mp4"

    assert resolve_repo_path(str(absolute)) == absolute


def test_resolve_repo_path_joins_relative_paths_to_repo_root(settings, tmp_path):
    settings.REPO_ROOT = tmp_path

    assert resolve_repo_path("data/manifest.csv") == tmp_path / "data" / "manifest.csv"


def test_is_within_repo_true_for_paths_under_repo_root(settings, tmp_path):
    settings.REPO_ROOT = tmp_path
    inside = tmp_path / "data" / "clip.mp4"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"x")

    assert is_within_repo(inside) is True


def test_is_within_repo_false_for_paths_escaping_repo_root(settings, tmp_path):
    settings.REPO_ROOT = tmp_path / "repo"
    settings.REPO_ROOT.mkdir()
    outside = tmp_path / "outside" / "secret.txt"
    outside.parent.mkdir()
    outside.write_bytes(b"x")

    assert is_within_repo(outside) is False


def test_is_within_repo_blocks_dot_dot_traversal(settings, tmp_path):
    settings.REPO_ROOT = tmp_path / "repo"
    settings.REPO_ROOT.mkdir()
    (settings.REPO_ROOT / "data").mkdir()
    traversal = settings.REPO_ROOT / "data" / ".." / ".." / "etc" / "passwd"

    assert is_within_repo(traversal) is False
