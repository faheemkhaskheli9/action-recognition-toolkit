import pytest

from core.models import Dataset
from core.services import datasets as datasets_service

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def _clear_migration_seeded_dataset():
    # Migration 0003 seeds a "Default" dataset for real installs upgrading
    # from before Dataset existed; clear it so each test starts clean.
    Dataset.objects.all().delete()


# --------------------------------------------------------------------- #
# create_dataset
# --------------------------------------------------------------------- #

def test_create_dataset_derives_slug_and_default_paths():
    dataset = datasets_service.create_dataset("Restaurant floor")

    assert dataset.slug == "restaurant-floor"
    assert dataset.video_dir == "data/datasets/restaurant-floor/raw"
    assert dataset.manifest_path == "data/datasets/restaurant-floor/manifest.csv"


def test_create_dataset_disambiguates_a_colliding_slug():
    datasets_service.create_dataset("Restaurant floor")
    second = datasets_service.create_dataset("Restaurant floor!")  # slugifies to the same base

    assert second.slug == "restaurant-floor-2"


# --------------------------------------------------------------------- #
# get_current / set_current / resolve
# --------------------------------------------------------------------- #

def test_get_current_is_none_with_no_datasets(rf):
    request = rf.get("/")
    request.session = {}

    assert datasets_service.get_current(request) is None


def test_get_current_falls_back_to_alphabetically_first(rf):
    datasets_service.create_dataset("Zebra")
    a = datasets_service.create_dataset("Alpha")
    request = rf.get("/")
    request.session = {}

    assert datasets_service.get_current(request) == a


def test_get_current_uses_the_sticky_session_selection(rf):
    datasets_service.create_dataset("Alpha")
    b = datasets_service.create_dataset("Beta")
    request = rf.get("/")
    request.session = {datasets_service.SESSION_KEY: b.slug}

    assert datasets_service.get_current(request) == b


def test_get_current_falls_back_when_the_session_slug_no_longer_exists(rf):
    a = datasets_service.create_dataset("Alpha")
    request = rf.get("/")
    request.session = {datasets_service.SESSION_KEY: "long-deleted"}

    assert datasets_service.get_current(request) == a


def test_set_current_stores_the_slug_in_the_session(rf):
    dataset = datasets_service.create_dataset("Alpha")
    request = rf.get("/")
    request.session = {}

    datasets_service.set_current(request, dataset)

    assert request.session[datasets_service.SESSION_KEY] == dataset.slug


def test_resolve_prefers_an_explicit_query_param_and_remembers_it(rf):
    a = datasets_service.create_dataset("Alpha")
    b = datasets_service.create_dataset("Beta")
    request = rf.get("/", {"dataset": b.slug})
    request.session = {datasets_service.SESSION_KEY: a.slug}

    resolved = datasets_service.resolve(request)

    assert resolved == b
    assert request.session[datasets_service.SESSION_KEY] == b.slug


def test_resolve_falls_back_to_get_current_without_a_query_param(rf):
    a = datasets_service.create_dataset("Alpha")
    request = rf.get("/")
    request.session = {datasets_service.SESSION_KEY: a.slug}

    assert datasets_service.resolve(request) == a


# --------------------------------------------------------------------- #
# delete_dataset
# --------------------------------------------------------------------- #

def test_delete_dataset_removes_only_the_row_by_default(settings, tmp_path):
    settings.REPO_ROOT = tmp_path
    dataset = datasets_service.create_dataset("Alpha")
    video_dir = tmp_path / dataset.video_dir
    video_dir.mkdir(parents=True)
    (video_dir / "a.mp4").write_bytes(b"x")

    datasets_service.delete_dataset(dataset)

    assert not Dataset.objects.filter(pk=dataset.pk).exists()
    assert (video_dir / "a.mp4").exists()


def test_delete_dataset_with_delete_files_removes_video_dir_and_manifest(settings, tmp_path):
    settings.REPO_ROOT = tmp_path
    dataset = datasets_service.create_dataset("Alpha")
    video_dir = tmp_path / dataset.video_dir
    video_dir.mkdir(parents=True)
    (video_dir / "a.mp4").write_bytes(b"x")
    manifest_path = tmp_path / dataset.manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("video_path,label\n")

    datasets_service.delete_dataset(dataset, delete_files=True)

    assert not video_dir.exists()
    assert not manifest_path.exists()
