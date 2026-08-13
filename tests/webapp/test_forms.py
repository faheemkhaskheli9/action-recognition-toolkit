import pytest

from core.forms import InferenceForm, ManifestFormSet, ManifestRowForm, StartExtractionForm, StartTrainingForm
from core.models import Dataset


@pytest.mark.django_db
def test_start_extraction_form_valid_with_just_a_dataset(settings, tmp_path):
    settings.CONFIGS_DIR = tmp_path
    dataset = Dataset.objects.create(
        name="Restaurant", slug="restaurant", video_dir="data/raw", manifest_path="data/manifest.csv"
    )

    form = StartExtractionForm(data={"name": "scene1", "dataset": dataset.slug, "config": ""})

    assert form.is_valid(), form.errors
    assert form.cleaned_data["dataset"] == dataset
    assert form.cleaned_data["video"] == ""


@pytest.mark.django_db
def test_start_extraction_form_valid_with_a_single_video(settings, tmp_path):
    settings.CONFIGS_DIR = tmp_path
    dataset = Dataset.objects.create(
        name="Restaurant", slug="restaurant", video_dir="data/raw", manifest_path="data/manifest.csv"
    )

    form = StartExtractionForm(
        data={"name": "scene1", "dataset": dataset.slug, "video": "data/raw/a.mp4", "config": ""}
    )

    assert form.is_valid(), form.errors
    assert form.cleaned_data["video"] == "data/raw/a.mp4"


@pytest.mark.django_db
def test_start_extraction_form_rejects_an_unknown_dataset_slug(settings, tmp_path):
    settings.CONFIGS_DIR = tmp_path

    form = StartExtractionForm(data={"name": "scene1", "dataset": "does-not-exist", "config": ""})

    assert not form.is_valid()
    assert "dataset" in form.errors


def test_start_training_form_lists_available_configs_as_choices(settings, tmp_path):
    settings.CONFIGS_DIR = tmp_path
    (tmp_path / "default.yaml").write_text("{}")
    (tmp_path / "r3d18.yaml").write_text("{}")

    form = StartTrainingForm()

    values = {value for value, _label in form.fields["config"].choices}
    assert str(tmp_path / "default.yaml") in values
    assert str(tmp_path / "r3d18.yaml") in values


def test_start_training_form_valid_with_only_required_fields(settings, tmp_path):
    settings.CONFIGS_DIR = tmp_path
    config_path = tmp_path / "default.yaml"
    config_path.write_text("{}")

    form = StartTrainingForm(data={"name": "my-run", "config": str(config_path)})

    assert form.is_valid(), form.errors


def test_start_training_form_rejects_non_slug_name(settings, tmp_path):
    settings.CONFIGS_DIR = tmp_path
    config_path = tmp_path / "default.yaml"
    config_path.write_text("{}")

    form = StartTrainingForm(data={"name": "not a slug!", "config": str(config_path)})

    assert not form.is_valid()
    assert "name" in form.errors


def test_start_training_form_rejects_config_not_in_available_choices(settings, tmp_path):
    settings.CONFIGS_DIR = tmp_path
    (tmp_path / "default.yaml").write_text("{}")

    form = StartTrainingForm(data={"name": "my-run", "config": "/not/a/real/config.yaml"})

    assert not form.is_valid()
    assert "config" in form.errors


def test_manifest_row_form_split_defaults_to_blank_choice():
    form = ManifestRowForm(data={"video_path": "/a.mp4", "label": "jump"})

    assert form.is_valid(), form.errors
    assert form.cleaned_data["split"] == ""


def test_manifest_row_form_rejects_unknown_split_value():
    form = ManifestRowForm(data={"video_path": "/a.mp4", "label": "jump", "split": "bogus"})

    assert not form.is_valid()
    assert "split" in form.errors


def test_manifest_formset_validates_multiple_rows():
    data = {
        "form-TOTAL_FORMS": "2",
        "form-INITIAL_FORMS": "0",
        "form-0-video_path": "/a.mp4",
        "form-0-label": "jump",
        "form-0-split": "train",
        "form-1-video_path": "/b.mp4",
        "form-1-label": "wave",
        "form-1-split": "",
    }

    formset = ManifestFormSet(data)

    assert formset.is_valid(), formset.errors


def test_inference_form_uses_supplied_checkpoint_choices():
    form = InferenceForm(checkpoint_choices=[("best.pt", "exp1 / best.pt")])

    assert form.fields["checkpoint"].choices == [("best.pt", "exp1 / best.pt")]


def test_inference_form_requires_an_uploaded_or_existing_video():
    form = InferenceForm(
        data={"checkpoint": "best.pt", "top_k": "3"},
        checkpoint_choices=[("best.pt", "exp1 / best.pt")],
    )

    assert not form.is_valid()
    assert form.non_field_errors()


def test_inference_form_valid_with_only_an_existing_video():
    form = InferenceForm(
        data={"checkpoint": "best.pt", "top_k": "3", "existing_video": "data/raw/a.mp4"},
        checkpoint_choices=[("best.pt", "exp1 / best.pt")],
    )

    assert form.is_valid(), form.errors


def test_inference_form_top_k_bounds_are_enforced():
    from django.core.files.uploadedfile import SimpleUploadedFile

    upload = SimpleUploadedFile("clip.mp4", b"x")
    form = InferenceForm(
        data={"checkpoint": "best.pt", "top_k": "20"},
        files={"video": upload},
        checkpoint_choices=[("best.pt", "exp1 / best.pt")],
    )

    assert not form.is_valid()
    assert "top_k" in form.errors
