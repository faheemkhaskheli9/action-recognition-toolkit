from __future__ import annotations

from django import forms

from .services import extraction as extraction_service
from .services import training as training_service


class StartExtractionForm(forms.Form):
    name = forms.SlugField(
        max_length=100,
        help_text="Used as the output folder name under data/tracks/ — letters, numbers, hyphens.",
    )
    video_dir = forms.CharField(
        initial="data/raw_scenes",
        help_text="Folder of raw multi-person videos to detect+track (searched recursively).",
    )
    config = forms.ChoiceField(
        required=False, help_text="Tracking config from configs/tracking/; defaults to default.yaml"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        configs = extraction_service.available_configs()
        self.fields["config"].choices = [("", "(default)")] + [(str(p), p.name) for p in configs]


class StartTrainingForm(forms.Form):
    name = forms.SlugField(
        max_length=100,
        help_text="Used as the run folder name under runs/ — letters, numbers, hyphens.",
    )
    config = forms.ChoiceField(help_text="Base config from configs/; fields below override it.")
    manifest_path = forms.CharField(
        required=False, initial="data/manifest.csv", help_text="Overrides data.manifest"
    )
    model_name = forms.CharField(
        required=False, help_text="Overrides model.name, e.g. cnn_lstm or r3d18"
    )
    epochs = forms.IntegerField(required=False, min_value=1)
    batch_size = forms.IntegerField(required=False, min_value=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        configs = training_service.available_configs()
        self.fields["config"].choices = [(str(p), p.name) for p in configs]


class ManifestRowForm(forms.Form):
    video_path = forms.CharField(widget=forms.TextInput(attrs={"readonly": "readonly"}))
    label = forms.CharField(max_length=200)
    split = forms.ChoiceField(
        choices=[("", "—"), ("train", "train"), ("val", "val"), ("test", "test")],
        required=False,
    )


ManifestFormSet = forms.formset_factory(ManifestRowForm, extra=0, can_delete=True)


class InferenceForm(forms.Form):
    checkpoint = forms.ChoiceField()
    video = forms.FileField()
    top_k = forms.IntegerField(initial=3, min_value=1, max_value=10)
    scene_mode = forms.BooleanField(
        required=False,
        help_text="Video has multiple people — detect+track each one and classify them separately.",
    )

    def __init__(self, *args, checkpoint_choices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["checkpoint"].choices = checkpoint_choices or []
