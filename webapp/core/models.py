from django.db import models


class Dataset(models.Model):
    """A named collection of videos with its own video folder and manifest
    CSV -- the grouping unit the Dataset/Label/Review pages operate within,
    and what extraction runs against (either every video under `video_dir`,
    or one video from it). Replaces the app's old scheme of one fixed
    `data/raw` folder shared by everything, with only the manifest CSV
    varying per session (see migration 0003_dataset, which preserves that
    old scheme's paths as one Dataset named "Default" so existing installs
    keep working)."""

    name = models.CharField(max_length=200, unique=True)
    slug = models.SlugField(max_length=100, unique=True)
    video_dir = models.CharField(max_length=500)
    manifest_path = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class TrackExtractionRun(models.Model):
    """One `ar-extract-tracks` invocation, launched as a background OS
    process — same liveness-tracking trade-offs as TrainingRun below (a
    process can die without updating this row, so status/pid are re-checked
    against the OS on each read; see services.extraction.refresh_status).
    """

    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    name = models.CharField(max_length=255)
    config_path = models.CharField(max_length=500)
    # The path actually handed to ar-extract-tracks -- either `dataset`'s own
    # video_dir (a whole-dataset run) or a single video file path within it
    # (a single-video run). See services.extraction.start_run.
    video_dir = models.CharField(max_length=500)
    dataset = models.ForeignKey(
        Dataset, null=True, blank=True, on_delete=models.SET_NULL, related_name="extraction_runs"
    )
    output_dir = models.CharField(max_length=500)
    log_file = models.CharField(max_length=500)

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.RUNNING)
    pid = models.IntegerField(null=True, blank=True)
    return_code = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name


class TrainingRun(models.Model):
    """One `ar-train` invocation, launched as a background OS process.

    Liveness isn't stored as a column — a process can die without updating
    the row (kill -9, crash, host restart), so `status`/`pid` are treated as
    last-known values and re-checked against the OS on each read
    (see services.training.refresh_status).
    """

    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    name = models.CharField(max_length=255)
    model_name = models.CharField(max_length=100)
    config_path = models.CharField(max_length=500)
    manifest_path = models.CharField(max_length=500)
    output_dir = models.CharField(max_length=500)
    log_file = models.CharField(max_length=500)

    status = models.CharField(max_length=20, choices=Status.choices, default=Status.RUNNING)
    pid = models.IntegerField(null=True, blank=True)
    return_code = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.model_name})"
