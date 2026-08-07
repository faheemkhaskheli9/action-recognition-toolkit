from django.db import models


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

    name = models.CharField(max_length=255)
    config_path = models.CharField(max_length=500)
    video_dir = models.CharField(max_length=500)
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
