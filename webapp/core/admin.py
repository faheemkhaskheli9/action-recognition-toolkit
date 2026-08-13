from django.contrib import admin

from .models import Dataset, TrackExtractionRun, TrainingRun


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ("name", "model_name", "status", "created_at", "finished_at")
    list_filter = ("status", "model_name")


@admin.register(TrackExtractionRun)
class TrackExtractionRunAdmin(admin.ModelAdmin):
    list_display = ("name", "video_dir", "dataset", "status", "created_at", "finished_at")
    list_filter = ("status",)


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "video_dir", "manifest_path", "created_at")
