from django.contrib import admin

from .models import TrackExtractionRun, TrainingRun


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ("name", "model_name", "status", "created_at", "finished_at")
    list_filter = ("status", "model_name")


@admin.register(TrackExtractionRun)
class TrackExtractionRunAdmin(admin.ModelAdmin):
    list_display = ("name", "video_dir", "status", "created_at", "finished_at")
    list_filter = ("status",)
