from django.contrib.auth import views as auth_views
from django.urls import path

from .views import dataset, extraction, inference, labeling, runs, training

app_name = "core"

urlpatterns = [
    path(
        "login/",
        auth_views.LoginView.as_view(template_name="core/login.html", redirect_authenticated_user=True),
        name="login",
    ),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("", dataset.dataset_list, name="dataset_list"),
    path("dataset/upload/", dataset.upload_videos, name="upload_videos"),
    path("dataset/label/", dataset.update_label, name="update_label"),
    path("dataset/delete/", dataset.delete_entry, name="delete_entry"),
    path("label/", labeling.label_videos, name="label_videos"),
    path("label/save/", labeling.save_label, name="save_label"),
    path("label/skip/", labeling.skip_video, name="skip_video"),
    path("label/video/", labeling.serve_video, name="serve_video"),
    path("label/review/", labeling.review_manifest, name="review_manifest"),
    path("extraction/new/", extraction.start_extraction, name="start_extraction"),
    path("extraction/", extraction.extraction_list, name="extraction_list"),
    path("extraction/<int:pk>/", extraction.extraction_detail, name="extraction_detail"),
    path("extraction/<int:pk>/log/", extraction.extraction_log_partial, name="extraction_log_partial"),
    path("training/new/", training.start_training, name="start_training"),
    path("runs/", runs.run_list, name="run_list"),
    path("runs/<int:pk>/", runs.run_detail, name="run_detail"),
    path("runs/<int:pk>/log/", runs.run_log_partial, name="run_log_partial"),
    path("inference/", inference.inference, name="inference"),
]
