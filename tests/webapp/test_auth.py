import pytest
from django.test import Client
from django.urls import reverse

pytestmark = pytest.mark.django_db


def test_anonymous_request_redirects_to_login():
    resp = Client().get(reverse("core:dataset_list"))

    assert resp.status_code == 302
    assert reverse("core:login") in resp.url


def test_login_page_reachable_without_auth():
    resp = Client().get(reverse("core:login"))

    assert resp.status_code == 200


def test_login_with_valid_credentials_grants_access(django_user_model):
    django_user_model.objects.create_user(username="alice", password="s3cret-pw")

    anon = Client()
    resp = anon.post(reverse("core:login"), {"username": "alice", "password": "s3cret-pw"})
    assert resp.status_code == 302  # redirected to LOGIN_REDIRECT_URL

    resp = anon.get(reverse("core:dataset_list"))
    assert resp.status_code == 200


def test_login_with_bad_credentials_is_rejected(django_user_model):
    django_user_model.objects.create_user(username="alice", password="s3cret-pw")

    resp = Client().post(reverse("core:login"), {"username": "alice", "password": "wrong"})

    assert resp.status_code == 200  # re-renders the form, no redirect
    assert resp.context["form"].errors

    resp = Client().get(reverse("core:dataset_list"))
    assert resp.status_code == 302  # still anonymous


def test_authenticated_client_fixture_reaches_a_page(client):
    resp = client.get(reverse("core:dataset_list"))

    assert resp.status_code == 200


def test_logout_requires_login_again(client):
    resp = client.post(reverse("core:logout"))
    assert resp.status_code in (200, 302)

    resp = client.get(reverse("core:dataset_list"))
    assert resp.status_code == 302


def test_admin_is_exempt_from_our_gate_but_still_gated_by_its_own():
    """`/admin/` is in LoginRequiredMiddleware's `_EXEMPT_PREFIXES` (it gates
    itself), not "open to anyone" — an anonymous request must still bounce
    to *some* login, just Django admin's own rather than ours. Guards
    against the exemption regressing into an actually-unauthenticated
    `/admin/` if the prefix list is ever broadened carelessly.
    """
    resp = Client().get("/admin/")

    assert resp.status_code == 302
    assert "/admin/login" in resp.url
