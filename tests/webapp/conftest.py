import pytest


@pytest.fixture
def client(client, django_user_model):
    """Every page now requires login (core.middleware.LoginRequiredMiddleware)
    — give the standard pytest-django `client` fixture an authenticated
    session so the existing view tests keep exercising view logic instead of
    all bouncing to the login redirect. Tests that specifically care about
    anonymous/login behavior use `django.test.Client()` directly instead of
    this fixture — see tests/webapp/test_auth.py.
    """
    user = django_user_model.objects.create_user(username="tester", password="pw")
    client.force_login(user)
    return client
