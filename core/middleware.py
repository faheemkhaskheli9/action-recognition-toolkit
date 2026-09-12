"""Site-wide login gate.

Uploaded inference video and per-person track data already identify real
people, and the planned camera/re-ID work (see docs/plans/) will add more
of that plus live camera credentials — every page requires an
authenticated session by default now, rather than opting each view in
individually (easy to forget one as new pages are added). The small
allowlist below covers what a not-yet-logged-in browser legitimately needs
to reach: the login page itself, static assets (the login page's own CSS),
and Django's own `/admin/` (which gates itself). Uploaded/inference media
is deliberately *not* exempted — it's requested by `<video>`/`<img>` tags
on already-authenticated pages, where the browser sends the session cookie
automatically, so gating it costs nothing for a logged-in user but stops a
guessed/shared media URL from working for anyone else.

Per-site/per-role scoping (who can see which `Site`) is a separate,
finer-grained layer that lands with the `Site`/`Camera` data model — this
middleware only answers "is anyone logged in at all".
"""
from __future__ import annotations

from django.conf import settings
from django.contrib.auth.views import redirect_to_login
from django.urls import reverse

_EXEMPT_PREFIXES = ("/admin/", "/static/")


class LoginRequiredMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if (
            request.user.is_authenticated
            or request.path.startswith(_EXEMPT_PREFIXES)
            or request.path == reverse(settings.LOGIN_URL)
        ):
            return self.get_response(request)
        return redirect_to_login(request.get_full_path(), login_url=settings.LOGIN_URL)
