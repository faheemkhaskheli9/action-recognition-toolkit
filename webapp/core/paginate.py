"""Shared pagination helper for list pages (Dataset table, Extraction/
Training run lists, Review manifest) -- a thin wrapper around Django's own
Paginator so every list page uses the same page size and `?page=` query
param, and renders through the same `_pager.html` partial.
"""
from __future__ import annotations

from django.core.paginator import Page, Paginator
from django.http import HttpRequest

PAGE_SIZE = 50


def paginate(request: HttpRequest, object_list, *, page_size: int = PAGE_SIZE) -> Page:
    """A Page over object_list for whichever `?page=` the request asks for.

    `Paginator.get_page` clamps an out-of-range or non-numeric value to the
    nearest valid page instead of raising -- this is list-page pagination,
    not URL routing, so a stale/bookmarked `?page=` link should just show
    the closest real page rather than 404ing. Works equally on a queryset
    (sliced with LIMIT/OFFSET at the DB level) or a plain list/range (e.g.
    review_manifest paginates row *positions*, not a queryset).
    """
    paginator = Paginator(object_list, page_size)
    return paginator.get_page(request.GET.get("page"))
