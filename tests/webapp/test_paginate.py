"""Covers core.paginate directly -- the shared Paginator wrapper used by
dataset_list, review_manifest, extraction_list, and run_list."""
from django.test import RequestFactory

from core.paginate import PAGE_SIZE, paginate

rf = RequestFactory()


def test_paginate_defaults_to_page_one():
    request = rf.get("/")
    page = paginate(request, range(5))
    assert page.number == 1
    assert list(page.object_list) == [0, 1, 2, 3, 4]


def test_paginate_reads_page_from_query_string():
    request = rf.get("/", {"page": "2"})
    page = paginate(request, range(PAGE_SIZE + 5))
    assert page.number == 2
    assert len(page.object_list) == 5


def test_paginate_clamps_an_out_of_range_page_to_the_last_one():
    request = rf.get("/", {"page": "999"})
    page = paginate(request, range(3))
    assert page.number == 1  # only one page exists


def test_paginate_clamps_a_non_numeric_page_to_the_first_one():
    request = rf.get("/", {"page": "not-a-number"})
    page = paginate(request, range(3))
    assert page.number == 1


def test_paginate_respects_a_custom_page_size():
    request = rf.get("/")
    page = paginate(request, range(10), page_size=4)
    assert len(page.object_list) == 4
    assert page.paginator.num_pages == 3
