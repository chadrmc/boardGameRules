"""
Structural API tests — verify endpoint contracts without touching ingested data.
All tests require the backend to be running (marked live via the `backend` fixture).
"""
import pytest
import requests


def test_list_rulebooks_shape(backend):
    resp = requests.get(f"{backend}/rulebooks", timeout=5)
    assert resp.status_code == 200
    body = resp.json()
    assert "rulebooks" in body
    assert isinstance(body["rulebooks"], list)
    for rb in body["rulebooks"]:
        assert "id" in rb
        assert "name" in rb


def test_search_missing_rulebook_id_returns_422(backend):
    resp = requests.get(f"{backend}/search", params={"q": "test"}, timeout=5)
    assert resp.status_code == 422


def test_search_missing_query_returns_422(backend):
    resp = requests.get(f"{backend}/search", params={"rulebook_id": "anything"}, timeout=5)
    assert resp.status_code == 422


def test_search_unknown_rulebook_returns_empty(backend):
    """Unknown rulebook ID should return 200 with empty results (not a 500)."""
    resp = requests.get(
        f"{backend}/search",
        params={"q": "anything", "rulebook_id": "does-not-exist-xyz"},
        timeout=5,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert body["results"] == []


def test_search_n_param_respected(backend, ingested_rulebook):
    rulebook_id, _ = ingested_rulebook
    for n in (1, 2, 3):
        resp = requests.get(
            f"{backend}/search",
            params={"q": "rule", "rulebook_id": rulebook_id, "n": n},
            timeout=10,
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) <= n, f"Expected ≤{n} results, got {len(results)}"


def test_elements_endpoint_shape(backend, ingested_rulebook):
    rulebook_id, done = ingested_rulebook
    resp = requests.get(
        f"{backend}/elements",
        params={"rulebook_id": rulebook_id, "page_number": 1},
        timeout=5,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "elements" in body
    assert isinstance(body["elements"], list)


def test_elements_missing_params_returns_422(backend):
    resp = requests.get(f"{backend}/elements", params={"rulebook_id": "x"}, timeout=5)
    assert resp.status_code == 422

    resp = requests.get(f"{backend}/elements", params={"page_number": 1}, timeout=5)
    assert resp.status_code == 422
