"""
Structural API tests — verify endpoint contracts without touching ingested data.
All tests require the backend to be running.
"""
import pytest
import requests

pytestmark = pytest.mark.live


def test_list_rulebooks_shape(backend):
    resp = requests.get(f"{backend}/rulebooks", timeout=5)
    assert resp.status_code == 200
    body = resp.json()
    assert "rulebooks" in body
    assert isinstance(body["rulebooks"], list)
    for rb in body["rulebooks"]:
        assert "id" in rb
        assert "name" in rb


def test_ask_missing_rulebook_id_returns_422(backend):
    resp = requests.get(f"{backend}/ask", params={"q": "test"}, timeout=5)
    assert resp.status_code == 422


def test_ask_missing_query_returns_422(backend):
    resp = requests.get(f"{backend}/ask", params={"rulebook_id": "anything"}, timeout=5)
    assert resp.status_code == 422


def test_ask_unknown_rulebook_returns_empty_results(backend):
    """Unknown rulebook ID should return 200 SSE stream with empty results event (not a 500)."""
    from conftest import parse_ask_sse
    resp = requests.get(
        f"{backend}/ask",
        params={"q": "anything", "rulebook_id": "does-not-exist-xyz"},
        stream=True,
        timeout=30,
    )
    assert resp.status_code == 200
    results, _ = parse_ask_sse(resp)
    assert results is not None, "No results event in SSE stream"
    assert results == [], f"Expected empty results for unknown rulebook, got {results}"


def test_ask_n_param_respected(backend, ingested_rulebook):
    from conftest import parse_ask_sse
    rulebook_id, _ = ingested_rulebook
    for n in (1, 2, 3):
        resp = requests.get(
            f"{backend}/ask",
            params={"q": "rule", "rulebook_id": rulebook_id, "n": n},
            stream=True,
            timeout=30,
        )
        assert resp.status_code == 200
        results, _ = parse_ask_sse(resp)
        assert results is not None
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
