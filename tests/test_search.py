"""
Search quality tests — run queries against an ingested rulebook and assert
on result structure and basic quality thresholds.

These are intentionally loose: they catch broken retrieval and obvious
regressions without being brittle to classifier label changes.
"""
import pytest
import requests


def _search(backend, rulebook_id, query, n=5):
    resp = requests.get(
        f"{backend}/search",
        params={"q": query, "rulebook_id": rulebook_id, "n": n},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def test_search_returns_results(backend, ingested_rulebook):
    rulebook_id, _ = ingested_rulebook
    body = _search(backend, rulebook_id, "rule", n=5)
    assert len(body["results"]) >= 1


def test_search_result_structure(backend, ingested_rulebook):
    rulebook_id, _ = ingested_rulebook
    body = _search(backend, rulebook_id, "rule", n=3)
    for result in body["results"]:
        assert "element" in result
        assert "score" in result
        el = result["element"]
        for field in ("id", "rulebook_id", "source_type", "page_number", "type", "label", "bbox"):
            assert field in el, f"Missing field: {field}"


def test_search_scores_in_range(backend, ingested_rulebook):
    rulebook_id, _ = ingested_rulebook
    body = _search(backend, rulebook_id, "rule", n=5)
    for result in body["results"]:
        score = result["score"]
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1]"


def test_search_top_result_reasonable_score(backend, ingested_rulebook):
    """Top result for a generic query should score above a very loose threshold."""
    rulebook_id, _ = ingested_rulebook
    body = _search(backend, rulebook_id, "rule", n=5)
    if not body["results"]:
        pytest.skip("No results returned")
    top_score = body["results"][0]["score"]
    assert top_score >= 0.15, f"Top result score too low: {top_score}"


def test_search_query_echoed(backend, ingested_rulebook):
    rulebook_id, _ = ingested_rulebook
    query = "some unique query text"
    body = _search(backend, rulebook_id, query, n=1)
    assert body.get("query") == query


def test_search_results_for_correct_rulebook(backend, ingested_rulebook):
    rulebook_id, _ = ingested_rulebook
    body = _search(backend, rulebook_id, "rule", n=5)
    for result in body["results"]:
        assert result["element"]["rulebook_id"] == rulebook_id
