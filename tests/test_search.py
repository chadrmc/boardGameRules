"""
Search quality tests — run queries via GET /ask and assert on the SSE
stream structure and basic quality thresholds.

GET /ask returns:
  - SSE event {"type": "results", "results": [...]} — structured search hits
  - SSE events {"type": "token", "text": "..."} — streaming LLM answer
  - SSE event {"type": "done"}

These are intentionally loose: they catch broken retrieval and obvious
regressions without being brittle to classifier label changes.
"""
import pytest
import requests
from conftest import parse_ask_sse

pytestmark = pytest.mark.live


def _ask(backend, rulebook_id, query, n=5):
    resp = requests.get(
        f"{backend}/ask",
        params={"q": query, "rulebook_id": rulebook_id, "n": n},
        stream=True,
        timeout=30,
    )
    resp.raise_for_status()
    results, answer = parse_ask_sse(resp)
    return results, answer


def test_ask_returns_results(backend, ingested_rulebook):
    rulebook_id, _ = ingested_rulebook
    results, _ = _ask(backend, rulebook_id, "rule", n=5)
    assert results is not None, "No results event in SSE stream"
    assert len(results) >= 1


def test_ask_result_structure(backend, ingested_rulebook):
    rulebook_id, _ = ingested_rulebook
    results, _ = _ask(backend, rulebook_id, "rule", n=3)
    for result in results:
        assert "element" in result
        assert "score" in result
        el = result["element"]
        for field in ("id", "rulebook_id", "source_type", "page_number", "type", "label", "bbox"):
            assert field in el, f"Missing field: {field}"


def test_ask_scores_in_range(backend, ingested_rulebook):
    rulebook_id, _ = ingested_rulebook
    results, _ = _ask(backend, rulebook_id, "rule", n=5)
    for result in results:
        score = result["score"]
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1]"


def test_ask_top_result_reasonable_score(backend, ingested_rulebook):
    """Top result for a generic query should score above a very loose threshold."""
    rulebook_id, _ = ingested_rulebook
    results, _ = _ask(backend, rulebook_id, "rule", n=5)
    if not results:
        pytest.skip("No results returned")
    top_score = results[0]["score"]
    assert top_score >= 0.15, f"Top result score too low: {top_score}"


def test_ask_results_for_correct_rulebook(backend, ingested_rulebook):
    rulebook_id, _ = ingested_rulebook
    results, _ = _ask(backend, rulebook_id, "rule", n=5)
    for result in results:
        assert result["element"]["rulebook_id"] == rulebook_id


def test_ask_streams_answer_text(backend, ingested_rulebook):
    """The SSE stream should produce a non-empty natural language answer."""
    rulebook_id, _ = ingested_rulebook
    _, answer = _ask(backend, rulebook_id, "rule", n=3)
    assert len(answer.strip()) > 10, f"Answer text too short: {repr(answer)}"
