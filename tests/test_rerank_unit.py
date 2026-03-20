"""Unit tests for rerank.py — index handling edge cases with mocked Anthropic client."""
import json
from unittest.mock import patch, MagicMock
from models import Element, BoundingBox, SearchResult
from rerank import rerank


def _make_result(id, label="Label", score=0.5):
    return SearchResult(
        element=Element(
            id=id,
            rulebook_id="test",
            source_type="core",
            page_number=1,
            page_image_path="/images/test.png",
            type="rule",
            label=label,
            description="A description long enough to be useful in testing.",
            bbox=BoundingBox(x=0.1, y=0.1, w=0.5, h=0.1),
        ),
        score=score,
    )


def _mock_rerank_response(ranked_indices, reasoning="test"):
    """Create a mock Anthropic response returning the given ranked_indices."""
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock()]
    mock_resp.content[0].text = json.dumps({
        "ranked_indices": ranked_indices,
        "reasoning": reasoning,
    })
    return mock_resp


class TestRerankIndexHandling:
    @patch("rerank.client")
    def test_normal_reranking(self, mock_client):
        results = [_make_result("a", score=0.9), _make_result("b", score=0.7), _make_result("c", score=0.5)]
        mock_client.messages.create.return_value = _mock_rerank_response([2, 0, 1])
        reranked = rerank("test query", results)
        assert [r.element.id for r in reranked] == ["c", "a", "b"]

    @patch("rerank.client")
    def test_out_of_range_indices_ignored(self, mock_client):
        results = [_make_result("a"), _make_result("b")]
        mock_client.messages.create.return_value = _mock_rerank_response([1, 99, -1, 0])
        reranked = rerank("test query", results)
        assert [r.element.id for r in reranked] == ["b", "a"]

    @patch("rerank.client")
    def test_duplicate_indices_deduped(self, mock_client):
        results = [_make_result("a"), _make_result("b"), _make_result("c")]
        mock_client.messages.create.return_value = _mock_rerank_response([1, 1, 0, 2, 0])
        reranked = rerank("test query", results)
        assert [r.element.id for r in reranked] == ["b", "a", "c"]

    @patch("rerank.client")
    def test_missing_indices_appended(self, mock_client):
        """Results not mentioned by Claude are appended at the end."""
        results = [_make_result("a"), _make_result("b"), _make_result("c")]
        mock_client.messages.create.return_value = _mock_rerank_response([2])
        reranked = rerank("test query", results)
        assert reranked[0].element.id == "c"
        assert set(r.element.id for r in reranked) == {"a", "b", "c"}

    @patch("rerank.client")
    def test_empty_indices_returns_original_order(self, mock_client):
        results = [_make_result("a"), _make_result("b")]
        mock_client.messages.create.return_value = _mock_rerank_response([])
        reranked = rerank("test query", results)
        assert [r.element.id for r in reranked] == ["a", "b"]

    @patch("rerank.client")
    def test_markdown_fenced_response(self, mock_client):
        """Claude sometimes wraps JSON in ```json ... ``` fences."""
        results = [_make_result("a"), _make_result("b")]
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock()]
        mock_resp.content[0].text = '```json\n{"ranked_indices": [1, 0], "reasoning": "test"}\n```'
        mock_client.messages.create.return_value = mock_resp
        reranked = rerank("test query", results)
        assert [r.element.id for r in reranked] == ["b", "a"]

    def test_single_result_no_api_call(self):
        """Single result should be returned as-is without calling Claude."""
        results = [_make_result("a")]
        reranked = rerank("test query", results)
        assert len(reranked) == 1
        assert reranked[0].element.id == "a"

    def test_empty_results(self):
        assert rerank("test query", []) == []
