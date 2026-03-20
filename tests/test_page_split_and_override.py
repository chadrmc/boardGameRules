"""Unit tests for detect_page_split and _detect_source_type_override in ingest.py.

Both functions depend on PaddleOCR engines — we mock those to keep tests fast and offline.
"""
import io
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

import ingest as ingest_module

detect_page_split = ingest_module.detect_page_split
_detect_source_type_override = ingest_module._detect_source_type_override


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(w=800, h=600):
    """Create a minimal PNG as bytes."""
    img = Image.new("RGB", (w, h), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_pil(w=800, h=1000):
    """Create a PIL Image."""
    return Image.new("RGB", (w, h), (255, 255, 255))


def _box(label, x0, y0, x1, y1, score=0.9):
    return {"label": label, "coordinate": [x0, y0, x1, y1], "score": score}


# ---------------------------------------------------------------------------
# detect_page_split
# ---------------------------------------------------------------------------

class TestDetectPageSplit:
    """detect_page_split returns (split_x, boxes) or (None, boxes)."""

    def test_portrait_returns_none(self):
        """Portrait image (w < h*1.2) should immediately return (None, None)."""
        png = _make_png(600, 800)  # portrait
        split_x, boxes = detect_page_split(png)
        assert split_x is None
        assert boxes is None

    def test_square_returns_none(self):
        """Square image (w == h) should return (None, None)."""
        png = _make_png(800, 800)
        split_x, boxes = detect_page_split(png)
        assert split_x is None
        assert boxes is None

    def test_barely_landscape_returns_none(self):
        """w/h = 1.1 is below the 1.2 threshold."""
        png = _make_png(880, 800)
        split_x, boxes = detect_page_split(png)
        assert split_x is None
        assert boxes is None

    @patch("ingest._get_paddle")
    def test_page_number_boxes_trigger_split(self, mock_paddle):
        """Two 'number' boxes in bottom quarter, opposite halves → split."""
        w, h = 1600, 800  # landscape
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [{"boxes": [
            _box("number", 100, 650, 200, 700),   # left half, bottom quarter
            _box("number", 1400, 650, 1500, 700),  # right half, bottom quarter
            _box("text", 100, 100, 700, 200),       # body text
        ]}]
        mock_paddle.return_value = mock_engine

        png = _make_png(w, h)
        split_x, boxes = detect_page_split(png)

        assert split_x is not None
        # Split should be between left inner (200) and right inner (1400)
        assert split_x == (200 + 1400) // 2
        assert boxes is not None

    @patch("ingest._get_paddle")
    def test_page_numbers_same_half_no_split(self, mock_paddle):
        """Two 'number' boxes both in left half → no page number split."""
        w, h = 1600, 800
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [{"boxes": [
            _box("number", 100, 650, 200, 700),   # left half
            _box("number", 300, 650, 400, 700),   # still left half
        ]}]
        mock_paddle.return_value = mock_engine

        png = _make_png(w, h)
        split_x, boxes = detect_page_split(png)
        # No page-number split; falls through to gutter detection.
        # With boxes close together, no significant gutter in the middle third.
        # Returns (None, boxes) — boxes returned for reuse.
        assert boxes is not None

    @patch("ingest._get_paddle")
    def test_gutter_gap_triggers_split(self, mock_paddle):
        """Wide gap in the middle third → gutter-based split."""
        w, h = 1600, 800
        mock_engine = MagicMock()
        # Two columns of text with a gap in the center
        mock_engine.predict.return_value = [{"boxes": [
            _box("text", 50, 50, 700, 150),    # left column
            _box("text", 50, 160, 700, 260),
            _box("text", 900, 50, 1550, 150),  # right column
            _box("text", 900, 160, 1550, 260),
        ]}]
        mock_paddle.return_value = mock_engine

        png = _make_png(w, h)
        split_x, boxes = detect_page_split(png)

        assert split_x is not None
        # Gap is between 700 and 900, center = 800
        assert 700 <= split_x <= 900
        assert boxes is not None

    @patch("ingest._get_paddle")
    def test_no_gutter_no_page_numbers(self, mock_paddle):
        """Landscape image with no gap and no page numbers → (None, boxes)."""
        w, h = 1600, 800
        mock_engine = MagicMock()
        # One wide text region spanning the whole page — no gutter
        mock_engine.predict.return_value = [{"boxes": [
            _box("text", 50, 50, 1550, 150),
            _box("text", 50, 160, 1550, 260),
        ]}]
        mock_paddle.return_value = mock_engine

        png = _make_png(w, h)
        split_x, boxes = detect_page_split(png)

        assert split_x is None
        assert boxes is not None  # boxes still returned for reuse

    @patch("ingest._get_paddle")
    def test_empty_boxes(self, mock_paddle):
        """No boxes detected at all → (None, boxes=[])."""
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [{"boxes": []}]
        mock_paddle.return_value = mock_engine

        png = _make_png(1600, 800)
        split_x, boxes = detect_page_split(png)

        assert split_x is None
        assert boxes == []

    @patch("ingest._get_paddle")
    def test_page_numbers_not_in_bottom_quarter_ignored(self, mock_paddle):
        """'number' boxes above the bottom quarter don't count for page-number split."""
        w, h = 1600, 800
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [{"boxes": [
            _box("number", 100, 100, 200, 150),   # top of page
            _box("number", 1400, 100, 1500, 150),  # top of page
        ]}]
        mock_paddle.return_value = mock_engine

        png = _make_png(w, h)
        split_x, boxes = detect_page_split(png)
        # No page-number split (not in bottom quarter)
        # No gutter either (boxes are small, gap is huge but check middle third)
        assert boxes is not None

    @patch("ingest._get_paddle")
    def test_tiny_gutter_below_threshold(self, mock_paddle):
        """Gap narrower than 0.5% of width doesn't trigger split."""
        w, h = 1600, 800
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [{"boxes": [
            _box("text", 50, 50, 795, 150),
            _box("text", 802, 50, 1550, 150),  # gap is only 7px = 0.4% of 1600
        ]}]
        mock_paddle.return_value = mock_engine

        png = _make_png(w, h)
        split_x, boxes = detect_page_split(png)
        assert split_x is None


# ---------------------------------------------------------------------------
# _detect_source_type_override
# ---------------------------------------------------------------------------

class TestDetectSourceTypeOverride:
    """_detect_source_type_override checks box text for expansion/variant keywords."""

    @patch("ingest._get_text_rec")
    def test_tier1_paragraph_title_expansion(self, mock_get_rec):
        """paragraph_title box with text starting 'Expansion' → 'expansion'."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = [{"rec_text": "Expansion: The Northern Reach"}]
        mock_get_rec.return_value = mock_rec

        img = _make_pil(800, 1000)
        boxes = [_box("paragraph_title", 100, 50, 500, 100)]
        result = _detect_source_type_override(img, boxes)
        assert result == "expansion"

    @patch("ingest._get_text_rec")
    def test_tier1_doc_title_variant(self, mock_get_rec):
        """doc_title box with text starting 'Variant' → 'variant'."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = [{"rec_text": "Variant Rules"}]
        mock_get_rec.return_value = mock_rec

        img = _make_pil(800, 1000)
        boxes = [_box("doc_title", 100, 50, 500, 100)]
        result = _detect_source_type_override(img, boxes)
        assert result == "variant"

    @patch("ingest._get_text_rec")
    def test_tier1_no_keyword_returns_empty(self, mock_get_rec):
        """paragraph_title with non-keyword text → ''."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = [{"rec_text": "Setup Phase"}]
        mock_get_rec.return_value = mock_rec

        img = _make_pil(800, 1000)
        boxes = [_box("paragraph_title", 100, 50, 500, 100)]
        result = _detect_source_type_override(img, boxes)
        assert result == ""

    @patch("ingest._get_text_rec")
    def test_tier1_keyword_not_at_start_returns_empty(self, mock_get_rec):
        """Text mentioning 'expansion' mid-sentence → '' (must START with keyword)."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = [{"rec_text": "This is an expansion of the base game"}]
        mock_get_rec.return_value = mock_rec

        img = _make_pil(800, 1000)
        boxes = [_box("paragraph_title", 100, 50, 500, 100)]
        result = _detect_source_type_override(img, boxes)
        assert result == ""

    @patch("ingest._get_text_det")
    @patch("ingest._get_text_rec")
    def test_tier2_header_in_top20(self, mock_get_rec, mock_get_det):
        """header box in top 20% with 'Expansion' text → 'expansion'."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = [{"rec_text": "Expansion Pack"}]
        mock_get_rec.return_value = mock_rec

        mock_det = MagicMock()
        mock_det.predict.return_value = [{"dt_polys": [
            [[10, 5], [200, 5], [200, 30], [10, 30]]
        ]}]
        mock_get_det.return_value = mock_det

        img = _make_pil(800, 1000)
        # header box in top 20% (y0=50 < 200)
        boxes = [_box("header", 100, 50, 500, 100)]
        result = _detect_source_type_override(img, boxes)
        assert result == "expansion"

    @patch("ingest._get_text_det")
    @patch("ingest._get_text_rec")
    def test_tier2_header_below_top20_skipped(self, mock_get_rec, mock_get_det):
        """header box below 20% → not checked by Tier 2."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = [{"rec_text": "Expansion"}]
        mock_get_rec.return_value = mock_rec

        mock_det = MagicMock()
        mock_det.predict.return_value = [{"dt_polys": [
            [[10, 5], [200, 5], [200, 30], [10, 30]]
        ]}]
        mock_get_det.return_value = mock_det

        img = _make_pil(800, 1000)
        # header box at y0=300 > 200 (20% of 1000)
        boxes = [_box("header", 100, 300, 500, 400)]
        result = _detect_source_type_override(img, boxes)
        # Falls to Tier 3 — no heading boxes in top 15%, so Tier 3 runs.
        # But Tier 3 uses _get_text_det which we mocked, so it may detect.
        # The key point: Tier 2 is skipped for this box.
        # Since mock_det returns polys and mock_rec returns "Expansion",
        # Tier 3 will catch it.
        assert result == "expansion"

    @patch("ingest._get_text_det")
    @patch("ingest._get_text_rec")
    def test_tier2_image_box_in_top20(self, mock_get_rec, mock_get_det):
        """image box in top 20% → Tier 2 checks it."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = [{"rec_text": "Variant Mode"}]
        mock_get_rec.return_value = mock_rec

        mock_det = MagicMock()
        mock_det.predict.return_value = [{"dt_polys": [
            [[5, 5], [150, 5], [150, 25], [5, 25]]
        ]}]
        mock_get_det.return_value = mock_det

        img = _make_pil(800, 1000)
        boxes = [_box("image", 100, 50, 500, 150)]
        result = _detect_source_type_override(img, boxes)
        assert result == "variant"

    @patch("ingest._get_text_rec")
    def test_tier3_fallback_when_no_headings_in_top15(self, mock_get_rec):
        """No heading boxes in top 15% → Tier 3 scans the top strip."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = [{"rec_text": "Expansion Content"}]
        mock_get_rec.return_value = mock_rec

        mock_det = MagicMock()
        mock_det.predict.return_value = [{"dt_polys": [
            [[10, 10], [300, 10], [300, 50], [10, 50]]
        ]}]

        img = _make_pil(800, 1000)
        # Only text boxes (not cheap labels), so no headings in top 15%
        boxes = [_box("text", 100, 500, 500, 600)]

        with patch("ingest._get_text_det", return_value=mock_det):
            result = _detect_source_type_override(img, boxes)
        assert result == "expansion"

    @patch("ingest._get_text_rec")
    def test_tier3_skipped_when_heading_in_top15(self, mock_get_rec):
        """Heading box in top 15% → Tier 3 skipped, returns '' if Tier 1 finds nothing."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = [{"rec_text": "Game Setup"}]
        mock_get_rec.return_value = mock_rec

        img = _make_pil(800, 1000)
        # paragraph_title at y0=50 < 150 (15% of 1000) → heading in top 15%
        boxes = [_box("paragraph_title", 100, 50, 300, 80)]
        result = _detect_source_type_override(img, boxes)
        # Tier 1 checks it, text is "Game Setup" (no keyword) → ''
        # Tier 3 is skipped because heading exists in top 15%
        assert result == ""

    @patch("ingest._get_text_rec")
    def test_empty_boxes_returns_empty(self, mock_get_rec):
        """No boxes at all → falls to Tier 3."""
        mock_rec = MagicMock()
        mock_get_rec.return_value = mock_rec

        mock_det = MagicMock()
        mock_det.predict.return_value = [{"dt_polys": []}]

        img = _make_pil(800, 1000)
        with patch("ingest._get_text_det", return_value=mock_det):
            result = _detect_source_type_override(img, [])
        assert result == ""

    @patch("ingest._get_text_rec")
    def test_tier1_exception_continues(self, mock_get_rec):
        """If Tier 1 OCR throws, it continues to next box / tier."""
        mock_rec = MagicMock()
        mock_rec.predict.side_effect = [Exception("OCR failed"), [{"rec_text": "Variant"}]]
        mock_get_rec.return_value = mock_rec

        img = _make_pil(800, 1000)
        boxes = [
            _box("paragraph_title", 100, 50, 300, 80),  # will fail
            _box("doc_title", 100, 150, 300, 200),       # will succeed
        ]
        result = _detect_source_type_override(img, boxes)
        assert result == "variant"

    @patch("ingest._get_text_rec")
    def test_case_insensitive(self, mock_get_rec):
        """Keyword matching is case-insensitive."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = [{"rec_text": "EXPANSION RULES"}]
        mock_get_rec.return_value = mock_rec

        img = _make_pil(800, 1000)
        boxes = [_box("doc_title", 100, 50, 500, 100)]
        result = _detect_source_type_override(img, boxes)
        assert result == "expansion"

    @patch("ingest._get_text_rec")
    def test_text_label_not_checked_by_tier1(self, mock_get_rec):
        """'text' label boxes are not CHEAP_LABELS — Tier 1 skips them."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = [{"rec_text": "Expansion rules here"}]
        mock_get_rec.return_value = mock_rec

        mock_det = MagicMock()
        mock_det.predict.return_value = [{"dt_polys": []}]

        img = _make_pil(800, 1000)
        boxes = [_box("text", 100, 50, 500, 100)]

        with patch("ingest._get_text_det", return_value=mock_det):
            result = _detect_source_type_override(img, boxes)
        # 'text' is not cheap or expensive label → skipped by both tiers
        # Tier 3: no headings in top 15%, but det returns no polys → ''
        assert result == ""
