"""Unit tests for ingest.py pure functions — no live backend or API calls needed."""
import io
import json
import pytest
from PIL import Image
from models import Element, BoundingBox

# Import the functions under test. These are module-private but we test them
# directly since they contain non-trivial logic.
import ingest as ingest_module

_deduplicate = ingest_module._deduplicate
_split_wide_images = ingest_module._split_wide_images
_find_image_context = ingest_module._find_image_context
_find_text_context = ingest_module._find_text_context
_build_elements_from_chunks = ingest_module._build_elements_from_chunks
_build_context_crop = ingest_module._build_context_crop
split_image = ingest_module.split_image
_resize_for_api = ingest_module._resize_for_api
_extract_icon_legend = ingest_module._extract_icon_legend
_profile_to_style_notes = ingest_module._profile_to_style_notes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _el(id, x, y, w, h, **kw):
    defaults = dict(
        rulebook_id="test", source_type="core", page_number=1,
        page_image_path="/images/test.png", type="rule",
        label="Label", description="Description text that is long enough.",
    )
    defaults.update(kw)
    return Element(id=id, bbox=BoundingBox(x=x, y=y, w=w, h=h), **defaults)


def _box(label, x0, y0, x1, y1, score=0.9):
    return {"label": label, "coordinate": [x0, y0, x1, y1], "score": score}


def _make_png(w=200, h=300):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _deduplicate
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_no_duplicates_kept(self):
        elements = [
            _el("a", 0.1, 0.1, 0.3, 0.05),
            _el("b", 0.1, 0.5, 0.3, 0.05),
        ]
        assert len(_deduplicate(elements)) == 2

    def test_near_duplicate_removed(self):
        """Two elements at nearly the same center should collapse to one."""
        elements = [
            _el("a", 0.10, 0.20, 0.30, 0.05),
            _el("b", 0.11, 0.21, 0.30, 0.05),  # center within 0.02 y, 0.10 x
        ]
        result = _deduplicate(elements)
        assert len(result) == 1
        assert result[0].id == "a"  # first one wins

    def test_different_columns_not_deduped(self):
        """Elements in different columns (far apart in x) are NOT duplicates."""
        elements = [
            _el("a", 0.05, 0.20, 0.30, 0.05),
            _el("b", 0.55, 0.20, 0.30, 0.05),  # same y, different x column
        ]
        assert len(_deduplicate(elements)) == 2

    def test_empty_list(self):
        assert _deduplicate([]) == []

    def test_single_element(self):
        elements = [_el("a", 0.1, 0.1, 0.3, 0.05)]
        assert len(_deduplicate(elements)) == 1


# ---------------------------------------------------------------------------
# _split_wide_images
# ---------------------------------------------------------------------------

class TestSplitWideImages:
    def test_no_split_when_image_not_wide(self):
        """Image box narrower than 1.8× avg text width stays intact."""
        boxes = [
            _box("image", 100, 10, 300, 100),   # 200px wide
            _box("text", 100, 120, 280, 160),    # 180px wide, below
            _box("text", 100, 170, 280, 210),    # another text
        ]
        result = _split_wide_images(boxes)
        image_boxes = [b for b in result if b["label"] == "image"]
        assert len(image_boxes) == 1

    def test_splits_wide_image_into_columns(self):
        """Image 2× wider than avg text columns should be split."""
        boxes = [
            _box("image", 50, 10, 850, 200),     # 800px wide
            _box("text", 50, 220, 250, 280),      # col1: 200px
            _box("text", 450, 220, 650, 280),     # col2: 200px
        ]
        result = _split_wide_images(boxes)
        image_boxes = [b for b in result if b["label"] == "image"]
        assert len(image_boxes) == 2

    def test_text_boxes_preserved(self):
        """Non-image boxes pass through unchanged."""
        boxes = [
            _box("text", 100, 10, 300, 50),
            _box("paragraph_title", 100, 60, 250, 80),
        ]
        result = _split_wide_images(boxes)
        assert len(result) == 2

    def test_no_text_below_keeps_image(self):
        """Image with no text columns below stays intact."""
        boxes = [_box("image", 50, 10, 850, 200)]
        result = _split_wide_images(boxes)
        assert len(result) == 1

    def test_single_text_below_keeps_image(self):
        """Need at least 2 text boxes below to trigger split."""
        boxes = [
            _box("image", 50, 10, 850, 200),
            _box("text", 50, 220, 250, 280),
        ]
        result = _split_wide_images(boxes)
        image_boxes = [b for b in result if b["label"] == "image"]
        assert len(image_boxes) == 1


# ---------------------------------------------------------------------------
# _find_image_context
# ---------------------------------------------------------------------------

class TestFindImageContext:
    def test_finds_image_above_text(self):
        """Text box should find the image directly above it."""
        img_box = _box("image", 100, 10, 400, 100)
        text_box = _box("text", 100, 110, 400, 160)
        result = _find_image_context([img_box, text_box], text_box)
        assert result is img_box

    def test_no_image_above(self):
        text_box = _box("text", 100, 10, 400, 60)
        assert _find_image_context([text_box], text_box) is None

    def test_picks_closest_image(self):
        """When multiple images are above, pick the closest one."""
        far_img = _box("image", 100, 10, 400, 50)
        near_img = _box("image", 100, 60, 400, 100)
        text_box = _box("text", 100, 110, 400, 160)
        result = _find_image_context([far_img, near_img, text_box], text_box)
        assert result is near_img

    def test_image_not_aligned_skipped(self):
        """Image whose x-range doesn't cover text center is ignored."""
        img_box = _box("image", 500, 10, 800, 100)  # far right
        text_box = _box("text", 100, 110, 300, 160)  # center x = 200
        assert _find_image_context([img_box, text_box], text_box) is None


# ---------------------------------------------------------------------------
# _find_text_context
# ---------------------------------------------------------------------------

class TestFindTextContext:
    def test_finds_text_adjacent_right(self):
        """Image with text to its right (side-by-side)."""
        img_box = _box("image", 50, 100, 200, 300)
        text_box = _box("text", 220, 120, 500, 280)  # right of image, overlapping y
        result = _find_text_context([img_box, text_box], img_box)
        assert result is text_box

    def test_finds_text_below(self):
        """Image with text directly below."""
        img_box = _box("image", 100, 10, 400, 100)
        text_box = _box("text", 100, 110, 400, 160)
        result = _find_text_context([img_box, text_box], img_box)
        assert result is text_box

    def test_no_text_nearby(self):
        img_box = _box("image", 100, 10, 400, 100)
        far_text = _box("text", 100, 500, 400, 560)  # too far below
        assert _find_text_context([img_box, far_text], img_box) is None

    def test_ignores_other_images(self):
        """Only non-image boxes are candidates."""
        img_box = _box("image", 100, 10, 400, 100)
        other_img = _box("image", 100, 110, 400, 200)
        assert _find_text_context([img_box, other_img], img_box) is None


# ---------------------------------------------------------------------------
# _build_elements_from_chunks
# ---------------------------------------------------------------------------

class TestBuildElementsFromChunks:
    def test_basic_chunk_to_element(self):
        chunks = ["Q: How many coins? A: Each player starts with 3 coins."]
        classifications = [{"type": "rule", "label": "Setup – Coins", "description": "Each player starts with 3 coins."}]
        elements, last_section = _build_elements_from_chunks(
            chunks, classifications, "test", "faq", 1, ""
        )
        assert len(elements) == 1
        assert elements[0].type == "rule"
        assert elements[0].label == "Setup – Coins"
        assert elements[0].display_mode == "text"
        assert last_section == "Setup"

    def test_skips_other_type(self):
        """Elements classified as 'other' are dropped (nav/footer noise)."""
        chunks = ["Page 1 of 5"]
        classifications = [{"type": "other", "label": "Nav", "description": "Page 1 of 5"}]
        elements, _ = _build_elements_from_chunks(chunks, classifications, "test", "faq", 1, "")
        assert len(elements) == 0

    def test_skips_short_descriptions(self):
        chunks = ["Short"]
        classifications = [{"type": "rule", "label": "X", "description": "Too short"}]
        elements, _ = _build_elements_from_chunks(chunks, classifications, "test", "faq", 1, "")
        assert len(elements) == 0

    def test_invalid_type_mapped_to_other_and_skipped(self):
        chunks = ["Some text content that is reasonably long enough to pass."]
        classifications = [{"type": "invalid_type", "label": "X – Y", "description": "Some text content that is reasonably long enough to pass."}]
        elements, _ = _build_elements_from_chunks(chunks, classifications, "test", "faq", 1, "")
        assert len(elements) == 0  # 'other' gets skipped

    def test_section_propagation(self):
        chunks = ["First chunk long enough to pass the min chars filter.", "Second chunk long enough to pass the min chars filter."]
        classifications = [
            {"type": "rule", "label": "Combat – Attack", "description": "First chunk long enough to pass the min chars filter."},
            {"type": "rule", "label": "Combat – Defense", "description": "Second chunk long enough to pass the min chars filter."},
        ]
        elements, last_section = _build_elements_from_chunks(
            chunks, classifications, "test", "faq", 1, "Setup"
        )
        assert len(elements) == 2
        assert last_section == "Combat"

    def test_preserves_initial_section_when_no_dash(self):
        chunks = ["A description that has enough characters to pass."]
        classifications = [{"type": "rule", "label": "No Dash Here", "description": "A description that has enough characters to pass."}]
        _, last_section = _build_elements_from_chunks(
            chunks, classifications, "test", "faq", 1, "Initial"
        )
        assert last_section == "Initial"

    def test_empty_input(self):
        elements, last_section = _build_elements_from_chunks([], [], "test", "faq", 1, "X")
        assert elements == []
        assert last_section == "X"


# ---------------------------------------------------------------------------
# _build_context_crop
# ---------------------------------------------------------------------------

class TestBuildContextCrop:
    def test_no_prev_returns_current_crop(self):
        img = Image.new("RGB", (500, 500), (255, 255, 255))
        box = _box("text", 10, 100, 200, 150)
        crop = _build_context_crop(img, None, box)
        assert crop.size == (190, 50)

    def test_with_prev_stacks_crops(self):
        img = Image.new("RGB", (500, 500), (255, 255, 255))
        prev = _box("text", 10, 10, 200, 50)
        cur = _box("text", 10, 60, 200, 100)
        crop = _build_context_crop(img, prev, cur)
        # prev height (40) + separator (2) + current height (40) = 82
        assert crop.size[1] == 82
        assert crop.size[0] == 190  # max of both widths


# ---------------------------------------------------------------------------
# split_image
# ---------------------------------------------------------------------------

class TestSplitImage:
    def test_splits_at_x(self):
        png = _make_png(400, 300)
        left, right = split_image(png, 200)
        left_img = Image.open(io.BytesIO(left))
        right_img = Image.open(io.BytesIO(right))
        assert left_img.size == (200, 300)
        assert right_img.size == (200, 300)

    def test_asymmetric_split(self):
        png = _make_png(400, 300)
        left, right = split_image(png, 100)
        left_img = Image.open(io.BytesIO(left))
        right_img = Image.open(io.BytesIO(right))
        assert left_img.size == (100, 300)
        assert right_img.size == (300, 300)


# ---------------------------------------------------------------------------
# _compute_layout_stats
# ---------------------------------------------------------------------------

_compute_layout_stats = ingest_module._compute_layout_stats


class TestComputeLayoutStats:
    def test_defaults_when_no_boxes(self):
        """Empty samples should return default thresholds."""
        stats = _compute_layout_stats([], [])
        assert stats["heading_max_width_pct"] == 0.25
        assert stats["estimated_columns"] == 1
        assert stats["sample_page_count"] == 0

    def test_defaults_when_no_heading_boxes(self):
        """Pages with only text boxes should use default heading threshold."""
        boxes = [_box("text", 100, 50, 500, 100)]
        stats = _compute_layout_stats([boxes], [1000])
        assert stats["heading_max_width_pct"] == 0.25  # default
        assert stats["text_width_median"] == 0.4
        assert stats["estimated_columns"] == 1  # 0.4 is not < 0.4

    def test_narrow_headings(self):
        """Narrow heading boxes → low heading_max_width_pct."""
        boxes = [
            _box("paragraph_title", 50, 10, 150, 40),  # 100px / 1000 = 0.10
            _box("paragraph_title", 50, 100, 180, 130),  # 130px / 1000 = 0.13
            _box("text", 50, 150, 450, 200),  # text at 0.40
        ]
        stats = _compute_layout_stats([boxes], [1000])
        # p90 of [0.10, 0.13] → 0.13, ×1.2 = 0.156
        assert 0.14 < stats["heading_max_width_pct"] < 0.18

    def test_wide_headings_capped(self):
        """Very wide heading boxes should be capped at 0.45."""
        boxes = [
            _box("doc_title", 50, 10, 550, 40),  # 500px / 1000 = 0.50
        ]
        stats = _compute_layout_stats([boxes], [1000])
        # p90 = 0.50, ×1.2 = 0.60, but capped at 0.45
        assert stats["heading_max_width_pct"] == 0.45

    def test_two_column_detection(self):
        """Narrow text boxes should signal 2-column layout."""
        boxes = [
            _box("text", 50, 50, 350, 100),   # 300px / 1000 = 0.30
            _box("text", 550, 50, 850, 100),  # 300px / 1000 = 0.30
        ]
        stats = _compute_layout_stats([boxes], [1000])
        assert stats["estimated_columns"] == 2
        assert stats["text_width_median"] == 0.3

    def test_single_column_detection(self):
        """Wide text boxes should signal 1-column layout."""
        boxes = [
            _box("text", 50, 50, 600, 100),  # 550px / 1000 = 0.55
            _box("text", 50, 150, 650, 200),  # 600px / 1000 = 0.60
        ]
        stats = _compute_layout_stats([boxes], [1000])
        assert stats["estimated_columns"] == 1

    def test_multiple_pages_combined(self):
        """Stats should aggregate across sampled pages."""
        page1_boxes = [
            _box("paragraph_title", 50, 10, 200, 40),  # 150/1000 = 0.15
            _box("text", 50, 50, 350, 100),  # 300/1000 = 0.30
        ]
        page2_boxes = [
            _box("paragraph_title", 50, 10, 170, 40),  # 120/1000 = 0.12
            _box("text", 50, 50, 380, 100),  # 330/1000 = 0.33
        ]
        stats = _compute_layout_stats([page1_boxes, page2_boxes], [1000, 1000])
        assert stats["estimated_columns"] == 2  # median text width ~0.315 < 0.4
        assert stats["sample_page_count"] == 2

    def test_zero_width_page_skipped(self):
        """Pages with zero width should be safely skipped."""
        stats = _compute_layout_stats([[_box("text", 0, 0, 100, 50)]], [0])
        assert stats["heading_max_width_pct"] == 0.25  # default


# ---------------------------------------------------------------------------
# _profile_to_style_notes
# ---------------------------------------------------------------------------

_profile_to_style_notes = ingest_module._profile_to_style_notes


class TestProfileToStyleNotes:
    def test_empty_profile(self):
        assert _profile_to_style_notes({}) == ""

    def test_callout_only(self):
        profile = {"callout_description": "yellow background boxes"}
        result = _profile_to_style_notes(profile)
        assert "yellow background boxes" in result
        assert "Document style note:" in result

    def test_layout_notes_only(self):
        profile = {"layout_notes": "2-column with examples on the right"}
        result = _profile_to_style_notes(profile)
        assert "Layout note:" in result
        assert "2-column" in result

    def test_both_fields(self):
        profile = {
            "callout_description": "bordered panels",
            "layout_notes": "rules in left column",
        }
        result = _profile_to_style_notes(profile)
        assert "bordered panels" in result
        assert "rules in left column" in result
        assert "\n" in result

    def test_empty_strings_ignored(self):
        profile = {"callout_description": "", "layout_notes": ""}
        assert _profile_to_style_notes(profile) == ""


# ---------------------------------------------------------------------------
# _build_document_profile validation (mock Haiku response)
# ---------------------------------------------------------------------------

_build_document_profile = ingest_module._build_document_profile


class TestBuildDocumentProfileValidation:
    """Test the validation/normalization logic inside _build_document_profile.
    We mock the anthropic client to avoid real API calls."""

    def _make_mock_response(self, json_text):
        """Create a mock that patches ingest_module.client.messages.create."""
        import unittest.mock as mock
        resp = mock.MagicMock()
        resp.content = [mock.MagicMock(text=json_text)]
        return resp

    def test_valid_profile(self, monkeypatch):
        profile_json = json.dumps({
            "heading_max_width_pct": 0.18,
            "sub_heading_pattern": "numbered_steps",
            "column_count": 2,
            "has_bold_callouts": True,
            "callout_description": "yellow boxes",
            "layout_notes": "two columns",
        })
        monkeypatch.setattr(
            ingest_module.client.messages, "create",
            lambda **kw: self._make_mock_response(profile_json),
        )
        stats = {"heading_width_p90": 0.15, "heading_max_width_pct": 0.18,
                 "text_width_median": 0.35, "estimated_columns": 2}
        result = _build_document_profile([_make_png()], stats)
        assert result["heading_max_width_pct"] == 0.18
        assert result["sub_heading_pattern"] == "numbered_steps"
        assert result["column_count"] == 2
        assert result["has_bold_callouts"] is True

    def test_invalid_sub_heading_pattern_normalized(self, monkeypatch):
        profile_json = json.dumps({
            "heading_max_width_pct": 0.20,
            "sub_heading_pattern": "fancy_bullets",
            "column_count": 1,
        })
        monkeypatch.setattr(
            ingest_module.client.messages, "create",
            lambda **kw: self._make_mock_response(profile_json),
        )
        result = _build_document_profile([_make_png()], {"heading_width_p90": 0.15})
        assert result["sub_heading_pattern"] == "none"

    def test_heading_width_capped(self, monkeypatch):
        profile_json = json.dumps({
            "heading_max_width_pct": 0.80,
            "sub_heading_pattern": "none",
            "column_count": 1,
        })
        monkeypatch.setattr(
            ingest_module.client.messages, "create",
            lambda **kw: self._make_mock_response(profile_json),
        )
        result = _build_document_profile([_make_png()], {"heading_width_p90": 0.15})
        assert result["heading_max_width_pct"] == 0.45  # capped

    def test_heading_width_floored(self, monkeypatch):
        profile_json = json.dumps({
            "heading_max_width_pct": 0.01,
            "sub_heading_pattern": "none",
            "column_count": 1,
        })
        monkeypatch.setattr(
            ingest_module.client.messages, "create",
            lambda **kw: self._make_mock_response(profile_json),
        )
        result = _build_document_profile([_make_png()], {"heading_width_p90": 0.15})
        assert result["heading_max_width_pct"] == 0.05  # floored

    def test_invalid_json_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            ingest_module.client.messages, "create",
            lambda **kw: self._make_mock_response("not json at all"),
        )
        result = _build_document_profile([_make_png()], {"heading_width_p90": 0.15})
        assert result == {}

    def test_markdown_wrapped_json(self, monkeypatch):
        profile_json = '```json\n{"heading_max_width_pct": 0.20, "sub_heading_pattern": "bold_inline", "column_count": 1}\n```'
        monkeypatch.setattr(
            ingest_module.client.messages, "create",
            lambda **kw: self._make_mock_response(profile_json),
        )
        result = _build_document_profile([_make_png()], {"heading_width_p90": 0.15})
        assert result["sub_heading_pattern"] == "bold_inline"

    def test_invalid_column_count_falls_back(self, monkeypatch):
        profile_json = json.dumps({
            "heading_max_width_pct": 0.20,
            "sub_heading_pattern": "none",
            "column_count": 3,
        })
        monkeypatch.setattr(
            ingest_module.client.messages, "create",
            lambda **kw: self._make_mock_response(profile_json),
        )
        stats = {"heading_width_p90": 0.15, "estimated_columns": 2}
        result = _build_document_profile([_make_png()], stats)
        assert result["column_count"] == 2  # falls back to stats


# ---------------------------------------------------------------------------
# _resize_for_api
# ---------------------------------------------------------------------------

class TestResizeForApi:
    def test_wide_image_resized_down(self):
        """Image wider than max_w gets resized to max_w."""
        png = _make_png(2000, 1000)
        result = _resize_for_api(png, max_w=800)
        img = Image.open(io.BytesIO(result))
        assert img.size[0] == 800
        assert img.size[1] == 400  # aspect ratio preserved

    def test_small_image_unchanged_dimensions(self):
        """Image already smaller than max_w passes through with same dimensions."""
        png = _make_png(500, 300)
        result = _resize_for_api(png, max_w=800)
        img = Image.open(io.BytesIO(result))
        assert img.size == (500, 300)

    def test_exact_max_w_unchanged(self):
        """Image exactly at max_w is not resized."""
        png = _make_png(800, 600)
        result = _resize_for_api(png, max_w=800)
        img = Image.open(io.BytesIO(result))
        assert img.size == (800, 600)

    def test_output_is_valid_png(self):
        """Output should be valid PNG bytes."""
        png = _make_png(1600, 900)
        result = _resize_for_api(png, max_w=1024)
        img = Image.open(io.BytesIO(result))
        assert img.format == "PNG"


# ---------------------------------------------------------------------------
# _extract_icon_legend
# ---------------------------------------------------------------------------

class TestExtractIconLegend:
    @staticmethod
    def _mock_response(text):
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.content = [MagicMock()]
        resp.content[0].text = text
        return resp

    def test_valid_json_dict_returned(self, monkeypatch):
        legend = json.dumps({"[coin]": "gold cost", "[sword]": "attack"})
        monkeypatch.setattr(
            ingest_module.client.messages, "create",
            lambda **kw: self._mock_response(legend),
        )
        result = _extract_icon_legend([_make_png()])
        assert result == {"[coin]": "gold cost", "[sword]": "attack"}

    def test_empty_dict_returned_as_empty(self, monkeypatch):
        monkeypatch.setattr(
            ingest_module.client.messages, "create",
            lambda **kw: self._mock_response("{}"),
        )
        result = _extract_icon_legend([_make_png()])
        assert result == {}

    def test_json_parse_failure_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            ingest_module.client.messages, "create",
            lambda **kw: self._mock_response("not valid json at all"),
        )
        result = _extract_icon_legend([_make_png()])
        assert result == {}

    def test_markdown_fenced_json_unwrapped(self, monkeypatch):
        fenced = '```json\n{"[shield]": "defense"}\n```'
        monkeypatch.setattr(
            ingest_module.client.messages, "create",
            lambda **kw: self._mock_response(fenced),
        )
        result = _extract_icon_legend([_make_png()])
        assert result == {"[shield]": "defense"}

    def test_non_dict_json_returns_empty(self, monkeypatch):
        """JSON that parses but isn't a dict should return {}."""
        monkeypatch.setattr(
            ingest_module.client.messages, "create",
            lambda **kw: self._mock_response('["a", "b"]'),
        )
        result = _extract_icon_legend([_make_png()])
        assert result == {}

    def test_multiple_pages_sent(self, monkeypatch):
        """All page images should be included in the API call."""
        captured_kwargs = {}
        def capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return self._mock_response("{}")
        monkeypatch.setattr(
            ingest_module.client.messages, "create", capture_create,
        )
        pages = [_make_png(200, 300), _make_png(200, 300), _make_png(200, 300)]
        _extract_icon_legend(pages)
        # Should have 3 text + 3 image + 1 prompt = 7 content blocks
        content = captured_kwargs["messages"][0]["content"]
        image_blocks = [c for c in content if c.get("type") == "image"]
        assert len(image_blocks) == 3


# ---------------------------------------------------------------------------
# _profile_to_style_notes
# ---------------------------------------------------------------------------

class TestProfileToStyleNotes:
    def test_with_icon_legend(self):
        profile = {
            "icon_legend": {"[coin]": "gold cost", "[sword]": "attack"},
            "callout_description": "",
            "layout_notes": "",
        }
        notes = _profile_to_style_notes(profile)
        assert "[coin] = gold cost" in notes
        assert "[sword] = attack" in notes
        assert "Use these exact bracket names" in notes

    def test_without_icon_legend(self):
        profile = {
            "callout_description": "",
            "layout_notes": "",
        }
        notes = _profile_to_style_notes(profile)
        assert "icon" not in notes.lower()
        assert "bracket" not in notes.lower()

    def test_empty_icon_legend(self):
        profile = {
            "icon_legend": {},
            "callout_description": "",
            "layout_notes": "",
        }
        notes = _profile_to_style_notes(profile)
        assert "bracket" not in notes.lower()

    def test_with_callout_and_layout(self):
        profile = {
            "callout_description": "yellow background boxes",
            "layout_notes": "rules in left column",
        }
        notes = _profile_to_style_notes(profile)
        assert "yellow background boxes" in notes
        assert "rules in left column" in notes

    def test_icon_legend_combined_with_other_notes(self):
        profile = {
            "icon_legend": {"[star]": "victory points"},
            "callout_description": "blue bordered panels",
            "layout_notes": "",
        }
        notes = _profile_to_style_notes(profile)
        assert "[star] = victory points" in notes
        assert "blue bordered panels" in notes

    def test_empty_profile(self):
        notes = _profile_to_style_notes({})
        assert notes == ""
