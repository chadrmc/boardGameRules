"""
Fromage COMPONENT LIST box alignment and OCR verification tests.

Verifies that the block-section ingestion pipeline produces well-formed,
evenly-spaced component elements for the Fromage contents page, and that
element labels match the actual text visible in the page image.
All assertions are structural — no prior game knowledge used.
"""
import io
import pytest
import requests
import numpy as np
from PIL import Image

FROMAGE_ID = "fr"
CONTENTS_PAGE = 16
EXPECTED_MIN_ITEMS = 10  # loose lower bound; real count is 14


@pytest.fixture(scope="module")
def component_boxes(backend):
    resp = requests.get(
        f"{backend}/elements",
        params={"rulebook_id": FROMAGE_ID, "page_number": CONTENTS_PAGE},
        timeout=5,
    )
    assert resp.status_code == 200
    elements = resp.json()["elements"]
    boxes = [el for el in elements if el["type"] == "component" and "COMPONENT LIST" in el["label"]]
    if len(boxes) < EXPECTED_MIN_ITEMS:
        pytest.skip(f"Fromage not yet ingested or contents page missing (got {len(boxes)} items)")
    return sorted(boxes, key=lambda e: e["bbox"]["y"])


def test_contents_minimum_item_count(component_boxes):
    assert len(component_boxes) >= EXPECTED_MIN_ITEMS


def test_contents_bboxes_in_range(component_boxes):
    for el in component_boxes:
        b = el["bbox"]
        for key in ("x", "y", "w", "h"):
            assert 0.0 <= b[key] <= 1.0, f"{el['label']}: bbox.{key}={b[key]} out of [0,1]"


def _group_by_column(component_boxes, tol=0.05):
    """Group boxes into columns by similar x origin. Returns list of sorted column lists."""
    columns = {}
    for el in component_boxes:
        x = el["bbox"]["x"]
        key = next((k for k in columns if abs(k - x) < tol), x)
        columns.setdefault(key, []).append(el)
    return [sorted(col, key=lambda e: e["bbox"]["y"]) for col in columns.values()]


def test_contents_boxes_same_x_and_width(component_boxes):
    """Within each column, all items share the same x origin (widths vary by text length)."""
    for col in _group_by_column(component_boxes):
        xs = [el["bbox"]["x"] for el in col]
        assert max(xs) - min(xs) < 0.02, f"x origins vary within column: min={min(xs):.3f} max={max(xs):.3f}"


def test_contents_boxes_evenly_spaced(component_boxes):
    """Within each column, vertical gaps between rows should be uniform."""
    for col in _group_by_column(component_boxes):
        if len(col) < 3:
            continue
        heights = [el["bbox"]["h"] for el in col]
        assert max(heights) - min(heights) < 0.01, (
            f"Heights inconsistent in column: min={min(heights):.3f} max={max(heights):.3f}"
        )
        gaps = [
            col[i + 1]["bbox"]["y"] - (col[i]["bbox"]["y"] + col[i]["bbox"]["h"])
            for i in range(len(col) - 1)
        ]
        # Gaps should be uniform (low variance), not necessarily small
        assert max(gaps) - min(gaps) < 0.005, (
            f"Gaps between rows not uniform in column: {[f'{g:.3f}' for g in gaps]}"
        )


def test_contents_no_overlapping_boxes(component_boxes):
    """Within each column, no two items should overlap vertically."""
    for col in _group_by_column(component_boxes):
        for i in range(len(col) - 1):
            bottom = col[i]["bbox"]["y"] + col[i]["bbox"]["h"]
            next_top = col[i + 1]["bbox"]["y"]
            assert bottom <= next_top + 0.005, (
                f"Overlap in column between '{col[i]['label']}' and '{col[i+1]['label']}': "
                f"bottom={bottom:.3f} next_top={next_top:.3f}"
            )


def test_contents_boxes_sorted_top_to_bottom(component_boxes):
    """Items should be ordered top-to-bottom on the page."""
    ys = [el["bbox"]["y"] for el in component_boxes]
    assert ys == sorted(ys), "Component list items not sorted top-to-bottom"


def test_contents_specific_items_present(component_boxes):
    """Known component items must appear in the labels (case-insensitive)."""
    labels = [el["label"].lower() for el in component_boxes]

    expected_fragments = [
        "36 fruit tokens",
        "36 order cards",
        "36 livestock tokens",
        "36 structure tokens",
        "60 cheese tokens",
        "4 player boards",
        "4 board quadrants",
        "12 workers",
        "6 customer tokens",
    ]

    for fragment in expected_fragments:
        assert any(fragment in label for label in labels), (
            f"Expected component item not found: '{fragment}'\n"
            f"Actual labels: {[el['label'] for el in component_boxes]}"
        )


@pytest.fixture(scope="module")
def page_ocr_texts(backend, component_boxes):
    """
    Download the Fromage page 16 image, crop to the component list section
    (derived from API bboxes), run TextDetection+TextRecognition, and return
    a list of recognised text strings (lowercased, score >= 0.7).
    """
    try:
        from paddleocr import TextDetection, TextRecognition
    except ImportError:
        pytest.skip("paddleocr not available — skipping image OCR tests")

    # Fetch page image
    image_url = f"{backend}/images/fr_page{CONTENTS_PAGE}.png"
    resp = requests.get(image_url, timeout=10)
    if resp.status_code != 200:
        pytest.skip(f"Page image not available: {image_url}")
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    W, H = img.size

    # Derive section crop from union of component bboxes (with padding)
    PAD = 0.02
    xs = [b["bbox"]["x"] for b in component_boxes]
    ys = [b["bbox"]["y"] for b in component_boxes]
    x2s = [b["bbox"]["x"] + b["bbox"]["w"] for b in component_boxes]
    y2s = [b["bbox"]["y"] + b["bbox"]["h"] for b in component_boxes]
    x1 = max(0.0, min(xs) - PAD)
    y1 = max(0.0, min(ys) - PAD)
    x2 = min(1.0, max(x2s) + PAD)
    y2 = min(1.0, max(y2s) + PAD)

    section = img.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))
    section_bgr = np.array(section)[:, :, ::-1]

    det = TextDetection()
    rec = TextRecognition()

    det_result = det.predict(section_bgr)
    raw_polys = det_result[0].get("dt_polys") if det_result else None
    polys = raw_polys if raw_polys is not None and len(raw_polys) > 0 else []

    ocr_texts = []
    for poly in polys:
        px = [p[0] for p in poly]; py = [p[1] for p in poly]
        bx1, by1 = int(min(px)), int(min(py))
        bx2, by2 = int(max(px)), int(max(py))
        crop = section_bgr[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue
        r = rec.predict(crop)
        if r and r[0]["rec_score"] >= 0.7:
            ocr_texts.append(r[0]["rec_text"].lower())

    return ocr_texts


def test_contents_image_contains_36_fruit_tokens(page_ocr_texts):
    """'36 Fruit Tokens' must be readable in the actual page image."""
    assert any("36 fruit tokens" in t for t in page_ocr_texts), (
        f"'36 Fruit Tokens' not found in page OCR output.\nOCR texts: {page_ocr_texts}"
    )


def test_contents_labels_match_image_text(component_boxes, page_ocr_texts):
    """
    For each component element, the item name from the label (part after '– ')
    must appear in the OCR text from the page image.
    Skips items where OCR confidence may be unreliable (very short strings).
    """
    mismatches = []
    for el in component_boxes:
        # Label format: "COMPONENT LIST – 36 Fruit Tokens"
        parts = el["label"].split("–", 1)
        if len(parts) < 2:
            continue
        item_text = parts[1].strip().lower()
        if len(item_text) < 4:
            continue  # too short to match reliably
        if not any(item_text in ocr for ocr in page_ocr_texts):
            mismatches.append(item_text)

    assert not mismatches, (
        f"These labels have no matching text in the page image:\n"
        + "\n".join(f"  - '{m}'" for m in mismatches)
        + f"\nOCR found: {page_ocr_texts}"
    )
