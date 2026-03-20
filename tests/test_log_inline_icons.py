"""
Lands of Galzyr inline icon tests — Scenario 3.

LOD rulebook has inline skill icons (⚔🗡👁💬🧠🌿) and gold coin icons
embedded mid-sentence in rule text. These tests verify that text is NOT
split into multiple elements around an icon — which would produce two
narrow elements at the same y-position with a small horizontal gap.

A split signature:
  Element A: "There are 6 different skills"  x=0.04 w=0.15
  Icon gap:   ← ~0.03 wide (the icon cluster) →
  Element B: "in the game..."                x=0.22 w=0.25

vs a legitimate two-column layout gap, which is >> 0.05.

All assertions are structural — no prior game knowledge used.
"""
import pytest
import requests

LOG_ID = "LOG"

# A gap between adjacent same-line elements this small is a single inline icon.
# Larger gaps (> 0.05) are column separators or icon clusters covered by specific tests.
MAX_ICON_GAP = 0.05
MIN_GAP = 0.005
MAX_SPLIT_WIDTH = 0.35
MIN_Y_OVERLAP = 0.5

# "other" is excluded: section headers and decorative elements are legitimately
# narrow and positioned adjacent to text, causing false split positives.
TEXT_TYPES = {"rule", "note", "example"}


def _y_overlap_fraction(el_a, el_b):
    """Fraction of the shorter element's height that overlaps with the other."""
    a_top, a_bot = el_a["bbox"]["y"], el_a["bbox"]["y"] + el_a["bbox"]["h"]
    b_top, b_bot = el_b["bbox"]["y"], el_b["bbox"]["y"] + el_b["bbox"]["h"]
    overlap = max(0, min(a_bot, b_bot) - max(a_top, b_top))
    shorter_h = min(el_a["bbox"]["h"], el_b["bbox"]["h"])
    return overlap / shorter_h if shorter_h > 0 else 0


def _find_splits(elements):
    """
    Return pairs of elements that look like a split around an inline icon:
    - Both text types
    - Vertically on the same line (y overlap >= MIN_Y_OVERLAP)
    - Horizontally adjacent with a small gap (MIN_GAP < gap < MAX_ICON_GAP)
    - Both narrow (w < MAX_SPLIT_WIDTH)
    """
    text_els = [e for e in elements if e["type"] in TEXT_TYPES]
    splits = []
    for i, a in enumerate(text_els):
        for b in text_els[i + 1:]:
            if _y_overlap_fraction(a, b) < MIN_Y_OVERLAP:
                continue
            if a["bbox"]["x"] > b["bbox"]["x"]:
                a, b = b, a
            gap = b["bbox"]["x"] - (a["bbox"]["x"] + a["bbox"]["w"])
            if MIN_GAP < gap < MAX_ICON_GAP:
                if a["bbox"]["w"] < MAX_SPLIT_WIDTH and b["bbox"]["w"] < MAX_SPLIT_WIDTH:
                    splits.append((a, b, gap))
    return splits


def _get_all_elements(backend):
    """Fetch all elements across all pages of LOG."""
    elements_by_page = {}
    page = 1
    while True:
        resp = requests.get(
            f"{backend}/elements",
            params={"rulebook_id": LOG_ID, "page_number": page},
            timeout=5,
        )
        if resp.status_code != 200:
            break
        els = resp.json()["elements"]
        if not els and page > 3:
            break
        if els:
            elements_by_page[page] = els
        page += 1
    return elements_by_page


@pytest.fixture(scope="module")
def log_all_elements(backend, ingested_log):
    pages = _get_all_elements(backend)
    if not pages:
        pytest.skip("LOG has no elements after ingestion")
    return pages


@pytest.fixture(scope="module")
def log_skill_page(log_all_elements):
    """Find the page containing the skill circle rule, skip if not ingested."""
    for page, elements in log_all_elements.items():
        for el in elements:
            desc = el.get("description", "").lower()
            if "skill" in desc and ("circle" in desc or ("6" in desc and "skill" in desc)):
                return page, elements
    pytest.skip("LOG skills page not yet ingested")


def test_no_icon_splits_on_all_log_pages(log_all_elements):
    """
    No text element should be split around a single inline icon on any LOG page.
    Detected as: two same-line, narrow, horizontally-adjacent elements
    with a gap small enough to be a single icon (< 0.05).
    """
    all_splits = []
    for page, elements in log_all_elements.items():
        for a, b, gap in _find_splits(elements):
            all_splits.append(
                f"  page {page}: gap={gap:.3f} between "
                f"'{a['label'][:40]}' (w={a['bbox']['w']:.3f}) "
                f"and '{b['label'][:40]}' (w={b['bbox']['w']:.3f})"
            )

    assert not all_splits, (
        "Possible icon-split elements found (text broken into fragments around an inline icon):\n"
        + "\n".join(all_splits)
    )


def test_skill_circle_rule_is_single_element(log_skill_page):
    """
    The 'skill circle' rule contains inline skill icons mid-sentence.
    It must be exactly one element — not split into fragments.
    """
    _, elements = log_skill_page
    matches = [
        el for el in elements
        if el["type"] in TEXT_TYPES
        and "skill" in el.get("description", "").lower()
        and ("circle" in el.get("description", "").lower()
             or "6" in el.get("description", ""))
    ]
    if len(matches) == 0:
        pytest.skip("Skill circle element not found in current ingest — skills page may not be ingested yet")
    assert len(matches) == 1, (
        f"Skill circle rule appears as {len(matches)} elements — possible split:\n"
        + "\n".join(f"  {el['label']} (w={el['bbox']['w']:.3f})" for el in matches)
    )
    el = matches[0]
    assert el["bbox"]["w"] >= 0.30, (
        f"Skill circle element too narrow (w={el['bbox']['w']:.3f}) — may be a fragment"
    )
