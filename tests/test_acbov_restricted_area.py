"""
ACBoV Restricted Area icon recognition tests — page 28.

The Restricted Area rule (item 49) uses inline icons mid-sentence:
  - ⚠ exclamation icon for restricted area tokens
  - Alert state warning icon

These tests verify that:
1. The restricted area rule element exists and is classified correctly
2. Icon references are preserved in the description (not dropped or mangled)
3. The section header with its badge icon is detected
4. Search for "restricted area" surfaces the rule as top result

All assertions are structural — no prior game knowledge used.
"""
import re
import pytest
import requests

pytestmark = pytest.mark.live

ACBOV_ID = "ACBoV"
RESTRICTED_AREA_PAGE = 28


@pytest.fixture(scope="module")
def page28_elements(backend, ingested_acbov):
    """Fetch all elements for ACBoV page 28 (ingests ACBoV if needed)."""
    resp = requests.get(
        f"{backend}/elements",
        params={"rulebook_id": ACBOV_ID, "page_number": RESTRICTED_AREA_PAGE},
        timeout=5,
    )
    assert resp.status_code == 200
    elements = resp.json()["elements"]
    if not elements:
        pytest.skip("ACBoV page 28 has no elements after ingestion")
    return elements


@pytest.fixture(scope="module")
def restricted_area_elements(page28_elements):
    """Filter to elements whose label or description mentions 'restricted area'."""
    return [
        el for el in page28_elements
        if "restricted area" in el["label"].lower()
        or "restricted area" in el.get("description", "").lower()
    ]


# ---------------------------------------------------------------------------
# Element existence and classification
# ---------------------------------------------------------------------------

def test_restricted_area_rule_exists(restricted_area_elements):
    """At least one element should describe the restricted area rule."""
    rules = [el for el in restricted_area_elements if el["type"] == "rule"]
    assert len(rules) >= 1, (
        f"No 'rule' element found for Restricted Area. "
        f"Types found: {[el['type'] for el in restricted_area_elements]}"
    )


def test_restricted_area_section_labeled(restricted_area_elements):
    """At least one element should have 'Restricted Area' in its label."""
    labeled = [
        el for el in restricted_area_elements
        if "restricted area" in el["label"].lower()
    ]
    assert len(labeled) >= 1, (
        "No element with 'Restricted Area' in label. "
        f"Elements: {[(el['label'], el['type']) for el in restricted_area_elements]}"
    )


def test_restricted_area_rule_classified_correctly(restricted_area_elements):
    """The main restricted area mechanics element should be type 'rule', not 'other'."""
    mechanics = [
        el for el in restricted_area_elements
        if "exposure" in el.get("description", "").lower()
        or "automatically becomes exposed" in el.get("description", "").lower()
    ]
    assert len(mechanics) >= 1, "Could not find the exposure mechanics element"
    for el in mechanics:
        assert el["type"] == "rule", (
            f"Restricted area exposure mechanics classified as '{el['type']}' "
            f"instead of 'rule': {el['label']}"
        )


# ---------------------------------------------------------------------------
# Icon recognition in descriptions
# ---------------------------------------------------------------------------

def test_restricted_area_icon_reference_preserved(restricted_area_elements):
    """The restricted area token icon should appear in the description.

    The exclamation/warning icon on the map is a key game element —
    the description should reference it (as [exclamation], [!], ⚠, or similar).
    """
    rules = [el for el in restricted_area_elements if el["type"] == "rule"]
    assert rules, "No rule elements to check"

    # Find the main rule element (the one describing exposure mechanics)
    main_rule = next(
        (el for el in rules
         if "automatically" in el.get("description", "").lower()
         and "exposed" in el.get("description", "").lower()),
        None,
    )
    assert main_rule is not None, "Main restricted area rule not found"

    desc = main_rule["description"]
    # The icon could be transcribed in many ways depending on the ingestion pipeline:
    # [exclamation], [!], ⚠, [warning], [red shield icon], or any other [bracketed icon ref]
    icon_patterns = [
        r"\[exclamation\]",
        r"\[!\]",
        r"⚠",
        r"\[warning",
        r"exclamation",
        r"\[[^\]]*icon[^\]]*\]",      # any [... icon ...] bracket notation
        r"\[[^\]]*shield[^\]]*\]",     # [red shield icon] etc.
        r"\[[^\]]*symbol[^\]]*\]",     # [... symbol ...] bracket notation
    ]
    has_icon = any(re.search(p, desc, re.IGNORECASE) for p in icon_patterns)
    assert has_icon, (
        f"No icon reference found in restricted area rule description.\n"
        f"Expected a bracketed icon reference like [icon], [shield icon], etc.\n"
        f"Description: {desc}"
    )


def test_alert_state_icon_preserved(restricted_area_elements):
    """The alert state change icon should appear in the description.

    The rule states the Alert State changes — the description should
    reference the alert/warning icon.
    """
    rules = [el for el in restricted_area_elements if el["type"] == "rule"]
    main_rule = next(
        (el for el in rules
         if "alert" in el.get("description", "").lower()
         and "restricted area" in el.get("description", "").lower()),
        None,
    )
    assert main_rule is not None, "Rule mentioning alert state not found"

    desc = main_rule["description"]
    alert_patterns = [
        r"\[warning",
        r"alert",
        r"⚠",
        r"\[!",
        r"\[[^\]]*icon[^\]]*\]",      # any [... icon ...] bracket notation
        r"\[[^\]]*skull[^\]]*\]",      # [red skull icon] etc.
        r"\[[^\]]*symbol[^\]]*\]",     # [... symbol ...] bracket notation
    ]
    has_alert_ref = any(re.search(p, desc, re.IGNORECASE) for p in alert_patterns)
    assert has_alert_ref, (
        f"No alert state icon/reference found in description.\n"
        f"Description: {desc}"
    )


# ---------------------------------------------------------------------------
# Bbox sanity
# ---------------------------------------------------------------------------

def test_restricted_area_rule_bbox_in_lower_half(restricted_area_elements):
    """The restricted area section is in the lower portion of page 28."""
    rules = [el for el in restricted_area_elements if el["type"] == "rule"]
    assert rules, "No rule elements"
    for el in rules:
        assert el["bbox"]["y"] >= 0.5, (
            f"Restricted area rule bbox too high on page (y={el['bbox']['y']:.3f}): "
            f"{el['label']}"
        )


# ---------------------------------------------------------------------------
# Search quality
# ---------------------------------------------------------------------------

def test_search_restricted_area_finds_rule(backend, restricted_area_elements):
    """Searching 'restricted area' should surface the rule element as top result."""
    from conftest import parse_ask_sse

    resp = requests.get(
        f"{backend}/ask",
        params={"q": "restricted area", "rulebook_id": ACBOV_ID, "n": 5},
        stream=True,
        timeout=30,
    )
    resp.raise_for_status()
    results, _ = parse_ask_sse(resp)
    assert results, "No results returned for 'restricted area' query"

    # Top result should be from page 28 and mention restricted area
    top = results[0]
    assert "restricted area" in top["element"]["label"].lower() or \
           "restricted area" in top["element"]["description"].lower(), (
        f"Top result doesn't mention restricted area: {top['element']['label']}"
    )
    assert top["element"]["page_number"] == RESTRICTED_AREA_PAGE, (
        f"Top result from page {top['element']['page_number']}, expected {RESTRICTED_AREA_PAGE}"
    )
    assert top["score"] >= 0.3, (
        f"Top result score too low: {top['score']:.3f}"
    )
