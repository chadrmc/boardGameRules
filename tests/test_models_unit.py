"""Unit tests for models.py — ResultSet.context and ResultSet.ux formatting."""
from models import Element, BoundingBox, SearchResult, ResultSet


def _make_element(**overrides):
    defaults = dict(
        id="el-1",
        rulebook_id="test",
        source_type="core",
        page_number=1,
        page_image_path="/images/test_page1.png",
        type="rule",
        label="Setup – Starting Resources",
        description="Each player receives 3 coins and 2 cards.",
        bbox=BoundingBox(x=0.1, y=0.2, w=0.8, h=0.1),
    )
    defaults.update(overrides)
    return Element(**defaults)


def _make_result(score=0.85, **el_overrides):
    return SearchResult(element=_make_element(**el_overrides), score=score)


# --- ResultSet.context ---

def test_context_single_result():
    r = _make_result()
    ctx = ResultSet([r]).context
    assert "[1] Setup – Starting Resources (page 1)" in ctx
    assert "Each player receives 3 coins and 2 cards." in ctx


def test_context_multiple_results_numbered():
    r1 = _make_result(id="el-1", label="Setup – Resources", description="Get 3 coins.")
    r2 = _make_result(id="el-2", label="Turn – Actions", description="Choose one action.", page_number=2)
    ctx = ResultSet([r1, r2]).context
    assert "[1] Setup – Resources (page 1)" in ctx
    assert "[2] Turn – Actions (page 2)" in ctx


def test_context_includes_errata():
    r = _make_result()
    errata_el = _make_element(id="er-1", source_type="errata", label="Setup – Correction", description="Actually 4 coins.")
    r.errata = [errata_el]
    ctx = ResultSet([r]).context
    assert "[Errata] Setup – Correction" in ctx
    assert "Actually 4 coins." in ctx


def test_context_includes_faq():
    r = _make_result()
    faq_el = _make_element(id="fq-1", source_type="faq", label="Setup – FAQ", description="Q: How many coins? A: 3.")
    r.faq = [faq_el]
    ctx = ResultSet([r]).context
    assert "[FAQ] Setup – FAQ" in ctx
    assert "Q: How many coins?" in ctx


def test_context_empty_results():
    assert ResultSet([]).context == ""


# --- ResultSet.ux ---

def test_ux_single_result_shape():
    r = _make_result(score=0.9)
    ux = ResultSet([r]).ux
    assert len(ux) == 1
    assert ux[0]["score"] == 0.9
    assert ux[0]["element"]["id"] == "el-1"
    assert ux[0]["element"]["label"] == "Setup – Starting Resources"
    assert ux[0]["element"]["bbox"] == {"x": 0.1, "y": 0.2, "w": 0.8, "h": 0.1}
    assert ux[0]["errata"] == []
    assert ux[0]["faq"] == []


def test_ux_includes_errata_and_faq():
    r = _make_result()
    r.errata = [_make_element(id="er-1", source_type="errata")]
    r.faq = [_make_element(id="fq-1", source_type="faq")]
    ux = ResultSet([r]).ux
    assert len(ux[0]["errata"]) == 1
    assert len(ux[0]["faq"]) == 1
    assert ux[0]["errata"][0]["id"] == "er-1"
    assert ux[0]["faq"][0]["id"] == "fq-1"


def test_ux_empty_results():
    assert ResultSet([]).ux == []
