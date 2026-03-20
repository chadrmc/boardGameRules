"""
Ingest pipeline tests — POST a small PDF and verify the SSE stream + stored elements.
These run as part of the session-scoped `ingested_rulebook` fixture; this file
adds assertions on top of what conftest already captured.
"""
import pytest
import requests


def test_ingest_sse_done_event(ingested_rulebook):
    """SSE stream must emit a done event with expected fields."""
    _, done = ingested_rulebook
    assert done.get("done") is True
    assert "rulebook_id" in done
    assert "pages_processed" in done
    assert "elements_found" in done


def test_ingest_pages_processed_positive(ingested_rulebook):
    _, done = ingested_rulebook
    assert done["pages_processed"] >= 1


def test_ingest_elements_found_positive(ingested_rulebook):
    _, done = ingested_rulebook
    assert done["elements_found"] >= 1, "Ingestion produced no elements"


def test_ingest_elements_stored(backend, ingested_rulebook):
    """After ingest, GET /elements for page 1 must return at least one element."""
    rulebook_id, done = ingested_rulebook
    resp = requests.get(
        f"{backend}/elements",
        params={"rulebook_id": rulebook_id, "page_number": 1},
        timeout=5,
    )
    assert resp.status_code == 200
    elements = resp.json()["elements"]
    assert len(elements) >= 1, "No elements stored for page 1 after ingest"


def test_ingest_element_fields(backend, ingested_rulebook):
    """Each stored element must have all required fields with correct types."""
    rulebook_id, _ = ingested_rulebook
    resp = requests.get(
        f"{backend}/elements",
        params={"rulebook_id": rulebook_id, "page_number": 1},
        timeout=5,
    )
    elements = resp.json()["elements"]
    valid_types = {"rule", "note", "illustration", "example", "diagram", "table", "component", "other"}
    valid_sources = {"core", "errata", "faq", "expansion", "variant"}

    for el in elements:
        assert el.get("rulebook_id") == rulebook_id
        assert el.get("type") in valid_types, f"Invalid type: {el.get('type')}"
        assert el.get("source_type") in valid_sources, f"Invalid source_type: {el.get('source_type')}"
        assert isinstance(el.get("page_number"), int)
        assert el.get("label"), "Element has empty label"
        bbox = el.get("bbox", {})
        for key in ("x", "y", "w", "h"):
            assert key in bbox, f"bbox missing {key}"
            val = bbox[key]
            assert 0.0 <= val <= 1.0, f"bbox.{key}={val} out of [0,1]"


def test_ingest_rulebook_appears_in_list(backend, ingested_rulebook):
    rulebook_id, _ = ingested_rulebook
    resp = requests.get(f"{backend}/rulebooks", timeout=5)
    ids = [rb["id"] for rb in resp.json()["rulebooks"]]
    assert rulebook_id in ids, f"{rulebook_id} not in rulebook list"
