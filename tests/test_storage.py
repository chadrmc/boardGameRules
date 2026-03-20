"""Unit tests for storage.py — ChromaDB operations and rulebook registry.

Uses a temporary directory for ChromaDB and rulebooks.json to avoid touching real data.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch

import chromadb

from models import Element, BoundingBox


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    """
    Redirect storage.py's ChromaDB client and rulebooks file to a temp dir.
    Also clear the _collections cache between tests.
    """
    import storage

    # Use an ephemeral ChromaDB client with a unique tenant per test
    ephemeral = chromadb.EphemeralClient()
    # Delete any leftover collections from previous test
    for col in ephemeral.list_collections():
        ephemeral.delete_collection(col.name)
    monkeypatch.setattr(storage, "client", ephemeral)
    monkeypatch.setattr(storage, "_collections", {})

    # Point rulebooks file to temp dir
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage, "RULEBOOKS_FILE", tmp_path / "rulebooks.json")

    yield


def _make_element(id="el-1", rulebook_id="test-rb", source_type="core",
                  page_number=1, label="Setup – Coins", **kw):
    defaults = dict(
        page_image_path="/images/test_page1.png",
        type="rule",
        description="Each player receives 3 coins and 2 cards at the start.",
        bbox=BoundingBox(x=0.1, y=0.2, w=0.8, h=0.1),
    )
    defaults.update(kw)
    return Element(
        id=id, rulebook_id=rulebook_id, source_type=source_type,
        page_number=page_number, label=label, **defaults,
    )


# ---------------------------------------------------------------------------
# add_elements + search_elements
# ---------------------------------------------------------------------------

class TestAddAndSearch:
    def test_add_and_search_returns_results(self):
        from storage import add_elements, search_elements

        elements = [
            _make_element(id="e1", label="Setup – Coins",
                          description="Each player starts with 3 gold coins."),
            _make_element(id="e2", label="Turn – Actions",
                          description="On your turn, choose one action to perform."),
        ]
        add_elements(elements)

        results = search_elements("coins", n_results=5, rulebook_id="test-rb")
        assert len(results) >= 1
        assert all(r.element.rulebook_id == "test-rb" for r in results)
        # Scores should be between 0 and 1
        for r in results:
            assert 0.0 <= r.score <= 2.0  # cosine distance can occasionally be > 1

    def test_add_empty_list_is_noop(self):
        from storage import add_elements, search_elements
        add_elements([])
        results = search_elements("anything", n_results=5, rulebook_id="test-rb")
        assert results == []

    def test_search_empty_collection(self):
        from storage import search_elements
        results = search_elements("query", n_results=5, rulebook_id="nonexistent")
        assert results == []

    def test_search_n_results_capped(self):
        from storage import add_elements, search_elements
        elements = [
            _make_element(id=f"e{i}", description=f"Rule number {i} about coins.")
            for i in range(10)
        ]
        add_elements(elements)
        results = search_elements("coins", n_results=3, rulebook_id="test-rb")
        assert len(results) <= 3

    def test_search_with_source_type_filter(self):
        from storage import add_elements, search_elements
        elements = [
            _make_element(id="core1", source_type="core", description="Core rule about movement and actions."),
            _make_element(id="faq1", source_type="faq", description="FAQ about movement and actions."),
        ]
        add_elements(elements)
        results = search_elements(
            "movement", n_results=5, rulebook_id="test-rb",
            source_types=["faq"],
        )
        assert all(r.element.source_type == "faq" for r in results)

    def test_search_with_page_number_filter(self):
        from storage import add_elements, search_elements
        elements = [
            _make_element(id="p1", page_number=1, description="Page 1 rule about setup."),
            _make_element(id="p2", page_number=2, description="Page 2 rule about setup."),
        ]
        add_elements(elements)
        results = search_elements(
            "setup", n_results=5, rulebook_id="test-rb",
            page_numbers=[1],
        )
        assert all(r.element.page_number == 1 for r in results)


# ---------------------------------------------------------------------------
# delete_rulebook_elements
# ---------------------------------------------------------------------------

class TestDeleteElements:
    def test_delete_all_elements(self):
        from storage import add_elements, search_elements, delete_rulebook_elements
        add_elements([_make_element(id="e1", description="A long enough rule description.")])
        delete_rulebook_elements("test-rb")
        results = search_elements("rule", n_results=5, rulebook_id="test-rb")
        assert results == []

    def test_delete_by_source_type_only_removes_matching(self):
        from storage import add_elements, search_elements, delete_rulebook_elements
        add_elements([
            _make_element(id="core1", source_type="core", description="Core rules about scoring points."),
            _make_element(id="faq1", source_type="faq", description="FAQ about scoring points."),
        ])
        delete_rulebook_elements("test-rb", source_type="faq")
        results = search_elements("scoring", n_results=5, rulebook_id="test-rb")
        ids = [r.element.id for r in results]
        assert "core1" in ids
        assert "faq1" not in ids

    def test_delete_empty_collection_no_error(self):
        from storage import delete_rulebook_elements
        # Should not raise
        delete_rulebook_elements("nonexistent")


# ---------------------------------------------------------------------------
# get_page_elements
# ---------------------------------------------------------------------------

class TestGetPageElements:
    def test_returns_elements_for_page(self):
        from storage import add_elements, get_page_elements
        add_elements([
            _make_element(id="p1e1", page_number=1, description="Rule on page 1 about cards."),
            _make_element(id="p2e1", page_number=2, description="Rule on page 2 about tokens."),
            _make_element(id="p1e2", page_number=1, description="Another rule on page 1 about dice."),
        ])
        elems = get_page_elements("test-rb", 1)
        assert len(elems) == 2
        assert all(e.page_number == 1 for e in elems)
        assert {e.id for e in elems} == {"p1e1", "p1e2"}

    def test_empty_page_returns_empty(self):
        from storage import add_elements, get_page_elements
        add_elements([_make_element(id="e1", page_number=1, description="A rule here.")])
        elems = get_page_elements("test-rb", 99)
        assert elems == []

    def test_element_fields_preserved(self):
        from storage import add_elements, get_page_elements
        original = _make_element(
            id="e1", label="Combat – Attack", type="rule",
            source_type="errata", page_number=3,
            description="Roll 2d6 and add your attack modifier.",
            bbox=BoundingBox(x=0.05, y=0.3, w=0.4, h=0.15),
        )
        add_elements([original])
        elems = get_page_elements("test-rb", 3)
        assert len(elems) == 1
        e = elems[0]
        assert e.id == "e1"
        assert e.label == "Combat – Attack"
        assert e.type == "rule"
        assert e.source_type == "errata"
        assert e.page_number == 3
        assert abs(e.bbox.x - 0.05) < 0.001
        assert abs(e.bbox.w - 0.4) < 0.001


# ---------------------------------------------------------------------------
# register_rulebook + list_rulebooks
# ---------------------------------------------------------------------------

class TestRulebookRegistry:
    def test_register_and_list(self):
        from storage import register_rulebook, list_rulebooks
        register_rulebook("rb-1", "Catan")
        register_rulebook("rb-2", "Wingspan")
        result = list_rulebooks()
        assert len(result) == 2
        ids = [r["id"] for r in result]
        assert "rb-1" in ids
        assert "rb-2" in ids

    def test_list_empty(self):
        from storage import list_rulebooks
        assert list_rulebooks() == []

    def test_register_overwrites_name(self):
        from storage import register_rulebook, list_rulebooks
        register_rulebook("rb-1", "Old Name")
        register_rulebook("rb-1", "New Name")
        result = list_rulebooks()
        assert len(result) == 1
        assert result[0]["name"] == "New Name"

    def test_list_sorted_by_id(self):
        from storage import register_rulebook, list_rulebooks
        register_rulebook("z-game", "Zebra")
        register_rulebook("a-game", "Aardvark")
        result = list_rulebooks()
        assert result[0]["id"] == "a-game"
        assert result[1]["id"] == "z-game"

    def test_registry_persists_to_file(self, tmp_path):
        from storage import register_rulebook, RULEBOOKS_FILE
        register_rulebook("rb-1", "Test Game")
        data = json.loads(Path(RULEBOOKS_FILE).read_text())
        assert data["rb-1"]["name"] == "Test Game"

    def test_legacy_string_format_compat(self, tmp_path):
        """Legacy rulebooks.json with plain string values should still work."""
        from storage import RULEBOOKS_FILE, list_rulebooks, register_rulebook
        # Write legacy format directly
        Path(RULEBOOKS_FILE).write_text(json.dumps({"old-rb": "Old Game"}))
        # list_rulebooks should still work
        result = list_rulebooks()
        assert len(result) == 1
        assert result[0]["name"] == "Old Game"
        # Registering again should migrate to new format
        register_rulebook("old-rb", "Old Game Updated")
        data = json.loads(Path(RULEBOOKS_FILE).read_text())
        assert isinstance(data["old-rb"], dict)
        assert data["old-rb"]["name"] == "Old Game Updated"


# ---------------------------------------------------------------------------
# save_document_profile / get_document_profile
# ---------------------------------------------------------------------------

class TestDocumentProfile:
    def test_save_and_get_profile(self):
        from storage import register_rulebook, save_document_profile, get_document_profile
        register_rulebook("rb-1", "Test Game")
        profile = {"heading_max_width_pct": 0.18, "sub_heading_pattern": "none"}
        save_document_profile("rb-1", profile)
        result = get_document_profile("rb-1")
        assert result["heading_max_width_pct"] == 0.18
        assert result["sub_heading_pattern"] == "none"

    def test_get_profile_nonexistent_returns_none(self):
        from storage import get_document_profile
        assert get_document_profile("nonexistent") is None

    def test_profile_survives_name_update(self):
        from storage import register_rulebook, save_document_profile, get_document_profile
        register_rulebook("rb-1", "Name v1")
        save_document_profile("rb-1", {"columns": 2})
        register_rulebook("rb-1", "Name v2")
        result = get_document_profile("rb-1")
        assert result == {"columns": 2}

    def test_profile_on_legacy_entry(self, tmp_path):
        """Saving a profile on a legacy string entry should migrate it."""
        from storage import RULEBOOKS_FILE, save_document_profile, get_document_profile, list_rulebooks
        Path(RULEBOOKS_FILE).write_text(json.dumps({"old-rb": "Legacy Game"}))
        save_document_profile("old-rb", {"heading_max_width_pct": 0.22})
        # Profile should be retrievable
        assert get_document_profile("old-rb")["heading_max_width_pct"] == 0.22
        # Name should be preserved
        result = list_rulebooks()
        assert result[0]["name"] == "Legacy Game"


# ---------------------------------------------------------------------------
# get_errata_for_pages
# ---------------------------------------------------------------------------

class TestGetErrataForPages:
    def test_returns_errata_grouped_by_page(self):
        from storage import add_elements, get_errata_for_pages
        add_elements([
            _make_element(id="er1", source_type="errata", page_number=1,
                          description="Errata: card count should be 4 not 3."),
            _make_element(id="er2", source_type="errata", page_number=2,
                          description="Errata: token color corrected to blue."),
            _make_element(id="core1", source_type="core", page_number=1,
                          description="Core rule about card setup and distribution."),
        ])
        result = get_errata_for_pages("test-rb", [1, 2])
        assert 1 in result
        assert 2 in result
        assert all(e.source_type == "errata" for e in result[1])
        assert all(e.source_type == "errata" for e in result[2])

    def test_empty_pages_returns_empty(self):
        from storage import get_errata_for_pages
        assert get_errata_for_pages("test-rb", []) == {}

    def test_no_errata_returns_empty(self):
        from storage import add_elements, get_errata_for_pages
        add_elements([_make_element(id="c1", source_type="core",
                                    description="Just a core rule about movement.")])
        result = get_errata_for_pages("test-rb", [1])
        assert result == {} or all(len(v) == 0 for v in result.values())
