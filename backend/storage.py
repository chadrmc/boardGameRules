import chromadb
import json
from pathlib import Path
from models import Element, BoundingBox, SearchResult, SourceType

DATA_DIR = Path("./data")
RULEBOOKS_FILE = DATA_DIR / "rulebooks.json"

client = chromadb.PersistentClient(path="./data/chroma")
_collections: dict[str, chromadb.Collection] = {}


def _get_collection(rulebook_id: str) -> chromadb.Collection:
    if rulebook_id not in _collections:
        _collections[rulebook_id] = client.get_or_create_collection(
            name=f"rulebook_{rulebook_id}",
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[rulebook_id]


def _load_rulebooks() -> dict:
    if RULEBOOKS_FILE.exists():
        return json.loads(RULEBOOKS_FILE.read_text())
    return {}


def _save_rulebooks(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RULEBOOKS_FILE.write_text(json.dumps(data, indent=2))


def _get_entry(data: dict, rulebook_id: str) -> dict:
    """Get a rulebook entry, handling legacy format (plain string → dict)."""
    val = data.get(rulebook_id)
    if isinstance(val, str):
        return {"name": val}
    return val or {}


def register_rulebook(rulebook_id: str, game_name: str) -> None:
    data = _load_rulebooks()
    entry = _get_entry(data, rulebook_id)
    entry["name"] = game_name
    data[rulebook_id] = entry
    _save_rulebooks(data)


def save_document_profile(rulebook_id: str, profile: dict) -> None:
    """Store a document profile alongside the rulebook entry."""
    data = _load_rulebooks()
    entry = _get_entry(data, rulebook_id)
    entry["profile"] = profile
    data[rulebook_id] = entry
    _save_rulebooks(data)


def get_document_profile(rulebook_id: str) -> dict | None:
    """Retrieve a stored document profile, or None if not set."""
    data = _load_rulebooks()
    entry = _get_entry(data, rulebook_id)
    return entry.get("profile")


def delete_rulebook_elements(rulebook_id: str, source_type: str | None = None) -> None:
    coll = _get_collection(rulebook_id)
    where = {"source_type": source_type} if source_type else None
    existing = coll.get(where=where, include=[])
    ids = existing["ids"]
    if ids:
        coll.delete(ids=ids)


def add_elements(elements: list[Element]) -> None:
    if not elements:
        return
    coll = _get_collection(elements[0].rulebook_id)
    coll.add(
        ids=[e.id for e in elements],
        documents=[f"{e.label}: {e.description}" for e in elements],
        metadatas=[
            {
                "rulebook_id": e.rulebook_id,
                "source_type": e.source_type,
                "page_number": e.page_number,
                "display_mode": e.display_mode,
                "page_image_path": e.page_image_path,
                "type": e.type,
                "label": e.label,
                "bbox_x": e.bbox.x,
                "bbox_y": e.bbox.y,
                "bbox_w": e.bbox.w,
                "bbox_h": e.bbox.h,
            }
            for e in elements
        ],
    )


def get_page_elements(rulebook_id: str, page_number: int) -> list[Element]:
    coll = _get_collection(rulebook_id)
    results = coll.get(
        where={"page_number": page_number},
        include=["metadatas", "documents"],
    )
    elements = []
    for i, (meta, doc) in enumerate(zip(results["metadatas"], results["documents"])):
        elements.append(Element(
            id=results["ids"][i],
            rulebook_id=meta["rulebook_id"],
            source_type=meta.get("source_type", "core"),
            page_number=meta["page_number"],
            display_mode=meta.get("display_mode", "image"),
            page_image_path=meta["page_image_path"],
            type=meta["type"],
            label=meta["label"],
            description=doc,
            bbox=BoundingBox(
                x=meta["bbox_x"], y=meta["bbox_y"],
                w=meta["bbox_w"], h=meta["bbox_h"],
            ),
        ))
    return elements


def get_page_count(rulebook_id: str) -> int:
    coll = _get_collection(rulebook_id)
    if coll.count() == 0:
        return 0
    results = coll.get(include=["metadatas"])
    return max((m["page_number"] for m in results["metadatas"]), default=0)


def list_rulebooks() -> list[dict]:
    data = _load_rulebooks()
    result = []
    for k, v in sorted(data.items()):
        name = v if isinstance(v, str) else v.get("name", k)
        result.append({"id": k, "name": name})
    return result


def _parse_results(results: dict, n_results: int) -> list[SearchResult]:
    search_results = []
    for i, (meta, doc, dist) in enumerate(
        zip(results["metadatas"][0], results["documents"][0], results["distances"][0])
    ):
        element = Element(
            id=results["ids"][0][i],
            rulebook_id=meta["rulebook_id"],
            source_type=meta.get("source_type", "core"),
            page_number=meta["page_number"],
            display_mode=meta.get("display_mode", "image"),
            page_image_path=meta["page_image_path"],
            type=meta["type"],
            label=meta["label"],
            description=doc,
            bbox=BoundingBox(
                x=meta["bbox_x"], y=meta["bbox_y"],
                w=meta["bbox_w"], h=meta["bbox_h"],
            ),
        )
        search_results.append(SearchResult(element=element, score=1 - dist))
    return search_results


def search_elements(
    query: str, n_results: int = 5, rulebook_id: str = "",
    source_types: list[str] | None = None,
    page_numbers: list[int] | None = None,
) -> list[SearchResult]:
    coll = _get_collection(rulebook_id)
    n = min(n_results, coll.count())
    if n == 0:
        return []
    clauses = []
    if source_types:
        clauses.append({"source_type": {"$in": source_types}})
    if page_numbers:
        clauses.append({"page_number": {"$in": page_numbers}})
    where = {"$and": clauses} if len(clauses) > 1 else clauses[0] if clauses else None
    kwargs: dict = {
        "query_texts": [query],
        "n_results": n,
        "include": ["metadatas", "documents", "distances"],
    }
    if where:
        kwargs["where"] = where
    results = coll.query(**kwargs)
    return _parse_results(results, n)


def get_errata_for_pages(rulebook_id: str, page_numbers: list[int]) -> dict[int, list[Element]]:
    if not page_numbers:
        return {}
    coll = _get_collection(rulebook_id)
    results = coll.get(
        where={"$and": [
            {"source_type": {"$eq": "errata"}},
            {"page_number": {"$in": page_numbers}},
        ]},
        include=["metadatas", "documents"],
    )
    by_page: dict[int, list[Element]] = {}
    for i, (meta, doc) in enumerate(zip(results["metadatas"], results["documents"])):
        elem = Element(
            id=results["ids"][i],
            rulebook_id=meta["rulebook_id"],
            source_type=meta.get("source_type", "errata"),
            page_number=meta["page_number"],
            page_image_path=meta["page_image_path"],
            type=meta["type"],
            label=meta["label"],
            description=doc,
            bbox=BoundingBox(
                x=meta["bbox_x"], y=meta["bbox_y"],
                w=meta["bbox_w"], h=meta["bbox_h"],
            ),
        )
        by_page.setdefault(elem.page_number, []).append(elem)
    return by_page
