from pydantic import BaseModel
from typing import Literal

SourceType = Literal["core", "errata", "faq", "expansion", "variant"]

# Higher number = more authoritative. Errata/FAQ supersede core rules.
SOURCE_PRIORITY: dict[str, int] = {
    "errata": 3,
    "faq": 2,
    "expansion": 1,
    "core": 1,
    "variant": 0,
}


class BoundingBox(BaseModel):
    x: float  # 0-1 normalized from left
    y: float  # 0-1 normalized from top
    w: float  # 0-1 normalized width
    h: float  # 0-1 normalized height


class Element(BaseModel):
    id: str
    rulebook_id: str
    source_type: SourceType
    page_number: int
    display_mode: Literal["image", "text"] = "image"
    page_image_path: str
    type: Literal["rule", "note", "illustration", "example", "diagram", "table", "component", "other"]
    label: str
    description: str
    bbox: BoundingBox


class SearchResult(BaseModel):
    element: Element
    score: float
    errata: list[Element] = []
    faq: list[Element] = []


class ResultSet:
    def __init__(self, results: list[SearchResult]):
        self.results = results

    @property
    def ux(self) -> list[dict]:
        return [
            {
                "element": r.element.model_dump(),
                "score": r.score,
                "errata": [e.model_dump() for e in r.errata],
                "faq": [e.model_dump() for e in r.faq],
            }
            for r in self.results
        ]

    @property
    def context(self) -> str:
        parts = []
        for i, r in enumerate(self.results, 1):
            part = f"[{i}] {r.element.label} ({r.element.type}, page {r.element.page_number})\n{r.element.description}"
            for e in r.errata:
                part += f"\n\n  [Errata] {e.label}\n  {e.description}"
            for e in r.faq:
                part += f"\n\n  [FAQ] {e.label}\n  {e.description}"
            parts.append(part)
        return "\n\n".join(parts)
