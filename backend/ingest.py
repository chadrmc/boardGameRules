import anthropic
import base64
import io
import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from models import Element, BoundingBox, SourceType

logger = logging.getLogger(__name__)

client = anthropic.Anthropic()

MIN_REGION_W_PX = 80  # skip regions narrower than this — catches single-word fragments
MIN_REGION_H_PX = 20  # skip regions shorter than this
MIN_DESCRIPTION_CHARS = 20  # skip elements whose description is too short to be useful

VALID_TYPES = {"rule", "note", "illustration", "example", "diagram", "table", "component", "other"}

REGION_PROMPT = """You are classifying a region from a board game rulebook.
The image shows a CONTEXT region (for section/topic context) above a grey separator line, then the CURRENT region to classify below it.
PaddleOCR classified the current region as: {paddle_type}
{section_line}{style_notes}

Return a JSON object for the CURRENT region only with these exact fields:
- "type": exactly one of: "rule", "note", "illustration", "example", "diagram", "table", "component", "other"
  - rule: primary rulebook text stating how the game is played — mechanics, procedures, scoring, turn structure
  - note: a supplementary callout, sidebar, tip, clarification, or "important" box that supports a rule but is not the rule itself
  - illustration: a picture, drawing, or photo supporting a rule or example
  - example: a worked example, scenario, or in-game dialogue showing rules in action — includes player speech bubbles, "would you..." questions, and specific game situations used to illustrate how a rule applies
  - diagram: a labeled visual showing layout, structure, or spatial relationships
  - table: a grid of values
  - component: a list or description of physical game components (cards, tokens, dice, boards, etc.)
  - other: page numbers, titles, decorative elements, anything that doesn't fit above
- "label": {label_instruction}
- "description": for text regions, quote or closely paraphrase the actual rule/note text — do NOT write meta-summaries like "rulebook text explaining X". WRONG: "Region header indicating section 49 covering restricted area rules, establishing which areas have limited access." RIGHT: "Any Assassin who enters a square containing a Restricted Area token AND 1+ Enemies automatically becomes exposed." When a region contains multiple sentences or paragraphs, include ALL of them — do not drop or skip any sentence. The full text of the region must be represented. For images/diagrams describe what is shown AND the game concept being illustrated. Before transcribing, identify ALL non-text symbols/icons in the region. Never use generic [icon] or [symbol] — always describe what you see (e.g. [exclamation], [heart], [dice]). Never omit icons.

Return ONLY valid JSON, no markdown, no explanation."""

_paddle_engine = None
_paddle_lock = threading.Lock()

_text_det = None
_text_rec = None
_ocr_lock = threading.Lock()


def _get_paddle():
    global _paddle_engine
    if _paddle_engine is None:
        from paddleocr import LayoutDetection
        _paddle_engine = LayoutDetection()
    return _paddle_engine


def _get_text_det():
    global _text_det
    with _ocr_lock:
        if _text_det is None:
            from paddleocr import TextDetection
            _text_det = TextDetection(box_thresh=0.4)
    return _text_det


def _get_text_rec():
    global _text_rec
    with _ocr_lock:
        if _text_rec is None:
            from paddleocr import TextRecognition
            _text_rec = TextRecognition()
    return _text_rec


def _extract_component_list(crop: Image.Image, section_name: str) -> list[dict]:
    """Extract component list items as individual dicts for per-item indexing."""
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": (
                "This is a component list from a board game rulebook. "
                "List every item exactly as shown (quantity and name), one per line. "
                "Do not add any other text, headings, or explanation."
            )},
        ]}],
    )
    items = [line.strip() for line in response.content[0].text.strip().splitlines() if line.strip()]
    return [
        {"type": "component", "label": f"{section_name} – {item}", "description": item}
        for item in items
    ]


def _extract_heading_text(crop: Image.Image) -> str:
    """Extract heading text using Claude Haiku for accuracy (section labels cascade to all elements)."""
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": "What text is shown in this heading? Reply with just the heading text, nothing else."},
        ]}],
    )
    return response.content[0].text.strip()


def _classify_region(
    crop: Image.Image, paddle_type: str, current_section: str = "",
    style_notes: str = "",
) -> dict:
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    if current_section:
        section_line = f"Current section heading: {current_section}"
        label_instruction = (
            f'short name in the form "{current_section} – <topic>" where <topic> describes the specific '
            f'mechanic, condition, or procedure this content covers (e.g. "{current_section} – Tie-Breaking Rule", '
            f'"{current_section} – Maximum Hand Size"). Do NOT use generic topics like "Core Mechanic", '
            f'"Overview", "Introduction", "Main Rule", "General Rules", or "Procedure".'
        )
    else:
        section_line = ""
        label_instruction = (
            'short name in the form "<Section> – <topic>" where Section is a heading visible in the text and '
            '<topic> describes the specific mechanic, condition, or procedure this content covers '
            '(e.g. "Villes – Tie-Breaking Rule", "Setup – Starting Resources"). '
            'Do NOT use generic topics like "Core Mechanic", "Overview", "Introduction", "Main Rule", or "General Rules". '
            'If no heading is visible, describe the content briefly.'
        )
    prompt = REGION_PROMPT.format(
        paddle_type=paddle_type,
        section_line=section_line + "\n" if section_line else "",
        label_instruction=label_instruction,
        style_notes=style_notes + "\n" if style_notes else "",
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


TEXT_CLASSIFY_PROMPT = """You are classifying a text excerpt from a board game FAQ or errata document.

Text:
{text}

{section_line}
Return a JSON object with these exact fields:
- "type": exactly one of: "rule", "note", "example", "table", "other"
  - rule: a Q&A pair or ruling that clarifies how a rule works
  - note: a supplementary clarification, errata notice, or correction
  - example: a worked example showing how a rule applies
  - table: a grid of values or structured comparison
  - other: navigation elements, page headers/footers, decorative text, store links, social media — anything non-substantive
- "label": short name in the form "<Section> – <Topic>" where Section is the game topic and Topic describes what this text covers (e.g. "Calendar – End-of-Game Timer Tokens", "Setup – Starting Resources"). Do NOT use "Table of Contents" as a section name. If no section is clear, infer one from the text content.
- "description": quote or closely paraphrase the actual text — do NOT write meta-summaries like "text explaining X"

Return ONLY valid JSON, no markdown, no explanation."""


def _classify_text_chunk(text: str, current_section: str = "") -> dict:
    section_line = f'Current section: "{current_section}"\n' if current_section else ""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": TEXT_CLASSIFY_PROMPT.format(
            text=text[:800],
            section_line=section_line,
        )}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _classify_chunks_parallel(chunks: list[str], current_section: str) -> list[dict]:
    """Classify text chunks in parallel via Claude Haiku."""
    def _do(text):
        try:
            return _classify_text_chunk(text, current_section)
        except Exception:
            return {"type": "other", "label": current_section or "Unknown", "description": text}

    with ThreadPoolExecutor(max_workers=len(chunks) or 1) as ex:
        return list(ex.map(_do, chunks))


def _build_elements_from_chunks(
    chunks: list[str],
    classifications: list[dict],
    rulebook_id: str,
    source_type: SourceType,
    page_number: int,
    initial_section: str,
) -> tuple[list[Element], str]:
    """Convert classified text chunks into Elements with display_mode='text'."""
    elements = []
    last_section = initial_section
    for text, data in zip(chunks, classifications):
        elem_type = data.get("type", "other")
        if elem_type not in VALID_TYPES:
            elem_type = "other"
        if elem_type == "other":
            continue  # skip nav/footer noise
        label = data.get("label", initial_section or "Unknown")
        description = data.get("description", text)
        if len(description) < MIN_DESCRIPTION_CHARS:
            continue
        elements.append(Element(
            id=str(uuid.uuid4()),
            rulebook_id=rulebook_id,
            source_type=source_type,
            page_number=page_number,
            display_mode="text",
            page_image_path="",
            type=elem_type,
            label=label,
            description=description,
            bbox=BoundingBox(x=0, y=0, w=1, h=1),
        ))
        if " \u2013 " in label:
            last_section = label.split(" \u2013 ")[0]
    return elements, last_section


HTML_EXTRACT_PROMPT = """Extract every Q&A pair from this FAQ/errata document.

Return a JSON array where each item has:
- "section": the section heading this Q&A falls under (e.g. "Saving the game", "Card questions")
- "question": the full question text
- "answer": the full answer text
- "type": one of "rule", "note", "example" — "rule" for rulings/clarifications, "note" for errata/corrections, "example" for worked examples

Include every Q&A. Skip navigation, cookie notices, store links, and other non-game-rules content.
Return ONLY a JSON array, no markdown, no explanation."""


def extract_elements_html(
    html: str,
    rulebook_id: str,
    source_type: SourceType,
) -> list[Element]:
    """Extract Q&A elements from an HTML document via a single Haiku call."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": f"{HTML_EXTRACT_PROMPT}\n\n{text}"}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    pairs = json.loads(raw)

    elements = []
    for i, pair in enumerate(pairs):
        section = pair.get("section", "")
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        elem_type = pair.get("type", "rule")
        if elem_type not in VALID_TYPES:
            elem_type = "rule"
        label = f"{section} \u2013 {question[:60]}" if section else question[:60]
        description = f"Q: {question}\nA: {answer}"
        if len(description) < MIN_DESCRIPTION_CHARS:
            continue
        elements.append(Element(
            id=str(uuid.uuid4()),
            rulebook_id=rulebook_id,
            source_type=source_type,
            page_number=i + 1,
            display_mode="text",
            page_image_path="",
            type=elem_type,
            label=label,
            description=description,
            bbox=BoundingBox(x=0, y=0, w=1, h=1),
        ))
    return elements


def extract_elements_text(
    text_blocks: list[dict],
    page_w: float,
    page_h: float,
    rulebook_id: str,
    source_type: SourceType,
    page_number: int,
    image_path: str,
    initial_section: str,
) -> tuple[list[Element], str]:
    """Extract Q&A elements from fitz text blocks (real PDF) via a single Haiku call."""
    page_text = "\n".join(b["text"].strip() for b in text_blocks if b["text"].strip())
    if not page_text.strip():
        return [], initial_section

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": f"{HTML_EXTRACT_PROMPT}\n\n{page_text}"}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        pairs = json.loads(raw)
    except Exception:
        return [], initial_section

    elements = []
    last_section = initial_section
    for pair in pairs:
        section = pair.get("section", "") or initial_section
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        elem_type = pair.get("type", "rule")
        if elem_type not in VALID_TYPES:
            elem_type = "rule"
        label = f"{section} \u2013 {question[:60]}" if section else question[:60]
        description = f"Q: {question}\nA: {answer}"
        if len(description) < MIN_DESCRIPTION_CHARS:
            continue
        elements.append(Element(
            id=str(uuid.uuid4()),
            rulebook_id=rulebook_id,
            source_type=source_type,
            page_number=page_number,
            display_mode="text",
            page_image_path=image_path,
            type=elem_type,
            label=label,
            description=description,
            bbox=BoundingBox(x=0, y=0, w=1, h=1),
        ))
        if section:
            last_section = section
    return elements, last_section


def _split_wide_images(boxes: list[dict]) -> list[dict]:
    """
    Split image/figure boxes that are much wider than the text columns below them.
    This handles multi-column layouts where a single wide illustration spans multiple
    column sections — e.g. four illustrations detected as one wide box above four text columns.
    Each strip is aligned to the x-boundaries of the text column beneath it.
    """
    IMAGE_LABELS = {"image", "figure", "picture"}
    image_boxes = [b for b in boxes if b.get("label", "").lower() in IMAGE_LABELS]
    text_boxes = [b for b in boxes if b.get("label", "").lower() not in IMAGE_LABELS]

    result = []
    processed_images = set()

    for box in boxes:
        label = box.get("label", "").lower()
        if label not in IMAGE_LABELS:
            result.append(box)
            continue

        box_id = id(box)
        if box_id in processed_images:
            continue
        processed_images.add(box_id)

        ix0, iy0, ix1, iy1 = [int(v) for v in box["coordinate"]]
        box_width = ix1 - ix0

        # Find text boxes that start within 150px below this image and overlap horizontally
        below_texts = [
            b for b in text_boxes
            if int(b["coordinate"][1]) >= iy1
            and int(b["coordinate"][1]) <= iy1 + 150
            and int(b["coordinate"][2]) > ix0
            and int(b["coordinate"][0]) < ix1
        ]

        if len(below_texts) < 2:
            result.append(box)
            continue

        avg_text_w = sum(
            int(b["coordinate"][2]) - int(b["coordinate"][0]) for b in below_texts
        ) / len(below_texts)

        if box_width < avg_text_w * 1.8:
            result.append(box)
            continue

        # Split into strips aligned to the text column x-boundaries
        sorted_texts = sorted(below_texts, key=lambda b: b["coordinate"][0])
        for tb in sorted_texts:
            tx0 = max(int(tb["coordinate"][0]), ix0)
            tx1 = min(int(tb["coordinate"][2]), ix1)
            if tx1 - tx0 < MIN_REGION_W_PX:
                continue
            result.append({
                "label": box["label"],
                "coordinate": [tx0, iy0, tx1, iy1],
                "score": box["score"],
            })

    return result


def _find_image_context(all_boxes: list[dict], box: dict) -> dict | None:
    """
    For a text box, find an image/figure box that is above it and horizontally aligned
    (center-x of the text box falls within the image box's x-range).
    Returns the closest such image box, or None.
    """
    IMAGE_LABELS = {"image", "figure", "picture"}
    x0, y0, x1, y1 = [int(v) for v in box["coordinate"]]
    cx = (x0 + x1) / 2

    candidates = [
        b for b in all_boxes
        if b.get("label", "").lower() in IMAGE_LABELS
        and int(b["coordinate"][3]) <= y0  # image ends above current box
        and int(b["coordinate"][0]) <= cx <= int(b["coordinate"][2])  # cx within image x-range
    ]

    if not candidates:
        return None
    return max(candidates, key=lambda b: b["coordinate"][3])  # closest above


def _find_text_context(all_boxes: list[dict], box: dict) -> dict | None:
    """
    For an image box, find the most closely associated text box — checking:
    1. Horizontally adjacent (side-by-side layout): text whose y-range overlaps the image's y-range
    2. Directly below (stacked layout): text within 150px below the image
    Returns the best candidate, or None.
    """
    IMAGE_LABELS = {"image", "figure", "picture"}
    x0, y0, x1, y1 = [int(v) for v in box["coordinate"]]
    cy = (y0 + y1) / 2
    img_h = y1 - y0

    non_image = [b for b in all_boxes if b.get("label", "").lower() not in IMAGE_LABELS]

    # Side-by-side: text whose vertical center falls within the image's y-range
    adjacent = [
        b for b in non_image
        if y0 <= (int(b["coordinate"][1]) + int(b["coordinate"][3])) / 2 <= y1
        and (int(b["coordinate"][2]) <= x0 + 20 or int(b["coordinate"][0]) >= x1 - 20)  # left or right
    ]
    if adjacent:
        # Pick the one with the most vertical overlap
        def overlap(b):
            by0, by1 = int(b["coordinate"][1]), int(b["coordinate"][3])
            return min(y1, by1) - max(y0, by0)
        return max(adjacent, key=overlap)

    # Stacked: text directly below
    below = [
        b for b in non_image
        if int(b["coordinate"][1]) >= y1
        and int(b["coordinate"][1]) <= y1 + 150
        and int(b["coordinate"][0]) < x1
        and int(b["coordinate"][2]) > x0
    ]
    if below:
        return min(below, key=lambda b: b["coordinate"][1])

    return None


def _build_context_crop(img: Image.Image, prev_box: dict | None, box: dict) -> Image.Image:
    """
    Build the image sent to Claude: the previous box (for section context) stacked
    above a 2px separator line, then the current box. If no previous box, just the
    current box crop.
    """
    x0, y0, x1, y1 = [int(v) for v in box["coordinate"]]
    current = img.crop((x0, y0, x1, y1))
    if prev_box is None:
        return current

    px0, py0, px1, py1 = [int(v) for v in prev_box["coordinate"]]
    prev_crop = img.crop((px0, py0, px1, py1))

    sep_h = 2
    combined_w = max(prev_crop.width, current.width)
    combined_h = prev_crop.height + sep_h + current.height
    combined = Image.new("RGB", (combined_w, combined_h), (200, 200, 200))
    combined.paste(prev_crop, (0, 0))
    # separator is left as the grey background
    combined.paste(current, (0, prev_crop.height + sep_h))
    return combined


def _detect_source_type_override(img: Image.Image, boxes: list[dict]) -> str:
    """
    Check if this page starts an expansion or variant section.
    Uses a two-tier approach scoped to LayoutDetection boxes:
    - Tier 1 (cheap): paragraph_title / doc_title boxes — run TextRecognition directly
    - Tier 2 (expensive): header / image boxes in the top 20% only — run TextDetection then TextRecognition
    A recognized string must START with the keyword to avoid false positives from
    body text that merely mentions "expansion" or "variant" in passing.
    Returns 'expansion', 'variant', or '' (no change).
    """
    CHEAP_LABELS = {"paragraph_title", "doc_title"}
    EXPENSIVE_LABELS = {"header", "image"}
    w_img, h_img = img.size
    top_threshold = int(h_img * 0.20)
    rec = _get_text_rec()

    def _check(text: str) -> str:
        t = text.strip().lower()
        if t.startswith("expansion"):
            return "expansion"
        if t.startswith("variant"):
            return "variant"
        return ""

    for box in boxes:
        label = box.get("label", "").lower()
        x0, y0, x1, y1 = [int(v) for v in box["coordinate"]]

        if label in CHEAP_LABELS:
            try:
                crop_bgr = np.array(img.crop((x0, y0, x1, y1)))[:, :, ::-1]
                result = rec.predict(crop_bgr)
                if result:
                    hit = _check(result[0]["rec_text"])
                    if hit:
                        return hit
            except Exception:
                continue

        elif label in EXPENSIVE_LABELS and y0 < top_threshold:
            try:
                crop = img.crop((x0, y0, x1, y1))
                crop_bgr = np.array(crop)[:, :, ::-1]
                det_result = _get_text_det().predict(crop_bgr)
                dt_polys = det_result[0]["dt_polys"] if det_result else []
                for poly in dt_polys:
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    sx0, sy0 = int(min(xs)), int(min(ys))
                    sx1, sy1 = int(max(xs)), int(max(ys))
                    sub_bgr = np.array(crop.crop((sx0, sy0, sx1, sy1)))[:, :, ::-1]
                    rec_result = rec.predict(sub_bgr)
                    if rec_result:
                        hit = _check(rec_result[0]["rec_text"])
                        if hit:
                            return hit
            except Exception:
                continue

    # Tier 3: scan the top 15% of the page with TextDetection.
    # Section-announcement banners always appear at the top; body text does not.
    # This catches banners that LayoutDetection doesn't box as headings/images.
    # Skip if heading boxes were already found in the top 15% — Tier 1 already checked them.
    w_img, h_img = img.size
    top_threshold = h_img * 0.15
    has_top_headings = any(
        box.get("label", "").lower() in CHEAP_LABELS and int(box["coordinate"][1]) < top_threshold
        for box in boxes
    )
    if has_top_headings:
        return ""
    try:
        w, h = img.size
        top_strip = img.crop((0, 0, w, int(h * 0.15)))
        top_bgr = np.array(top_strip)[:, :, ::-1]
        det_result = _get_text_det().predict(top_bgr)
        dt_polys = det_result[0]["dt_polys"] if det_result else []
        for poly in dt_polys:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            sx0, sy0 = int(min(xs)), int(min(ys))
            sx1, sy1 = int(max(xs)), int(max(ys))
            sub_bgr = np.array(top_strip.crop((sx0, sy0, sx1, sy1)))[:, :, ::-1]
            rec_result = rec.predict(sub_bgr)
            if rec_result:
                hit = _check(rec_result[0]["rec_text"])
                if hit:
                    return hit
    except Exception:
        pass

    return ""


def detect_game_name(image_bytes: bytes) -> str | None:
    """
    Detect the game name from a cover/first page using PaddleOCR + Claude Haiku.
    Looks for a doc_title region; falls back to the largest detected box.
    Returns the name string or None if detection fails.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_bgr = np.array(img)[:, :, ::-1]

    engine = _get_paddle()
    det_results = engine.predict(img_bgr)
    boxes = det_results[0]["boxes"] if det_results else []
    if not boxes:
        return None

    title_boxes = [b for b in boxes if b["label"] == "doc_title"]
    if not title_boxes:
        # Fall back to the largest region by area
        title_boxes = sorted(
            boxes,
            key=lambda b: (b["coordinate"][2] - b["coordinate"][0]) * (b["coordinate"][3] - b["coordinate"][1]),
            reverse=True,
        )[:1]

    box = max(title_boxes, key=lambda b: b["score"])
    x0, y0, x1, y1 = [int(v) for v in box["coordinate"]]
    crop = img.crop((x0, y0, x1, y1))

    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": "What is the name of this board game? Reply with just the game name, nothing else."},
            ]}],
        )
        return response.content[0].text.strip()
    except Exception:
        return None


def _deduplicate(elements: list[Element]) -> list[Element]:
    """Remove near-duplicates that can occur if PaddleOCR returns overlapping regions."""
    kept: list[Element] = []
    for e in elements:
        cy = e.bbox.y + e.bbox.h / 2
        cx = e.bbox.x + e.bbox.w / 2
        is_dup = any(
            abs(cy - (k.bbox.y + k.bbox.h / 2)) < 0.02
            and abs(cx - (k.bbox.x + k.bbox.w / 2)) < 0.10
            for k in kept
        )
        if not is_dup:
            kept.append(e)
    return kept


def detect_page_split(image_bytes: bytes) -> tuple[int | None, list[dict] | None]:
    """
    Detect if an image is a 2-up (two logical pages side by side) layout.
    Returns (split_x, boxes) where boxes are the PaddleOCR layout boxes in
    full-image coordinates, or (None, None) if single page.

    Detection order:
    1. Aspect ratio: portrait → never 2-up, fast exit
    2. Page number boxes: two `number`-labeled boxes in opposite halves of the bottom
       quarter → split between their inner edges
    3. Vertical gutter: largest gap in merged box x-coverage within the middle third
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    if w < h * 1.2:
        return None, None

    img_bgr = np.array(img)[:, :, ::-1]
    with _paddle_lock:
        det_results = _get_paddle().predict(img_bgr)
    boxes = det_results[0]["boxes"] if det_results else []

    # 1. Page number detection
    bottom_y = h * 0.75
    numbers = [b for b in boxes if b.get("label") == "number" and int(b["coordinate"][1]) >= bottom_y]
    left_nums  = [b for b in numbers if (int(b["coordinate"][0]) + int(b["coordinate"][2])) / 2 <  w / 2]
    right_nums = [b for b in numbers if (int(b["coordinate"][0]) + int(b["coordinate"][2])) / 2 >= w / 2]
    if left_nums and right_nums:
        left_inner  = max(int(b["coordinate"][2]) for b in left_nums)
        right_inner = min(int(b["coordinate"][0]) for b in right_nums)
        split_x = (left_inner + right_inner) // 2
        return split_x, boxes

    # 2. Vertical gutter: largest gap in merged x-coverage within the middle third
    center = w // 2
    search_half = w // 6
    intervals = sorted((max(0, int(b["coordinate"][0])), min(w, int(b["coordinate"][2]))) for b in boxes)
    merged: list[list[int]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    best_gap_x, best_gap_w = None, 0
    prev_end = center - search_half
    for s, e in merged:
        if s > center + search_half:
            break
        gap_s = max(prev_end, center - search_half)
        gap_e = min(s, center + search_half)
        if gap_e > gap_s and (gap_e - gap_s) > best_gap_w:
            best_gap_w = gap_e - gap_s
            best_gap_x = (gap_s + gap_e) // 2
        prev_end = max(prev_end, e)

    if best_gap_x is not None and best_gap_w >= w * 0.005:
        return best_gap_x, boxes

    # No split found, but return boxes so caller can pass them as precomputed_boxes
    # and avoid re-running LayoutDetection in extract_elements.
    return None, boxes


def _remap_boxes_to_half(
    boxes: list[dict], split_x: int, half: str, margin: int = 10,
) -> list[dict]:
    """
    Remap full-image PaddleOCR boxes to a half-image coordinate system.

    half="left":  keep boxes with x1 < split_x + margin, clamp x2 to split_x
    half="right": keep boxes with x2 > split_x - margin, clamp x1 to split_x,
                  then shift all x coordinates left by split_x
    """
    remapped = []
    for box in boxes:
        x0, y0, x1, y1 = box["coordinate"]
        if half == "left":
            if x0 >= split_x + margin:
                continue  # entirely in right half
            new_x1 = min(x1, split_x)
            if new_x1 - x0 < MIN_REGION_W_PX:
                continue
            remapped.append({
                "label": box["label"],
                "coordinate": [x0, y0, new_x1, y1],
                "score": box["score"],
            })
        else:  # right
            if x1 <= split_x - margin:
                continue  # entirely in left half
            new_x0 = max(x0, split_x)
            if x1 - new_x0 < MIN_REGION_W_PX:
                continue
            remapped.append({
                "label": box["label"],
                "coordinate": [new_x0 - split_x, y0, x1 - split_x, y1],
                "score": box["score"],
            })
    return remapped


def split_image(image_bytes: bytes, split_x: int) -> tuple[bytes, bytes]:
    """Crop image into left and right halves at split_x."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    def to_bytes(crop):
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()

    return to_bytes(img.crop((0, 0, split_x, h))), to_bytes(img.crop((split_x, 0, w, h)))


# ---------------------------------------------------------------------------
# Phase A: statistical layout calibration from sampled pages
# ---------------------------------------------------------------------------

# Default thresholds (used when sample has no heading boxes)
_DEFAULT_HEADING_MAX_W_PCT = 0.25
_DEFAULT_MAYBE_HEADING_MAX_W_PCT = 0.45


def _compute_layout_stats(
    sampled_boxes: list[list[dict]],
    page_widths: list[int],
) -> dict:
    """Compute layout statistics from PaddleOCR boxes on sampled pages.

    Returns a dict with calibrated thresholds — designed to be extensible
    for Phase B (LLM document profile).
    """
    heading_widths: list[float] = []  # as fraction of page width
    text_widths: list[float] = []

    for boxes, pw in zip(sampled_boxes, page_widths):
        if pw <= 0:
            continue
        for b in boxes:
            label = b.get("label", "").lower()
            bw = (int(b["coordinate"][2]) - int(b["coordinate"][0])) / pw
            if label in ("paragraph_title", "doc_title"):
                heading_widths.append(bw)
            elif label == "text":
                text_widths.append(bw)

    # heading_max_width_pct: 90th percentile of heading widths × 1.2,
    # capped at 45% to avoid over-permissive threshold
    if heading_widths:
        sorted_hw = sorted(heading_widths)
        p90_idx = int(len(sorted_hw) * 0.9)
        p90_idx = min(p90_idx, len(sorted_hw) - 1)
        heading_width_p90 = sorted_hw[p90_idx]
        heading_max_width_pct = min(heading_width_p90 * 1.2, _DEFAULT_MAYBE_HEADING_MAX_W_PCT)
    else:
        heading_max_width_pct = _DEFAULT_HEADING_MAX_W_PCT

    # estimated_columns: median text width < 0.4 → likely 2-column
    if text_widths:
        sorted_tw = sorted(text_widths)
        text_width_median = sorted_tw[len(sorted_tw) // 2]
        estimated_columns = 2 if text_width_median < 0.4 else 1
    else:
        text_width_median = 0.45
        estimated_columns = 1

    return {
        "heading_max_width_pct": heading_max_width_pct,
        "estimated_columns": estimated_columns,
        "heading_width_p90": heading_widths and sorted_hw[p90_idx] or _DEFAULT_HEADING_MAX_W_PCT,
        "text_width_median": text_width_median,
        "sample_page_count": len(sampled_boxes),
    }


ICON_LEGEND_PROMPT = """Look at these early pages from a board game rulebook. If any page shows a symbols/icons glossary or legend (a section that explains what game icons or symbols mean), extract the vocabulary as a JSON object mapping bracket names to meanings.

Example output: {"[coin]": "gold cost", "[sword]": "attack strength", "[shield]": "defense value"}

Rules:
- Use lowercase bracket names derived from the icon's visual appearance or label in the rulebook.
- Only include icons that are explicitly defined/labeled in the rulebook.
- Return {} if no symbols/icons legend page is present.

Return ONLY valid JSON, no markdown, no explanation."""


_ICON_LEGEND_MAX_W = 1024  # resize pages before sending to avoid 5MB API limit


def _resize_for_api(img_bytes: bytes, max_w: int = _ICON_LEGEND_MAX_W) -> bytes:
    """Resize image to max_w wide (preserving aspect ratio) if wider, return PNG bytes."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    if w > max_w:
        img = img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _extract_icon_legend(page_images: list[bytes]) -> dict:
    """Send early page images to Claude Haiku to extract any icon/symbol legend vocabulary."""
    content: list[dict] = []
    for i, img_bytes in enumerate(page_images):
        resized = _resize_for_api(img_bytes)
        b64 = base64.standard_b64encode(resized).decode()
        content.append({"type": "text", "text": f"Page {i + 1} of {len(page_images)}:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })
    content.append({"type": "text", "text": ICON_LEGEND_PROMPT})

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": content}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        logger.warning(f"  [icon_legend] Failed to parse Haiku response: {raw[:200]}")
    return {}


PROFILE_PROMPT = """Analyze these sample pages from a board game rulebook and characterize the document's layout.

Measured layout statistics from PaddleOCR box detection:
{stats_summary}

Based on these pages, return a JSON object with these exact fields:
- "heading_max_width_pct": float — maximum width of true section headings as a fraction of page width (e.g. 0.20 for narrow left-aligned headings, 0.40 for centered chapter titles). Use the measured stats above as a guide but override if the visual evidence disagrees.
- "sub_heading_pattern": exactly one of "numbered_steps", "bold_inline", "none" — how sub-sections within a main section are formatted. "numbered_steps" means headings like "1. Setup", "2. Draw Cards"; "bold_inline" means bold text at the start of a paragraph that acts as a sub-heading; "none" means no sub-heading pattern.
- "column_count": 1 or 2 or "mixed" — the dominant column layout across these pages.
- "has_bold_callouts": boolean — whether there are wide bold text blocks that look like headings but are actually rule callouts or emphasis text within the body.
- "callout_description": string — how callout boxes, sidebars, tips, or emphasized rules visually differ from section headings (e.g. "yellow background boxes", "indented with a lightbulb icon", "bordered panels"). Empty string if none.
- "layout_notes": string — any other distinctive layout patterns that would help correctly classify regions (e.g. "rules are in the left column, examples in the right", "player aids are boxed in blue").

Return ONLY valid JSON, no markdown, no explanation."""


def _build_document_profile(
    sample_images: list[bytes],
    stats: dict,
    early_page_images: list[bytes] | None = None,
) -> dict:
    """Send sampled page images + stats to Claude Haiku for a document profile."""
    stats_summary = (
        f"- Heading width p90: {stats.get('heading_width_p90', 0.25):.2f} of page width\n"
        f"- Heading max width (statistical): {stats.get('heading_max_width_pct', 0.25):.2f}\n"
        f"- Text width median: {stats.get('text_width_median', 0.45):.2f}\n"
        f"- Estimated columns: {stats.get('estimated_columns', 1)}"
    )

    content: list[dict] = []
    for i, img_bytes in enumerate(sample_images):
        b64 = base64.standard_b64encode(img_bytes).decode()
        content.append({"type": "text", "text": f"Page {i + 1} of {len(sample_images)}:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })

    content.append({
        "type": "text",
        "text": PROFILE_PROMPT.format(stats_summary=stats_summary),
    })

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": content}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        profile = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"  [profile] Failed to parse Haiku response: {raw[:200]}")
        return {}

    # Validate and normalize
    valid_sub_patterns = {"numbered_steps", "bold_inline", "none"}
    if profile.get("sub_heading_pattern") not in valid_sub_patterns:
        profile["sub_heading_pattern"] = "none"

    if isinstance(profile.get("heading_max_width_pct"), (int, float)):
        profile["heading_max_width_pct"] = min(
            max(float(profile["heading_max_width_pct"]), 0.05),
            _DEFAULT_MAYBE_HEADING_MAX_W_PCT,
        )
    else:
        profile.pop("heading_max_width_pct", None)

    if profile.get("column_count") not in (1, 2, "mixed"):
        profile["column_count"] = stats.get("estimated_columns", 1)

    profile.setdefault("has_bold_callouts", False)
    profile.setdefault("callout_description", "")
    profile.setdefault("layout_notes", "")

    if early_page_images:
        icon_legend = _extract_icon_legend(early_page_images)
        if icon_legend:
            profile["icon_legend"] = icon_legend
            logger.info(f"  [icon_legend] Extracted {len(icon_legend)} icon(s): {list(icon_legend.keys())}")
        else:
            logger.info("  [icon_legend] No icon legend found in early pages")

    logger.info(f"  [profile] heading_max_w={profile.get('heading_max_width_pct', 'n/a')}, "
                f"sub_heading={profile['sub_heading_pattern']}, "
                f"columns={profile['column_count']}, "
                f"bold_callouts={profile['has_bold_callouts']}")
    return profile


def _profile_to_style_notes(profile: dict) -> str:
    """Build a style_notes string from a document profile for injection into REGION_PROMPT."""
    parts = []
    icon_legend = profile.get("icon_legend")
    if icon_legend:
        vocab = ", ".join(f"{k} = {v}" for k, v in icon_legend.items())
        parts.append(f"Icon vocabulary for this game: {vocab}. Use these exact bracket names.")
    if profile.get("callout_description"):
        parts.append(f"Document style note: {profile['callout_description']}")
    if profile.get("layout_notes"):
        parts.append(f"Layout note: {profile['layout_notes']}")
    return "\n".join(parts)


def compute_layout_stats(
    page_images: list[tuple[int, bytes]],
    total_pages: int,
) -> tuple[dict, dict[int, list[dict]]]:
    """Sample up to 3 pages and compute layout statistics.

    Args:
        page_images: list of (page_index, image_bytes) for ALL pages
        total_pages: total number of pages in the document

    Returns:
        (stats_dict, precomputed_boxes) where precomputed_boxes maps
        page_index → PaddleOCR boxes for reuse downstream.
    """
    # Pick sample indices: pages 0, n//3, 2n//3 (capped to what's available)
    if total_pages <= 3:
        sample_indices = list(range(total_pages))
    else:
        sample_indices = [0, total_pages // 3, 2 * total_pages // 3]

    # Build lookup from provided page_images
    images_by_idx = {idx: img_bytes for idx, img_bytes in page_images}

    sampled_boxes: list[list[dict]] = []
    page_widths: list[int] = []
    precomputed: dict[int, list[dict]] = {}

    for idx in sample_indices:
        if idx not in images_by_idx:
            continue
        img = Image.open(io.BytesIO(images_by_idx[idx])).convert("RGB")
        w, _h = img.size
        img_bgr = np.array(img)[:, :, ::-1]
        with _paddle_lock:
            det_results = _get_paddle().predict(img_bgr)
        boxes = det_results[0]["boxes"] if det_results else []

        sampled_boxes.append(boxes)
        page_widths.append(w)
        precomputed[idx] = boxes

    stats = _compute_layout_stats(sampled_boxes, page_widths)
    logger.info(f"  [calibration] sampled {len(sampled_boxes)} pages → "
                f"heading_max_w={stats['heading_max_width_pct']:.2f}, "
                f"columns={stats['estimated_columns']}, "
                f"heading_p90={stats['heading_width_p90']:.2f}, "
                f"text_median={stats['text_width_median']:.2f}")
    return stats, precomputed


def extract_elements(
    image_bytes: bytes,
    rulebook_id: str,
    source_type: SourceType,
    page_number: int,
    image_path: str,
    initial_section: str = "",
    initial_source_type: str = "",
    precomputed_boxes: list[dict] | None = None,
    layout_stats: dict | None = None,
    document_profile: dict | None = None,
) -> tuple[list[Element], str, str]:
    _page_t0 = time.perf_counter()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    HEADING_PADDLE_LABELS = {"paragraph_title", "doc_title"}
    IMAGE_PADDLE_LABELS = {"image", "figure", "picture"}
    BLOCK_SECTION_KEYWORDS = {"component", "components", "contents"}

    # Build style notes from document profile for injection into REGION_PROMPT
    _style_notes = _profile_to_style_notes(document_profile) if document_profile else ""

    # numbered_steps: paragraph_title boxes matching "1.", "2." etc. are rule steps, not headings
    _numbered_steps = (document_profile or {}).get("sub_heading_pattern") == "numbered_steps"

    _t0 = time.perf_counter()
    if precomputed_boxes is not None:
        boxes = precomputed_boxes
        logger.info(f"  [timing] page {page_number} paddle layout: skipped (precomputed {len(boxes)} boxes)")
    else:
        img_bgr = np.array(img)[:, :, ::-1]  # RGB → BGR for PaddleOCR
        with _paddle_lock:
            det_results = _get_paddle().predict(img_bgr)
        logger.info(f"  [timing] page {page_number} paddle layout: {time.perf_counter()-_t0:.1f}s")
        boxes = det_results[0]["boxes"] if det_results else []

    # Gate _split_wide_images on column count: profile overrides stats
    _profile_cols = (document_profile or {}).get("column_count")
    _stats_cols = (layout_stats or {}).get("estimated_columns", 1)
    _effective_cols = _profile_cols if _profile_cols in (1, 2, "mixed") else _stats_cols
    if layout_stats is None or _effective_cols in (2, "mixed"):
        boxes = _split_wide_images(boxes)
    sorted_boxes = sorted(boxes, key=lambda b: b["coordinate"][1])

    # Detect if this page begins an expansion or variant section.
    # initial_source_type carries a prior detection from a previous page.
    _t0 = time.perf_counter()
    # FAQ/errata never contain expansion/variant sections; skip detection entirely.
    # If a previous page already detected expansion/variant, inherit it — no need to re-detect.
    if source_type in ("faq", "errata") or initial_source_type:
        source_type_override = ""
    else:
        source_type_override = _detect_source_type_override(img, boxes)
    logger.info(f"  [timing] page {page_number} source_type_override: {time.perf_counter()-_t0:.1f}s")
    effective_source_type: str = source_type_override or initial_source_type or source_type

    valid_boxes = [
        b for b in sorted_boxes
        if (
            # Headings can be narrow single words — skip width filter for them
            b.get("label", "").lower() in HEADING_PADDLE_LABELS
            or (int(b["coordinate"][2]) - int(b["coordinate"][0])) >= MIN_REGION_W_PX
        )
        and (int(b["coordinate"][3]) - int(b["coordinate"][1])) >= MIN_REGION_H_PX
    ]

    # Phase 1 + 1b: classify heading boxes in parallel to build the section map.
    # Body elements depend on knowing section names before they can be labeled correctly.
    # Wide "paragraph_title" boxes (> 25% of page width) are bold rule callouts, not section headings.
    # Use calibrated threshold: profile overrides stats, stats override default.
    _heading_pct = (
        (document_profile or {}).get("heading_max_width_pct")
        or (layout_stats or {}).get("heading_max_width_pct")
        or _DEFAULT_HEADING_MAX_W_PCT
    )
    MAX_HEADING_W = w * _heading_pct
    MAYBE_HEADING_MAX_W = w * _DEFAULT_MAYBE_HEADING_MAX_W_PCT

    # Collect heading candidates in one pass
    phase1_boxes = []   # definite headings
    phase1b_boxes = []  # slightly-wide candidates (need keyword check)
    for box in valid_boxes:
        label = box.get("label", "").lower()
        if label not in HEADING_PADDLE_LABELS:
            continue
        box_w = int(box["coordinate"][2]) - int(box["coordinate"][0])
        if label == "paragraph_title" and box_w > MAX_HEADING_W:
            if box_w <= MAYBE_HEADING_MAX_W:
                phase1b_boxes.append(box)
            # else: wide callout, skip entirely
        else:
            phase1_boxes.append(box)

    all_heading_boxes = phase1_boxes + phase1b_boxes

    def _extract_for_box(box):
        crop = _build_context_crop(img, None, box)
        try:
            return box, _extract_heading_text(crop)
        except Exception:
            return box, "unknown"

    _t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(all_heading_boxes) or 1) as ex:
        heading_results = list(ex.map(_extract_for_box, all_heading_boxes))
    logger.info(f"  [timing] page {page_number} phase1 headings ({len(all_heading_boxes)}): {time.perf_counter()-_t0:.1f}s")

    # Seed with the last section from the previous page so content that continues
    # across a page break still gets the correct section label.
    heading_sections: list[tuple[dict, str]] = (
        [({"coordinate": [0, 0, w, 0]}, initial_section)] if initial_section else []
    )
    phase1_ids = {id(b) for b in phase1_boxes}
    _numbered_step_re = re.compile(r"^\d+\.")
    # Track boxes demoted from heading → body due to numbered_steps pattern
    _demoted_step_boxes: set[int] = set()
    for box, name in heading_results:
        # numbered_steps: paragraph_title boxes like "1. Setup" are rule steps, not headings
        if (
            _numbered_steps
            and box.get("label", "").lower() == "paragraph_title"
            and _numbered_step_re.match(name)
        ):
            _demoted_step_boxes.add(id(box))
            continue
        if id(box) in phase1_ids:
            heading_sections.append((box, name))
        else:  # phase1b: only include if matches block-section keyword
            if any(kw in name.lower() for kw in BLOCK_SECTION_KEYWORDS):
                heading_sections.append((box, name))

    # Keep heading_sections sorted by y so last_section is correct at page end.
    heading_sections.sort(key=lambda t: t[0]["coordinate"][1])

    # Index heading boxes as their own elements so section-name text is directly searchable.
    # Headings are used for section propagation but skipped by Phase 2/3, so without this
    # their text only appears as a label prefix on child elements, never in its own bbox.
    heading_elements: list[Element] = []
    for hbox, hname in heading_sections:
        hx0, hy0, hx1, hy1 = [int(v) for v in hbox["coordinate"]]
        if hy1 <= hy0:
            continue  # skip seed entry (coordinate [0,0,w,0])
        heading_elements.append(Element(
            id=str(uuid.uuid4()),
            rulebook_id=rulebook_id,
            source_type=effective_source_type,
            page_number=page_number,
            page_image_path=image_path,
            type="other",
            label=f"{hname} – Section Heading",
            description=f"Section heading: {hname}",
            bbox=BoundingBox(
                x=hx0 / w, y=hy0 / h,
                w=(hx1 - hx0) / w, h=(hy1 - hy0) / h,
            ),
        ))

    # Phase 1.5: for each block-section heading, crop its entire body as a single element.
    # PaddleOCR fragments component lists into icon + short-text pairs; treating the whole
    # region as one crop gives Claude enough context to produce a useful description.
    # Skipped for faq/errata — often website dumps where nav "Contents" links appear on
    # every page and would incorrectly trigger block-section detection.
    block_section_skip: set[int] = set()
    block_elements: list[Element] = []

    if effective_source_type not in ("faq", "errata"):
        for hbox, hname in heading_sections:
            if not any(kw in hname.lower() for kw in BLOCK_SECTION_KEYWORDS):
                continue
            hy1 = int(hbox["coordinate"][3])
            # Find y-top of the next heading that starts below this one.
            next_y = h
            for nhbox, _ in heading_sections:
                ny0 = int(nhbox["coordinate"][1])
                if ny0 > hy1:
                    next_y = min(next_y, ny0)
            # Collect valid_box indices that fall inside this section's y-range.
            section_indices = [
                i for i, b in enumerate(valid_boxes)
                if int(b["coordinate"][1]) >= hy1 and int(b["coordinate"][1]) < next_y
            ]
            if not section_indices:
                continue
            for i in section_indices:
                block_section_skip.add(i)
            # Union bbox: heading box + all section boxes.
            all_coords = [hbox["coordinate"]] + [valid_boxes[i]["coordinate"] for i in section_indices]
            ux0 = int(min(c[0] for c in all_coords))
            uy0 = int(min(c[1] for c in all_coords))
            ux1 = int(max(c[2] for c in all_coords))
            uy1 = int(max(c[3] for c in all_coords))
            section_crop = img.crop((ux0, uy0, ux1, uy1))
            _t0 = time.perf_counter()
            try:
                items = _extract_component_list(section_crop, hname)
            except Exception:
                items = []
            logger.info(f"  [timing] page {page_number} phase1.5 component list '{hname}': {time.perf_counter()-_t0:.1f}s")
            description = ", ".join(item["description"] for item in items) if items else f"Component list for {hname}"
            block_elements.append(Element(
                id=str(uuid.uuid4()),
                rulebook_id=rulebook_id,
                source_type=effective_source_type,
                page_number=page_number,
                page_image_path=image_path,
                type="component",
                label=f"{hname} – Component List",
                description=description,
                bbox=BoundingBox(
                    x=ux0 / w, y=uy0 / h,
                    w=(ux1 - ux0) / w, h=(uy1 - uy0) / h,
                ),
            ))

    def section_for_box(target: dict) -> str:
        tx0, ty0, tx1, ty1 = [int(v) for v in target["coordinate"]]
        tcx = (tx0 + tx1) / 2

        # Headings extend full page width: return the closest heading above by y-bottom.
        # Tiebreak by x-center distance to handle side-by-side headings at the same y.
        above = [
            (hbox, hname) for hbox, hname in heading_sections
            if int(hbox["coordinate"][1]) < ty0
        ]
        if not above:
            return ""
        return max(
            above,
            key=lambda t: (
                t[0]["coordinate"][3],
                -abs(tcx - (t[0]["coordinate"][0] + t[0]["coordinate"][2]) / 2),
            ),
        )[1]

    # Phase 2: build work items for all non-heading boxes (context crops + section assignment).
    work_items: list[tuple[dict, Image.Image, str, str]] = []  # (box, crop, paddle_type, section)
    for i, box in enumerate(valid_boxes):
        if i in block_section_skip:
            continue
        box_w = int(box["coordinate"][2]) - int(box["coordinate"][0])
        is_heading = (
            box.get("label", "").lower() in HEADING_PADDLE_LABELS
            and (box.get("label", "").lower() != "paragraph_title" or box_w <= MAX_HEADING_W)
            and id(box) not in _demoted_step_boxes
        )
        if not is_heading:
            is_image_box = box.get("label", "").lower() in IMAGE_PADDLE_LABELS
            section = section_for_box(box)
            if is_image_box:
                ctx = _find_text_context(valid_boxes, box)
            elif not section:
                # No section known — image context may help identify the topic
                ctx = _find_image_context(valid_boxes, box)
            else:
                # Section already known — skip image context; venue illustrations
                # contain styled logos that cause Claude to emit "Logo/Brand" labels
                ctx = None
            crop = _build_context_crop(img, ctx, box)
            work_items.append((box, crop, box.get("label", "Text"), section))

    # Phase 3: classify all body boxes in parallel.
    def classify(item: tuple) -> dict:
        box, crop, paddle_type, section = item
        for attempt in range(3):
            try:
                return _classify_region(crop, paddle_type, section, style_notes=_style_notes)
            except Exception:
                if attempt == 2:
                    return {"type": "other", "label": "Unknown Region", "description": ""}
                time.sleep(1 * (attempt + 1))

    _t0 = time.perf_counter()
    if work_items:
        with ThreadPoolExecutor(max_workers=min(len(work_items), 15)) as executor:
            classifications = list(executor.map(classify, work_items))
    else:
        classifications = []
    logger.info(f"  [timing] page {page_number} phase3 classify ({len(work_items)} items): {time.perf_counter()-_t0:.1f}s")

    elements = []
    for (box, _crop, _paddle_type, _section), result in zip(work_items, classifications):
        x0, y0, x1, y1 = [int(v) for v in box["coordinate"]]
        description = result.get("description", "")
        elem_type = result.get("type", "other")
        if elem_type not in VALID_TYPES:
            elem_type = "other"
        # Component items often have short descriptions (e.g. "36 Order Cards").
        # Fall back to the topic part of the label rather than dropping the element.
        if elem_type == "component" and len(description) < MIN_DESCRIPTION_CHARS:
            label_text = result.get("label", "")
            description = label_text.split(" – ", 1)[-1] if " – " in label_text else label_text
        if len(description) < MIN_DESCRIPTION_CHARS:
            continue
        elements.append(Element(
            id=str(uuid.uuid4()),
            rulebook_id=rulebook_id,
            source_type=effective_source_type,
            page_number=page_number,
            page_image_path=image_path,
            type=elem_type,
            label=result.get("label", "Unlabeled"),
            description=description,
            bbox=BoundingBox(
                x=x0 / w, y=y0 / h,
                w=(x1 - x0) / w, h=(y1 - y0) / h,
            ),
        ))

    last_section = heading_sections[-1][1] if heading_sections else ""
    logger.info(f"  [timing] page {page_number} TOTAL: {time.perf_counter()-_page_t0:.1f}s")
    # block_elements share a bbox (per-item indexing of component lists) — skip dedup for them.
    return _deduplicate(heading_elements + elements) + block_elements, last_section, effective_source_type
