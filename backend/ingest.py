import anthropic
import base64
import io
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from models import Element, BoundingBox, SourceType

client = anthropic.Anthropic()

MIN_REGION_W_PX = 80  # skip regions narrower than this — catches single-word fragments
MIN_REGION_H_PX = 20  # skip regions shorter than this
MIN_DESCRIPTION_CHARS = 20  # skip elements whose description is too short to be useful

VALID_TYPES = {"rule", "note", "illustration", "example", "diagram", "table", "component", "other"}

REGION_PROMPT = """You are classifying a region from a board game rulebook.
The image shows a CONTEXT region (for section/topic context) above a grey separator line, then the CURRENT region to classify below it.
PaddleOCR classified the current region as: {paddle_type}
{section_line}

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
- "description": for text regions, quote or closely paraphrase the actual rule/note text — do NOT write meta-summaries like "rulebook text explaining X"; for images/diagrams describe what is shown AND the game concept being illustrated

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


def _classify_region(crop: Image.Image, paddle_type: str, current_section: str = "") -> dict:
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
    - Tier 2 (expensive): header / image boxes — run TextDetection then TextRecognition
    A recognized string must START with the keyword to avoid false positives from
    body text that merely mentions "expansion" or "variant" in passing.
    Returns 'expansion', 'variant', or '' (no change).
    """
    CHEAP_LABELS = {"paragraph_title", "doc_title"}
    EXPENSIVE_LABELS = {"header", "image"}
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

        elif label in EXPENSIVE_LABELS:
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


def detect_page_split(image_bytes: bytes) -> int | None:
    """
    Detect if an image is a 2-up (two logical pages side by side) layout.
    Returns the x-coordinate to split on, or None if single page.

    Detection order:
    1. Aspect ratio: portrait → never 2-up, fast exit
    2. Page number boxes: two `number`-labeled boxes in opposite halves of the bottom
       quarter → split between their inner edges
    3. Vertical gutter: largest gap in merged box x-coverage within the middle third
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    if w < h * 1.2:
        return None

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
        return split_x

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
        return best_gap_x

    return None


def split_image(image_bytes: bytes, split_x: int) -> tuple[bytes, bytes]:
    """Crop image into left and right halves at split_x."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    def to_bytes(crop):
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()

    return to_bytes(img.crop((0, 0, split_x, h))), to_bytes(img.crop((split_x, 0, w, h)))


def extract_elements(
    image_bytes: bytes,
    rulebook_id: str,
    source_type: SourceType,
    page_number: int,
    image_path: str,
    initial_section: str = "",
    initial_source_type: str = "",
) -> tuple[list[Element], str, str]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    img_bgr = np.array(img)[:, :, ::-1]  # RGB → BGR for PaddleOCR

    HEADING_PADDLE_LABELS = {"paragraph_title", "doc_title"}
    IMAGE_PADDLE_LABELS = {"image", "figure", "picture"}
    BLOCK_SECTION_KEYWORDS = {"component", "components", "contents"}

    with _paddle_lock:
        det_results = _get_paddle().predict(img_bgr)
    boxes = det_results[0]["boxes"] if det_results else []

    boxes = _split_wide_images(boxes)
    sorted_boxes = sorted(boxes, key=lambda b: b["coordinate"][1])

    # Detect if this page begins an expansion or variant section.
    # initial_source_type carries a prior detection from a previous page.
    source_type_override = _detect_source_type_override(img, boxes)
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
    MAX_HEADING_W = w * 0.25
    MAYBE_HEADING_MAX_W = w * 0.45

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

    with ThreadPoolExecutor(max_workers=len(all_heading_boxes) or 1) as ex:
        heading_results = list(ex.map(_extract_for_box, all_heading_boxes))

    # Seed with the last section from the previous page so content that continues
    # across a page break still gets the correct section label.
    heading_sections: list[tuple[dict, str]] = (
        [({"coordinate": [0, 0, w, 0]}, initial_section)] if initial_section else []
    )
    phase1_ids = {id(b) for b in phase1_boxes}
    for box, name in heading_results:
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
            try:
                items = _extract_component_list(section_crop, hname)
            except Exception:
                items = []
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
                return _classify_region(crop, paddle_type, section)
            except Exception:
                if attempt == 2:
                    return {"type": "other", "label": "Unknown Region", "description": ""}
                time.sleep(1 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=8) as executor:
        classifications = list(executor.map(classify, work_items))

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
    # block_elements share a bbox (per-item indexing of component lists) — skip dedup for them.
    return _deduplicate(heading_elements + elements) + block_elements, last_section, effective_source_type
