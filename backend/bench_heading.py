"""
Side-by-side comparison: Haiku vs PaddleOCR TextDetection+TextRecognition for heading extraction.

Usage:
    python bench_heading.py <pdf_path> [page_numbers...]

Example:
    python bench_heading.py ../examples/log.pdf 2 3 4
    python bench_heading.py ../examples/log.pdf   # uses pages 1-5
"""
import io
import sys
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("/Users/chad/personal/bgr/.env"))

import fitz
from PIL import Image
from paddleocr import LayoutDetection, TextRecognition, TextDetection
import anthropic

PDF_RENDER_SCALE = 2.0
HEADING_LABELS = {"paragraph_title", "doc_title"}
MAX_HEADING_W_FRAC = 0.25

client = anthropic.Anthropic()
layout_engine = LayoutDetection()
rec_engine = TextRecognition()
det_engine = TextDetection()


def render_page(pdf_path: str, page_num: int) -> bytes:
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)
    pix = doc[page_num - 1].get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def get_heading_boxes(img: Image.Image) -> list[dict]:
    w, h = img.size
    img_bgr = np.array(img)[:, :, ::-1]
    det = layout_engine.predict(img_bgr)
    boxes = det[0]["boxes"] if det else []
    return [
        b for b in boxes
        if b["label"].lower() in HEADING_LABELS
        and (
            b["label"].lower() == "doc_title"
            or (int(b["coordinate"][2]) - int(b["coordinate"][0])) <= w * MAX_HEADING_W_FRAC
        )
    ]


def extract_with_haiku(img: Image.Image, box: dict) -> tuple[str, float]:
    import base64
    x0, y0, x1, y1 = [int(v) for v in box["coordinate"]]
    crop = img.crop((x0, y0, x1, y1))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()

    t0 = time.perf_counter()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": "What text is shown in this heading? Reply with just the heading text, nothing else."},
        ]}],
    )
    elapsed = time.perf_counter() - t0
    return response.content[0].text.strip(), elapsed


def extract_with_ocr(img: Image.Image, box: dict) -> tuple[str, float]:
    """TextDetection finds text start x (skips icon), then single TextRecognition on trimmed crop."""
    x0, y0, x1, y1 = [int(v) for v in box["coordinate"]]
    crop = img.crop((x0, y0, x1, y1))
    crop_bgr = np.array(crop)[:, :, ::-1]

    t0 = time.perf_counter()
    det_result = det_engine.predict(crop_bgr)
    dt_polys = det_result[0]["dt_polys"] if det_result else []
    if len(dt_polys) == 0:
        elapsed = time.perf_counter() - t0
        return "", elapsed
    text_start_x = int(min(pt[0] for poly in dt_polys for pt in poly))
    trimmed_bgr = crop_bgr[:, text_start_x:]
    if trimmed_bgr.size == 0:
        elapsed = time.perf_counter() - t0
        return "", elapsed
    r = rec_engine.predict(trimmed_bgr)
    elapsed = time.perf_counter() - t0
    text = r[0]["rec_text"].strip() if r else ""
    return text, elapsed


def run_comparison(pdf_path: str, page_nums: list[int]):
    print(f"\nPDF: {pdf_path}")
    print(f"Pages: {page_nums}\n")
    print(f"{'Page':<6} {'Box':<5} {'Label':<18} {'Haiku result':<35} {'ms':>6}  {'Det+Rec result':<35} {'ms':>6}  Match")
    print("-" * 130)

    totals = {"haiku": 0.0, "ocr": 0.0, "boxes": 0}

    for page_num in page_nums:
        image_bytes = render_page(pdf_path, page_num)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        boxes = get_heading_boxes(img)

        if not boxes:
            print(f"{page_num:<6} {'—':<5} (no heading boxes found)")
            continue

        boxes.sort(key=lambda b: b["coordinate"][1])
        for i, box in enumerate(boxes):
            haiku_text, haiku_ms = extract_with_haiku(img, box)
            ocr_text, ocr_ms = extract_with_ocr(img, box)
            match = "✓" if haiku_text.strip().lower() == ocr_text.strip().lower() else "~"
            label = box["label"]
            print(
                f"{page_num:<6} {i+1:<5} {label:<18} "
                f"{haiku_text[:33]:<35} {haiku_ms*1000:>6.0f}  "
                f"{ocr_text[:33]:<35} {ocr_ms*1000:>6.0f}  {match}"
            )
            totals["haiku"] += haiku_ms
            totals["ocr"] += ocr_ms
            totals["boxes"] += 1

    if totals["boxes"]:
        print("-" * 130)
        n = totals["boxes"]
        print(
            f"{'TOTAL':<6} {n} boxes  "
            f"{'Haiku avg:':<24} {totals['haiku']/n*1000:>6.0f}ms total={totals['haiku']*1000:.0f}ms  "
            f"{'OCR avg:':<24} {totals['ocr']/n*1000:>6.0f}ms total={totals['ocr']*1000:.0f}ms  "
            f"Speedup: {totals['haiku']/totals['ocr']:.1f}×"
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python bench_heading.py <pdf_path> [page_numbers...]")
        sys.exit(1)
    pdf_path = args[0]
    page_nums = [int(p) for p in args[1:]] if len(args) > 1 else list(range(1, 6))
    run_comparison(pdf_path, page_nums)
