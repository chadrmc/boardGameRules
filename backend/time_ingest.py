"""Standalone timing script for ingest.py — bypasses uvicorn/SSE."""
import sys
import time
from pathlib import Path

# Load .env before importing ingest (needs ANTHROPIC_API_KEY)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import fitz
import ingest as ingest_module

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else "../examples/f_rulebook.pdf"
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 3  # default: first 3 pages
PDF_RENDER_SCALE = 2.0

doc = fitz.open(PDF_PATH)
mat = fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)
total = min(doc.page_count, MAX_PAGES)

print(f"Timing {total} pages from {PDF_PATH} ({doc.page_count} total)\n")

last_section = ""
last_detected_source = ""
overall_t0 = time.perf_counter()

for i in range(total):
    page = doc[i]
    logical_page = i + 1

    t0 = time.perf_counter()
    pix = page.get_pixmap(matrix=mat, alpha=False)
    image_bytes = pix.tobytes("png")
    render_time = time.perf_counter() - t0

    print(f"--- Page {logical_page} (render: {render_time:.2f}s) ---")

    # Check for 2-up split (now returns boxes too)
    t0 = time.perf_counter()
    split_x, full_boxes = ingest_module.detect_page_split(image_bytes)
    split_time = time.perf_counter() - t0
    if split_x:
        print(f"  [timing] split detection: {split_time:.2f}s (split at x={split_x}, {len(full_boxes)} boxes)")
    else:
        print(f"  [timing] split detection: {split_time:.2f}s (no split)")

    if split_x:
        halves = ingest_module.split_image(image_bytes, split_x)
        half_boxes = [
            ingest_module._remap_boxes_to_half(full_boxes, split_x, "left"),
            ingest_module._remap_boxes_to_half(full_boxes, split_x, "right"),
        ]
    else:
        halves = (image_bytes,)
        half_boxes = [full_boxes]  # reuse boxes from split detection if available

    for half_idx, half_bytes in enumerate(halves):
        if len(halves) > 1:
            print(f"  -- Half {half_idx + 1} ({len(half_boxes[half_idx])} remapped boxes) --")

        t0 = time.perf_counter()
        elements, last_section, detected_source = ingest_module.extract_elements(
            image_bytes=half_bytes,
            rulebook_id="timing_test",
            source_type="core",
            page_number=logical_page,
            image_path=f"/images/timing_test_page{logical_page}.png",
            initial_section=last_section,
            initial_source_type=last_detected_source,
            precomputed_boxes=half_boxes[half_idx],
        )
        page_total = time.perf_counter() - t0

        if detected_source:
            last_detected_source = detected_source

        print(f"  => {len(elements)} elements, extract_elements total: {page_total:.2f}s")
    print()

overall = time.perf_counter() - overall_t0
print(f"=== Overall: {overall:.1f}s for {total} pages ===")
