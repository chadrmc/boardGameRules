"""
One-time script to capture PaddleOCR LayoutDetection output from ACBoV sample pages.

Renders pages 1, 10, 19 of examples/AC.pdf at 2x scale, runs LayoutDetection,
and saves raw box dicts + page dimensions as JSON fixtures for unit tests.

Usage:
    cd /Users/chad/personal/bgr
    source backend/.venv/bin/activate
    python tests/capture_fixtures.py
"""
import json
import os
import sys
from pathlib import Path

# Load .env before any paddle imports
from dotenv import load_dotenv
load_dotenv(Path("/Users/chad/personal/bgr/.env"))

import fitz
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PDF_PATH = PROJECT_ROOT / "examples" / "AC.pdf"
FIXTURES_DIR = SCRIPT_DIR / "fixtures"
SAMPLE_PAGES = [1, 10, 19]  # 1-based page numbers
PDF_RENDER_SCALE = 2


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    # Lazy import — requires paddleocr in the venv
    from paddleocr import LayoutDetection
    engine = LayoutDetection()

    doc = fitz.open(str(PDF_PATH))
    mat = fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)

    page_dims = {}

    for page_num in SAMPLE_PAGES:
        if page_num > doc.page_count:
            print(f"WARNING: Page {page_num} exceeds PDF page count ({doc.page_count}), skipping")
            continue

        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to PIL → numpy BGR for PaddleOCR
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img_bgr = np.array(img)[:, :, ::-1]

        print(f"Running LayoutDetection on page {page_num} ({pix.width}x{pix.height})...")
        det_results = engine.predict(img_bgr)
        boxes = det_results[0]["boxes"] if det_results else []

        # Save boxes — convert numpy types to native Python for JSON serialization
        clean_boxes = []
        for b in boxes:
            clean_boxes.append({
                "label": str(b["label"]),
                "coordinate": [float(v) for v in b["coordinate"]],
                "score": float(b["score"]),
            })

        boxes_path = FIXTURES_DIR / f"acbov_page{page_num}_boxes.json"
        boxes_path.write_text(json.dumps(clean_boxes, indent=2))
        print(f"  Saved {len(clean_boxes)} boxes to {boxes_path.name}")

        page_dims[str(page_num)] = {"width": pix.width, "height": pix.height}

    dims_path = FIXTURES_DIR / "acbov_page_dims.json"
    dims_path.write_text(json.dumps(page_dims, indent=2))
    print(f"Saved page dimensions to {dims_path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
