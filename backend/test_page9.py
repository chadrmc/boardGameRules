"""Quick test: run extract_elements on page 5 of f_rulebook.pdf and print results."""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import fitz
import ingest
from PIL import Image
import io
import numpy as np

PDF = Path(__file__).parent.parent / "examples/f_rulebook.pdf"
PAGE_NUMBER = 5
PDF_RENDER_SCALE = 2.0

doc = fitz.open(str(PDF))
page = doc[PAGE_NUMBER - 1]
mat = fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)
pix = page.get_pixmap(matrix=mat, alpha=False)
image_bytes = pix.tobytes("png")

split_x = ingest.detect_page_split(image_bytes)
print(f"detect_page_split → split_x={split_x}")

# Save a few context crops for inspection
def save_crops(half_bytes, suffix, n=6):
    img = Image.open(io.BytesIO(half_bytes)).convert("RGB")
    w, h = img.size
    img_bgr = np.array(img)[:, :, ::-1]
    with ingest._paddle_lock:
        det = ingest._get_paddle().predict(img_bgr)
    boxes = det[0]["boxes"] if det else []
    boxes = ingest._split_wide_images(boxes)
    sorted_boxes = sorted(boxes, key=lambda b: b["coordinate"][1])
    valid = [b for b in sorted_boxes
             if (int(b["coordinate"][2]) - int(b["coordinate"][0])) >= ingest.MIN_REGION_W_PX
             and (int(b["coordinate"][3]) - int(b["coordinate"][1])) >= ingest.MIN_REGION_H_PX]
    IMAGE_LABELS = {"image", "figure", "picture"}
    prev_box = None
    for i, box in enumerate(valid[:n]):
        is_image = box.get("label", "").lower() in IMAGE_LABELS
        ctx = (ingest._find_text_context(valid, box) if is_image
               else ingest._find_image_context(valid, box))
        crop = ingest._build_context_crop(img, ctx, box)
        out = Path(f"/tmp/crop_{suffix}_{i}_{box.get('label','?')}.png")
        crop.save(out)
        print(f"  saved {out} ({crop.width}x{crop.height})")
        prev_box = box

if split_x:
    left_bytes, right_bytes = ingest.split_image(image_bytes, split_x)
    halves = [("L", left_bytes), ("R", right_bytes)]
else:
    halves = [("", image_bytes)]

import os
os.makedirs("debug_crops", exist_ok=True)

# Patch _classify_region to save crops before sending
original_classify = ingest._classify_region
crop_index = [0]
def saving_classify(crop, paddle_type, current_section=""):
    crop.save(f"debug_crops/{crop_index[0]:03d}_{paddle_type}.png")
    crop_index[0] += 1
    return original_classify(crop, paddle_type, current_section)
ingest._classify_region = saving_classify

all_elements = []
for suffix, half_bytes in halves:
    label = f"page {PAGE_NUMBER}{suffix}"
    print(f"\nSaving crops for {label}...")
    save_crops(half_bytes, f"{PAGE_NUMBER}{suffix}")
    print(f"\nRunning extract_elements on {label}...")
    elements, _ = ingest.extract_elements(
        image_bytes=half_bytes,
        rulebook_id="fr",
        source_type="core",
        page_number=PAGE_NUMBER,
        image_path=f"/images/fr_page{PAGE_NUMBER}{suffix}.png",
    )
    all_elements.extend(elements)
    print(f"  {len(elements)} elements")

print(f"\n{len(all_elements)} total elements:\n")
for e in all_elements:
    print(f"  [{e.type:12s}] {e.label}")
    print(f"             bbox: x={e.bbox.x:.2f} y={e.bbox.y:.2f} w={e.bbox.w:.2f} h={e.bbox.h:.2f}")
    print(f"             desc: {e.description[:120]}")
    print()
