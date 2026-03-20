"""
Test _detect_source_type_override on a range of ityofd pages.
Usage: python test_expansion.py <start_page> <end_page>
Pages are 1-indexed (matching page_number in the DB).
"""
import sys
from pathlib import Path

# Load .env before importing ingest
from dotenv import load_dotenv
load_dotenv(Path("/Users/chad/personal/bgr/.env"))

sys.path.insert(0, str(Path(__file__).parent / "backend"))

import numpy as np
from PIL import Image
from ingest import _detect_source_type_override, _get_paddle

start = int(sys.argv[1]) if len(sys.argv) > 1 else 9
end = int(sys.argv[2]) if len(sys.argv) > 2 else 12

images_dir = Path(__file__).parent / "backend/data/images"
engine = _get_paddle()

for page_num in range(start, end + 1):
    img_path = images_dir / f"ityofd_page{page_num}.png"
    if not img_path.exists():
        print(f"Page {page_num}: image not found at {img_path}")
        continue
    img = Image.open(img_path).convert("RGB")
    img_bgr = np.array(img)[:, :, ::-1]
    det_results = engine.predict(img_bgr)
    boxes = det_results[0]["boxes"] if det_results else []
    labels = [b["label"] for b in boxes]
    result = _detect_source_type_override(img, boxes)
    print(f"Page {page_num}: detected_source={result!r}  boxes={labels}")
