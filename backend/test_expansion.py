"""Test expansion/variant detection on ityofd pages 8+"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import fitz
import ingest

PDF_PATH = Path(__file__).parent.parent / "examples" / "ityofd.pdf"
if not PDF_PATH.exists():
    print(f"PDF not found: {PDF_PATH}")
    sys.exit(1)

doc = fitz.open(str(PDF_PATH))
mat = fitz.Matrix(2.0, 2.0)

start_page = int(sys.argv[1]) if len(sys.argv) > 1 else 8
end_page = int(sys.argv[2]) if len(sys.argv) > 2 else min(start_page + 3, doc.page_count)

last_section = ""
last_detected_source = ""

for pdf_page_num in range(start_page, end_page + 1):
    page = doc[pdf_page_num - 1]
    pix = page.get_pixmap(matrix=mat, alpha=False)
    image_bytes = pix.tobytes("png")

    elements, last_section, detected_source = ingest.extract_elements(
        image_bytes=image_bytes,
        rulebook_id="ityofd",
        source_type="core",
        page_number=pdf_page_num,
        image_path=f"/images/ityofd_page{pdf_page_num}.png",
        initial_section=last_section,
        initial_source_type=last_detected_source,
    )
    if detected_source:
        last_detected_source = detected_source

    print(f"\n=== PDF page {pdf_page_num} | detected_source={detected_source!r} | effective={last_detected_source!r} ===")
    for e in elements:
        print(f"  [{e.source_type}] {e.type} | {e.label}")
