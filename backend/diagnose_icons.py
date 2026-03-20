"""Diagnostic script: detect inline icons in a PDF via PyMuPDF structure.

Checks three icon embedding methods:
  A. XObject/inline images (get_image_info)
  B. PUA font glyphs (rawdict character scan)
  C. Text gaps suggesting inline non-text content

Usage:
  python diagnose_icons.py [path/to/file.pdf]
"""
import sys
import statistics
from collections import Counter
from pathlib import Path

import fitz

DEFAULT_PDF = str(Path(__file__).parent.parent / "examples" / "bb.pdf")
MAX_PAGES = 5


def analyze_page(page, page_num):
    """Analyze one page for icon indicators. Returns a results dict."""
    results = {
        "page": page_num,
        "xobject_images": [],
        "pua_chars": [],
        "text_gaps": [],
    }

    # --- A. Embedded images ---
    for img in page.get_image_info(xrefs=True):
        results["xobject_images"].append({
            "bbox": tuple(round(v, 1) for v in img["bbox"]),
            "size": (img["width"], img["height"]),
            "digest": img.get("digest"),
        })

    # --- B & C. Scan rawdict for PUA chars and text gaps ---
    rd = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_IMAGES)

    for block in rd.get("blocks", []):
        if block.get("type") != 0:  # text blocks only
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_name = span.get("font", "")
                chars = span.get("chars", [])
                if not chars:
                    continue

                # B. PUA character detection
                for ch in chars:
                    cp = ord(ch["c"]) if ch["c"] else 0
                    if 0xE000 <= cp <= 0xF8FF:
                        results["pua_chars"].append({
                            "char": ch["c"],
                            "codepoint": f"U+{cp:04X}",
                            "font": font_name,
                            "bbox": tuple(round(v, 1) for v in ch["bbox"]),
                        })

                # C. Text gap detection
                if len(chars) < 3:
                    continue
                # Compute inter-character x-gaps
                gaps = []
                for i in range(1, len(chars)):
                    prev_right = chars[i - 1]["bbox"][2]
                    curr_left = chars[i]["bbox"][0]
                    gaps.append(curr_left - prev_right)

                positive_gaps = [g for g in gaps if g > 0.5]
                if len(positive_gaps) < 2:
                    continue
                median_gap = statistics.median(positive_gaps)
                threshold = max(median_gap * 2.5, 5.0)  # at least 5pt to avoid false positives

                for i, gap in enumerate(gaps):
                    if gap > threshold:
                        prev_char = chars[i]
                        next_char = chars[i + 1]
                        results["text_gaps"].append({
                            "gap_width": round(gap, 1),
                            "median_gap": round(median_gap, 1),
                            "between": f"'{prev_char['c']}' and '{next_char['c']}'",
                            "x_pos": round(prev_char["bbox"][2], 1),
                            "y_pos": round(prev_char["bbox"][1], 1),
                            "font": font_name,
                        })

    return results


def analyze_fonts(doc, max_pages):
    """Check all fonts in the first N pages for PUA codepoints."""
    font_info = {}
    seen_xrefs = set()

    for page_num in range(min(max_pages, doc.page_count)):
        page = doc[page_num]
        for xref, _ext, _type, _basefont, name, _encoding in page.get_fonts():
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                _name, _ext2, _subtype, content = doc.extract_font(xref)
                if not content:
                    continue
                font = fitz.Font(fontbuffer=content)
                codepoints = font.valid_codepoints()
                pua_cps = [cp for cp in codepoints if 0xE000 <= cp <= 0xF8FF]
                if pua_cps:
                    glyph_names = {}
                    for cp in pua_cps[:20]:  # sample up to 20
                        gn = font.unicode_to_glyph_name(cp)
                        glyph_names[f"U+{cp:04X}"] = gn
                    font_info[name] = {
                        "xref": xref,
                        "total_glyphs": len(codepoints),
                        "pua_count": len(pua_cps),
                        "sample_glyph_names": glyph_names,
                    }
            except Exception as e:
                font_info[name] = {"error": str(e)}

    return font_info


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    print(f"Analyzing: {pdf_path}")
    print(f"Pages to scan: first {MAX_PAGES}\n")

    doc = fitz.open(pdf_path)
    pages_to_scan = min(MAX_PAGES, doc.page_count)

    all_results = []
    all_digests = Counter()

    for i in range(pages_to_scan):
        page = doc[i]
        res = analyze_page(page, i + 1)
        all_results.append(res)
        for img in res["xobject_images"]:
            if img["digest"]:
                all_digests[img["digest"]] += 1

    # --- Summary table ---
    print("=" * 70)
    print(f"{'Page':>4}  {'XObj Images':>11}  {'PUA Chars':>9}  {'Text Gaps':>9}")
    print("-" * 70)
    for res in all_results:
        print(f"{res['page']:>4}  {len(res['xobject_images']):>11}  "
              f"{len(res['pua_chars']):>9}  {len(res['text_gaps']):>9}")
    print("=" * 70)

    # --- Details per page ---
    for res in all_results:
        print(f"\n--- Page {res['page']} ---")

        if res["xobject_images"]:
            print(f"  XObject images ({len(res['xobject_images'])}):")
            for img in res["xobject_images"][:10]:
                print(f"    {img['size'][0]}×{img['size'][1]}  bbox={img['bbox']}  "
                      f"digest={img['digest'][:12] if img['digest'] else 'none'}...")
            if len(res["xobject_images"]) > 10:
                print(f"    ... and {len(res['xobject_images']) - 10} more")

        if res["pua_chars"]:
            print(f"  PUA characters ({len(res['pua_chars'])}):")
            for ch in res["pua_chars"][:10]:
                print(f"    {ch['codepoint']}  font={ch['font']}  bbox={ch['bbox']}")
            if len(res["pua_chars"]) > 10:
                print(f"    ... and {len(res['pua_chars']) - 10} more")

        if res["text_gaps"]:
            print(f"  Text gaps ({len(res['text_gaps'])}):")
            for gap in res["text_gaps"][:10]:
                print(f"    gap={gap['gap_width']}pt (median={gap['median_gap']}pt)  "
                      f"between {gap['between']}  at x={gap['x_pos']} y={gap['y_pos']}  "
                      f"font={gap['font']}")
            if len(res["text_gaps"]) > 10:
                print(f"    ... and {len(res['text_gaps']) - 10} more")

        if not any([res["xobject_images"], res["pua_chars"], res["text_gaps"]]):
            print("  (no icon indicators found)")

    # --- Font analysis ---
    print("\n" + "=" * 70)
    print("FONT ANALYSIS (PUA codepoints)")
    print("=" * 70)
    font_info = analyze_fonts(doc, pages_to_scan)
    if font_info:
        for name, info in font_info.items():
            if "error" in info:
                print(f"  {name}: ERROR - {info['error']}")
            else:
                print(f"  {name}: {info['pua_count']} PUA glyphs "
                      f"(of {info['total_glyphs']} total)")
                for cp, gn in info["sample_glyph_names"].items():
                    print(f"    {cp} → {gn}")
    else:
        print("  No fonts with PUA codepoints found.")

    # --- Image digest frequency ---
    print("\n" + "=" * 70)
    print("IMAGE DIGEST FREQUENCY (repeated = likely icons)")
    print("=" * 70)
    if all_digests:
        for digest, count in all_digests.most_common(20):
            label = "REPEATED" if count > 1 else "unique"
            print(f"  {digest[:16]}...  ×{count}  [{label}]")
        unique = sum(1 for c in all_digests.values() if c == 1)
        repeated = sum(1 for c in all_digests.values() if c > 1)
        print(f"\n  Total unique digests: {len(all_digests)} "
              f"({repeated} repeated, {unique} unique)")
    else:
        print("  No images found.")

    doc.close()


if __name__ == "__main__":
    main()
