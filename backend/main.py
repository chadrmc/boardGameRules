import asyncio
import fitz
import json
import os
import anthropic
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from models import ResultSet, SourceType
import ingest as ingest_module
import storage
import rerank as rerank_module

DATA_DIR = Path(os.environ.get("BGR_DATA_DIR", "./data"))
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

PDF_RENDER_SCALE = 2.0  # 2× scale → ~1680px wide for a typical rulebook page

app = FastAPI(title="Board Game Rulebook Search")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.post("/rulebooks/{rulebook_id}")
async def ingest_rulebook(
    rulebook_id: str,
    file: UploadFile = File(...),
    source_type: SourceType = "core",
    game_name: str | None = None,
):
    raw_bytes = await file.read()
    filename_lower = (file.filename or "").lower()
    is_html = filename_lower.endswith(".html") or filename_lower.endswith(".htm")

    if is_html:
        async def generate_html():
            nonlocal game_name
            html = raw_bytes.decode("utf-8", errors="replace")
            if game_name is None:
                game_name = rulebook_id
            storage.register_rulebook(rulebook_id, game_name)

            elements = await asyncio.to_thread(
                ingest_module.extract_elements_html,
                html=html,
                rulebook_id=rulebook_id,
                source_type=source_type,
            )
            # Delete old elements only after new ones are ready
            storage.delete_rulebook_elements(rulebook_id, source_type)
            storage.add_elements(elements)
            yield f"data: {json.dumps({'done': True, 'rulebook_id': rulebook_id, 'game_name': game_name, 'pages_processed': 1, 'elements_found': len(elements)})}\n\n"

        return StreamingResponse(generate_html(), media_type="text/event-stream")

    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
    except Exception:
        raise HTTPException(400, "File must be a valid PDF or HTML")

    mat = fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)
    total_pages = doc.page_count

    async def generate():
        nonlocal game_name
        all_elements = []
        logical_page = 0  # increments per logical page (each half of a 2-up counts as one)
        last_section = ""  # threads the active section across page boundaries
        last_detected_source = ""  # threads expansion/variant detection across page boundaries

        use_text_path = source_type in ("faq", "errata")

        # --- Layout calibration + document profile ---
        layout_stats: dict | None = None
        document_profile: dict | None = None
        cached_split_results: dict[int, tuple] = {}  # pdf_page_num → (split_x, boxes, img_bytes)
        if not use_text_path and total_pages > 0:
            # Check for a stored profile from a previous ingest of this rulebook
            stored_profile = storage.get_document_profile(rulebook_id)

            # Pick sample indices (1-based): page 1, n//3, 2n//3
            if total_pages <= 3:
                sample_page_nums = list(range(1, total_pages + 1))
            else:
                sample_page_nums = [1, total_pages // 3, 2 * total_pages // 3]

            sampled_boxes: list[list[dict]] = []
            sampled_widths: list[int] = []
            sample_images_for_profile: list[bytes] = []  # page images for Haiku profiling

            for spn in sample_page_nums:
                page = doc[spn - 1]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")

                split_x, full_boxes = await asyncio.to_thread(
                    ingest_module.detect_page_split, img_bytes,
                )
                cached_split_results[spn] = (split_x, full_boxes, img_bytes)
                sample_images_for_profile.append(img_bytes)

                if split_x is not None and full_boxes:
                    # 2-up: use remapped half-boxes so width ratios match logical pages
                    for half in ("left", "right"):
                        half_boxes = ingest_module._remap_boxes_to_half(full_boxes, split_x, half)
                        half_w = split_x if half == "left" else (
                            int(pix.width) - split_x
                        )
                        sampled_boxes.append(half_boxes)
                        sampled_widths.append(half_w)
                elif full_boxes:
                    sampled_boxes.append(full_boxes)
                    sampled_widths.append(int(pix.width))

            layout_stats = ingest_module._compute_layout_stats(sampled_boxes, sampled_widths)

            # Phase B: Claude document profile (reuse stored if available)
            if stored_profile:
                document_profile = stored_profile
            else:
                # Render early pages (1–5) for icon legend extraction, reusing cached where available
                early_page_nums = list(range(1, min(5, total_pages) + 1))
                early_page_images: list[bytes] = []
                for epn in early_page_nums:
                    if epn in cached_split_results:
                        _, _, ep_bytes = cached_split_results[epn]
                        early_page_images.append(ep_bytes)
                    else:
                        ep_page = doc[epn - 1]
                        ep_pix = ep_page.get_pixmap(matrix=mat, alpha=False)
                        ep_bytes = ep_pix.tobytes("png")
                        early_page_images.append(ep_bytes)

                document_profile = await asyncio.to_thread(
                    ingest_module._build_document_profile,
                    sample_images_for_profile,
                    layout_stats,
                    early_page_images,
                )
                if document_profile:
                    storage.save_document_profile(rulebook_id, document_profile)

        for pdf_page_num, page in enumerate(doc, start=1):
            # Reuse rendered image from calibration cache if available
            if pdf_page_num in cached_split_results:
                _, _, image_bytes = cached_split_results[pdf_page_num]
            else:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                image_bytes = pix.tobytes("png")

            if pdf_page_num == 1:
                if game_name is None:
                    game_name = await asyncio.to_thread(ingest_module.detect_game_name, image_bytes) or rulebook_id
                storage.register_rulebook(rulebook_id, game_name)

            page_elements = []

            if use_text_path:
                logical_page += 1
                filename = f"{rulebook_id}_{source_type}_page{logical_page}.png"
                (IMAGES_DIR / filename).write_bytes(image_bytes)

                fitz_blocks = page.get_text("blocks")
                text_blocks = [
                    {"text": b[4], "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]}
                    for b in fitz_blocks if b[6] == 0  # text blocks only
                ]
                elements, last_section = await asyncio.to_thread(
                    ingest_module.extract_elements_text,
                    text_blocks=text_blocks,
                    page_w=page.rect.width,
                    page_h=page.rect.height,
                    rulebook_id=rulebook_id,
                    source_type=source_type,
                    page_number=logical_page,
                    image_path=f"/images/{filename}",
                    initial_section=last_section,
                )
                all_elements.extend(elements)
                page_elements.extend(elements)
            else:
                # Reuse detect_page_split results from calibration cache if available
                if pdf_page_num in cached_split_results:
                    split_x, full_boxes, _ = cached_split_results[pdf_page_num]
                else:
                    split_x, full_boxes = await asyncio.to_thread(ingest_module.detect_page_split, image_bytes)

                if split_x is not None:
                    halves = await asyncio.to_thread(ingest_module.split_image, image_bytes, split_x)
                    # Remap full-image boxes to each half's coordinate system
                    half_boxes = [
                        ingest_module._remap_boxes_to_half(full_boxes, split_x, "left"),
                        ingest_module._remap_boxes_to_half(full_boxes, split_x, "right"),
                    ]
                else:
                    halves = (image_bytes,)
                    half_boxes = [full_boxes]  # reuse boxes from split detection if available

                for half_idx, half_bytes in enumerate(halves):
                    logical_page += 1
                    filename = f"{rulebook_id}_{source_type}_page{logical_page}.png"
                    (IMAGES_DIR / filename).write_bytes(half_bytes)

                    elements, last_section, detected_source = await asyncio.to_thread(
                        ingest_module.extract_elements,
                        image_bytes=half_bytes,
                        rulebook_id=rulebook_id,
                        source_type=source_type,
                        page_number=logical_page,
                        image_path=f"/images/{filename}",
                        initial_section=last_section,
                        initial_source_type=last_detected_source,
                        precomputed_boxes=half_boxes[half_idx],
                        layout_stats=layout_stats,
                        document_profile=document_profile,
                    )
                    if detected_source:
                        last_detected_source = detected_source
                    all_elements.extend(elements)
                    page_elements.extend(elements)

            yield f"data: {json.dumps({'page': pdf_page_num, 'total': total_pages, 'elements': len(page_elements)})}\n\n"

        # Delete old elements only after all pages processed successfully, then bulk add
        storage.delete_rulebook_elements(rulebook_id, source_type)
        storage.add_elements(all_elements)

        yield f"data: {json.dumps({'done': True, 'rulebook_id': rulebook_id, 'game_name': game_name, 'pages_processed': logical_page, 'elements_found': len(all_elements)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/ask")
async def ask_question(
    q: str = Query(..., min_length=1),
    rulebook_id: str = Query(...),
    n: int = Query(default=5, ge=1, le=20),
):
    import queue, threading

    # Phase 1: core rules
    candidates = storage.search_elements(
        query=q, n_results=max(n * 3, 15), rulebook_id=rulebook_id,
        source_types=["core", "expansion", "variant"],
    )
    results = rerank_module.rerank(query=q, results=candidates)[:n]

    # Attach errata by page number
    page_numbers = [r.element.page_number for r in results]
    errata_by_page = storage.get_errata_for_pages(rulebook_id, page_numbers)
    for r in results:
        r.errata = errata_by_page.get(r.element.page_number, [])

    # Phase 2: FAQ/errata — find candidates then link each to the most similar core result
    faq_candidates = storage.search_elements(
        query=q, n_results=5, rulebook_id=rulebook_id,
        source_types=["faq", "errata"],
    )
    faq_candidates = [f for f in faq_candidates if f.score >= 0.35]
    if faq_candidates and results:
        result_by_page = {r.element.page_number: r for r in results}
        result_page_numbers = list(result_by_page.keys())
        for faq in faq_candidates:
            match = storage.search_elements(
                query=faq.element.description,
                n_results=1,
                rulebook_id=rulebook_id,
                source_types=["core", "expansion", "variant"],
                page_numbers=result_page_numbers,
            )
            target = result_by_page.get(match[0].element.page_number) if match else results[0]
            target.faq.append(faq.element)

    game_name = next((r["name"] for r in storage.list_rulebooks() if r["id"] == rulebook_id), rulebook_id)

    result_set = ResultSet(results)
    context = result_set.context
    system = (
        f"You are a rules assistant for the board game {game_name}. "
        "Answer the player's question using ONLY the rulebook excerpts provided. "
        "Do not use any prior knowledge about this game. "
        "If the answer isn't clearly supported by the excerpts, say so briefly. "
        "Be concise and direct. Cite section name or page number when helpful."
    )
    user_msg = f"Rulebook excerpts:\n{context}\n\nQuestion: {q}"

    ac = anthropic.Anthropic()

    async def generate():
        token_queue: queue.Queue[str | None] = queue.Queue()

        def run():
            with ac.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                for text in stream.text_stream:
                    token_queue.put(text)
            token_queue.put(None)

        # 1. Emit results immediately before streaming starts
        results_data = result_set.ux
        yield f"data: {json.dumps({'type': 'results', 'results': results_data})}\n\n"

        # 2. Start streaming thread and emit tokens
        t = threading.Thread(target=run, daemon=True)
        t.start()

        while True:
            token = await asyncio.to_thread(token_queue.get)
            if token is None:
                break
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        t.join()

        # 3. Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/rulebooks")
async def list_rulebooks():
    return {"rulebooks": storage.list_rulebooks()}  # [{id, name}, ...]


@app.get("/rulebooks/{rulebook_id}/page_count")
async def get_page_count(rulebook_id: str):
    return {"page_count": storage.get_page_count(rulebook_id)}


@app.get("/elements")
async def get_page_elements(rulebook_id: str, page_number: int):
    elements = storage.get_page_elements(rulebook_id, page_number)
    return {"elements": [e.model_dump() for e in elements]}


@app.get("/health")
async def health():
    return {"status": "ok"}
