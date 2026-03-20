# Board Game Rulebook Search — System Documentation

## 1. What It Does

Board Game Rulebook Search (bgr) lets you upload a board game rulebook (PDF or HTML) and then search it by natural-language question. The system extracts spatially-aware elements from each page using PaddleOCR layout detection and Claude Haiku classification, stores them in ChromaDB with their bounding boxes, and returns highlighted page image crops showing exactly where in the rulebook the answer comes from. A second Claude Haiku call streams a direct answer to the question, grounded only in the indexed rulebook content.

---

## 2. Architecture Overview

**Stack**
- Backend: Python 3.12, FastAPI, ChromaDB (vector DB), PaddleOCR 3.x (layout/OCR), Claude Haiku (classification, reranking, answering), PyMuPDF (PDF rendering)
- Frontend: Next.js 14, TypeScript, Tailwind CSS
- Storage: ChromaDB on disk (`./data/chroma`), page images as PNGs (`./data/images`), rulebook registry as JSON (`./data/rulebooks.json`)

**Data flow — ingestion**

```
User uploads PDF/HTML
        │
        ▼
POST /rulebooks/{id}    (FastAPI, SSE stream)
        │
        ├─ PDF path:
        │   PyMuPDF renders each page → PNG at 2× scale (~1680px wide)
        │   detect_page_split() → split landscape 2-up pages into halves
        │   extract_elements() per half:
        │       PaddleOCR LayoutDetection → bounding boxes with labels
        │       Phase 1: Claude Haiku reads heading boxes → section name map
        │       Phase 1.5: component-list sections → _extract_component_list()
        │       Phase 2: build work items (context crops)
        │       Phase 3: parallel Claude Haiku classifies each region
        │       → list[Element] with type, label, description, bbox
        │
        └─ HTML path (FAQ/errata):
            BeautifulSoup strips tags → plain text
            Single Claude Haiku call → list of Q&A pairs
            → list[Element] with display_mode="text"
        │
        ▼
ChromaDB: add_elements()
    document = "{label}: {description}"   (embedded by ChromaDB)
    metadata = page_number, bbox, type, source_type, image_path, ...
        │
        ▼
SSE progress events → frontend (page N of M, elements found)
```

**Data flow — search/ask**

```
User types question
        │
        ▼
GET /ask?q=...&rulebook_id=...    (FastAPI, SSE stream)
        │
        ├─ Phase 1: vector search (core/expansion/variant)
        │   ChromaDB cosine similarity → 3× candidate pool
        │   Claude Haiku reranker → ranked list
        │   Attach errata elements by page number
        │
        ├─ Phase 2: FAQ/errata search
        │   Vector search → link each FAQ to closest core result
        │
        ├─ SSE event: { type: "results", results: [...] }   ← sent immediately
        │
        └─ Claude Haiku streaming answer (grounded in indexed text)
            SSE events: { type: "token", text: "..." } × N
            SSE event:  { type: "done" }
```

---

## 3. Ingestion Pipeline

### PDF path (`extract_elements` in `backend/ingest.py`)

**Pre-processing and calibration (runs before the main page loop)**

Before iterating pages, `main.py` runs a pre-loop calibration block for PDF ingestion (skipped for FAQ/errata text paths):

1. Three sample pages (page 1, n//3, 2n//3) are rendered and `detect_page_split()` is called on each. Results are cached in `cached_split_results` to avoid re-rendering in the main loop.
2. `_compute_layout_stats(sampled_boxes, page_widths)` → `layout_stats`: derives calibrated thresholds from PaddleOCR box statistics across the sampled pages:
   - `heading_max_width_pct`: 90th percentile of heading box widths × 1.2, capped at 45%.
   - `estimated_columns`: 1 or 2, based on median text box width (< 0.4 → 2-column).
3. `_build_document_profile(sample_images, layout_stats, early_page_images)` → `document_profile`: sends the same sample page images to Claude Haiku with PROFILE_PROMPT and the measured stats. Returns a dict with:
   - `heading_max_width_pct`: visual override of the statistical threshold.
   - `sub_heading_pattern`: `"numbered_steps"`, `"bold_inline"`, or `"none"`.
   - `column_count`: 1, 2, or `"mixed"`.
   - `has_bold_callouts`: boolean.
   - `callout_description`, `layout_notes`: free text injected into REGION_PROMPT as style notes.
   - `icon_legend`: dict of `{"[icon_name]": "meaning"}` pairs extracted by `_extract_icon_legend()` from early pages (pages 1–5). Empty if no icon/symbol glossary is found.
   - The profile is saved to `rulebooks.json` via `storage.save_document_profile()`. On re-ingest, the stored profile is reused (skipping the Haiku call).

`_profile_to_style_notes(profile)` converts the profile into a string injected into `REGION_PROMPT` as `{style_notes}`, giving Claude context about icon vocabulary, callout visual style, and layout notes during Phase 3 classification.

The `document_profile` also drives two Phase 0 decisions in `extract_elements`:
- If `sub_heading_pattern == "numbered_steps"`, `paragraph_title` boxes matching numbered patterns (e.g. "1.") are treated as rule steps rather than section headings.
- `column_count` from the profile takes precedence over `layout_stats.estimated_columns` when deciding whether to apply `_split_wide_images`.

**Per-page pre-processing**

1. PyMuPDF renders each PDF page to a PNG at 2× scale (typically ~1680 px wide). If the page was in the calibration sample, the cached render is reused.
2. `detect_page_split()` checks whether the image is a 2-up (landscape, two logical pages side by side). Detection order:
   - If width < 1.2× height → single page, skip.
   - PaddleOCR `LayoutDetection` runs on the full image.
   - Look for `number`-labeled boxes in opposite halves of the bottom quarter → split at the midpoint of their inner edges.
   - Fallback: find the largest gap in merged x-coverage within the middle third of the image.
3. If 2-up detected, `split_image()` crops into left and right halves. `_remap_boxes_to_half()` remaps the full-image PaddleOCR boxes to each half's coordinate system — avoiding a second Paddle run on already-sampled pages. Each half is treated as an independent logical page. Source type (`expansion`/`variant`) detection is threaded across pages via `last_detected_source`.

**Phase 1 — Section headings (serial, parallelized per-box via ThreadPoolExecutor)**

PaddleOCR box labels `paragraph_title` and `doc_title` are candidates for section headings.

- `paragraph_title` boxes wider than 25% of page width are bold rule callouts, not section headings — excluded from Phase 1.
- `paragraph_title` boxes between 25–45% wide (Phase 1b) are checked: if the extracted text contains a block-section keyword (`component`, `components`, `contents`), they become headings.
- `doc_title` boxes are always treated as headings regardless of width.
- All heading candidates are extracted in parallel: `_extract_heading_text(crop)` sends a minimal prompt to Claude Haiku and returns the verbatim heading string.
- Headings are stored as `other`-type `Element` records with label `"{name} – Section Heading"` so the section name text is directly searchable.
- Section propagation: a heading covers everything below it on the page (full page width — no column scoping). The previous page's last section is threaded in as `initial_section` so cross-page sections work.

**Source type override detection (`_detect_source_type_override`)**

Three-tier scan to detect when a page begins an `expansion` or `variant` section:
- Tier 1 (cheap): `paragraph_title`/`doc_title` boxes → TextRecognition directly.
- Tier 2 (expensive): `header`/`image` boxes → TextDetection then TextRecognition.
- Tier 3 (fallback): top 15% page strip → TextDetection → TextRecognition. Skipped if heading boxes already found in top 15%.

A recognized string must start with `"expansion"` or `"variant"` to avoid false positives from body text. Returns `"expansion"`, `"variant"`, or `""`.

**Phase 1.5 — Block sections (component lists)**

For headings matching block-section keywords whose body boxes are collected:
- All valid boxes within the section's y-range are marked to skip in Phase 2.
- The entire section region (union bbox of heading + all boxes) is cropped and sent to `_extract_component_list()`.
- Claude Haiku returns one line per component item (quantity + name). Each item becomes a separate `component`-type `Element` sharing the section's bbox (intentionally not deduplicated).

**Phase 2 — Work item construction**

For each non-heading, non-block-section box:
- Image boxes (`image`, `figure`, `picture`): `_find_text_context()` finds the closest adjacent or below text box to stack as context.
- Text boxes with no known section: `_find_image_context()` finds a horizontally-aligned image box above to stack as context.
- Text boxes with a known section: no image context (avoids styled logos producing bad labels).
- `_build_context_crop()` combines the context box and current box with a 2 px grey separator into a single image crop sent to Claude.
- Wide image boxes spanning multiple columns are pre-split by `_split_wide_images()` into per-column strips aligned to the text column x-boundaries below them.

**Fragment filtering**

- `MIN_REGION_W_PX = 80`: boxes narrower than 80 px dropped (exemption: `paragraph_title` boxes, which can be narrow single-word headings).
- `MIN_REGION_H_PX = 20`: boxes shorter than 20 px dropped.
- `MIN_DESCRIPTION_CHARS = 20`: elements whose Claude-generated description is fewer than 20 characters are dropped after classification.

**Phase 3 — Parallel classification (8 workers)**

`_classify_region(crop, paddle_type, current_section, style_notes)` sends each work item's crop to Claude Haiku with the `REGION_PROMPT`. The prompt instructs Claude to return a JSON object with:
- `type`: one of `rule`, `note`, `illustration`, `example`, `diagram`, `table`, `component`, `other`.
- `label`: `"{Section} – {specific topic}"` — section name injected when known.
- `description`: verbatim or close paraphrase of the text (not a meta-summary).

`style_notes` (derived from the document profile via `_profile_to_style_notes`) is injected into the prompt when available, providing icon vocabulary, callout descriptions, and layout notes.

Any type not in `VALID_TYPES` is mapped to `"other"`. Three retries with back-off on Claude API errors.

**Deduplication**

`_deduplicate()` removes near-duplicate elements where two elements share a center y within 2% and center x within 10% of the page. Block elements (component list items) bypass deduplication because they intentionally share a bbox.

**Storage**

After all pages are processed successfully, old elements for that `rulebook_id` + `source_type` combination are deleted, and the new elements are bulk-added to ChromaDB. This means a re-ingest of a FAQ won't touch core elements.

`storage.save_document_profile(rulebook_id, profile)` and `storage.get_document_profile(rulebook_id)` persist the document profile as a `"profile"` key inside the rulebook's entry in `rulebooks.json`. `list_rulebooks()` strips the profile key — it returns only `{id, name}` to callers. `get_page_count(rulebook_id)` returns the max `page_number` across all elements in the collection, used by the PageBrowser to bound navigation.

### HTML path (`extract_elements_html` in `backend/ingest.py`)

For `.html`/`.htm` uploads:
1. BeautifulSoup strips script/style tags and extracts plain text.
2. A single Claude Haiku call extracts all Q&A pairs from the text as a JSON array.
3. Each pair becomes an `Element` with `display_mode="text"` and `page_image_path=""` (no page image).

### Text path — real PDF FAQ/errata (`extract_elements_text`)

For PDFs uploaded with `source_type` of `faq` or `errata`:
1. PyMuPDF `page.get_text("blocks")` extracts text blocks with coordinates.
2. Block text is joined and sent to Claude Haiku with the same Q&A extraction prompt used for HTML.
3. Results are `Element` records with `display_mode="text"`.

### Game name detection

On the first page of any PDF, if `game_name` was not passed as a query param, `detect_game_name()` runs PaddleOCR layout detection, picks the `doc_title` box (or largest box as fallback), and sends the crop to Claude Haiku to read the game title. The result is registered in `rulebooks.json`.

---

## 4. Search Flow

### `GET /ask?q=&rulebook_id=&n=`

Returns an SSE stream. All search is scoped to a single rulebook.

1. **Core/expansion/variant search**: `storage.search_elements()` queries ChromaDB with cosine similarity, retrieving `max(n*3, 15)` candidates filtered to `source_types=["core", "expansion", "variant"]`.

2. **Reranking** (`rerank.rerank()`): Claude Haiku receives the query and all candidate labels/descriptions. It returns a ranked index array based on query intent:
   - Queries like "example of X" → notes/illustrations ranked above rule text.
   - Queries like "how does X work" → rule element first.
   - Variant results are pushed to the bottom unless the query explicitly mentions a variant.
   - Falls back to original order if parsing fails.

3. **Errata attachment**: `get_errata_for_pages()` retrieves all errata elements matching the result page numbers and attaches them to the corresponding `SearchResult` objects.

4. **FAQ linking**: FAQ/errata candidates are vector-searched separately. Each FAQ is linked to the most similar core result (by re-querying the FAQ's description against the result page set). Linked FAQs are attached to the target `SearchResult`.

5. **SSE `results` event** is emitted immediately before the answer stream starts, so the frontend can show result cards while the answer is still generating.

6. **Answer streaming**: Claude Haiku is given a system prompt (`"Answer using ONLY the rulebook excerpts provided"`) and a user message containing numbered excerpts (labels, descriptions, attached errata/FAQ). Tokens are streamed via SSE `token` events. A `done` event signals completion.

### ChromaDB storage details

Each element is stored as:
- `document`: `"{label}: {description}"` — this is what ChromaDB embeds and queries against.
- `metadata` flat dict: `rulebook_id`, `source_type`, `page_number`, `display_mode`, `page_image_path`, `type`, `label`, `bbox_x/y/w/h`.

One ChromaDB collection per rulebook (`rulebook_{rulebook_id}`), using cosine distance (`hnsw:space: cosine`). Scores returned to the frontend are `1 - cosine_distance`.

---

## 5. Frontend Flow

### Rulebook selection

On load, `listRulebooks()` fetches `GET /rulebooks`. If only one rulebook is available, it is auto-selected. Rulebooks are shown as pill buttons; clicking one selects it and focuses the search bar.

When a rulebook is selected, a "Browse pages" button appears. Clicking it toggles the `PageBrowser` component inline (below the rulebook pills). Selecting a different rulebook collapses the browser.

### ConnectionStatus (`components/ConnectionStatus.tsx`)

Displays a colored dot + label in the page header. Polls `GET /health` every 10 seconds with a 3-second timeout. States: `checking` (gray pulsing), `connected` (green), `disconnected` (red). `API_BASE` is read from `lib/constants.ts`.

### PageBrowser (`components/PageBrowser.tsx`)

A standalone dev tool for auditing ingestion results page by page, accessible from the main UI without opening a search result.

- Fetches total page count via `getPageCount(rulebookId)` on mount.
- Fetches all elements for the current page via `getPageElements(rulebookId, page)` on page change.
- Renders the full page image via `HighlightedImage` with one colored highlight box per element. Colors come from `ELEMENT_COLORS` in `lib/constants.ts`.
- Clicking a highlight selects the element and opens an inspection panel below the image showing ID, type badge, source type, bbox percentages, and description.
- "Overlays" toggle button hides/shows all highlight boxes. Clicking a different rulebook pill collapses the browser.

### `lib/constants.ts`

Central location for two shared values:
- `API_BASE = "http://localhost:8000"` — replaces the inline string previously scattered across `api.ts` and component files.
- `ELEMENT_COLORS` — maps each element type to a hex color string. Used by both `PageBrowser` and `ResultCard` for consistent type coloring.

### Search

The search bar is disabled until a rulebook is selected. On submit, `ask()` in `api.ts` opens an SSE connection to `GET /ask`. The client processes SSE events in order:
- `results` event → `setResults()` and `setLoading(false)` (result cards appear).
- `token` events → appended to the answer string (streaming answer box).
- `done` event → `setAnswerLoading(false)`.

### Result grouping

`groupResults()` in `page.tsx` clusters results by rulebook + page number, then greedily merges results whose top edge is within `GAP_THRESHOLD = 0.06` (6% of page height) of the cluster's current bottom. Results on the same page within that gap appear in one card with a merged crop showing multiple highlight boxes. Groups are sorted by the best rerank position of any result in the group.

### Result cards (`ResultCard.tsx`)

Each card represents one group of spatially-proximate results from the same page.

- For `display_mode="image"` elements: an `ExcerptImage` component renders a CSS crop of the full page image to the union bounding box of all results in the group, with one colored highlight box per result.
- For `display_mode="text"` elements: the description text is shown directly.
- Each result in the group has a label row showing source badge (errata=red, faq=amber, expansion=violet, variant=gray), element type badge (color-coded), and label text.
- Attached errata and FAQ elements appear as clickable buttons at the bottom of the card. Clicking one opens the PageModal for that annotation element.
- Card border is red for errata source, amber for FAQ source, gray otherwise.
- Clicking the card image or a label row opens `PageModal` for that result.

### Page modal (`PageModal.tsx`)

Opens on card click, showing the full page with the result highlighted.

Two display modes (toggled by the user):
- **Focus mode** (default): CSS crop of the page image centered on the union bbox of all results in the group, with 4%/7% padding and a minimum height of 25% of the page. Highlight boxes are positioned relative to the crop window.
- **Full page mode**: full-page image with zoom controls (1×–4×, 0.5× steps). Auto-scrolls to center the primary result's bbox. Highlight boxes rendered by `HighlightedImage`.

**Dev mode** (toggled by "dev" button): in full-page mode, fetches all elements for the page via `GET /elements` and renders them as dimmed overlays behind the selected element highlights, allowing inspection of the full indexed element map.

---

## 6. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/rulebooks/{rulebook_id}` | Ingest a PDF or HTML file. Query params: `source_type` (core\|errata\|faq\|expansion\|variant, default `core`), `game_name` (optional). Body: multipart `file`. Returns SSE stream of progress events, ending with `{done: true, ...}`. Re-ingest deletes only elements of the same `source_type`. |
| `GET` | `/ask` | Search + stream answer. Params: `q` (required), `rulebook_id` (required), `n` (1–20, default 5). SSE stream: `{type:"results",...}` then `{type:"token",...}` × N then `{type:"done"}`. |
| `GET` | `/rulebooks` | List all registered rulebooks. Returns `{rulebooks: [{id, name}]}`. |
| `GET` | `/elements` | All indexed elements for one page. Params: `rulebook_id`, `page_number`. Returns `{elements: Element[]}`. Used by PageModal dev overlay and PageBrowser. |
| `GET` | `/rulebooks/{rulebook_id}/page_count` | Returns `{page_count: N}` — the max page number with indexed elements. Used by PageBrowser to bound navigation. |
| `GET` | `/images/{filename}` | Static file serving for page images stored at `./data/images/`. |
| `GET` | `/health` | Returns `{"status": "ok"}`. Polled every 10 s by the ConnectionStatus component. |

---

## 7. Key Design Decisions

### Image-first storage

Pages are stored as rendered PNG images rather than extracted text. This preserves the spatial layout, works with scan-only PDFs, and allows the frontend to show the exact page region as a visual crop — the user sees what the rulebook actually says, not a text reconstruction.

### Source type hierarchy

Five source types with priority: `errata` (3) > `faq` (2) > `expansion`/`core` (1) > `variant` (0). Errata and FAQ elements are found by separate vector searches and linked to core results rather than competing in the same ranked list. Variant results are deprioritized by the reranker unless the query explicitly mentions a variant. Source type is per-element, not per-rulebook — a single PDF can contain core + expansion + variant sections, detected automatically via `_detect_source_type_override`.

### Section propagation

PaddleOCR detects heading boxes as separate elements from their body text. Phase 1 reads each heading and builds a sorted list of `(box, section_name)` pairs. `section_for_box()` assigns any body element to the closest heading above it, treating headings as extending full page width (no column scoping). This means a heading in the left column applies to right-column content below the same y-coordinate. The last detected section threads across page boundaries.

### Context crops for classification

Rather than classifying each region in isolation, Phase 2 builds a composite image: the most relevant context box (a nearby image or the previous text box) stacked above a 2 px grey separator, then the target box. This gives Claude Haiku enough context to produce accurate labels and descriptions for content that would otherwise be ambiguous in isolation.

### Layout calibration + document profile

A per-rulebook document profile is computed once (before the main page loop) and reused on re-ingest. It serves two purposes:

1. **Calibrated heading thresholds**: the fixed `MAX_HEADING_W = 25%` is replaced by a threshold derived from the actual heading box width distribution on sampled pages (p90 × 1.2, capped at 45%). The Claude Haiku visual pass can override the statistical value when the visual evidence disagrees.

2. **Style injection into REGION_PROMPT**: icon vocabulary (`[coin] = gold cost`), callout descriptions, and layout notes are injected as `style_notes` into every Phase 3 classification call, improving labeling accuracy for games with specialized iconography or unusual layouts.

The profile persists in `rulebooks.json` under the rulebook's entry so subsequent ingests (e.g. re-ingest after a bug fix) skip the Haiku profile call entirely.

### Component list handling (Phase 1.5)

PaddleOCR fragments component lists (which typically have icons + short text) into many small boxes. Phase 1.5 detects sections headed by component/contents keywords, crops the entire section as one unit, and asks Claude to enumerate items. Each item is stored as a separate `component` element (sharing the section bbox) to avoid embedding dilution from a single element listing 40 game components.

### Result grouping

Results from the same page within a 6% vertical gap are merged into one card. This avoids showing three separate cards for a heading, a rule callout, and a diagram that visually belong to the same rulebook section. The merged card shows a single image crop covering all matched elements, with per-element highlight boxes.

### Re-ingest safety

All pages are processed before any existing elements are deleted. If ingestion fails mid-PDF, the old index is untouched. Deletion is scoped to `(rulebook_id, source_type)` so re-ingesting a FAQ supplement won't remove the core rulebook index.
