import { Element, SearchResult } from "@/lib/api";
import { API_BASE, ELEMENT_COLORS } from "@/lib/constants";
import { ExcerptImage } from "./ExcerptImage";

const SOURCE_BADGES: Record<string, { label: string; className: string }> = {
  errata: { label: "Errata", className: "bg-red-600 text-white" },
  faq: { label: "FAQ", className: "bg-amber-500 text-white" },
  expansion: { label: "Expansion", className: "bg-violet-600 text-white" },
  variant: { label: "Variant", className: "bg-gray-400 text-white" },
  core: { label: "", className: "" },
};

// Threshold for coalescing results into a single tile (fraction of page height).
// Elements within this vertical gap read as continuous flowing text.
const TILE_GAP = 0.03;

/** Sub-group results into tiles: nearby elements merge, distant ones get separate tiles. */
function clusterIntoTiles(results: SearchResult[]): SearchResult[][] {
  const sorted = [...results].sort((a, b) => a.element.bbox.y - b.element.bbox.y);
  const tiles: SearchResult[][] = [[sorted[0]]];
  let bottom = sorted[0].element.bbox.y + sorted[0].element.bbox.h;

  for (let i = 1; i < sorted.length; i++) {
    const r = sorted[i];
    if (r.element.bbox.y - bottom <= TILE_GAP) {
      tiles[tiles.length - 1].push(r);
      bottom = Math.max(bottom, r.element.bbox.y + r.element.bbox.h);
    } else {
      tiles.push([r]);
      bottom = r.element.bbox.y + r.element.bbox.h;
    }
  }
  return tiles;
}

interface Props {
  results: SearchResult[];
  onViewPage: (primary: SearchResult) => void;
  onViewAnnotation: (element: Element) => void;
}

export function ResultCard({ results, onViewPage, onViewAnnotation }: Props) {
  const primary = results[0];
  const { element } = primary;
  const tiles = clusterIntoTiles(results);

  const sourceType = element.source_type;
  const borderClass =
    sourceType === "errata" ? "border-red-400" :
    sourceType === "faq" ? "border-amber-400" :
    "border-gray-200";

  return (
    <div
      className={`bg-white rounded-lg overflow-hidden shadow-sm border hover:shadow-md transition-shadow ${borderClass}`}
    >
      {/* Per-tile excerpt sections */}
      {tiles.map((tile, tileIdx) => {
        const highlights = tile.map((r) => ({
          bbox: r.element.bbox,
          color: ELEMENT_COLORS[r.element.type] ?? ELEMENT_COLORS.other,
        }));
        // All text-mode results in this tile?
        const allText = tile.every((r) => r.element.display_mode === "text");

        return (
          <div
            key={tile.map((r) => r.element.id).join("+")}
            className={tileIdx > 0 ? "border-t border-gray-100" : ""}
          >
            {/* Tile image: merged crop for coalesced results, or text fallback */}
            <div
              role="button"
              tabIndex={0}
              className="cursor-pointer hover:bg-gray-50 transition-colors"
              onClick={() => onViewPage(tile[0])}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onViewPage(tile[0]); } }}
            >
              {allText ? (
                <div className="px-3 pt-3 pb-1">
                  {tile.map((r) => (
                    <p key={r.element.id} className="text-sm text-gray-700 leading-snug line-clamp-6 whitespace-pre-wrap">
                      {r.element.description}
                    </p>
                  ))}
                </div>
              ) : (
                <ExcerptImage
                  src={`${API_BASE}${tile[0].element.page_image_path}`}
                  highlights={highlights}
                  maxHeight="144px"
                  alt={`${tile.map((r) => r.element.label).join(", ")} — Page ${tile[0].element.page_number}`}
                />
              )}
            </div>
            {/* Label rows — one per result, individually clickable */}
            {tile.map((r) => {
              const color = ELEMENT_COLORS[r.element.type] ?? ELEMENT_COLORS.other;
              const sourceBadge = SOURCE_BADGES[r.element.source_type] ?? SOURCE_BADGES.core;
              return (
                <div
                  key={r.element.id}
                  role="button"
                  tabIndex={0}
                  className="px-3 py-1.5 space-y-0.5 cursor-pointer hover:bg-gray-50 transition-colors"
                  onClick={() => onViewPage(r)}
                  onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onViewPage(r); } }}
                >
                  <div className="flex items-center gap-2 flex-wrap">
                    {sourceBadge.label && (
                      <span
                        className={`text-xs font-bold px-2 py-0.5 rounded uppercase tracking-wide ${sourceBadge.className}`}
                      >
                        {sourceBadge.label}
                      </span>
                    )}
                    <span
                      className="text-xs font-semibold px-2 py-0.5 rounded text-white uppercase tracking-wide"
                      style={{ backgroundColor: color }}
                    >
                      {r.element.type}
                    </span>
                    <span className="font-medium text-gray-900 text-sm">{r.element.label}</span>
                  </div>
                  {r.element.description && (
                    <p className="text-xs text-gray-600 leading-snug line-clamp-2">{r.element.description}</p>
                  )}
                </div>
              );
            })}
          </div>
        );
      })}

      {/* Errata/FAQ annotation buttons */}
      {(results.some(r => r.errata?.length > 0) || results.some(r => r.faq?.length > 0)) && (
        <div className="flex flex-wrap gap-1.5 px-3 py-2 border-t border-gray-100">
          {results.flatMap(r => r.errata ?? []).map(e => (
            <button
              key={e.id}
              onClick={(ev) => { ev.stopPropagation(); onViewAnnotation(e); }}
              className="text-xs font-bold px-2 py-0.5 rounded uppercase tracking-wide bg-red-600 text-white hover:bg-red-700 transition-colors"
            >
              ⚠ Errata · p.{e.page_number}
            </button>
          ))}
          {results.flatMap(r => r.faq ?? []).map(e => (
            <button
              key={e.id}
              onClick={(ev) => { ev.stopPropagation(); onViewAnnotation(e); }}
              className="text-xs font-bold px-2 py-0.5 rounded uppercase tracking-wide bg-amber-500 text-white hover:bg-amber-600 transition-colors"
            >
              ? FAQ · p.{e.page_number}
            </button>
          ))}
        </div>
      )}

      {/* Footer: page metadata */}
      <div className="flex items-center justify-between gap-2 px-3 py-2 border-t border-gray-100">
        <p className="text-xs text-gray-400">
          {element.rulebook_id} · Page {element.page_number} · {Math.round(primary.score * 100)}%
        </p>
        <button
          className="text-xs text-gray-300 font-mono hover:text-gray-500 transition-colors"
          onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(element.id); }}
          title="Click to copy full ID"
        >
          {element.id.slice(0, 8)}
        </button>
      </div>
    </div>
  );
}
