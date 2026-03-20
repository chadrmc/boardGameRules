import { Element, SearchResult } from "@/lib/api";
import { ExcerptImage } from "./ExcerptImage";

const ELEMENT_COLORS: Record<string, string> = {
  rule: "#2563eb",
  note: "#d97706",
  illustration: "#16a34a",
  example: "#9333ea",
  diagram: "#0891b2",
  table: "#dc2626",
  component: "#0d9488",
  other: "#6b7280",
};

const SOURCE_BADGES: Record<string, { label: string; className: string }> = {
  errata: { label: "Errata", className: "bg-red-600 text-white" },
  faq: { label: "FAQ", className: "bg-amber-500 text-white" },
  expansion: { label: "Expansion", className: "bg-violet-600 text-white" },
  variant: { label: "Variant", className: "bg-gray-400 text-white" },
  core: { label: "", className: "" },
};

interface Props {
  results: SearchResult[];
  onViewPage: (primary: SearchResult) => void;
  onViewAnnotation: (element: Element) => void;
}

export function ResultCard({ results, onViewPage, onViewAnnotation }: Props) {
  // Primary result (highest score) drives the card's image and metadata.
  const primary = results[0];
  const { element } = primary;
  const highlights = results.map((r) => ({
    bbox: r.element.bbox,
    color: ELEMENT_COLORS[r.element.type] ?? ELEMENT_COLORS.other,
  }));

  const sourceType = element.source_type;
  const borderClass =
    sourceType === "errata" ? "border-red-400" :
    sourceType === "faq" ? "border-amber-400" :
    "border-gray-200";

  return (
    <div
      className={`bg-white rounded-lg overflow-hidden shadow-sm border cursor-pointer hover:shadow-md transition-shadow ${borderClass}`}
    >
      <div onClick={() => onViewPage(primary)}>
        {element.display_mode === "text" ? (
          <div className="px-3 pt-3 pb-1">
            <p className="text-sm text-gray-700 leading-snug line-clamp-6 whitespace-pre-wrap">
              {element.description}
            </p>
          </div>
        ) : (
          <ExcerptImage
            src={`http://localhost:8000${element.page_image_path}`}
            highlights={highlights}
          />
        )}
      </div>
      <div className="p-3 space-y-1.5">
        {results.map((r) => {
          const color = ELEMENT_COLORS[r.element.type] ?? ELEMENT_COLORS.other;
          const sourceBadge = SOURCE_BADGES[r.element.source_type] ?? SOURCE_BADGES.core;
          return (
            <div key={r.element.id} onClick={() => onViewPage(r)} className="cursor-pointer hover:bg-gray-50 rounded px-1 -mx-1 space-y-0.5">
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
                <p className="text-xs text-gray-600 leading-snug">{r.element.description}</p>
              )}
            </div>
          );
        })}
        {(results.some(r => r.errata?.length > 0) || results[0].faq?.length > 0) && (
          <div className="flex flex-wrap gap-1.5 pt-1 border-t border-gray-100">
            {results.flatMap(r => r.errata ?? []).map(e => (
              <button
                key={e.id}
                onClick={(ev) => { ev.stopPropagation(); onViewAnnotation(e); }}
                className="text-xs font-bold px-2 py-0.5 rounded uppercase tracking-wide bg-red-600 text-white hover:bg-red-700 transition-colors"
              >
                ⚠ Errata · p.{e.page_number}
              </button>
            ))}
            {(results[0].faq ?? []).map(e => (
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
        <p className="text-xs text-gray-400">
          {element.rulebook_id} · Page {element.page_number} · {Math.round(primary.score * 100)}%
        </p>
      </div>
    </div>
  );
}
