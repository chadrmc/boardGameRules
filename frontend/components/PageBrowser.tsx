"use client";

import { useState, useEffect } from "react";
import { Element, getPageElements, getPageCount } from "@/lib/api";
import { API_BASE, ELEMENT_COLORS } from "@/lib/constants";
import { HighlightedImage } from "./HighlightedImage";

interface Highlight {
  bbox: { x: number; y: number; w: number; h: number };
  color: string;
  label: string;
  elementId: string;
}

interface Props {
  rulebookId: string;
  onClose: () => void;
}

export function PageBrowser({ rulebookId, onClose }: Props) {
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState<number | null>(null);
  const [elements, setElements] = useState<Element[]>([]);
  const [loading, setLoading] = useState(false);
  const [inspectedElement, setInspectedElement] = useState<Element | null>(null);
  const [showOverlays, setShowOverlays] = useState(true);

  // Fetch total page count once
  useEffect(() => {
    getPageCount(rulebookId)
      .then(setTotalPages)
      .catch(() => setTotalPages(0));
  }, [rulebookId]);

  // Fetch elements for current page
  useEffect(() => {
    setLoading(true);
    setInspectedElement(null);
    getPageElements(rulebookId, page)
      .then(setElements)
      .catch(() => setElements([]))
      .finally(() => setLoading(false));
  }, [rulebookId, page]);

  const highlights: Highlight[] = elements.map((e) => ({
    bbox: e.bbox,
    color: ELEMENT_COLORS[e.type] ?? ELEMENT_COLORS.other,
    label: e.label,
    elementId: e.id,
  }));

  // Derive image URL from element data (filenames include source_type)
  const imgSrc = elements.length > 0
    ? `${API_BASE}${elements[0].page_image_path}`
    : `${API_BASE}/images/${rulebookId}_core_page${page}.png`;

  function handleHighlightClick(index: number) {
    const h = highlights[index];
    const el = elements.find((e) => e.id === h.elementId);
    if (el) setInspectedElement((prev) => prev?.id === el.id ? null : el);
  }

  const activeIndex = inspectedElement
    ? highlights.findIndex((h) => h.elementId === inspectedElement.id)
    : undefined;

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden shadow-sm">
      {/* Header with navigation */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="px-2 py-1 rounded border border-gray-300 text-sm text-gray-600 disabled:opacity-30 hover:bg-white transition-colors"
            aria-label="Previous page"
          >
            ← Prev
          </button>
          <span className="text-sm text-gray-600 tabular-nums min-w-[80px] text-center">
            Page {page}{totalPages !== null ? ` of ${totalPages}` : ""}
          </span>
          <button
            onClick={() => setPage((p) => totalPages ? Math.min(totalPages, p + 1) : p + 1)}
            disabled={totalPages !== null && page >= totalPages}
            className="px-2 py-1 rounded border border-gray-300 text-sm text-gray-600 disabled:opacity-30 hover:bg-white transition-colors"
            aria-label="Next page"
          >
            Next →
          </button>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">
            {elements.length} element{elements.length !== 1 ? "s" : ""}
          </span>
          <button
            onClick={() => { setShowOverlays((v) => !v); if (showOverlays) setInspectedElement(null); }}
            aria-label="Toggle element overlays"
            aria-pressed={showOverlays}
            className={`px-2 py-1 rounded border text-xs font-mono transition-colors ${
              showOverlays
                ? "bg-amber-100 border-amber-400 text-amber-800"
                : "bg-gray-100 border-gray-300 text-gray-500"
            }`}
          >
            overlays
          </button>
          <button
            onClick={onClose}
            aria-label="Close page browser"
            className="text-gray-400 hover:text-gray-700 text-lg leading-none"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Page image with overlays */}
      <div className="relative">
        {loading && (
          <div className="absolute inset-0 bg-white/70 z-10 flex items-center justify-center">
            <span className="text-sm text-gray-400 animate-pulse">Loading...</span>
          </div>
        )}
        <HighlightedImage
          src={imgSrc}
          highlights={showOverlays ? highlights : []}
          onHighlightClick={showOverlays ? handleHighlightClick : undefined}
          activeIndex={showOverlays ? activeIndex : undefined}
          alt={`Rulebook page ${page}`}
        />
      </div>

      {/* Inspection panel */}
      {inspectedElement && (
        <div className="border-t border-amber-300 bg-amber-50 px-4 py-3 text-xs">
          <div className="flex items-start justify-between gap-2 mb-2">
            <span className="font-bold text-amber-900 text-sm">{inspectedElement.label}</span>
            <button
              onClick={() => setInspectedElement(null)}
              aria-label="Close inspection panel"
              className="text-amber-400 hover:text-amber-700 text-lg leading-none shrink-0"
            >
              ✕
            </button>
          </div>
          <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-amber-900">
            <span className="font-semibold text-amber-600">ID</span>
            <span className="font-mono select-all break-all">{inspectedElement.id}</span>
            <span className="font-semibold text-amber-600">Type</span>
            <span>
              <span
                className="inline-block text-xs font-semibold px-1.5 py-0.5 rounded text-white uppercase"
                style={{ backgroundColor: ELEMENT_COLORS[inspectedElement.type] ?? ELEMENT_COLORS.other }}
              >
                {inspectedElement.type}
              </span>
            </span>
            <span className="font-semibold text-amber-600">Source</span>
            <span>{inspectedElement.source_type}</span>
            <span className="font-semibold text-amber-600">BBox</span>
            <span className="font-mono tabular-nums">
              x:{(inspectedElement.bbox.x * 100).toFixed(1)}%
              {" "}y:{(inspectedElement.bbox.y * 100).toFixed(1)}%
              {" "}w:{(inspectedElement.bbox.w * 100).toFixed(1)}%
              {" "}h:{(inspectedElement.bbox.h * 100).toFixed(1)}%
            </span>
            {inspectedElement.description && (
              <>
                <span className="font-semibold text-amber-600 self-start">Description</span>
                <p className="whitespace-pre-wrap leading-relaxed">{inspectedElement.description}</p>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
