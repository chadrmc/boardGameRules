"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { SearchResult, Element, getPageElements } from "@/lib/api";
import { API_BASE, ELEMENT_COLORS } from "@/lib/constants";
import { HighlightedImage } from "./HighlightedImage";

// Focus view crop padding (fraction of page dimensions)
const FOCUS_PAD_X = 0.04;
const FOCUS_PAD_Y = 0.07;
const MIN_FOCUS_H = 0.25; // always show at least 25% of page height for context

// Full-page zoom controls
const ZOOM_MIN = 1;
const ZOOM_MAX = 4;
const ZOOM_STEP = 0.5;

interface Highlight {
  bbox: { x: number; y: number; w: number; h: number };
  color: string;
  label: string;
  dim?: boolean;
  elementId?: string;
}

interface Props {
  primary: SearchResult;
  group: SearchResult[];
  onClose: () => void;
}

export function PageModal({ primary, group, onClose }: Props) {
  const { element } = primary;
  const color = ELEMENT_COLORS[element.type] ?? ELEMENT_COLORS.other;
  const scrollRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLElement | null>(null);
  const [pageElements, setPageElements] = useState<Element[]>([]);
  const [devMode, setDevMode] = useState(false);
  const [focusMode, setFocusMode] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const [inspectedElement, setInspectedElement] = useState<Element | null>(null);

  const groupIds = new Set(group.map((r) => r.element.id));
  const imgSrc = `${API_BASE}${element.page_image_path}`;

  // Union bbox of all results in the group (for focus crop center)
  const unionMinX = Math.min(...group.map((r) => r.element.bbox.x));
  const unionMinY = Math.min(...group.map((r) => r.element.bbox.y));
  const unionMaxX = Math.max(...group.map((r) => r.element.bbox.x + r.element.bbox.w));
  const unionMaxY = Math.max(...group.map((r) => r.element.bbox.y + r.element.bbox.h));

  // Focus crop: union bbox + padding, expanded to MIN_FOCUS_H if needed
  const rawH = Math.max(MIN_FOCUS_H, unionMaxY - unionMinY + FOCUS_PAD_Y * 2);
  const centerY = (unionMinY + unionMaxY) / 2;
  const focusCrop = {
    x: Math.max(0, unionMinX - FOCUS_PAD_X),
    y: Math.max(0, Math.min(1 - rawH, centerY - rawH / 2)),
    w: Math.min(1, unionMaxX - unionMinX + FOCUS_PAD_X * 2),
    h: rawH,
  };

  useEffect(() => {
    getPageElements(element.rulebook_id, element.page_number)
      .then(setPageElements)
      .catch(() => setPageElements([]));
  }, [element.rulebook_id, element.page_number]);

  // Save trigger element and move focus into modal on open; restore on close
  useEffect(() => {
    triggerRef.current = document.activeElement as HTMLElement;
    const modal = modalRef.current;
    if (modal) {
      const firstFocusable = modal.querySelector<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      firstFocusable?.focus();
    }
    return () => {
      triggerRef.current?.focus();
    };
  }, []);

  // Keyboard handling: Escape to close + focus trap
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key === "Tab") {
        const modal = modalRef.current;
        if (!modal) return;
        const focusable = modal.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey) {
          if (document.activeElement === first) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (document.activeElement === last) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  // --- Full-page mode: scroll to center the primary bbox ---
  const scrollToBbox = useCallback(
    (z: number, nat: { w: number; h: number }) => {
      const container = scrollRef.current;
      if (!container) return;
      const renderedW = container.clientWidth * z;
      const renderedH = renderedW * (nat.h / nat.w);
      container.scrollLeft =
        (element.bbox.x + element.bbox.w / 2) * renderedW - container.clientWidth / 2;
      container.scrollTop =
        (element.bbox.y + element.bbox.h / 2) * renderedH - container.clientHeight / 2;
    },
    [element.bbox]
  );

  function handleImageLoad(e: React.SyntheticEvent<HTMLImageElement>) {
    const img = e.currentTarget;
    const nat = { w: img.naturalWidth, h: img.naturalHeight };
    setNaturalSize(nat);
    if (!focusMode) scrollToBbox(zoom, nat);
  }

  useEffect(() => {
    if (!focusMode && naturalSize) scrollToBbox(zoom, naturalSize);
  }, [zoom, focusMode, naturalSize, scrollToBbox]);

  function adjustZoom(delta: number) {
    setZoom((z) =>
      Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, Math.round((z + delta) / ZOOM_STEP) * ZOOM_STEP))
    );
  }

  // Highlights for full-page mode (HighlightedImage)
  const fullPageHighlights: Highlight[] = [
    ...(devMode
      ? pageElements
          .filter((e) => !groupIds.has(e.id))
          .map((e) => ({
            bbox: e.bbox,
            color: ELEMENT_COLORS[e.type] ?? ELEMENT_COLORS.other,
            label: e.label,
            dim: true,
            elementId: e.id,
          }))
      : []),
    ...group
      .filter((r) => r.element.id !== element.id)
      .map((r) => ({
        bbox: r.element.bbox,
        color: ELEMENT_COLORS[r.element.type] ?? ELEMENT_COLORS.other,
        label: r.element.label,
        elementId: r.element.id,
      })),
    { bbox: element.bbox, color, label: element.label, elementId: element.id },
  ];

  // All elements by ID for quick lookup when inspecting
  const allElementsById = new Map<string, Element>();
  for (const e of pageElements) allElementsById.set(e.id, e);
  for (const r of group) allElementsById.set(r.element.id, r.element);

  function handleHighlightClick(index: number) {
    const h = fullPageHighlights[index];
    if (!h.elementId) return;
    const el = allElementsById.get(h.elementId);
    if (el) setInspectedElement((prev) => prev?.id === el.id ? null : el);
  }

  // Highlights for focus mode (positioned relative to crop window)
  const focusHighlights = group.map((r) => ({
    bbox: r.element.bbox,
    color: ELEMENT_COLORS[r.element.type] ?? ELEMENT_COLORS.other,
    isPrimary: r.element.id === element.id,
  }));

  // Focus crop aspect ratio (needs naturalSize to avoid squashing)
  const focusAspectRatio = naturalSize
    ? (focusCrop.w * naturalSize.w) / (focusCrop.h * naturalSize.h)
    : null;
  const focusMarginTopPct = naturalSize
    ? -(focusCrop.y * naturalSize.h / naturalSize.w) * 100
    : 0;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 flex items-start justify-center p-4 overflow-y-auto"
      onClick={onClose}
    >
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-label={`${element.label} — Page ${element.page_number}`}
        className="relative bg-white rounded-lg max-w-3xl w-full my-8 overflow-hidden shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
          <div className="flex items-baseline gap-2 min-w-0">
            <span className="font-medium text-gray-900 truncate">
              {group.length > 1
                ? group.map((r) => r.element.label).join(" · ")
                : element.label}
            </span>
            <span className="text-sm text-gray-400 shrink-0">
              {element.rulebook_id} · Page {element.page_number}
            </span>
          </div>
          <div className="ml-4 flex items-center gap-3 shrink-0">
            {/* Focus / Full page toggle — only for image-mode elements */}
            {element.display_mode !== "text" && (
            <div className="flex rounded border border-gray-300 overflow-hidden text-xs">
              <button
                onClick={() => setFocusMode(true)}
                className={`px-2 py-1 transition-colors ${
                  focusMode ? "bg-blue-600 text-white" : "text-gray-500 hover:bg-gray-50"
                }`}
              >
                Focus
              </button>
              <button
                onClick={() => setFocusMode(false)}
                className={`px-2 py-1 border-l border-gray-300 transition-colors ${
                  !focusMode ? "bg-blue-600 text-white" : "text-gray-500 hover:bg-gray-50"
                }`}
              >
                Full page
              </button>
            </div>
            )}
            {/* Zoom controls — only in full-page mode for image elements */}
            {element.display_mode !== "text" && !focusMode && (
              <div className="flex items-center gap-1">
                <button
                  onClick={() => adjustZoom(-ZOOM_STEP)}
                  disabled={zoom <= ZOOM_MIN}
                  aria-label="Zoom out"
                  className="w-6 h-6 flex items-center justify-center rounded border border-gray-300 text-gray-600 text-sm disabled:opacity-30 hover:bg-gray-50"
                >
                  −
                </button>
                <span className="text-xs text-gray-500 w-8 text-center tabular-nums">
                  {zoom === 1 ? "1×" : `${zoom}×`}
                </span>
                <button
                  onClick={() => adjustZoom(ZOOM_STEP)}
                  disabled={zoom >= ZOOM_MAX}
                  aria-label="Zoom in"
                  className="w-6 h-6 flex items-center justify-center rounded border border-gray-300 text-gray-600 text-sm disabled:opacity-30 hover:bg-gray-50"
                >
                  +
                </button>
              </div>
            )}
            <button
              onClick={() => { setDevMode((v) => !v); setInspectedElement(null); }}
              aria-label="Toggle developer overlay"
              aria-pressed={devMode}
              className={`text-xs px-2 py-1 rounded border font-mono transition-colors ${
                devMode
                  ? "bg-amber-100 border-amber-400 text-amber-800"
                  : "bg-gray-100 border-gray-300 text-gray-500"
              }`}
            >
              dev
            </button>
            <button
              onClick={onClose}
              aria-label="Close modal"
              className="text-gray-400 hover:text-gray-700 text-xl leading-none"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Text view — for FAQ/errata elements with no page image */}
        {element.display_mode === "text" && (
          <div className="px-5 py-4 max-h-[70vh] overflow-y-auto">
            <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">
              {element.description}
            </p>
          </div>
        )}

        {/* Focus view — cropped around the result, no scrolling */}
        {element.display_mode !== "text" && focusMode && (
          <div
            className="relative overflow-hidden bg-gray-100"
            style={
              focusAspectRatio
                ? { aspectRatio: String(focusAspectRatio) }
                : { paddingTop: "30%" }
            }
          >
            <div
              style={{
                position: "absolute",
                width: `${(1 / focusCrop.w) * 100}%`,
                left: `${-(focusCrop.x / focusCrop.w) * 100}%`,
                top: 0,
              }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imgSrc}
                alt={`${element.label} — Page ${element.page_number}`}
                ref={imgRef}
                style={{ display: "block", width: "100%", marginTop: `${focusMarginTopPct}%` }}
                onLoad={handleImageLoad}
              />
            </div>
            {/* Highlight boxes positioned relative to the crop window */}
            {naturalSize &&
              focusHighlights.map((h, i) => (
                <div
                  key={i}
                  className="absolute border-2 pointer-events-none"
                  style={{
                    left: `${((h.bbox.x - focusCrop.x) / focusCrop.w) * 100}%`,
                    top: `${((h.bbox.y - focusCrop.y) / focusCrop.h) * 100}%`,
                    width: `${(h.bbox.w / focusCrop.w) * 100}%`,
                    height: `${(h.bbox.h / focusCrop.h) * 100}%`,
                    borderColor: h.color,
                    backgroundColor: h.isPrimary ? `${h.color}30` : `${h.color}18`,
                  }}
                />
              ))}
          </div>
        )}

        {/* Full page view — scrollable with zoom controls */}
        {element.display_mode !== "text" && !focusMode && (
          <div ref={scrollRef} className="overflow-auto max-h-[80vh]">
            <div style={{ width: `${zoom * 100}%` }}>
              <HighlightedImage
                src={imgSrc}
                imgRef={imgRef}
                highlights={fullPageHighlights}
                onLoad={handleImageLoad}
                alt={`${element.label} — Page ${element.page_number}`}
                onHighlightClick={devMode ? handleHighlightClick : undefined}
                activeIndex={inspectedElement ? fullPageHighlights.findIndex((h) => h.elementId === inspectedElement.id) : undefined}
              />
            </div>
          </div>
        )}

        {/* Element inspection panel — visible in dev mode when an element is clicked */}
        {devMode && inspectedElement && (
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
    </div>
  );
}
