"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { SearchResult, Element, getPageElements } from "@/lib/api";
import { HighlightedImage } from "./HighlightedImage";

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
  const [pageElements, setPageElements] = useState<Element[]>([]);
  const [devMode, setDevMode] = useState(false);
  const [focusMode, setFocusMode] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);

  const groupIds = new Set(group.map((r) => r.element.id));
  const imgSrc = `http://localhost:8000${element.page_image_path}`;

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
          }))
      : []),
    ...group
      .filter((r) => r.element.id !== element.id)
      .map((r) => ({
        bbox: r.element.bbox,
        color: ELEMENT_COLORS[r.element.type] ?? ELEMENT_COLORS.other,
        label: r.element.label,
      })),
    { bbox: element.bbox, color, label: element.label },
  ];

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
                  className="w-6 h-6 flex items-center justify-center rounded border border-gray-300 text-gray-600 text-sm disabled:opacity-30 hover:bg-gray-50"
                >
                  +
                </button>
              </div>
            )}
            <button
              onClick={() => setDevMode((v) => !v)}
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
                alt="Rulebook page"
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
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
