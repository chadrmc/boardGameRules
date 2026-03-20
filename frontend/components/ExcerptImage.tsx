"use client";

import { useState } from "react";
import { BBox } from "@/lib/api";

interface Highlight {
  bbox: BBox;
  color: string;
}

interface Props {
  src: string;
  highlights: Highlight[];
  /** CSS max-height for the container (e.g. "144px"). Adds a bottom fade when content overflows. */
  maxHeight?: string;
  /** Descriptive alt text for the excerpt image. */
  alt?: string;
}

// Padding added around the crop window so single-line excerpts show enough context.
const PAD_X = 0.03;
const PAD_Y = 0.03;
// Minimum crop height (as fraction of page height) so tiny results are still readable.
const MIN_CROP_H = 0.12;

export function ExcerptImage({ src, highlights, maxHeight, alt }: Props) {
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);

  // Union bbox of all highlights defines the crop window.
  const minX = Math.min(...highlights.map((h) => h.bbox.x));
  const minY = Math.min(...highlights.map((h) => h.bbox.y));
  const maxX = Math.max(...highlights.map((h) => h.bbox.x + h.bbox.w));
  const maxY = Math.max(...highlights.map((h) => h.bbox.y + h.bbox.h));

  // Compute padded edges, clamp each to [0, 1], then derive w/h.
  // This ensures x+w <= 1 and y+h <= 1 even when bbox is near a page edge.
  const px0 = Math.max(0, minX - PAD_X);
  const py0 = Math.max(0, minY - PAD_Y);
  const px1 = Math.min(1, maxX + PAD_X);
  const py1 = Math.min(1, maxY + PAD_Y);
  const padded = { x: px0, y: py0, w: px1 - px0, h: py1 - py0 };

  // Expand crop vertically if the region is too short to be readable.
  const cropH = Math.max(padded.h, MIN_CROP_H);
  const centerY = padded.y + padded.h / 2;
  const display = {
    x: padded.x,
    w: padded.w,
    y: Math.max(0, Math.min(1 - cropH, centerY - cropH / 2)),
    h: cropH,
  };

  // Container aspect ratio uses the page's actual pixel dimensions so the crop
  // isn't vertically squashed on tall pages.
  const aspectRatio = naturalSize
    ? (display.w * naturalSize.w) / (display.h * naturalSize.h)
    : null;

  // Vertical offset uses margin-top (% of containing block WIDTH per CSS spec),
  // avoiding the unreliable top: X% when container height comes from aspect-ratio.
  const marginTopPct = naturalSize
    ? -(display.y * naturalSize.h / naturalSize.w) * 100
    : 0;

  const containerStyle: React.CSSProperties = aspectRatio
    ? { aspectRatio: String(aspectRatio), ...(maxHeight ? { maxHeight } : {}) }
    : { paddingTop: "30%" };

  return (
    <div
      className="relative overflow-hidden bg-gray-100"
      style={containerStyle}
    >
      {/* Inner div handles horizontal placement; image margin-top handles vertical */}
      <div
        style={{
          position: "absolute",
          width: `${(1 / display.w) * 100}%`,
          left: `${-(display.x / display.w) * 100}%`,
          top: 0,
        }}
      >
        <img
          src={src}
          alt={alt || "Rulebook excerpt"}
          style={{
            display: "block",
            width: "100%",
            marginTop: `${marginTopPct}%`,
          }}
          onLoad={(e) => {
            const el = e.currentTarget;
            setNaturalSize({ w: el.naturalWidth, h: el.naturalHeight });
          }}
        />
      </div>
      {naturalSize &&
        highlights.map((h, i) => {
          const style = {
            left: `${((h.bbox.x - display.x) / display.w) * 100}%`,
            top: `${((h.bbox.y - display.y) / display.h) * 100}%`,
            width: `${(h.bbox.w / display.w) * 100}%`,
            height: `${(h.bbox.h / display.h) * 100}%`,
            borderColor: h.color,
            backgroundColor: `${h.color}18`,
          };
          return (
            <div key={i} className="absolute border-2 pointer-events-none" style={style} />
          );
        })}
      {/* Bottom fade gradient when content is height-capped */}
      {maxHeight && (
        <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-gray-100 to-transparent pointer-events-none" />
      )}
    </div>
  );
}
