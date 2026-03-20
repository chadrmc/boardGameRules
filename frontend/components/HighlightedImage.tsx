"use client";

import { BBox } from "@/lib/api";

interface Highlight {
  bbox: BBox;
  color: string;
  label: string;
  dim?: boolean;
}

interface Props {
  src: string;
  highlights: Highlight[];
  activeIndex?: number;
  onHighlightClick?: (index: number) => void;
  onLoad?: (e: React.SyntheticEvent<HTMLImageElement>) => void;
  imgRef?: React.RefObject<HTMLImageElement>;
  alt?: string;
}

export function HighlightedImage({
  src,
  highlights,
  activeIndex,
  onHighlightClick,
  onLoad,
  imgRef,
  alt,
}: Props) {
  return (
    <div className="relative w-full">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img ref={imgRef} src={src} alt={alt || "Rulebook page"} className="w-full h-auto block" onLoad={onLoad} />
      {highlights.map((h, i) => (
        <div
          key={i}
          onClick={() => onHighlightClick?.(i)}
          className="absolute border-2 transition-colors"
          style={{
            left: `${h.bbox.x * 100}%`,
            top: `${h.bbox.y * 100}%`,
            width: `${h.bbox.w * 100}%`,
            height: `${h.bbox.h * 100}%`,
            borderColor: h.color,
            backgroundColor: h.dim ? `${h.color}15` : activeIndex === i ? `${h.color}40` : `${h.color}25`,
            opacity: h.dim ? 0.6 : 1,
            cursor: onHighlightClick ? "pointer" : "default",
          }}
        >
          <span
            className="absolute -top-5 left-0 text-xs font-bold px-1 py-0.5 rounded text-white whitespace-nowrap leading-none"
            style={{ backgroundColor: h.color }}
          >
            {h.label}
          </span>
        </div>
      ))}
    </div>
  );
}
