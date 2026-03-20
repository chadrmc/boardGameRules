"use client";

import { useState } from "react";
import { ingestRulebook } from "@/lib/api";

export function UploadPanel({ onIngested }: { onIngested?: () => void }) {
  const [open, setOpen] = useState(false);
  const [rulebookId, setRulebookId] = useState("");
  const [gameName, setGameName] = useState("");
  const [sourceType, setSourceType] = useState("core");
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<{ page: number; total: number } | null>(null);
  const [result, setResult] = useState<{ pages_processed: number; elements_found: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file || !rulebookId) return;

    setUploading(true);
    setProgress(null);
    setError(null);
    setResult(null);

    try {
      const data = await ingestRulebook(
        rulebookId, file, sourceType, gameName || undefined,
        (page, total) => setProgress({ page, total }),
      );
      setResult({ pages_processed: data.pages_processed, elements_found: data.elements_found });
      onIngested?.();
    } catch {
      setError("Upload failed. Is the backend running?");
    } finally {
      setUploading(false);
      setProgress(null);
    }
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-4 py-3 text-left flex justify-between items-center hover:bg-gray-50 transition-colors rounded-lg"
      >
        <span className="font-medium text-gray-700">Add Rulebook</span>
        <span className="text-gray-400 text-sm">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <form
          onSubmit={handleSubmit}
          className="px-4 pb-4 pt-3 space-y-3 border-t border-gray-100"
        >
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-sm text-gray-600 mb-1">
                Game Name
              </label>
              <input
                type="text"
                value={gameName}
                onChange={(e) => setGameName(e.target.value)}
                placeholder="e.g. Catan"
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">
                Rulebook ID
              </label>
              <input
                type="text"
                value={rulebookId}
                onChange={(e) => setRulebookId(e.target.value)}
                placeholder="e.g. catan"
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">
                Source Type
              </label>
              <select
                value={sourceType}
                onChange={(e) => setSourceType(e.target.value)}
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="core">Core</option>
                <option value="errata">Errata</option>
                <option value="faq">FAQ</option>
                <option value="expansion">Expansion</option>
                <option value="variant">Variant</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-600 mb-1">
              Rulebook PDF
            </label>
            <input
              type="file"
              accept="application/pdf,.pdf"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="w-full text-sm text-gray-600"
              required
            />
          </div>

          {progress && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-gray-500">
                <span>Processing page {progress.page} of {progress.total}…</span>
                <span>{Math.round((progress.page / progress.total) * 100)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-1.5">
                <div
                  className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${(progress.page / progress.total) * 100}%` }}
                />
              </div>
            </div>
          )}

          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={uploading || !file || !rulebookId}
              className="bg-blue-600 text-white px-4 py-2 rounded text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {uploading ? "Ingesting…" : "Ingest Rulebook"}
            </button>
            {result && (
              <span className="text-sm text-green-600">
                Processed {result.pages_processed} pages, found {result.elements_found} elements
              </span>
            )}
            {error && <span className="text-sm text-red-600">{error}</span>}
          </div>
        </form>
      )}
    </div>
  );
}
