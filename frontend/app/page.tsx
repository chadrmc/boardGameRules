"use client";

import { useState, useEffect, useRef } from "react";
import { SearchBar, SearchBarHandle } from "@/components/SearchBar";
import { ResultCard } from "@/components/ResultCard";
import { PageModal } from "@/components/PageModal";
import { UploadPanel } from "@/components/UploadPanel";
import { ConnectionStatus } from "@/components/ConnectionStatus";
import { PageBrowser } from "@/components/PageBrowser";
import { ask, listRulebooks, Element, SearchResult, Rulebook } from "@/lib/api";

// Cluster results on the same page whose bboxes are within GAP_THRESHOLD vertically.
const GAP_THRESHOLD = 0.06;

function groupResults(results: SearchResult[]): SearchResult[][] {
  const byPage = new Map<string, SearchResult[]>();
  for (const r of results) {
    const key = `${r.element.rulebook_id}::${r.element.page_number}`;
    if (!byPage.has(key)) byPage.set(key, []);
    byPage.get(key)!.push(r);
  }

  const groups: SearchResult[][] = [];
  for (const pageResults of byPage.values()) {
    const sorted = [...pageResults].sort((a, b) => a.element.bbox.y - b.element.bbox.y);
    let cluster: SearchResult[] = [sorted[0]];
    let clusterBottom = sorted[0].element.bbox.y + sorted[0].element.bbox.h;

    for (let i = 1; i < sorted.length; i++) {
      const r = sorted[i];
      if (r.element.bbox.y - clusterBottom <= GAP_THRESHOLD) {
        cluster.push(r);
        clusterBottom = Math.max(clusterBottom, r.element.bbox.y + r.element.bbox.h);
      } else {
        groups.push(cluster);
        cluster = [r];
        clusterBottom = r.element.bbox.y + r.element.bbox.h;
      }
    }
    groups.push(cluster);
  }

  // Preserve rerank order: sort groups by the best-ranked result within each group.
  return groups.sort((a, b) => {
    const bestA = Math.min(...a.map(r => results.indexOf(r)));
    const bestB = Math.min(...b.map(r => results.indexOf(r)));
    return bestA - bestB;
  });
}

export default function Home() {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);
  const [answer, setAnswer] = useState("");
  const [answerLoading, setAnswerLoading] = useState(false);
  const [rulebooks, setRulebooks] = useState<Rulebook[]>([]);
  const [selectedRulebook, setSelectedRulebook] = useState<string>("");
  const [modalState, setModalState] = useState<{ primary: SearchResult; group: SearchResult[] } | null>(null);
  const [browsing, setBrowsing] = useState(false);
  const searchBarRef = useRef<SearchBarHandle>(null);

  useEffect(() => {
    listRulebooks()
      .then((books) => {
        setRulebooks(books);
        if (books.length === 1) setSelectedRulebook(books[0].id);
      })
      .catch(() => {});
  }, []);

  async function handleSearch(q: string) {
    setQuery(q);
    setLoading(true);
    setAnswerLoading(true);
    setAnswer("");
    setError(null);
    setSearched(true);
    setResults([]);

    try {
      await ask(
        q,
        selectedRulebook,
        (token) => setAnswer((prev) => prev + token),
        (searchResults) => { setResults(searchResults); setLoading(false); },
      );
    } catch {
      setError("Search failed. Is the backend running on localhost:8000?");
    } finally {
      setAnswerLoading(false);
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">
            Board Game Rulebook Search
          </h1>
          <ConnectionStatus />
        </div>
        <p className="text-sm text-gray-500 mt-0.5">
          Answers sourced directly from the rulebook — no assumptions
        </p>
      </header>

      <div className="max-w-5xl mx-auto px-6 py-8 space-y-6">
        <UploadPanel onIngested={() => listRulebooks().then(setRulebooks)} />

        {/* Step 1: pick a game */}
        <div>
          {rulebooks.length === 0 ? (
            <p className="text-sm text-gray-400">No rulebooks yet — upload one above to get started.</p>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              {rulebooks.map((book) => {
                const active = selectedRulebook === book.id;
                return (
                  <button
                    key={book.id}
                    onClick={() => {
                      setSelectedRulebook(book.id);
                      setBrowsing(false);
                      setTimeout(() => searchBarRef.current?.focus(), 0);
                    }}
                    className={`px-4 py-1.5 rounded-full border text-sm font-medium transition-colors ${
                      active
                        ? "bg-blue-600 border-blue-600 text-white shadow-sm"
                        : "bg-white border-gray-300 text-gray-600 hover:border-blue-400 hover:text-blue-600"
                    }`}
                  >
                    {book.name}
                  </button>
                );
              })}
              {selectedRulebook && (
                <button
                  onClick={() => setBrowsing((v) => !v)}
                  className={`px-3 py-1.5 rounded-full border text-xs font-mono transition-colors ${
                    browsing
                      ? "bg-amber-100 border-amber-400 text-amber-800"
                      : "bg-gray-100 border-gray-300 text-gray-500 hover:border-amber-400 hover:text-amber-700"
                  }`}
                >
                  {browsing ? "Close browser" : "Browse pages"}
                </button>
              )}
            </div>
          )}
        </div>

        {/* Page browser — dev tool for auditing ingestion */}
        {browsing && selectedRulebook && (
          <PageBrowser rulebookId={selectedRulebook} onClose={() => setBrowsing(false)} />
        )}

        {/* Step 2: search — only active once a game is selected */}
        <div>
          <SearchBar ref={searchBarRef} onSearch={handleSearch} loading={loading} disabled={!selectedRulebook} />
          {!selectedRulebook && (
            <p id="search-disabled-hint" className="text-xs text-gray-400 mt-1.5">Select a game above to search</p>
          )}
          {selectedRulebook && !searched && (
            <p className="text-xs text-gray-400 mt-1.5">
              Try searching for setup, scoring, movement, or any specific rule
            </p>
          )}
        </div>

        {(answerLoading || answer) && (
          <div className="bg-white border border-gray-200 rounded-xl px-5 py-4 shadow-sm">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Answer</p>
            <p className="text-gray-800 text-sm leading-relaxed whitespace-pre-wrap">
              {answer}
              {answerLoading && (
                <span className="inline-block w-2 h-4 bg-gray-400 ml-0.5 animate-pulse rounded-sm align-middle" />
              )}
            </p>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
            {error}
          </div>
        )}

        {results.length > 0 && (
          <div>
            {(() => {
              const groups = groupResults(results);
              return (
                <>
                  <p className="text-sm text-gray-500 mb-4">
                    {groups.length} result{groups.length !== 1 ? "s" : ""} for
                    &ldquo;{query}&rdquo;
                    {selectedRulebook && (
                      <span className="ml-1">
                        in {rulebooks.find((b) => b.id === selectedRulebook)?.name ?? selectedRulebook}
                      </span>
                    )}
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {groups.map((group) => (
                      <ResultCard
                        key={group.map((r) => r.element.id).join("+")}
                        results={group}
                        onViewPage={(primary) => setModalState({ primary, group })}
                        onViewAnnotation={(el: Element) => {
                          const synthetic: SearchResult = { element: el, score: 1, errata: [], faq: [] };
                          setModalState({ primary: synthetic, group: [synthetic] });
                        }}
                      />
                    ))}
                  </div>
                </>
              );
            })()}
          </div>
        )}

        {modalState && (
          <PageModal primary={modalState.primary} group={modalState.group} onClose={() => setModalState(null)} />
        )}

        {searched && !loading && results.length === 0 && !error && (
          <div className="text-center text-gray-500 py-16">
            No results found for &ldquo;{query}&rdquo;
          </div>
        )}
      </div>
    </main>
  );
}
