"use client";

import { useState, useRef, useImperativeHandle, forwardRef } from "react";

interface Props {
  onSearch: (query: string) => void;
  loading?: boolean;
  disabled?: boolean;
}

export interface SearchBarHandle {
  focus: () => void;
}

export const SearchBar = forwardRef<SearchBarHandle, Props>(function SearchBar(
  { onSearch, loading, disabled },
  ref,
) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useImperativeHandle(ref, () => ({
    focus: () => inputRef.current?.focus(),
  }));

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (value.trim()) onSearch(value.trim());
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-3">
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder='Search rules... e.g. "how do I move?" or "what happens when..."'
        disabled={disabled}
        aria-disabled={disabled || undefined}
        aria-describedby={disabled ? "search-disabled-hint" : undefined}
        className="flex-1 border border-gray-300 rounded-lg px-4 py-3 text-base focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white disabled:opacity-40 disabled:cursor-not-allowed"
      />
      <button
        type="submit"
        disabled={loading || disabled || !value.trim()}
        className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-colors"
      >
        {loading ? "Searching..." : "Search"}
      </button>
    </form>
  );
});
