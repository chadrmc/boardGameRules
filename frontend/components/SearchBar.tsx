"use client";

import { useState } from "react";

interface Props {
  onSearch: (query: string) => void;
  loading?: boolean;
  disabled?: boolean;
}

export function SearchBar({ onSearch, loading, disabled }: Props) {
  const [value, setValue] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (value.trim()) onSearch(value.trim());
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-3">
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder='Search rules... e.g. "how do I move?" or "what happens when..."'
        className="flex-1 border border-gray-300 rounded-lg px-4 py-3 text-base focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
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
}
