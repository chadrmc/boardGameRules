"use client";

import { useState, useEffect } from "react";
import { API_BASE } from "@/lib/constants";

const POLL_INTERVAL = 10_000; // 10 seconds
const TIMEOUT = 3_000; // 3 second timeout per check

export function ConnectionStatus() {
  const [status, setStatus] = useState<"connected" | "disconnected" | "checking">("checking");

  useEffect(() => {
    let mounted = true;

    async function check() {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), TIMEOUT);
        const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
        clearTimeout(timer);
        if (mounted) setStatus(res.ok ? "connected" : "disconnected");
      } catch {
        if (mounted) setStatus("disconnected");
      }
    }

    check();
    const id = setInterval(check, POLL_INTERVAL);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  const dot =
    status === "connected"
      ? "bg-green-500"
      : status === "disconnected"
        ? "bg-red-500"
        : "bg-gray-400 animate-pulse";

  const label =
    status === "connected"
      ? "Backend connected"
      : status === "disconnected"
        ? "Backend unavailable"
        : "Checking...";

  return (
    <div className="flex items-center gap-1.5" title={label}>
      <span className={`inline-block w-2 h-2 rounded-full ${dot}`} />
      <span
        className={`text-xs ${
          status === "connected"
            ? "text-gray-400"
            : status === "disconnected"
              ? "text-red-600 font-medium"
              : "text-gray-400"
        }`}
      >
        {status === "connected"
          ? "Connected"
          : status === "disconnected"
            ? "Backend offline"
            : "Checking…"}
      </span>
    </div>
  );
}
