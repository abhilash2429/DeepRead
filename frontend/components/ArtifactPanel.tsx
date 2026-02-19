"use client";

import { useEffect, useState } from "react";
import { getArtifacts } from "@/lib/api";
import { ArtifactItem } from "@/lib/types";

export default function ArtifactPanel({ sessionId }: { sessionId: string }) {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<ArtifactItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await getArtifacts(sessionId);
      setItems(res.items as ArtifactItem[]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load artifacts");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (open && items.length === 0) void load();
  }, [open, items.length]);

  return (
    <div className="relative">
      <button className="rounded border bg-white px-2 py-1 text-xs font-semibold" onClick={() => setOpen((v) => !v)}>
        Artifacts
      </button>
      {open && (
        <div className="absolute right-0 z-20 mt-1 min-w-[260px] space-y-2 rounded border bg-white p-2 text-sm shadow-lg">
          {loading && <div className="text-slate-600">Loading artifacts...</div>}
          {error && <div className="text-rose-700">{error}</div>}
          {items.map((item) => (
            <button
              key={item.filename}
              className="block rounded border px-2 py-1"
              onClick={() => {
                const blob = new Blob([item.content], { type: item.content_type });
                const href = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = href;
                a.download = item.filename;
                a.click();
                URL.revokeObjectURL(href);
              }}
            >
              Download {item.filename}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
