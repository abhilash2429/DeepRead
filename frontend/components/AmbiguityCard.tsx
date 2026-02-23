"use client";

import { useState } from "react";
import { AmbiguityEntry } from "@/lib/types";

type Props = Readonly<{
  ambiguity: AmbiguityEntry;
  onResolve: (id: string, resolution: string) => Promise<void>;
}>;

export default function AmbiguityCard({ ambiguity, onResolve }: Props) {
  const [value, setValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [collapsed, setCollapsed] = useState(ambiguity.resolved);

  if (collapsed) {
    return (
      <article className="surface px-4 py-3 text-sm text-zinc-400">
        <span className="text-zinc-500 mr-2">Resolved:</span>
        {ambiguity.title || ambiguity.ambiguity_id}
      </article>
    );
  }

  return (
    <article className="surface p-4 space-y-4">
      <header className="flex items-start justify-between gap-3">
        <span className="rounded bg-zinc-800 px-2 py-1 text-[10px] font-medium uppercase tracking-wider text-zinc-300">
          {ambiguity.ambiguity_type.replace(/_/g, " ")}
        </span>
        <span className="text-[10px] uppercase tracking-wider text-zinc-500">
          Confidence {Math.round((ambiguity.confidence || 0) * 100)}%
        </span>
      </header>

      <div>
        <h4 className="text-sm font-medium text-zinc-100">{ambiguity.title || ambiguity.ambiguous_point}</h4>
        <p className="mt-1.5 text-sm leading-relaxed text-zinc-400">{ambiguity.implementation_consequence}</p>
      </div>

      <div className="rounded-lg bg-zinc-900/50 p-3 text-sm border border-zinc-800/50">
        <span className="text-zinc-500 font-medium mr-2">Agent resolution:</span>
        <span className="text-zinc-300">{ambiguity.agent_resolution}</span>
      </div>

      <div className="flex gap-2">
        <input
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="Override with your decision"
          className="w-full rounded bg-zinc-900 px-3 py-2 text-sm text-zinc-200 border border-zinc-800 placeholder:text-zinc-600 focus:border-zinc-500 focus:outline-none transition-colors"
        />
        <button
          className="rounded bg-zinc-800 px-4 py-2 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-700 disabled:opacity-40"
          disabled={!value.trim() || saving}
          onClick={async () => {
            setSaving(true);
            try {
              await onResolve(ambiguity.ambiguity_id, value.trim());
              setCollapsed(true);
            } finally {
              setSaving(false);
            }
          }}
        >
          {saving ? "..." : "Confirm"}
        </button>
      </div>
    </article>
  );
}
