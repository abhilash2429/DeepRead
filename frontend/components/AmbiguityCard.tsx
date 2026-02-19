"use client";

import { useState } from "react";
import { AmbiguityEntry } from "@/lib/types";

type Props = {
  ambiguity: AmbiguityEntry;
  onResolve: (id: string, resolution: string) => Promise<void>;
};

export default function AmbiguityCard({ ambiguity, onResolve }: Props) {
  const [value, setValue] = useState("");
  const [saving, setSaving] = useState(false);

  if (ambiguity.resolved) {
    return (
      <div className="rounded border bg-emerald-50 p-3 text-sm">
        <div className="font-semibold">Resolved: {ambiguity.ambiguity_id}</div>
      </div>
    );
  }

  return (
    <div className="rounded border bg-rose-50 p-3">
      <div className="text-sm font-semibold">{ambiguity.ambiguous_point}</div>
      <div className="mt-2 text-xs">
        <div>Impact: {ambiguity.implementation_consequence}</div>
        <div className="mt-1">Agent resolution: {ambiguity.best_guess_resolution}</div>
      </div>
      <div className="mt-3 flex gap-2">
        <input
          className="w-full rounded border px-2 py-1 text-sm"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Your resolution"
        />
        <button
          className="rounded bg-slate-900 px-3 py-1 text-sm text-white disabled:opacity-50"
          disabled={!value || saving}
          onClick={async () => {
            setSaving(true);
            try {
              await onResolve(ambiguity.ambiguity_id, value);
            } finally {
              setSaving(false);
            }
          }}
        >
          Confirm
        </button>
      </div>
    </div>
  );
}

