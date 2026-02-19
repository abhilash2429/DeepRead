"use client";

import { useMemo, useState } from "react";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import "highlight.js/styles/github.css";

hljs.registerLanguage("python", python);

const badgeClass: Record<string, string> = {
  "paper-stated": "bg-emerald-600 text-white",
  inferred: "bg-amber-500 text-black",
  assumed: "bg-rose-600 text-white",
  missing: "bg-rose-700 text-white",
};

type Props = {
  code: string;
  provenance: "paper-stated" | "inferred" | "assumed" | "missing";
};

export default function CodeBlock({ code, provenance }: Props) {
  const [showNotes, setShowNotes] = useState(false);
  const highlighted = useMemo(() => hljs.highlight(code, { language: "python" }).value, [code]);
  const notes = code
    .split("\n")
    .filter((line) => line.includes("# ASSUMED:") || line.includes("# INFERRED:"))
    .map((line) => line.trim());

  return (
    <div className="relative rounded-md border bg-white">
      <div className="absolute right-2 top-2 flex gap-2">
        <span className={`rounded px-2 py-1 text-xs ${badgeClass[provenance]}`}>{provenance}</span>
        <button
          className="rounded border px-2 py-1 text-xs"
          onClick={() => navigator.clipboard.writeText(code)}
        >
          Copy
        </button>
      </div>
      <pre className="overflow-x-auto p-4 pt-10 text-sm">
        <code dangerouslySetInnerHTML={{ __html: highlighted }} />
      </pre>
      <div className="border-t p-2">
        <button className="text-xs underline" onClick={() => setShowNotes((s) => !s)}>
          {showNotes ? "Hide notes" : "Show assumptions/inferences"}
        </button>
        {showNotes && (
          <ul className="mt-2 list-disc pl-5 text-xs">
            {notes.map((note, i) => (
              <li key={`${note}-${i}`}>{note}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

