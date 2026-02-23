"use client";

import { useMemo, useState } from "react";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";

hljs.registerLanguage("python", python);

type Props = Readonly<{
  code: string;
  provenance?: "paper-stated" | "inferred" | "assumed" | "missing";
  language?: string;
  title?: string;
}>;

export default function CodeBlock({ code, provenance = "paper-stated", language = "python", title }: Props) {
  const [open, setOpen] = useState(false);

  const highlighted = useMemo(() => {
    try {
      if (language === "python" || language.includes("py")) {
        return hljs.highlight(code, { language: "python" }).value;
      }
      return hljs.highlightAuto(code).value;
    } catch {
      return code;
    }
  }, [code, language]);

  const notes = code
    .split("\n")
    .filter((line) => line.includes("# ASSUMED:") || line.includes("# INFERRED:"))
    .map((line) => line.trim());

  return (
    <article className="surface overflow-hidden">
      <header className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/50 px-4 py-2.5">
        <div className="flex items-center gap-3">
          {title && <span className="font-mono text-xs text-zinc-300">{title}</span>}
          {provenance && provenance !== "paper-stated" && (
            <span className="rounded-full border border-zinc-700 bg-zinc-800 px-2 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400">
              {provenance}
            </span>
          )}
        </div>
        <button
          className="rounded text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          onClick={() => navigator.clipboard.writeText(code)}
        >
          Copy
        </button>
      </header>

      <pre className="thin-scroll overflow-x-auto p-4 text-sm bg-zinc-950">
        <code dangerouslySetInnerHTML={{ __html: highlighted }} />
      </pre>

      {notes.length > 0 && (
        <footer className="border-t border-zinc-800 bg-zinc-900/50 px-4 py-3 text-xs">
          <button
            className="text-[11px] text-zinc-500 hover:text-zinc-300 transition-colors"
            onClick={() => setOpen((value) => !value)}
          >
            {open ? "− Hide analytical notes" : "+ Show analytical notes"}
          </button>
          {open && (
            <ul className="mt-2 space-y-1.5 border-l-2 border-zinc-700 pl-3">
              {notes.map((note) => (
                <li key={note} className="text-zinc-400 font-mono text-[10px]">
                  {note}
                </li>
              ))}
            </ul>
          )}
        </footer>
      )}
    </article>
  );
}
