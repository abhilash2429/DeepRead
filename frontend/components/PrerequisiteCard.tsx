"use client";

import { useState } from "react";
import { PrerequisiteEntry } from "@/lib/types";

export default function PrerequisiteCard({ item }: Readonly<{ item: PrerequisiteEntry }>) {
  const [open, setOpen] = useState(true);

  return (
    <article className="surface transition-all duration-300">
      <button
        className="flex w-full items-center justify-between p-4 cursor-pointer hover:bg-zinc-900/30 transition-colors"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <h4 className="text-sm font-medium text-zinc-100 text-left">{item.concept}</h4>
        <div className="text-xs font-mono tracking-wider text-zinc-500">
          {open ? "[COLLAPSE]" : "[EXPAND]"}
        </div>
      </button>

      {open && (
        <div className="border-t border-zinc-900/50 p-4 space-y-4">
          <div className="space-y-1">
            <h5 className="text-[10px] uppercase tracking-widest text-zinc-500 font-medium">Problem it solves</h5>
            <p className="text-sm text-zinc-400 leading-relaxed">{item.problem}</p>
          </div>
          <div className="space-y-1">
            <h5 className="text-[10px] uppercase tracking-widest text-zinc-500 font-medium">Core Mechanism</h5>
            <p className="text-sm text-zinc-400 leading-relaxed">{item.solution}</p>
          </div>
          <div className="space-y-1">
            <h5 className="text-[10px] uppercase tracking-widest text-zinc-500 font-medium">Usage in this paper</h5>
            <p className="text-sm text-zinc-400 leading-relaxed">{item.usage_in_paper}</p>
          </div>
        </div>
      )}
    </article>
  );
}
