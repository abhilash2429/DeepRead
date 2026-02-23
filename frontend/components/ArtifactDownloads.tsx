"use client";

import { useEffect, useMemo, useState } from "react";
import { getArtifacts } from "@/lib/api";
import { BriefingSectionKey } from "@/lib/types";

type Props = {
  paperId: string;
  sections: Record<BriefingSectionKey, boolean>;
};

export default function ArtifactDownloads({ paperId, sections }: Props) {
  const [items, setItems] = useState<{ kind: string; filename: string; content_type: string; content: string }[]>([]);
  const [loading, setLoading] = useState(false);

  const readiness = useMemo(
    () => ({
      briefing: sections.section_1,
      code: sections.section_4,
      hyperparams: sections.section_6,
      ambiguity_report: sections.section_5,
    }),
    [sections]
  );

  useEffect(() => {
    if (!Object.values(readiness).some(Boolean)) return;
    void (async () => {
      setLoading(true);
      try {
        const payload = await getArtifacts(paperId);
        setItems(payload.items || []);
      } finally {
        setLoading(false);
      }
    })();
  }, [paperId, readiness]);

  function download(kind: string) {
    const item = items.find((entry) => entry.kind === kind || entry.filename.includes(kind));
    if (!item) return;
    const blob = new Blob([item.content], { type: item.content_type });
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = item.filename;
    anchor.click();
    URL.revokeObjectURL(href);
  }

  const buttons = [
    { label: "Briefing (.md)", kind: "briefing", enabled: readiness.briefing },
    { label: "Code (.py)", kind: "code", enabled: readiness.code },
    { label: "Hyperparams (.csv)", kind: "hyperparams", enabled: readiness.hyperparams },
    { label: "Ambiguity Report (.md)", kind: "ambiguity_report", enabled: readiness.ambiguity_report },
  ];

  return (
    <section className="mt-8 pt-6 border-t border-zinc-800">
      <h4 className="text-[10px] uppercase tracking-widest text-zinc-600 mb-3">Downloads</h4>
      <div className="space-y-1.5">
        {buttons.map((button) => (
          <button
            key={button.kind}
            disabled={!button.enabled || loading}
            onClick={() => download(button.kind)}
            className="flex w-full items-center justify-between rounded px-2 py-1.5 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:bg-transparent"
          >
            <span>{button.label}</span>
            <span className="text-[10px] font-mono">v</span>
          </button>
        ))}
      </div>
    </section>
  );
}

