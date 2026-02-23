"use client";

import { HyperparameterEntry } from "@/lib/types";

export default function HyperparamTable({ rows }: Readonly<{ rows: HyperparameterEntry[] }>) {
  function exportCsv() {
    const lines = [
      "Name,Value,Source,Status,Suggested Default,Suggested Reasoning",
      ...rows.map((row) =>
        [
          row.name,
          row.value ?? "",
          row.source_section,
          row.status,
          row.suggested_default ?? "",
          row.suggested_reasoning ?? "",
        ]
          .map((part) => `"${String(part).replace(/"/g, '""')}"`)
          .join(",")
      ),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = "hyperparameters.csv";
    anchor.click();
    URL.revokeObjectURL(href);
  }

  return (
    <section className="surface overflow-hidden">
      <header className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/40 px-4 py-3">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">Hyperparameters</h4>
        <button
          className="rounded border border-zinc-700 bg-zinc-800 px-2.5 py-1 text-[11px] font-medium text-zinc-300 hover:bg-zinc-700 transition-colors"
          onClick={exportCsv}
        >
          Export CSV
        </button>
      </header>

      <div className="thin-scroll overflow-x-auto">
        <table className="min-w-full text-sm text-left">
          <thead className="bg-zinc-900/20 text-xs uppercase tracking-wider text-zinc-500">
            <tr>
              <th className="px-4 py-3 font-medium">Name</th>
              <th className="px-4 py-3 font-medium">Value</th>
              <th className="px-4 py-3 font-medium">Source</th>
              <th className="px-4 py-3 font-medium">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {rows.map((row) => {
              const missing = !row.value;
              return (
                <tr key={`${row.name}-${row.source_section}`} className="hover:bg-zinc-900/30 transition-colors">
                  <td className="px-4 py-3 font-medium text-zinc-200">{row.name}</td>
                  <td className="px-4 py-3 text-zinc-300">
                    {row.value ? (
                      row.value
                    ) : (
                      <span
                        className="italic text-zinc-500"
                        title={row.suggested_reasoning || "Agent-suggested default due to missing value"}
                      >
                        {row.suggested_default || "missing"}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-zinc-500">{row.source_section}</td>
                  <td className={`px-4 py-3 text-xs uppercase tracking-wide ${missing ? "text-zinc-500 font-medium" : "text-zinc-400"}`}>
                    {missing ? `${row.status} (default)` : row.status}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
