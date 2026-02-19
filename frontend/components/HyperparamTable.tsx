"use client";

import { HyperparameterEntry } from "@/lib/types";

const statusColor: Record<string, string> = {
  "paper-stated": "text-emerald-700",
  inferred: "text-amber-700",
  missing: "text-rose-700",
  assumed: "text-rose-700",
};

export default function HyperparamTable({ rows }: { rows: HyperparameterEntry[] }) {
  return (
    <div className="overflow-x-auto rounded-md border bg-white">
      <table className="min-w-full text-left text-sm">
        <thead className="bg-slate-50">
          <tr>
            <th className="p-2">Name</th>
            <th className="p-2">Value</th>
            <th className="p-2">Source Section</th>
            <th className="p-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.name}-${r.source_section}`} className="border-t">
              <td className="p-2">{r.name}</td>
              <td className="p-2">
                {r.value || (
                  <span className="italic text-slate-500">{r.suggested_default || "missing"}</span>
                )}
              </td>
              <td className="p-2">{r.source_section}</td>
              <td className={`p-2 font-medium ${statusColor[r.status] || ""}`}>{r.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

