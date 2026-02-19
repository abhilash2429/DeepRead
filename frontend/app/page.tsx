"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ingestArxiv, ingestUpload } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function HomePage() {
  const router = useRouter();
  const [arxivRef, setArxivRef] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("Ready");
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const [events, setEvents] = useState<string[]>([]);

  function pushEvent(message: string) {
    const ts = new Date().toLocaleTimeString();
    setEvents((prev) => [`[${ts}] ${message}`, ...prev].slice(0, 30));
  }

  function followProgress(sessionId: string, statusUrl: string) {
    pushEvent(`Listening for status: ${statusUrl}`);
    const es = new EventSource(`${API_BASE}${statusUrl}`);
    es.addEventListener("status", (e) => {
      const data = JSON.parse((e as MessageEvent).data);
      setStatus(data.message || data.step);
      setProgress(data.progress || 0);
      pushEvent(`${data.step || "status"}: ${data.message || "update"}`);
    });
    es.addEventListener("done", (e) => {
      pushEvent("Ingestion finished");
      es.close();
      try {
        const data = JSON.parse((e as MessageEvent).data);
        if (data.failed) {
          setStatus("Ingestion failed");
          setLoading(false);
          return;
        }
      } catch {
        // ignore parse failure and continue to session page
      }
      setLoading(false);
      router.push(`/session/${sessionId}`);
    });
    es.addEventListener("error", (e) => {
      try {
        const data = JSON.parse((e as MessageEvent).data);
        setStatus(data.message || "error");
        pushEvent(`error: ${data.message || "unknown"}`);
      } catch {
        setStatus("Connection issue while streaming status");
        pushEvent("error: streaming connection issue");
      }
      es.close();
      setLoading(false);
    });
  }

  return (
    <main className="mx-auto max-w-3xl space-y-6 p-6">
      <h1 className="text-2xl font-bold">DeepRead</h1>
      <p className="text-sm text-slate-700">Upload a PDF or paste an arXiv ID/URL to start a guided paper walkthrough.</p>

      <div className="rounded border bg-white p-4">
        <label className="mb-1 block text-sm font-medium">arXiv ID or URL</label>
        <div className="flex gap-2">
          <input
            className="w-full rounded border px-3 py-2 text-sm"
            value={arxivRef}
            onChange={(e) => setArxivRef(e.target.value)}
            placeholder="2310.06825 or https://arxiv.org/abs/2310.06825"
          />
          <button
            className="rounded bg-slate-900 px-3 py-2 text-sm text-white disabled:opacity-50"
            disabled={!arxivRef || loading}
            onClick={async () => {
              try {
                setLoading(true);
                setEvents([]);
                setProgress(0);
                setStatus("Starting arXiv ingestion...");
                pushEvent(`Requested arXiv ingest: ${arxivRef}`);
                const result = await ingestArxiv(arxivRef);
                followProgress(result.session_id, result.status_stream_url);
              } catch (err) {
                const message = err instanceof Error ? err.message : "Failed to start arXiv ingestion";
                setStatus(message);
                pushEvent(`error: ${message}`);
                setLoading(false);
              }
            }}
          >
            Ingest arXiv
          </button>
        </div>
      </div>

      <div className="rounded border bg-white p-4">
        <label className="mb-1 block text-sm font-medium">Upload PDF</label>
        <div className="flex gap-2">
          <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <button
            className="rounded bg-slate-900 px-3 py-2 text-sm text-white disabled:opacity-50"
            disabled={!file || loading}
            onClick={async () => {
              if (!file) return;
              try {
                setLoading(true);
                setEvents([]);
                setProgress(0);
                setStatus("Starting PDF upload...");
                pushEvent(`Uploading PDF: ${file.name}`);
                const result = await ingestUpload(file);
                followProgress(result.session_id, result.status_stream_url);
              } catch (err) {
                const message = err instanceof Error ? err.message : "Failed to upload PDF";
                setStatus(message);
                pushEvent(`error: ${message}`);
                setLoading(false);
              }
            }}
          >
            Upload
          </button>
        </div>
      </div>

      <div className="rounded border bg-white p-4">
        <div className="mb-2 text-sm font-medium">Progress</div>
        <div className="h-2 w-full rounded bg-slate-200">
          <div className="h-2 rounded bg-blue-600" style={{ width: `${progress}%` }} />
        </div>
        <div className="mt-2 text-sm text-slate-700">{status}</div>
      </div>

      <div className="rounded border bg-white p-4">
        <div className="mb-2 text-sm font-medium">Live Updates</div>
        <div className="max-h-52 overflow-auto rounded border bg-slate-50 p-2 text-xs text-slate-700">
          {events.length === 0 ? (
            <div>No events yet. Start ingestion to see status updates.</div>
          ) : (
            events.map((entry, i) => (
              <div key={`${entry}-${i}`} className="border-b border-slate-200 py-1 last:border-b-0">
                {entry}
              </div>
            ))
          )}
        </div>
      </div>
    </main>
  );
}
