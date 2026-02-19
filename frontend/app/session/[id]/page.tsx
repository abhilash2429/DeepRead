"use client";

import { useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { Document, Page, pdfjs } from "react-pdf";
import ArtifactPanel from "@/components/ArtifactPanel";
import ChatPanel from "@/components/ChatPanel";
import ComponentGraph from "@/components/ComponentGraph";
import { getConversationState } from "@/lib/api";
import { ConversationState, Stage } from "@/lib/types";

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/legacy/build/pdf.worker.min.mjs`;

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function SessionPage() {
  const params = useParams<{ id: string }>();
  const sessionId = params.id;
  const [state, setState] = useState<ConversationState | null>(null);
  const [stage, setStage] = useState<Stage>("orientation");
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const [pdfOpen, setPdfOpen] = useState(true);
  const [graphOpen, setGraphOpen] = useState(false);
  const [pdfWidth, setPdfWidth] = useState(460);
  const [resizing, setResizing] = useState(false);
  const [pageNumber, setPageNumber] = useState(1);
  const [numPages, setNumPages] = useState(0);
  const [pageWidth, setPageWidth] = useState(420);
  const viewerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const s = await getConversationState(sessionId);
        setState(s);
        setStage(s.current_stage);
      } catch (err) {
        setSessionError(err instanceof Error ? err.message : "Failed to load session");
      }
    })();
  }, [sessionId]);

  useEffect(() => {
    const node = viewerRef.current;
    if (!node) return;
    const resizeObserver = new ResizeObserver((entries) => {
      const nextWidth = Math.max(260, Math.floor(entries[0].contentRect.width - 24));
      setPageWidth(nextWidth);
    });
    resizeObserver.observe(node);
    return () => resizeObserver.disconnect();
  }, [pdfOpen]);

  useEffect(() => {
    const onMove = (event: MouseEvent) => {
      if (!resizing) return;
      const min = 320;
      const max = Math.floor(window.innerWidth * 0.68);
      setPdfWidth(Math.min(max, Math.max(min, event.clientX)));
    };
    const onUp = () => setResizing(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [resizing]);

  if (sessionError) return <div className="p-6 text-rose-700">{sessionError}</div>;
  if (!state) return <div className="p-6">Loading session...</div>;

  return (
    <main className="h-screen overflow-hidden bg-slate-100 p-2">
      <div className="flex h-full min-h-0 overflow-hidden rounded-xl border border-slate-300 bg-white">
        {pdfOpen ? (
          <aside
            className="relative flex h-full min-w-[320px] flex-col border-r border-slate-300 bg-slate-50"
            style={{ width: `${pdfWidth}px` }}
          >
            <div className="flex items-center justify-between border-b border-slate-300 px-3 py-2">
              <div className="text-sm font-semibold text-slate-800">PDF Preview</div>
              <div className="flex items-center gap-1">
                <button className="rounded border px-2 py-1 text-xs" onClick={() => setPageNumber((p) => Math.max(1, p - 1))}>
                  Prev
                </button>
                <button
                  className="rounded border px-2 py-1 text-xs"
                  onClick={() => setPageNumber((p) => Math.min(numPages || p + 1, p + 1))}
                >
                  Next
                </button>
                <button className="rounded border px-2 py-1 text-xs" onClick={() => setPdfOpen(false)}>
                  Hide
                </button>
              </div>
            </div>
            <div ref={viewerRef} className="min-h-0 flex-1 overflow-auto p-3">
              {pdfError && <div className="mb-2 rounded border border-rose-200 bg-rose-50 p-2 text-xs text-rose-700">{pdfError}</div>}
              <Document
                file={`${API_BASE}/ingest/${sessionId}/pdf`}
                loading={<div className="text-sm text-slate-600">Loading PDF...</div>}
                onLoadSuccess={(doc) => {
                  setNumPages(doc.numPages);
                  setPdfError(null);
                }}
                onLoadError={(err) => setPdfError(String(err))}
                onSourceError={(err) => setPdfError(String(err))}
              >
                <Page
                  pageNumber={pageNumber}
                  width={pageWidth}
                  renderAnnotationLayer={false}
                  renderTextLayer={false}
                  loading={<div className="text-sm text-slate-600">Rendering page...</div>}
                />
              </Document>
            </div>
            <div className="border-t border-slate-300 px-3 py-2 text-xs text-slate-600">
              Page {pageNumber} / {numPages || "?"}
            </div>
            <div
              role="separator"
              aria-label="Resize PDF panel"
              onMouseDown={() => setResizing(true)}
              className="absolute right-0 top-0 h-full w-2 cursor-col-resize bg-transparent hover:bg-slate-300"
            />
          </aside>
        ) : (
          <button
            className="flex w-10 items-center justify-center border-r border-slate-300 bg-slate-100 text-xs font-semibold text-slate-700"
            onClick={() => setPdfOpen(true)}
          >
            PDF
          </button>
        )}

        <section className="flex min-w-0 flex-1 flex-col">
          <div className="flex flex-wrap items-center gap-2 border-b border-slate-300 px-3 py-2">
            <span className="rounded bg-slate-900 px-2 py-1 text-xs text-white">Stage: {stage}</span>
            {state.user_level && (
              <span className="rounded bg-slate-200 px-2 py-1 text-xs text-slate-700">User level: {state.user_level}</span>
            )}
            <button className="rounded border px-2 py-1 text-xs" onClick={() => setGraphOpen((v) => !v)}>
              {graphOpen ? "Hide" : "Show"} Component Graph
            </button>
            {!pdfOpen && (
              <button className="rounded border px-2 py-1 text-xs" onClick={() => setPdfOpen(true)}>
                Open PDF Panel
              </button>
            )}
            <div className="ml-auto">
              <ArtifactPanel sessionId={sessionId} />
            </div>
          </div>

          {graphOpen && (
            <div className="max-h-[260px] overflow-auto border-b border-slate-300 p-2">
              <ComponentGraph
                edgesData={state.internal_representation.component_graph}
                onSelectComponent={(component) => {
                  setStage("architecture");
                  window.dispatchEvent(new CustomEvent("deepread-component-focus", { detail: { component } }));
                }}
              />
            </div>
          )}

          <div className="min-h-0 flex-1">
            <ChatPanel sessionId={sessionId} initialState={state} onStageChange={setStage} />
          </div>
        </section>
      </div>
    </main>
  );
}
