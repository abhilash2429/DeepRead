"use client";

import { useEffect, useMemo, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";

import { pdfUrl } from "@/lib/api";

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/legacy/build/pdf.worker.min.mjs`;

type Props = {
  paperId: string;
  open: boolean;
  onClose: () => void;
  pdfReady: boolean;
};

export default function PdfPanel({ paperId, open, onClose, pdfReady }: Props) {
  const [zoom, setZoom] = useState(1);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    if (pdfReady && open) {
      setRetryKey((k) => k + 1);
      setError(null);
    }
  }, [pdfReady, open]);

  useEffect(() => {
    if (pages <= 0) return;
    setPage((prev) => Math.min(Math.max(prev, 1), pages));
  }, [pages]);

  function prevPage() {
    setPage((p) => Math.max(1, p - 1));
  }

  function nextPage() {
    if (pages <= 0) return;
    setPage((p) => Math.min(pages, p + 1));
  }

  const panelClass = useMemo(
    () =>
      `surface border-y-0 border-l-0 rounded-none fixed left-0 top-0 z-40 h-screen w-full transition-transform duration-300 ease-in-out sm:w-[36vw] ${
        open ? "translate-x-0" : "-translate-x-full"
      }`,
    [open]
  );

  return (
    <aside className={panelClass}>
      <header className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/50 px-4 py-3">
        <div className="text-xs font-medium uppercase tracking-wider text-zinc-300">Original Document</div>
        <button className="text-zinc-500 transition-colors hover:text-zinc-200" onClick={onClose} aria-label="Close PDF panel">
          x
        </button>
      </header>

      <div className="thin-scroll h-[calc(100%-110px)] overflow-auto bg-zinc-950 p-4">
        {!pdfReady && (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <div className="mb-4 h-6 w-6 animate-spin rounded-full border-2 border-zinc-800 border-t-zinc-400" />
            <div className="text-sm text-zinc-500">PDF will be available once ingestion completes...</div>
          </div>
        )}
        {pdfReady && error && (
          <div className="mb-4 flex flex-col items-center gap-3">
            <div className="rounded border border-zinc-800 bg-zinc-900 px-4 py-3 text-sm text-zinc-400">{error}</div>
            <button
              className="rounded border border-zinc-700 bg-zinc-800 px-4 py-1.5 text-xs font-medium text-zinc-300 transition-colors hover:bg-zinc-700"
              onClick={() => {
                setError(null);
                setRetryKey((k) => k + 1);
              }}
            >
              Retry
            </button>
          </div>
        )}
        {pdfReady && !error && (
          <Document
            key={retryKey}
            file={pdfUrl(paperId)}
            options={{ withCredentials: true }}
            onLoadSuccess={(doc) => {
              setPages(doc.numPages);
              setPage(1);
              setError(null);
            }}
            onLoadError={(err) => setError(String(err))}
            className="flex flex-col items-center"
          >
            <Page
              pageNumber={page}
              scale={zoom}
              renderAnnotationLayer={false}
              renderTextLayer={false}
              className="border border-zinc-800 shadow-xl"
              loading={<div className="my-10 text-sm text-zinc-600">Rendering page...</div>}
              onRenderError={(err) => setError(String(err))}
            />
          </Document>
        )}
      </div>

      <footer className="border-t border-zinc-800 bg-zinc-900/50 px-4 py-3">
        <div className="flex items-center justify-between font-mono text-xs text-zinc-400">
          <div className="flex items-center gap-3">
            <button className="rounded px-1.5 py-0.5 transition-colors hover:bg-zinc-800 hover:text-zinc-200" onClick={() => setZoom((z) => Math.max(0.6, z - 0.1))}>
              -
            </button>
            <span>{Math.round(zoom * 100)}%</span>
            <button className="rounded px-1.5 py-0.5 transition-colors hover:bg-zinc-800 hover:text-zinc-200" onClick={() => setZoom((z) => Math.min(2.2, z + 0.1))}>
              +
            </button>
          </div>
          <div className="flex items-center gap-3">
            <button
              className="font-sans text-[10px] uppercase tracking-widest transition-colors hover:text-zinc-200 disabled:cursor-not-allowed disabled:opacity-40"
              onClick={prevPage}
              disabled={page <= 1 || pages <= 0}
            >
              Prev
            </button>
            <span>
              {page} / {pages || "?"}
            </span>
            <button
              className="font-sans text-[10px] uppercase tracking-widest transition-colors hover:text-zinc-200 disabled:cursor-not-allowed disabled:opacity-40"
              onClick={nextPage}
              disabled={pages <= 0 || page >= pages}
            >
              Next
            </button>
          </div>
        </div>
      </footer>
    </aside>
  );
}
