"use client";

import { DragEvent, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { API_BASE, getAuthProfile, ingestArxiv, ingestUpload, isUnauthorizedError } from "@/lib/api";
import { AuthProfile } from "@/lib/types";

export default function UploadPage() {
  const router = useRouter();
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [arxivRef, setArxivRef] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [profile, setProfile] = useState<AuthProfile | null>(null);
  const [authChecked, setAuthChecked] = useState(false);

  const canSubmit = useMemo(() => Boolean(file || arxivRef.trim()), [file, arxivRef]);

  function isValidArxivReference(input: string): boolean {
    const value = input.trim();
    if (!value) return false;
    const idPattern = /^(?:arXiv:)?(?:(?:\d{4}\.\d{4,5})|(?:[a-z\-]+(?:\.[A-Z]{2})?\/\d{7}))(?:v\d+)?$/i;
    const urlPattern = /^https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf)\/(?:(?:\d{4}\.\d{4,5})|(?:[a-z\-]+(?:\.[A-Z]{2})?\/\d{7}))(?:v\d+)?(?:\.pdf)?$/i;
    return idPattern.test(value) || urlPattern.test(value);
  }

  useEffect(() => {
    getAuthProfile()
      .then((p) => { setProfile(p); setAuthChecked(true); })
      .catch(() => {
        setProfile(null);
        setAuthChecked(true);
        router.replace("/signin");
      });
  }, [router]);

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragActive(false);
    const dropped = event.dataTransfer.files?.[0];
    if (!dropped) return;
    if (!dropped.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }
    setError(null);
    setFile(dropped);
  }

  async function onAnalyze() {
    if (!canSubmit || busy) return;
    if (!profile) { setError("Please sign in first."); return; }
    if (!file && !isValidArxivReference(arxivRef)) {
      setError("Enter a valid research paper ID or URL.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = file ? await ingestUpload(file) : await ingestArxiv(arxivRef.trim());
      sessionStorage.setItem("deepread_active_paper_id", result.paper_id);
      router.push("/dashboard");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to start analysis.";
      if (isUnauthorizedError(err) || msg.includes("Authentication")) {
        setError("Session expired. Please sign in again.");
        setProfile(null);
        router.replace("/signin");
      } else {
        setError(msg);
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4">
      <section className="surface w-full max-w-md p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-zinc-50">Analyze a Paper</h1>
            <p className="mt-1 text-sm text-zinc-500">Upload a PDF or paste an arXiv link.</p>
          </div>
          <div className="flex items-center gap-4">
            {authChecked && profile && (
              <div className="text-right">
                <div className="text-xs font-medium text-zinc-300">{profile.name}</div>
                <div className="text-[10px] text-zinc-600">
                  {profile.plan === "PRO" ? "Pro" : `${profile.papers_analyzed}/${profile.limit}`}
                </div>
              </div>
            )}
          </div>
        </div>

        <label
          className={`mt-6 block cursor-pointer rounded-lg border border-dashed px-4 py-8 text-center transition-colors ${dragActive ? "border-zinc-500 bg-zinc-800/50" : "border-zinc-700 bg-zinc-900/50"
            }`}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={onDrop}
        >
          <input
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={(e) => {
              const selected = e.target.files?.[0] || null;
              if (selected && !selected.name.toLowerCase().endsWith(".pdf")) {
                setError("Only PDF files are supported.");
                return;
              }
              setError(null);
              setFile(selected);
            }}
          />
          <div className="text-sm text-zinc-500">
            {file ? file.name : "Drag and drop PDF here, or click to browse"}
          </div>
        </label>

        <div className="my-4 text-center text-xs text-zinc-600">or</div>

        <input
          className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-zinc-600 focus:outline-none"
          value={arxivRef}
          onChange={(e) => setArxivRef(e.target.value)}
          placeholder="arxiv.org/abs/... or 2310.06825"
        />

        {error && <div className="mt-4 rounded-lg bg-zinc-900 border border-zinc-700 px-3 py-2 text-sm text-zinc-400">{error}</div>}

        <button
          className="mt-6 w-full rounded-lg bg-zinc-800 px-4 py-3 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
          disabled={!canSubmit || busy}
          onClick={onAnalyze}
        >
          {busy ? "Starting..." : "Analyze"}
        </button>

        {authChecked && !profile && (
          <a
            href="/signin"
            className="mt-4 flex items-center justify-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 text-sm text-zinc-300 transition-colors hover:bg-zinc-800"
          >
            Go to sign in
          </a>
        )}

        {authChecked && profile && (
          <button
            onClick={async () => {
              await fetch(`${API_BASE}/auth/logout`, { method: "POST", credentials: "include" });
              setProfile(null);
            }}
            className="mt-3 block w-full text-center text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
          >
            Sign out
          </button>
        )}
      </section>
    </main>
  );
}
