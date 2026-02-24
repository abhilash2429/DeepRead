"use client";

import { DragEvent, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

import { API_BASE, getAuthProfile, ingestArxiv, ingestUpload, isUnauthorizedError } from "@/lib/api";
import { AuthProfile } from "@/lib/types";
import styles from "./upload.module.css";

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
      setError("Enter a valid arXiv paper ID or URL.");
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
    <div className={styles.page}>
      {/* Nav */}
      <nav className={styles.nav}>
        <Link href="/" className={styles.logo}>
          <span className={styles.logoDot} />
          DeepRead
        </Link>

        <div className={styles.navRight}>
          {authChecked && profile && (
            <>
              <Link href="/dashboard" className={styles.dashboardBtn}>
                dashboard
              </Link>
              <span className={styles.userName}>{profile.name}</span>
              <span className={styles.usageTag}>
                {profile.plan === "PRO" ? "Pro" : `${profile.papers_analyzed} / ${profile.limit} papers`}
              </span>
              <button
                className={styles.signOutBtn}
                onClick={async () => {
                  await fetch(`${API_BASE}/auth/logout`, { method: "POST", credentials: "include" });
                  setProfile(null);
                  router.replace("/signin");
                }}
              >
                sign out
              </button>
            </>
          )}
          {authChecked && !profile && (
            <Link href="/signin" className={styles.signOutBtn}>sign in</Link>
          )}
        </div>
      </nav>

      {/* Content */}
      <div className={styles.content}>
        {/* Left — upload form */}
        <div className={styles.left}>
          <div className={styles.eyebrow}>upload paper</div>
          <h1 className={styles.heading}>Drop your paper.</h1>
          <p className={styles.sub}>Upload a PDF or paste an arXiv link to get started.</p>

          {/* Drop zone */}
          <label
            className={`${styles.dropZone} ${dragActive ? styles.dropZoneActive : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
          >
            <input
              type="file"
              accept="application/pdf"
              className="sr-only"
              style={{ display: "none" }}
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
            <div className={styles.dropIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            {file ? (
              <div className={styles.fileChip}>
                <span className={styles.fileChipDot} />
                {file.name}
              </div>
            ) : (
              <div className={styles.dropText}>
                Drag a PDF here, or click to browse
              </div>
            )}
          </label>

          {/* or */}
          <div className={styles.orDivider}>
            <span className={styles.orText}>or</span>
          </div>

          {/* arXiv input */}
          <input
            className={styles.arxivInput}
            value={arxivRef}
            onChange={(e) => setArxivRef(e.target.value)}
            placeholder="arxiv.org/abs/... or 2310.06825"
            disabled={!!file}
          />

          {/* Error */}
          {error && <div className={styles.error}>{error}</div>}

          {/* Submit */}
          <button
            className={styles.btnSubmit}
            disabled={!canSubmit || busy}
            onClick={onAnalyze}
          >
            {busy ? "Starting analysis..." : "Analyze Paper →"}
          </button>
        </div>

        {/* Right — tips */}
        <aside className={styles.right}>
          <div className={styles.rightLabel}> tips</div>
          <div className={styles.tips}>
            <div className={styles.tip}>
              <span className={styles.tipNum}>01</span>
              <span className={styles.tipText}>PDF upload works best with papers that have embedded text layers, not scanned images.</span>
            </div>
            <div className={styles.tip}>
              <span className={styles.tipNum}>02</span>
              <span className={styles.tipText}>Paste an arXiv ID like <code>2310.06825</code> or the full abstract URL.</span>
            </div>
            <div className={styles.tip}>
              <span className={styles.tipNum}>03</span>
              <span className={styles.tipText}>After analysis, you'll get implementation notes, architecture breakdown, and key claims.</span>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
