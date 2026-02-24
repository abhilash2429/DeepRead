"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { API_BASE, getAuthProfile, isUnauthorizedError } from "@/lib/api";
import { AuthProfile } from "@/lib/types";
import styles from "./dashboard.module.css";

export default function DashboardPage() {
  const router = useRouter();
  const [profile, setProfile] = useState<AuthProfile | null>(null);
  const [activePaperId, setActivePaperId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAuthProfile()
      .then((next) => {
        setProfile(next);
        setActivePaperId(sessionStorage.getItem("deepread_active_paper_id"));
        setLoading(false);
      })
      .catch((error) => {
        if (isUnauthorizedError(error)) {
          router.replace("/signin");
          return;
        }
        router.replace("/signin");
      });
  }, [router]);

  if (loading) {
    return <div className={styles.loading}>loading dashboard...</div>;
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
          {profile && (
            <>
              <span className={styles.planBadge}>
                {profile.plan === "PRO" ? "Pro" : "Free"}
              </span>
              <span className={styles.userName}>{profile.name}</span>
              <button
                className={styles.signOutBtn}
                onClick={async () => {
                  await fetch(`${API_BASE}/auth/logout`, { method: "POST", credentials: "include" });
                  router.replace("/signin");
                }}
              >
                sign out
              </button>
            </>
          )}
          <Link href="/upload" className={styles.uploadLink}>
            + New Paper
          </Link>
        </div>
      </nav>

      {/* Main */}
      <div className={styles.content}>
        {/* Greeting */}
        <div className={styles.eyebrow}>dashboard</div>
        <h1 className={styles.greeting}>
          {profile ? `Welcome back, ${profile.name.split(" ")[0]}.` : "Welcome back."}
        </h1>

        {/* Stats row */}
        <div className={styles.statsRow}>
          <div className={styles.statCard}>
            <div className={styles.statLabel}>Papers Analyzed</div>
            <div className={styles.statValue}>{profile?.papers_analyzed ?? 0}</div>
            <div className={styles.statSub}>of {profile?.limit ?? 10} on free plan</div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statLabel}>Active paper</div>
            <div className={styles.statValue}>{activePaperId ? "1" : "—"}</div>
            <div className={styles.statSub}>{activePaperId ? "ready to open" : "none uploaded"}</div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statLabel}>Plan</div>
            <div className={styles.statValue}>{profile?.plan === "PRO" ? "Pro" : "Free"}</div>
            <div className={styles.statSub}>{profile?.plan === "PRO" ? "unlimited papers" : `${(typeof profile?.limit === "number" ? profile.limit : 10) - (profile?.papers_analyzed ?? 0)} remaining`}</div>
          </div>
        </div>

        {/* Active paper */}
        <div className={styles.sectionLabel}>active paper</div>

        {activePaperId ? (
          <div className={styles.paperCard}>
            <div className={styles.paperMeta}>{activePaperId}</div>
            <div className={styles.paperTitle}>Paper loaded — open the analysis session to view insights.</div>
            <div className={styles.paperActions}>
              <Link href="/session/analyze" className={styles.btnPrimary}>
                Open Analysis →
              </Link>
              <Link href="/upload" className={styles.btnGhost}>
                Upload Another
              </Link>
            </div>
          </div>
        ) : (
          <div className={styles.paperEmpty}>
            <div className={styles.paperMeta}>No active paper. Upload one to get started.</div>
          </div>
        )}

        {/* Recent sessions placeholder */}
        <div className={styles.sectionLabel}>recent sessions</div>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Paper ID</th>
              <th>Date</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {activePaperId ? (
              <tr>
                <td>{activePaperId}</td>
                <td>Today</td>
                <td><span className={`${styles.pill} ${styles.pillReady}`}>Ready</span></td>
                <td>
                  <Link href="/session/analyze" className={styles.btnGhost} style={{ padding: "5px 12px", fontSize: "12px" }}>
                    Open
                  </Link>
                </td>
              </tr>
            ) : (
              <tr>
                <td colSpan={4} style={{ color: "#3f3f46", fontFamily: "var(--font-mono)", fontSize: "12px", paddingTop: "20px" }}>
                  no sessions yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
