"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { getAuthProfile, isUnauthorizedError } from "@/lib/api";
import { AuthProfile } from "@/lib/types";

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
    return (
      <main className="flex min-h-screen items-center justify-center px-4">
        <div className="text-sm text-zinc-500">Loading dashboard...</div>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4">
      <section className="surface w-full max-w-2xl p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold text-zinc-50">Dashboard</h1>
            <p className="mt-1 text-sm text-zinc-500">
              {profile ? `Welcome back, ${profile.name}.` : "Welcome back."}
            </p>
          </div>
          <span className="rounded-full border border-zinc-700 px-2.5 py-1 text-[10px] uppercase tracking-wider text-zinc-400">
            {profile?.plan ?? "Free"}
          </span>
        </div>

        <div className="mt-6 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
          <div className="text-xs uppercase tracking-wider text-zinc-500">Current Paper</div>
          <div className="mt-2 text-sm text-zinc-300">
            {activePaperId ? `Paper ID: ${activePaperId}` : "No active paper yet. Start a new analysis from upload."}
          </div>
        </div>

        <div className="mt-6 flex flex-wrap gap-3">
          <Link
            href="/upload"
            className="rounded-lg border border-zinc-700 px-4 py-2.5 text-sm text-zinc-200 transition-colors hover:bg-zinc-800"
          >
            Upload Another Paper
          </Link>
          <Link
            href={activePaperId ? "/session/analyze" : "/upload"}
            className="rounded-lg bg-zinc-100 px-4 py-2.5 text-sm font-medium text-zinc-900 transition-colors hover:bg-zinc-200"
          >
            {activePaperId ? "Open Live Analysis" : "Go To Upload"}
          </Link>
        </div>
      </section>
    </main>
  );
}
