"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { API_BASE, getAuthProfile } from "@/lib/api";

export default function SignInPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAuthProfile()
      .then(() => router.replace("/upload"))
      .catch(() => setLoading(false));
  }, [router]);

  if (loading) {
    return (
      <main className="flex min-h-screen items-center justify-center px-4">
        <div className="text-sm text-zinc-500">Checking session...</div>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4">
      <section className="surface w-full max-w-md p-6">
        <h1 className="text-lg font-semibold text-zinc-50">Sign in to continue</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Start with Google, then upload your paper and move into the dashboard.
        </p>

        <a
          href={`${API_BASE}/auth/google`}
          className="mt-6 flex w-full items-center justify-center rounded-lg bg-zinc-100 px-4 py-3 text-sm font-medium text-zinc-900 transition-colors hover:bg-zinc-200"
        >
          Continue with Google
        </a>

        <a
          href={`${API_BASE}/auth/github`}
          className="mt-3 flex w-full items-center justify-center rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-3 text-sm text-zinc-200 transition-colors hover:bg-zinc-800"
        >
          Continue with GitHub
        </a>

        <Link href="/" className="mt-4 block text-center text-xs text-zinc-600 transition-colors hover:text-zinc-400">
          Back to home
        </Link>
      </section>
    </main>
  );
}
