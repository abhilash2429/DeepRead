"use client";

import { useRouter } from "next/navigation";
import ThemeToggle from "@/components/ThemeToggle";

export default function HeroPage() {
  const router = useRouter();

  return (
    <main className="flex min-h-screen items-center justify-center px-6">
      <div className="absolute right-6 top-6">
        <ThemeToggle />
      </div>
      <section className="flex max-w-xl flex-col items-center text-center">
        <h1 className="text-5xl font-semibold tracking-tight text-zinc-50 sm:text-6xl">
          DeepRead
        </h1>
        <p className="mt-4 text-base text-zinc-500">
          Drop a research paper. Get everything you need to implement it.
        </p>
        <button
          className="mt-10 rounded-lg bg-zinc-800 px-6 py-3 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-700"
          onClick={() => router.push("/upload")}
        >
          Get Started
        </button>
      </section>
    </main>
  );
}
