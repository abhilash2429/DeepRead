import Link from "next/link";

import { EXAMPLE_LIST } from "@/lib/examples";

export default function ExamplesIndexPage() {
  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-12">
      <div className="mx-auto w-full max-w-5xl">
        <div className="mb-8 flex items-center justify-between">
          <h1 className="text-3xl font-semibold text-zinc-100">Implementation Walkthroughs</h1>
          <Link href="/" className="text-sm text-zinc-400 hover:text-zinc-200">
            Back to home
          </Link>
        </div>

        <p className="mb-10 max-w-3xl text-sm leading-7 text-zinc-400">
          Live paper analysis is temporarily paused due to API capacity. Explore these static walkthroughs to see how
          DeepRead structures architecture, implementation, ambiguity resolution, and training guidance.
        </p>

        <div className="grid gap-4 md:grid-cols-3">
          {EXAMPLE_LIST.map((example) => (
            <article key={example.slug} className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
              <div className="mb-2 text-xs uppercase tracking-[0.18em] text-zinc-500">{example.title}</div>
              <h2 className="mb-2 text-lg font-semibold text-zinc-100">{example.paperTitle}</h2>
              <p className="mb-4 text-sm leading-6 text-zinc-400">{example.summary}</p>
              <Link
                href={`/examples/${example.slug}`}
                className="inline-flex rounded-md border border-zinc-700 px-3 py-2 text-xs text-zinc-200 transition-colors hover:bg-zinc-800"
              >
                Open Walkthrough
              </Link>
            </article>
          ))}
        </div>
      </div>
    </main>
  );
}
