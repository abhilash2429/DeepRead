import Link from "next/link";
import { notFound } from "next/navigation";

import ExampleWalkthroughDocument from "@/components/ExampleWalkthroughDocument";
import { EXAMPLE_LIST, EXAMPLE_WALKTHROUGHS, ExampleSlug } from "@/lib/examples";

type ExamplePageProps = {
  params: {
    slug: string;
  };
};

export function generateStaticParams(): Array<{ slug: string }> {
  return EXAMPLE_LIST.map((example) => ({ slug: example.slug }));
}

export default function ExampleWalkthroughPage({ params }: ExamplePageProps) {
  const slug = params.slug as ExampleSlug;
  const walkthrough = EXAMPLE_WALKTHROUGHS[slug];
  if (!walkthrough) notFound();

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-10">
      <div className="mx-auto w-full max-w-4xl">
        <div className="mb-8 flex items-center justify-between">
          <Link href="/examples" className="text-sm text-zinc-400 hover:text-zinc-200">
            Back to examples
          </Link>
          <Link href="/" className="text-sm text-zinc-400 hover:text-zinc-200">
            Home
          </Link>
        </div>

        <header className="mb-10 rounded-xl border border-zinc-800 bg-zinc-900/60 p-6">
          <div className="mb-2 text-xs uppercase tracking-[0.2em] text-zinc-500">{walkthrough.title}</div>
          <h1 className="mb-3 text-2xl font-semibold text-zinc-100">{walkthrough.paperTitle}</h1>
          <p className="mb-4 text-sm leading-7 text-zinc-300">{walkthrough.summary}</p>
          <div className="flex flex-wrap gap-2">
            {walkthrough.badges.map((badge) => (
              <span
                key={badge}
                className="rounded border border-zinc-700 bg-zinc-800/70 px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-zinc-300"
              >
                {badge}
              </span>
            ))}
          </div>
        </header>

        <section className="mb-10">
          <ExampleWalkthroughDocument sections={walkthrough.sections} />
        </section>

        <section>
          <h2 className="mb-4 text-sm uppercase tracking-[0.2em] text-zinc-500">Sample Artifacts</h2>
          <div className="grid gap-3 md:grid-cols-2">
            {walkthrough.downloads.map((download) => (
              <a
                key={download.href}
                href={download.href}
                download
                className="rounded-md border border-zinc-800 px-4 py-3 text-sm text-zinc-200 transition-colors hover:bg-zinc-900"
              >
                Download {download.label}
              </a>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
