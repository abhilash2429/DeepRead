"use client";

import BriefingSection from "@/components/BriefingSection";
import { ExampleSection } from "@/lib/examples";

type Props = {
  sections: ExampleSection[];
};

export default function ExampleWalkthroughDocument({ sections }: Props) {
  return (
    <article className="mx-auto w-full max-w-3xl space-y-16 pb-20">
      {sections.map((section, index) => (
        <BriefingSection
          key={section.id}
          sectionId={section.id}
          sectionNumber={index + 1}
          title={section.title.replace(/^\d+\.\s*/, "")}
          content={section.content}
          hyperparameters={[]}
          ambiguities={[]}
          prerequisites={[]}
          onResolveAmbiguity={async () => {}}
          codeSnippets={[]}
        />
      ))}
    </article>
  );
}
