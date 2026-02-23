"use client";

import BriefingSection from "@/components/BriefingSection";
import {
  AmbiguityEntry,
  BriefingSectionKey,
  HyperparameterEntry,
  PrerequisiteEntry,
} from "@/lib/types";

const SECTION_META: { key: BriefingSectionKey; number: number; title: string }[] = [
  { key: "section_1", number: 1, title: "What This Paper Actually Does" },
  { key: "section_2", number: 2, title: "The Mechanism" },
  { key: "section_3", number: 3, title: "What You Need To Already Know" },
  { key: "section_4", number: 4, title: "The Full Implementation Map" },
  { key: "section_5", number: 5, title: "What The Paper Left Out" },
  { key: "section_6", number: 6, title: "How To Train It" },
];

type Props = {
  sections: Record<BriefingSectionKey, string | null>;
  streamingSections: Record<number, string>;
  hyperparameters: HyperparameterEntry[];
  ambiguities: AmbiguityEntry[];
  prerequisites: PrerequisiteEntry[];
  onResolveAmbiguity: (id: string, resolution: string) => Promise<void>;
  codeSnippets?: any[];
};

export default function BriefingDocument({
  sections,
  streamingSections,
  hyperparameters,
  ambiguities,
  prerequisites,
  onResolveAmbiguity,
  codeSnippets = [],
}: Props) {
  return (
    <article className="mx-auto w-full max-w-3xl space-y-16 pb-20">
      {SECTION_META.map((meta) => {
        const content = sections[meta.key] || streamingSections[meta.number] || "";
        if (!content.trim() && !(meta.number === 4 && codeSnippets.length > 0)) return null;
        return (
          <BriefingSection
            key={meta.key}
            sectionId={meta.key}
            sectionNumber={meta.number}
            title={meta.title}
            content={content}
            hyperparameters={hyperparameters}
            ambiguities={ambiguities}
            prerequisites={prerequisites}
            onResolveAmbiguity={onResolveAmbiguity}
            codeSnippets={meta.number === 4 ? codeSnippets : []}
          />
        );
      })}
    </article>
  );
}

