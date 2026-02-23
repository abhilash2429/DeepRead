"use client";

import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import AmbiguityCard from "@/components/AmbiguityCard";
import CodeBlock from "@/components/CodeBlock";
import HyperparamTable from "@/components/HyperparamTable";
import PrerequisiteCard from "@/components/PrerequisiteCard";
import { AmbiguityEntry, HyperparameterEntry, PrerequisiteEntry } from "@/lib/types";

type Props = {
  sectionId: string;
  title: string;
  content: string;
  sectionNumber: number;
  hyperparameters: HyperparameterEntry[];
  ambiguities: AmbiguityEntry[];
  prerequisites: PrerequisiteEntry[];
  onResolveAmbiguity: (id: string, resolution: string) => Promise<void>;
  codeSnippets?: any[];
};

export default function BriefingSection({
  sectionId,
  title,
  content,
  sectionNumber,
  hyperparameters,
  ambiguities,
  prerequisites,
  onResolveAmbiguity,
  codeSnippets = [],
}: Props) {

  const components: Components = {
    code(props) {
      const { children, className, node, ...rest } = props;
      const match = /language-(\w+)/.exec(className || "");
      const isInline = !match && !className;

      if (isInline) {
        return <code className={className} {...rest}>{children}</code>;
      }

      const codeText = String(children).replace(/\n$/, "");

      return (
        <CodeBlock
          code={codeText}
          provenance="paper-stated"
          language={match?.[1] || "text"}
        />
      );
    }
  };

  return (
    <section id={sectionId} className="fade-in scroll-mt-24">
      <header className="mb-6 border-b border-zinc-800 pb-3">
        <div className="text-[10px] uppercase tracking-[0.2em] text-zinc-500">Section {sectionNumber}</div>
        <h2 className="mt-1 text-2xl font-semibold text-zinc-50">{title}</h2>
      </header>

      <div className="space-y-8">
        <div className="prose-mono">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={components}
          >
            {content}
          </ReactMarkdown>
        </div>

        {sectionNumber === 3 && prerequisites.length > 0 && (
          <div className="space-y-4">
            {prerequisites.map((item) => (
              <PrerequisiteCard key={item.concept} item={item} />
            ))}
          </div>
        )}

        {sectionNumber === 4 && codeSnippets.length > 0 && (
          <div className="space-y-6 pt-2">
            <h3 className="text-[11px] uppercase tracking-[0.15em] text-zinc-500">Generated Implementation</h3>
            {codeSnippets.map((snippet, idx) => (
              <CodeBlock
                key={`${snippet.component_name}-${idx}`}
                code={snippet.code}
                provenance="inferred"
                language="python"
                title={snippet.component_name}
              />
            ))}
          </div>
        )}

        {sectionNumber === 5 && ambiguities.length > 0 && (
          <div className="space-y-4">
            {ambiguities.map((item) => (
              <AmbiguityCard
                key={item.ambiguity_id}
                ambiguity={item}
                onResolve={onResolveAmbiguity}
              />
            ))}
          </div>
        )}

        {sectionNumber === 6 && hyperparameters.length > 0 && (
          <HyperparamTable rows={hyperparameters} />
        )}
      </div>
    </section>
  );
}
