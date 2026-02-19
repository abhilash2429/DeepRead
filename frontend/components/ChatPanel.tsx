"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { resolveAmbiguity } from "@/lib/api";
import { ConversationState, Stage } from "@/lib/types";
import AmbiguityCard from "@/components/AmbiguityCard";
import CodeBlock from "@/components/CodeBlock";
import HyperparamTable from "@/components/HyperparamTable";
import { useSSE } from "@/hooks/useSSE";

const stages: Stage[] = ["orientation", "architecture", "implementation", "ambiguity", "training"];
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type Segment = { kind: "text"; value: string } | { kind: "code"; value: string };

function splitMessageSegments(content: string): Segment[] {
  const regex = /```(?:python)?\s*([\s\S]*?)```/gim;
  const segments: Segment[] = [];
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(content)) !== null) {
    if (match.index > last) {
      const text = content.slice(last, match.index).trim();
      if (text) segments.push({ kind: "text", value: text });
    }
    const code = (match[1] || "").trim();
    if (code) segments.push({ kind: "code", value: code });
    last = regex.lastIndex;
  }
  if (last < content.length) {
    const text = content.slice(last).trim();
    if (text) segments.push({ kind: "text", value: text });
  }
  return segments.length ? segments : [{ kind: "text", value: content }];
}

export default function ChatPanel({
  sessionId,
  initialState,
  onStageChange,
}: {
  sessionId: string;
  initialState: ConversationState;
  onStageChange: (s: Stage) => void;
}) {
  const [state, setState] = useState(initialState);
  const [input, setInput] = useState("");
  const [draftAnswer, setDraftAnswer] = useState("");
  const [clarifying, setClarifying] = useState<string | null>(initialState.pending_question || null);
  const [sendError, setSendError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const { connected, streamPost } = useSSE();
  const listRef = useRef<HTMLDivElement | null>(null);
  const draftRef = useRef("");

  useEffect(() => {
    setState(initialState);
    setClarifying(initialState.pending_question || null);
  }, [initialState]);

  const refreshState = useCallback(async () => {
    try {
      const refreshed = await fetch(`${API_BASE}/conversation/${sessionId}/state`, { cache: "no-store" });
      const json = (await refreshed.json()) as ConversationState;
      setState(json);
      setClarifying(json.pending_question || null);
    } catch (err) {
      setSendError(err instanceof Error ? err.message : "Failed to refresh conversation state");
    }
  }, [sessionId]);

  const messages = useMemo(
    () =>
      draftAnswer
        ? [...state.message_history, { role: "assistant" as const, content: draftAnswer }]
        : state.message_history,
    [state.message_history, draftAnswer]
  );

  useEffect(() => {
    const node = listRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [messages, clarifying, progress]);

  const send = useCallback(
    async (message: string, stageOverride?: Stage, optimisticUserMessage = true) => {
      setDraftAnswer("");
      draftRef.current = "";
      setSendError(null);
      setClarifying(null);
      setProgress("Preparing response...");

      if (optimisticUserMessage) {
        setState((prev) => ({
          ...prev,
          message_history: [...prev.message_history, { role: "user", content: message }],
        }));
      }

      await streamPost(
        `${API_BASE}/conversation/${sessionId}/message`,
        { message, stage_override: stageOverride ?? null },
        {
          onToken: (t) => {
            setProgress(null);
            setDraftAnswer((prev) => {
              const next = prev + t;
              draftRef.current = next;
              return next;
            });
          },
          onStage: (s) => {
            if (stages.includes(s as Stage)) {
              onStageChange(s as Stage);
              setState((prev) => ({ ...prev, current_stage: s as Stage }));
            }
          },
          onProgress: (msg) => setProgress(msg || "Thinking..."),
          onClarifying: (q) => setClarifying(q || null),
          onDone: () => {
            setProgress(null);
            const finalDraft = draftRef.current.trim();
            if (finalDraft) {
              setState((prev) => ({
                ...prev,
                message_history: [...prev.message_history, { role: "assistant", content: finalDraft }],
              }));
            }
            setDraftAnswer("");
            draftRef.current = "";
            void refreshState();
          },
          onError: (msg) => {
            setProgress(null);
            setSendError(msg || "Failed to stream response");
          },
        }
      );
    },
    [sessionId, streamPost, onStageChange, refreshState]
  );

  useEffect(() => {
    const handler = (event: Event) => {
      const custom = event as CustomEvent<{ component?: string }>;
      const component = custom.detail?.component;
      if (!component) return;
      onStageChange("architecture");
      void send(`Explain ${component} in depth using equations and implementation consequences.`, "architecture", false);
    };
    window.addEventListener("deepread-component-focus", handler);
    return () => window.removeEventListener("deepread-component-focus", handler);
  }, [onStageChange, send]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap gap-2 border-b border-slate-300 bg-white px-3 py-2">
        {stages.map((s) => (
          <button
            key={s}
            className={`rounded px-2 py-1 text-xs ${state.current_stage === s ? "bg-slate-900 text-white" : "bg-slate-200 text-slate-800"}`}
            onClick={() => void send(`Continue in ${s} stage with detailed explanation.`, s, false)}
          >
            {s}
          </button>
        ))}
      </div>

      {clarifying && <div className="border-b border-amber-300 bg-amber-50 px-3 py-2 text-sm">{clarifying}</div>}
      {sendError && <div className="border-b border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-700">{sendError}</div>}
      {progress && <div className="border-b border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-700">{progress}</div>}

      <div ref={listRef} className="min-h-0 flex-1 space-y-3 overflow-auto bg-slate-100 px-3 py-3">
        {messages.map((m, i) => {
          const segments = splitMessageSegments(m.content);
          const provenance =
            m.content.includes("# ASSUMED:")
              ? "assumed"
              : m.content.includes("# INFERRED:")
                ? "inferred"
                : "paper-stated";
          return (
            <div key={i} className={`rounded-md p-3 text-sm ${m.role === "user" ? "ml-8 bg-blue-50" : "mr-8 bg-white"}`}>
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">{m.role}</div>
              {segments.map((seg, idx) =>
                seg.kind === "text" ? (
                  <div key={`${i}-txt-${idx}`} className="mb-2 whitespace-pre-wrap text-slate-900">
                    {seg.value}
                  </div>
                ) : (
                  <div key={`${i}-code-${idx}`} className="mb-2">
                    <CodeBlock code={seg.value} provenance={provenance} />
                  </div>
                )
              )}
            </div>
          );
        })}
      </div>

      {state.current_stage === "training" && (
        <div className="border-t border-slate-300 bg-white p-3">
          <HyperparamTable rows={state.internal_representation.hyperparameter_registry} />
        </div>
      )}

      {state.current_stage === "ambiguity" && (
        <div className="max-h-[280px] space-y-2 overflow-auto border-t border-slate-300 bg-white p-3">
          {state.internal_representation.ambiguity_log.map((a) => (
            <AmbiguityCard
              key={a.ambiguity_id}
              ambiguity={a}
              onResolve={async (id, resolution) => {
                await resolveAmbiguity(sessionId, id, resolution);
                await refreshState();
              }}
            />
          ))}
        </div>
      )}

      <form
        className="border-t border-slate-300 bg-white p-3"
        onSubmit={(e) => {
          e.preventDefault();
          const msg = input.trim();
          if (!msg) return;
          setInput("");
          void send(msg);
        }}
      >
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask for deep explanation, equations, code, or training recipe."
            className="w-full rounded border border-slate-300 px-3 py-2 text-sm"
          />
          <button className="rounded bg-slate-900 px-3 py-2 text-sm text-white disabled:opacity-50" disabled={connected}>
            {connected ? "Streaming..." : "Send"}
          </button>
        </div>
      </form>
    </div>
  );
}
