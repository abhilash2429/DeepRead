"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";

import ArtifactDownloads from "@/components/ArtifactDownloads";
import BriefingDocument from "@/components/BriefingDocument";
import ChatInput from "@/components/ChatInput";
import PdfPanel from "@/components/PdfPanel";
import ThemeToggle from "@/components/ThemeToggle";
import { getConversationState, ingestSseUrl, resolveAmbiguity, conversationSseUrl } from "@/lib/api";
import { ChatMessage, ConversationState } from "@/lib/types";
import { useSSE } from "@/hooks/useSSE";

const SECTION_TITLES = [
  { key: "section_1", label: "1 · What It Does" },
  { key: "section_2", label: "2 · The Mechanism" },
  { key: "section_3", label: "3 · Prerequisites" },
  { key: "section_4", label: "4 · Implementation" },
  { key: "section_5", label: "5 · What's Missing" },
  { key: "section_6", label: "6 · How To Train" },
] as const;

export default function SessionPage() {
  const params = useParams<{ id: string }>();
  const paperId = params.id;

  const [state, setState] = useState<ConversationState | null>(null);
  const [streamingSections, setStreamingSections] = useState<Record<number, string>>({});
  const [streamingCodeSnippets, setStreamingCodeSnippets] = useState<any[]>([]);
  const [statusMessage, setStatusMessage] = useState<string>("Connecting...");
  const [pdfOpen, setPdfOpen] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [chatError, setChatError] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatDraft, setChatDraft] = useState("");
  const [generating, setGenerating] = useState(true);
  const { streamPost, connected } = useSSE();
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const draftRef = useRef("");

  async function refreshState() {
    const next = await getConversationState(paperId);
    setState(next);
    if (next.status === "COMPLETE" || next.status === "FAILED") {
      setGenerating(false);
      setStatusMessage(next.status === "COMPLETE" ? "Complete" : "Failed");
    }
  }

  useEffect(() => { void refreshState(); }, [paperId]);

  useEffect(() => {
    const source = new EventSource(ingestSseUrl(paperId), { withCredentials: true });

    source.addEventListener("thinking", (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent).data) as { message?: string };
        if (payload.message) setStatusMessage(payload.message);
      } catch { /* ignore */ }
    });

    source.addEventListener("code_snippet", (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent).data) as { snippet?: any };
        if (payload.snippet) {
          setStreamingCodeSnippets((prev) => [...prev, payload.snippet]);
        }
      } catch { /* ignore */ }
    });

    source.addEventListener("status", (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent).data) as { message?: string };
        if (payload.message) setStatusMessage(payload.message);
      } catch { /* ignore */ }
      setGenerating(true);
    });

    source.addEventListener("section_token", (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent).data) as { section_number?: number; text?: string };
        const num = Number(payload.section_number || 0);
        if (!num) return;
        setStreamingSections((prev) => ({
          ...prev,
          [num]: `${prev[num] || ""}${payload.text || ""}`,
        }));
      } catch { /* ignore */ }
    });

    source.addEventListener("progress", (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent).data) as { generation_progress?: number };
        if (payload.generation_progress) {
          const sec = SECTION_TITLES[payload.generation_progress - 1];
          if (sec) setStatusMessage(`Completed ${sec.label}`);
        }
      } catch { /* ignore */ }
      setGenerating(true);
    });

    source.addEventListener("error", (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent).data) as { message?: string };
        setIngestError(payload.message || "Connection error.");
      } catch {
        setIngestError("Connection error.");
      }
    });

    source.addEventListener("done", async (event) => {
      source.close();
      try {
        const payload = JSON.parse((event as MessageEvent).data) as { failed?: boolean };
        if (payload.failed) {
          setIngestError("Generation failed. Check backend logs.");
          setStatusMessage("Failed");
        } else {
          setStatusMessage("Complete");
        }
      } catch { /* ignore */ }
      await refreshState();
    });

    return () => source.close();
  }, [paperId]);

  useEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [state, streamingSections, chatMessages, chatDraft]);

  const canChat = useMemo(() => {
    return Boolean(state?.briefing.section_1?.trim() || streamingSections[1]?.trim());
  }, [state, streamingSections]);

  async function sendMessage() {
    const message = chatInput.trim();
    if (!message || !canChat || connected) return;
    setChatInput("");
    setChatError(null);
    setChatDraft("");
    draftRef.current = "";
    setChatMessages((prev) => [...prev, { role: "user", content: message }]);

    await streamPost(conversationSseUrl(paperId), { message }, {
      onToken: (token) => {
        setChatDraft((prev) => {
          const next = `${prev}${token}`;
          draftRef.current = next;
          return next;
        });
      },
      onDone: () => {
        const final = draftRef.current.trim();
        if (final) setChatMessages((prev) => [...prev, { role: "assistant", content: final }]);
        setChatDraft("");
        draftRef.current = "";
      },
      onError: (msg) => setChatError(msg),
    });
  }

  if (!state) {
    return (
      <main className="flex min-h-screen items-center justify-center">
        <div className="text-sm text-zinc-500">Loading session...</div>
      </main>
    );
  }

  const sections = state.sections;

  return (
    <main className="relative flex h-screen overflow-hidden">
      {/* PDF Panel */}
      <PdfPanel paperId={paperId} open={pdfOpen} onClose={() => setPdfOpen(false)} pdfReady={state.status === "COMPLETE"} />

      {/* Main content */}
      <div className={`flex flex-1 flex-col transition-all duration-300 ${pdfOpen ? "ml-[36vw]" : "ml-0"}`}>

        {/* Top bar */}
        <header className="flex items-center justify-between border-b border-zinc-800 px-6 py-3">
          <div className="flex items-center gap-4">
            <button
              className="rounded-md border border-zinc-800 px-3 py-1.5 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
              onClick={() => setPdfOpen((v) => !v)}
            >
              {pdfOpen ? "Close PDF" : "PDF"}
            </button>
            <div className="text-xs text-zinc-600">
              {state.status === "PROCESSING" ? "Processing" : state.status === "COMPLETE" ? "Complete" : "Failed"}
            </div>
          </div>
          <ThemeToggle />
        </header>

        {/* Content area: 2 columns */}
        <div className="flex flex-1 min-h-0">

          {/* Main scroll area */}
          <div ref={scrollRef} className="thin-scroll flex-1 overflow-auto px-6 pt-6">

            <div className="mx-auto w-full max-w-3xl">
              {ingestError && (
                <div className="mb-4 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-400">
                  {ingestError}
                </div>
              )}
            </div>

            <BriefingDocument
              sections={state.briefing}
              streamingSections={streamingSections}
              hyperparameters={state.hyperparameters}
              ambiguities={state.ambiguities}
              prerequisites={state.prerequisites}
              onResolveAmbiguity={async (id, resolution) => {
                await resolveAmbiguity(paperId, id, resolution);
                await refreshState();
              }}
              codeSnippets={state.code_snippets?.length ? state.code_snippets : streamingCodeSnippets}
            />

            <div className="mx-auto w-full max-w-3xl pb-24">
              {/* Streaming status line — shown inline under content */}
              {generating && (
                <div className="status-line mt-4 fade-in">
                  <span className="status-dot" />
                  <span>{statusMessage}</span>
                </div>
              )}

              {/* Chat messages */}
              <div className="mt-8 space-y-3">
                {chatMessages.map((msg, i) => (
                  <article
                    key={`${msg.role}-${i}`}
                    className={`fade-in max-w-[85%] rounded-lg px-4 py-3 text-sm ${msg.role === "user"
                      ? "ml-auto bg-zinc-800 text-zinc-200"
                      : "mr-auto border border-zinc-800 text-zinc-400"
                      }`}
                  >
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                  </article>
                ))}
                {chatDraft && (
                  <article className="fade-in mr-auto max-w-[85%] rounded-lg border border-zinc-800 px-4 py-3 text-sm text-zinc-400">
                    <div className="whitespace-pre-wrap">{chatDraft}</div>
                  </article>
                )}
              </div>
            </div>

            {/* Floating Chat input */}
            <div className="sticky bottom-6 mx-auto w-full max-w-2xl z-10">
              {chatError && (
                <div className="mb-2 rounded-lg border border-rose-900/50 bg-rose-950/30 px-3 py-2 text-sm text-rose-300">
                  {chatError}
                </div>
              )}
              <ChatInput
                value={chatInput}
                onChange={setChatInput}
                onSubmit={sendMessage}
                disabled={!canChat || connected}
                busy={connected}
              />
            </div>
          </div>

          {/* Right sidebar */}
          <aside className="hidden w-52 shrink-0 border-l border-zinc-800 p-4 lg:block">
            <div className="sticky top-4 space-y-6">
              <section>
                <h4 className="text-[10px] uppercase tracking-widest text-zinc-600">Sections</h4>
                <nav className="mt-3 space-y-1">
                  {SECTION_TITLES.map((sec) => {
                    const ready = sections[sec.key as keyof typeof sections];
                    const streaming = streamingSections[SECTION_TITLES.indexOf(sec) + 1];
                    return (
                      <button
                        key={sec.key}
                        className={`block w-full rounded px-2 py-1 text-left text-xs transition-colors ${ready ? "text-zinc-200 font-medium" : streaming ? "text-zinc-400" : "text-zinc-600"
                          } hover:bg-zinc-800/50`}
                        onClick={() => document.getElementById(sec.key)?.scrollIntoView({ behavior: "smooth", block: "start" })}
                      >
                        {sec.label}
                      </button>
                    );
                  })}
                </nav>
              </section>

              <ArtifactDownloads paperId={paperId} sections={sections} />
            </div>
          </aside>
        </div>
      </div>
    </main>
  );
}
