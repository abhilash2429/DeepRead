"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import ArtifactDownloads from "@/components/ArtifactDownloads";
import BriefingDocument from "@/components/BriefingDocument";
import ChatInput from "@/components/ChatInput";
import PdfPanel from "@/components/PdfPanel";
import ThemeToggle from "@/components/ThemeToggle";
import { getConversationState, ingestSseUrl, resolveAmbiguity, conversationSseUrl, isUnauthorizedError } from "@/lib/api";
import { ChatMessage, ConversationState } from "@/lib/types";
import { useSSE } from "@/hooks/useSSE";

const SECTION_TITLES = [
  { key: "section_1", label: "1 - What It Does" },
  { key: "section_2", label: "2 - The Mechanism" },
  { key: "section_3", label: "3 - Prerequisites" },
  { key: "section_4", label: "4 - Implementation" },
  { key: "section_5", label: "5 - What's Missing" },
  { key: "section_6", label: "6 - How To Train" },
] as const;

export default function SessionPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const routeId = params.id;
  const [paperId, setPaperId] = useState<string | null>(null);

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

  async function refreshState(activePaperId: string) {
    try {
      const next = await getConversationState(activePaperId);
      setState(next);
      if (next.status === "COMPLETE" || next.status === "FAILED") {
        setGenerating(false);
        setStatusMessage(next.status === "COMPLETE" ? "Complete" : "Failed");
      }
    } catch (error) {
      if (isUnauthorizedError(error)) {
        router.replace("/");
        return;
      }
      throw error;
    }
  }

  useEffect(() => {
    if (!routeId) return;
    if (routeId === "analyze") {
      const stored = sessionStorage.getItem("deepread_active_paper_id");
      if (!stored) {
        router.replace("/upload");
        return;
      }
      setPaperId(stored);
      return;
    }
    setPaperId(routeId);
    sessionStorage.setItem("deepread_active_paper_id", routeId);
  }, [routeId, router]);

  useEffect(() => {
    if (!paperId) return;
    void refreshState(paperId).catch(() => {
      setIngestError("Failed to load session state.");
    });
  }, [paperId]);

  useEffect(() => {
    if (!paperId) return;
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
      void refreshState(paperId).catch(() => { });
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
      await refreshState(paperId).catch(() => { });
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

  const showFollowUpInput = state?.status === "COMPLETE";

  const displayPaperTitle = useMemo(() => {
    const raw = (state?.paper_title || "").trim();
    if (!raw) return "Research Paper";
    const max = 96;
    if (raw.length <= max) return raw;
    return `${raw.slice(0, max - 1).trimEnd()}...`;
  }, [state?.paper_title]);

  async function sendMessage() {
    if (!paperId) return;
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
      onError: (msg, status) => {
        if (status === 401) {
          router.replace("/");
          return;
        }
        setChatError(msg);
      },
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
      <PdfPanel paperId={paperId || ""} open={pdfOpen} onClose={() => setPdfOpen(false)} pdfReady={state.status === "COMPLETE"} />

      {/* Main content */}
      <div className={`relative flex flex-1 flex-col transition-all duration-300 ${pdfOpen ? "ml-[36vw]" : "ml-0"}`}>

        {/* Top bar (Absolute so scroll passes under it for glass effect) */}
        <header className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between border-b border-zinc-700/30 bg-zinc-900/50 px-6 py-3 shadow-md backdrop-blur-xl">
          <div className="flex items-center gap-4">
            <button
              className="rounded-md border border-zinc-700/50 bg-zinc-800/40 px-3 py-1.5 text-xs text-zinc-200 transition-colors hover:bg-zinc-700/60"
              onClick={() => setPdfOpen((v) => !v)}
            >
              {pdfOpen ? "Close PDF" : "PDF"}
            </button>
            <div className="rounded-full border border-zinc-700/50 bg-zinc-800/40 px-2.5 py-1 text-[11px] text-zinc-300">
              {state.status === "PROCESSING" ? "Processing" : state.status === "COMPLETE" ? "Complete" : "Failed"}
            </div>
          </div>
          <ThemeToggle />
        </header>

        {/* Content area: 2 columns */}
        <div className="flex flex-1 min-h-0 pt-16">

          {/* Main scroll area */}
          <div ref={scrollRef} className="thin-scroll flex-1 overflow-auto px-6 pt-6">

            <div className="mx-auto w-full max-w-3xl">
              <h1 className="mb-5 text-2xl font-semibold tracking-tight text-zinc-100 sm:text-3xl">
                {displayPaperTitle}
              </h1>
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
                if (!paperId) return;
                try {
                  await resolveAmbiguity(paperId, id, resolution);
                  await refreshState(paperId);
                } catch (error) {
                  if (isUnauthorizedError(error)) {
                    router.replace("/");
                    return;
                  }
                  setChatError(error instanceof Error ? error.message : "Failed to resolve ambiguity.");
                }
              }}
              codeSnippets={state.code_snippets?.length ? state.code_snippets : streamingCodeSnippets}
            />

            <div className="mx-auto w-full max-w-3xl pb-24">
              {/* Streaming status line - shown inline under content */}
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
            {showFollowUpInput && (
              <div className="sticky bottom-6 z-10 mx-auto w-full max-w-2xl">
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
            )}
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

              <ArtifactDownloads paperId={paperId || ""} sections={sections} />
            </div>
          </aside>
        </div>
      </div>
    </main>
  );
}
