import { ConversationState } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export async function ingestArxiv(arxiv_ref: string) {
  const res = await fetch(`${API_BASE}/ingest/arxiv`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ arxiv_ref }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ session_id: string; status_stream_url: string }>;
}

export async function ingestUpload(file: File) {
  const fd = new FormData();
  fd.append("pdf", file);
  const res = await fetch(`${API_BASE}/ingest/upload`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ session_id: string; status_stream_url: string }>;
}

export async function getConversationState(sessionId: string) {
  const res = await fetch(`${API_BASE}/conversation/${sessionId}/state`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<ConversationState>;
}

export async function getArtifacts(sessionId: string) {
  const res = await fetch(`${API_BASE}/conversation/${sessionId}/artifacts`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ items: { kind: string; filename: string; content_type: string; content: string }[] }>;
}

export async function resolveAmbiguity(sessionId: string, ambiguity_id: string, resolution: string) {
  const res = await fetch(`${API_BASE}/conversation/${sessionId}/resolve-ambiguity`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ambiguity_id, resolution }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function conversationSSEUrl(sessionId: string) {
  return `${API_BASE}/conversation/${sessionId}/message`;
}
