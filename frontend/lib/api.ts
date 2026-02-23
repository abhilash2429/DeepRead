import { AuthProfile, ConversationState } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type IngestResult = { paper_id: string; status_stream_url: string };

async function handleResponse<T>(res: Response): Promise<T> {
  if (res.ok) {
    return (await res.json()) as T;
  }

  let message = await res.text();
  try {
    const parsed = JSON.parse(message) as { detail?: unknown; message?: string };
    if (typeof parsed.message === "string") {
      message = parsed.message;
    } else if (typeof parsed.detail === "string") {
      message = parsed.detail;
    } else if (parsed.detail && typeof parsed.detail === "object") {
      const detail = parsed.detail as Record<string, unknown>;
      message = String(detail.message || detail.error || message);
    }
  } catch {
    // keep raw body
  }
  throw new Error(message);
}

export async function ingestUpload(file: File): Promise<IngestResult> {
  const form = new FormData();
  form.append("pdf", file);
  const res = await fetch(`${API_BASE}/ingest/upload`, {
    method: "POST",
    body: form,
    credentials: "include",
  });
  return handleResponse<IngestResult>(res);
}

export async function ingestArxiv(arxivRef: string): Promise<IngestResult> {
  const res = await fetch(`${API_BASE}/ingest/arxiv`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ arxiv_ref: arxivRef }),
  });
  return handleResponse<IngestResult>(res);
}

export async function getConversationState(paperId: string): Promise<ConversationState> {
  const res = await fetch(`${API_BASE}/conversation/${paperId}/state`, {
    cache: "no-store",
    credentials: "include",
  });
  return handleResponse<ConversationState>(res);
}

export async function resolveAmbiguity(
  paperId: string,
  ambiguityId: string,
  resolution: string
): Promise<{ updated: unknown }> {
  const res = await fetch(`${API_BASE}/conversation/${paperId}/resolve-ambiguity`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ ambiguity_id: ambiguityId, resolution }),
  });
  return handleResponse<{ updated: unknown }>(res);
}

export async function getArtifacts(paperId: string): Promise<{ items: { kind: string; filename: string; content_type: string; content: string }[] }> {
  const res = await fetch(`${API_BASE}/conversation/${paperId}/artifacts`, {
    cache: "no-store",
    credentials: "include",
  });
  return handleResponse<{ items: { kind: string; filename: string; content_type: string; content: string }[] }>(res);
}

export async function getAuthProfile(): Promise<AuthProfile> {
  const res = await fetch(`${API_BASE}/auth/me`, {
    cache: "no-store",
    credentials: "include",
  });
  return handleResponse<AuthProfile>(res);
}

export function ingestSseUrl(paperId: string): string {
  return `${API_BASE}/ingest/${paperId}/events`;
}

export function pdfUrl(paperId: string): string {
  return `${API_BASE}/ingest/${paperId}/pdf`;
}

export function conversationSseUrl(paperId: string): string {
  return `${API_BASE}/conversation/${paperId}/message`;
}

