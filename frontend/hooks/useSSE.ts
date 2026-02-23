"use client";

import { useEffect, useRef, useState } from "react";

type SSEHandlers = {
  onToken?: (text: string) => void;
  onThinking?: (message: string) => void;
  onSectionToken?: (sectionNumber: number, text: string) => void;
  onProgress?: (payload: Record<string, unknown>) => void;
  onDone?: (payload: Record<string, unknown>) => void;
  onError?: (message: string, status?: number) => void;
};

export function useSSE() {
  const [connected, setConnected] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function streamPost(url: string, payload: unknown, handlers: SSEHandlers): Promise<void> {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setConnected(true);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        credentials: "include",
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!response.ok) {
        const body = await response.text();
        const error = new Error(body || `Request failed with ${response.status}`) as Error & { status?: number };
        error.status = response.status;
        throw error;
      }
      if (!response.body) {
        throw new Error("No SSE response body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";

        for (const frame of frames) {
          if (!frame.trim()) continue;
          let event = "message";
          const dataLines: string[] = [];
          for (const line of frame.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
          }
          const rawData = dataLines.join("\n");
          if (!rawData) continue;

          let payloadObj: Record<string, unknown>;
          try {
            payloadObj = JSON.parse(rawData) as Record<string, unknown>;
          } catch {
            payloadObj = { text: rawData };
          }

          if (event === "token") handlers.onToken?.(String(payloadObj.text || ""));
          if (event === "thinking") handlers.onThinking?.(String(payloadObj.message || ""));
          if (event === "section_token")
            handlers.onSectionToken?.(
              Number(payloadObj.section_number || 0),
              String(payloadObj.text || "")
            );
          if (event === "progress") handlers.onProgress?.(payloadObj);
          if (event === "done") handlers.onDone?.(payloadObj);
          if (event === "error") handlers.onError?.(String(payloadObj.message || "Stream error"));
        }
      }
    } catch (error) {
      const maybeStatus = typeof error === "object" && error !== null && "status" in error
        ? Number((error as { status?: number }).status)
        : undefined;
      handlers.onError?.(error instanceof Error ? error.message : "SSE request failed", maybeStatus);
    } finally {
      setConnected(false);
    }
  }

  useEffect(() => () => abortRef.current?.abort(), []);

  return { connected, streamPost };
}
