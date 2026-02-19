"use client";

import { useEffect, useRef, useState } from "react";

type SSEHandlers = {
  onToken?: (text: string) => void;
  onStage?: (stage: string) => void;
  onProgress?: (message: string) => void;
  onClarifying?: (question: string) => void;
  onDone?: () => void;
  onError?: (msg: string) => void;
};

export function useSSE() {
  const [connected, setConnected] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  async function streamPost(url: string, payload: unknown, handlers: SSEHandlers) {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setConnected(true);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      if (!response.body) throw new Error("No stream body returned");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        const chunks = buf.split("\n\n");
        buf = chunks.pop() || "";

        for (const chunk of chunks) {
          let event = "message";
          const dataParts: string[] = [];
          for (const line of chunk.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            if (line.startsWith("data:")) dataParts.push(line.slice(5).trim());
          }
          const data = dataParts.join("\n");
          if (!data) continue;
          let parsed: Record<string, unknown> = {};
          try {
            parsed = JSON.parse(data);
          } catch {
            parsed = { text: data };
          }
          if (event === "token") handlers.onToken?.(String(parsed.text || ""));
          if (event === "stage") handlers.onStage?.(String(parsed.current_stage || ""));
          if (event === "progress") handlers.onProgress?.(String(parsed.message || "Thinking..."));
          if (event === "clarifying") handlers.onClarifying?.(String(parsed.question || ""));
          if (event === "error") handlers.onError?.(String(parsed.message || "stream error"));
          if (event === "done") handlers.onDone?.();
        }
      }

      const tail = buf.trim();
      if (tail) {
        let event = "message";
        const dataParts: string[] = [];
        for (const line of tail.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) dataParts.push(line.slice(5).trim());
        }
        const data = dataParts.join("\n");
        if (data) {
          try {
            const parsed = JSON.parse(data) as Record<string, unknown>;
            if (event === "done") handlers.onDone?.();
            if (event === "error") handlers.onError?.(String(parsed.message || "stream error"));
          } catch {
            // no-op
          }
        }
      }
    } catch (err) {
      handlers.onError?.(err instanceof Error ? err.message : "stream failed");
    } finally {
      setConnected(false);
    }
  }

  useEffect(() => () => controllerRef.current?.abort(), []);

  return { connected, streamPost };
}
