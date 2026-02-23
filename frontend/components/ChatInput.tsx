"use client";

import { FormEvent } from "react";

type Props = Readonly<{
  value: string;
  disabled: boolean;
  busy: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
}>;

export default function ChatInput({ value, disabled, busy, onChange, onSubmit }: Props) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (disabled) return;
    onSubmit();
  }

  return (
    <form
      className="relative overflow-hidden rounded-2xl bg-zinc-500/10 p-2 shadow-[0_8px_32px_0_rgba(0,0,0,0.1)] backdrop-blur-3xl saturate-150"
      onSubmit={handleSubmit}
    >
      <div className="flex items-end gap-2">
        <textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e as unknown as FormEvent<HTMLFormElement>);
            }
          }}
          disabled={disabled}
          placeholder="Ask a question about this paper..."
          className="thin-scroll min-h-[40px] max-h-[120px] w-full resize-none bg-transparent px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none"
          rows={Math.min(Math.max(value.split("\n").length, 1), 5)}
        />
        <button
          disabled={disabled || !value.trim()}
          className="shrink-0 rounded-xl bg-zinc-500/10 px-4 py-2 text-sm font-medium text-zinc-100 transition-colors hover:bg-zinc-500/20 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "..." : "Send"}
        </button>
      </div>
    </form>
  );
}
