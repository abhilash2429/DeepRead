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
    <form className="relative overflow-hidden rounded-2xl bg-zinc-900/60 p-2 shadow-2xl shadow-black/20 backdrop-blur-xl" onSubmit={handleSubmit}>
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
          className="thin-scroll w-full resize-none rounded bg-transparent px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none min-h-[40px] max-h-[120px]"
          rows={Math.min(Math.max(value.split("\n").length, 1), 5)}
        />
        <button
          disabled={disabled || !value.trim()}
          className="rounded bg-zinc-800 px-4 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-700 transition-colors disabled:cursor-not-allowed disabled:opacity-40 shrink-0"
        >
          {busy ? "..." : "Send"}
        </button>
      </div>
    </form>
  );
}
