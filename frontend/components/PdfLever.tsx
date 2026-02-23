"use client";

type Props = {
  open: boolean;
  onToggle: () => void;
};

export default function PdfLever({ open, onToggle }: Props) {
  return (
    <button
      className="flex h-full w-8 items-center justify-center border-r border-zinc-800 bg-zinc-950 text-xs font-medium uppercase tracking-[0.2em] text-zinc-600 transition-colors hover:bg-zinc-900 hover:text-zinc-300"
      onClick={onToggle}
      title={open ? "Close PDF" : "Open PDF"}
      style={{ writingMode: "vertical-rl", textOrientation: "mixed" }}
    >
      {open ? "Close PDF" : "View PDF"}
    </button>
  );
}
