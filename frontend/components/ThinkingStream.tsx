"use client";

export default function ThinkingStream({ messages, active }: { messages: string[]; active: boolean }) {
  const visible = messages.slice(0, 5);

  return (
    <section className="glass fade-rise space-y-3 p-4">
      <h3 className="text-xs uppercase tracking-[0.24em] text-white/45">Thinking Stream</h3>
      <div className="space-y-2">
        {visible.map((message, index) => (
          <div
            key={`${message}-${index}`}
            className="text-sm transition-all duration-300 ease-in-out"
            style={{
              opacity: 1 - index * 0.18,
              transform: `translateY(${index * 2}px)`,
            }}
          >
            {message}
          </div>
        ))}
        {visible.length === 0 && <div className="text-sm text-white/50">Waiting for generation events...</div>}
      </div>
      {active && (
        <div className="inline-flex items-center gap-2 text-xs text-white/50">
          <span className="pulse-dot" />
          model is running
        </div>
      )}
    </section>
  );
}

