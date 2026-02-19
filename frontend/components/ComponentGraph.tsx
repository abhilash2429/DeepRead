"use client";

import ReactFlow, { Background, Controls, Edge, Node } from "reactflow";
import "reactflow/dist/style.css";

export default function ComponentGraph({
  edgesData,
  onSelectComponent,
}: {
  edgesData: { parent: string; child: string }[];
  onSelectComponent: (name: string) => void;
}) {
  const names = Array.from(new Set(edgesData.flatMap((e) => [e.parent, e.child])));
  if (names.length === 0) {
    return <div className="rounded border bg-white p-3 text-sm text-slate-600">Component graph unavailable.</div>;
  }
  const nodes: Node[] = names.map((name, i) => ({
    id: name,
    data: { label: name },
    position: { x: (i % 3) * 220, y: Math.floor(i / 3) * 120 },
  }));
  const edges: Edge[] = edgesData.map((e, idx) => ({ id: `e-${idx}`, source: e.parent, target: e.child }));

  return (
    <div className="h-[220px] min-h-[220px] rounded border bg-white">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        onNodeClick={(_, n) => onSelectComponent(String(n.id))}
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
}
