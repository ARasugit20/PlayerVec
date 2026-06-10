import { useMemo, useState } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { UMAPPoint } from "../types";

const POSITION_COLORS: Record<string, string> = {
  GK: "#f59e0b",
  DF: "#60a5fa",
  MF: "#34d399",
  FW: "#fb7185",
  UNK: "#9ca3af",
};

interface Props {
  points: UMAPPoint[];
  selectedPlayer: string | null;
  onSelectPlayer: (player: string) => void;
  highlightIds?: Set<number>;
}

interface TooltipPayload {
  payload: UMAPPoint;
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayload[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-lg border border-pitch-700 bg-pitch-900 px-3 py-2 text-sm shadow-lg">
      <p className="font-medium text-white">
        {p.jersey_number != null && `#${p.jersey_number} `}
        {p.player}
      </p>
      <p className="text-pitch-300">
        {p.nation} · {p.position_detail || p.position}
      </p>
    </div>
  );
}

export default function ClusterMap({
  points,
  selectedPlayer,
  onSelectPlayer,
  highlightIds,
}: Props) {
  const [hovered, setHovered] = useState<string | null>(null);

  const data = useMemo(
    () =>
      points.map((p) => ({
        ...p,
        fill: POSITION_COLORS[p.position] ?? POSITION_COLORS.UNK,
        opacity:
          !highlightIds || highlightIds.size === 0 || highlightIds.has(p.id) ? 0.85 : 0.15,
        radius:
          p.player === selectedPlayer || p.player === hovered ? 8 : highlightIds?.has(p.id) ? 6 : 4,
      })),
    [points, selectedPlayer, hovered, highlightIds]
  );

  if (points.length === 0) {
    return (
      <div className="flex h-80 items-center justify-center rounded-2xl border border-pitch-700 bg-pitch-800/50 text-pitch-300/60">
        Loading cluster map...
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-pitch-700 bg-pitch-800/50 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-medium text-white">World Cup Cluster Map</h2>
        <div className="flex gap-3 text-xs">
          {Object.entries(POSITION_COLORS).map(([pos, color]) =>
            pos !== "UNK" ? (
              <span key={pos} className="flex items-center gap-1 text-pitch-300">
                <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />
                {pos}
              </span>
            ) : null
          )}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1a4d32" />
          <XAxis type="number" dataKey="x" hide />
          <YAxis type="number" dataKey="y" hide />
          <ZAxis type="number" dataKey="radius" range={[40, 200]} />
          <Tooltip content={<CustomTooltip />} />
          <Scatter
            data={data}
            onClick={(d) => onSelectPlayer(d.player)}
            onMouseEnter={(d) => setHovered(d.player)}
            onMouseLeave={() => setHovered(null)}
          />
        </ScatterChart>
      </ResponsiveContainer>
      <p className="mt-2 text-center text-xs text-pitch-300/50">
        UMAP projection of 32-dim style embeddings · click a dot to search
      </p>
    </div>
  );
}
