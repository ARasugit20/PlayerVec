import type { SimilarPlayer } from "../types";

const POSITION_COLORS: Record<string, string> = {
  GK: "bg-amber-500/20 text-amber-300",
  DF: "bg-blue-500/20 text-blue-300",
  MF: "bg-emerald-500/20 text-emerald-300",
  FW: "bg-rose-500/20 text-rose-300",
  UNK: "bg-gray-500/20 text-gray-300",
};

function formatStat(key: string, value: number): string {
  const labels: Record<string, string> = {
    goals_per90: "Goals/90",
    xg_per90: "xG/90",
    assists_per90: "Ast/90",
    xa_per90: "xA/90",
    key_passes_per90: "KP/90",
    progressive_passes_per90: "PrgP/90",
  };
  return `${labels[key] ?? key}: ${value.toFixed(2)}`;
}

interface Props {
  query: string | null;
  results: SimilarPlayer[];
  loading?: boolean;
  error?: string | null;
}

export default function SimilarPlayers({ query, results, loading, error }: Props) {
  if (loading) {
    return (
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-40 animate-pulse rounded-2xl bg-pitch-800" />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-2xl border border-red-500/30 bg-red-500/10 px-6 py-4 text-red-200">
        {error}
      </div>
    );
  }

  if (!query) {
    return (
      <p className="text-pitch-300/70">
        Type a player name to find the 5 closest style-matches in the tournament.
      </p>
    );
  }

  if (results.length === 0) {
    return <p className="text-pitch-300/70">No similar players found.</p>;
  }

  return (
    <div>
      <h2 className="mb-4 text-lg font-medium text-pitch-300">
        Players most similar to <span className="text-white">{query}</span>
      </h2>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {results.map((p) => (
          <article
            key={`${p.player}-${p.team}`}
            className="rounded-2xl border border-pitch-700 bg-pitch-800/80 p-5 transition hover:border-pitch-500"
          >
            <div className="mb-3 flex items-start justify-between gap-2">
              <div>
                <h3 className="font-semibold text-white">
                  {p.jersey_number != null && (
                    <span className="mr-2 text-pitch-500">#{p.jersey_number}</span>
                  )}
                  {p.player}
                </h3>
                <p className="text-sm text-pitch-300">
                  {p.team} · {p.nation}
                  {p.position_detail ? ` · ${p.position_detail}` : ""}
                </p>
              </div>
              <span
                className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                  POSITION_COLORS[p.position] ?? POSITION_COLORS.UNK
                }`}
              >
                {p.position}
              </span>
            </div>
            <div className="mb-3 text-2xl font-bold text-pitch-300">
              {(p.similarity * 100).toFixed(1)}%
              <span className="ml-1 text-sm font-normal text-pitch-300/60">similar</span>
            </div>
            <div className="flex flex-wrap gap-2 text-xs text-pitch-300/80">
              {Object.entries(p.key_stats).slice(0, 4).map(([k, v]) => (
                <span key={k} className="rounded-md bg-pitch-900/60 px-2 py-1">
                  {formatStat(k, v)}
                </span>
              ))}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
