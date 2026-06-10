import { useEffect, useState } from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { fetchFixtureBrief, fetchTeams, fetchTournamentOutlook } from "../api";
import type { FixtureBrief as FixtureBriefType, TournamentOutlook } from "../types";

const DNA_LABELS: Record<string, string> = {
  press_intensity: "Press",
  progression: "Progression",
  aerial_direct: "Aerial/Direct",
  chance_creation: "Chance Creation",
  finishing: "Finishing",
  width: "Width",
};

function StyleDNARadar({
  teamA,
  teamB,
  dnaA,
  dnaB,
}: {
  teamA: string;
  teamB: string;
  dnaA: Record<string, number>;
  dnaB: Record<string, number>;
}) {
  const maxVal = Math.max(
    ...Object.values(dnaA),
    ...Object.values(dnaB),
    0.01
  );
  const data = Object.keys(DNA_LABELS).map((key) => ({
    dimension: DNA_LABELS[key],
    [teamA]: (dnaA[key] ?? 0) / maxVal,
    [teamB]: (dnaB[key] ?? 0) / maxVal,
  }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <RadarChart data={data}>
        <PolarGrid stroke="#1a4d32" />
        <PolarAngleAxis dataKey="dimension" tick={{ fill: "#6fcf97", fontSize: 11 }} />
        <Radar name={teamA} dataKey={teamA} stroke="#6fcf97" fill="#6fcf97" fillOpacity={0.25} />
        <Radar name={teamB} dataKey={teamB} stroke="#fb7185" fill="#fb7185" fillOpacity={0.2} />
        <Legend />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function TeamSelect({
  label,
  value,
  onChange,
  teams,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  teams: string[];
}) {
  return (
    <div className="flex-1">
      <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-pitch-300/70">
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-xl border border-pitch-700 bg-pitch-800 px-4 py-3 text-white focus:border-pitch-500 focus:outline-none"
      >
        <option value="">Select team...</option>
        {teams.map((t) => (
          <option key={t} value={t}>
            {t}
          </option>
        ))}
      </select>
    </div>
  );
}

export default function FixtureBriefPanel() {
  const [teams, setTeams] = useState<string[]>([]);
  const [teamA, setTeamA] = useState("");
  const [teamB, setTeamB] = useState("");
  const [brief, setBrief] = useState<FixtureBriefType | null>(null);
  const [outlook, setOutlook] = useState<TournamentOutlook | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTeams().then(setTeams).catch(() => setTeams([]));
  }, []);

  const runDiagnosis = async () => {
    if (!teamA || !teamB) return;
    setLoading(true);
    setError(null);
    try {
      const [briefRes, outlookRes] = await Promise.all([
        fetchFixtureBrief(teamA, teamB),
        fetchTournamentOutlook(teamA),
      ]);
      setBrief(briefRes);
      setOutlook(outlookRes);
    } catch (err) {
      setBrief(null);
      setOutlook(null);
      setError(err instanceof Error ? err.message : "Diagnosis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
        <TeamSelect label="Your team" value={teamA} onChange={setTeamA} teams={teams} />
        <TeamSelect label="Opponent" value={teamB} onChange={setTeamB} teams={teams} />
        <button
          type="button"
          onClick={runDiagnosis}
          disabled={loading || !teamA || !teamB || teamA === teamB}
          className="rounded-xl bg-pitch-500 px-6 py-3 font-medium text-white transition hover:bg-pitch-300 disabled:opacity-50 sm:mb-0"
        >
          {loading ? "Analyzing..." : "Generate Brief"}
        </button>
      </div>

      {error && (
        <div className="rounded-2xl border border-red-500/30 bg-red-500/10 px-6 py-4 text-red-200">
          {error}
        </div>
      )}

      {brief && (
        <>
          <div className="rounded-2xl border border-pitch-700 bg-pitch-800/50 p-6">
            <h2 className="mb-2 text-xl font-semibold text-white">
              {brief.team_a} vs {brief.team_b}
            </h2>
            <p className="text-sm text-pitch-300/80">{brief.summary}</p>
          </div>

          <div className="rounded-2xl border border-pitch-700 bg-pitch-800/50 p-6">
            <h3 className="mb-4 text-lg font-medium text-white">Style DNA Comparison</h3>
            <StyleDNARadar
              teamA={brief.team_a}
              teamB={brief.team_b}
              dnaA={brief.team_a_fingerprint.style_dna}
              dnaB={brief.team_b_fingerprint.style_dna}
            />
          </div>

          {brief.style_clashes.length > 0 && (
            <section>
              <h3 className="mb-3 text-lg font-medium text-white">Style Clashes</h3>
              <div className="grid gap-3 sm:grid-cols-2">
                {brief.style_clashes.map((c) => (
                  <div
                    key={c.dimension}
                    className="rounded-xl border border-pitch-700 bg-pitch-800/80 p-4"
                  >
                    <p className="text-sm text-pitch-300">{c.description}</p>
                    <p className="mt-1 text-xs text-pitch-300/50">
                      Δ {c.delta_pct > 0 ? "+" : ""}
                      {c.delta_pct.toFixed(0)}%
                    </p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {brief.structural_gaps.length > 0 && (
            <section>
              <h3 className="mb-3 text-lg font-medium text-white">Structural Gaps</h3>
              <div className="space-y-2">
                {brief.structural_gaps.map((g) => (
                  <div
                    key={g.archetype}
                    className="rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-sm text-amber-100/90"
                  >
                    {g.description}
                  </div>
                ))}
              </div>
            </section>
          )}

          {brief.exploit_vectors.length > 0 && (
            <section>
              <h3 className="mb-3 text-lg font-medium text-white">Exploit Vectors</h3>
              <ul className="list-inside list-disc space-y-1 text-sm text-pitch-300">
                {brief.exploit_vectors.map((v) => (
                  <li key={v}>{v}</li>
                ))}
              </ul>
            </section>
          )}

          <section>
            <h3 className="mb-3 text-lg font-medium text-white">Adjustment Cards</h3>
            <div className="grid gap-4 sm:grid-cols-2">
              {brief.adjustments.map((a) => (
                <article
                  key={a.title}
                  className="rounded-2xl border border-pitch-700 bg-pitch-800/80 p-5"
                >
                  <span className="mb-2 inline-block rounded-full bg-pitch-700 px-2.5 py-0.5 text-xs text-pitch-300">
                    {a.category}
                  </span>
                  <h4 className="font-semibold text-white">{a.title}</h4>
                  <p className="mt-2 text-sm text-pitch-300/80">{a.detail}</p>
                  {a.suggested_players.length > 0 && (
                    <p className="mt-2 text-xs text-pitch-500">
                      Suggested: {a.suggested_players.join(", ")}
                    </p>
                  )}
                </article>
              ))}
            </div>
          </section>

          {brief.wildcard_picks.length > 0 && (
            <section>
              <h3 className="mb-3 text-lg font-medium text-white">Wildcard Picks</h3>
              <div className="grid gap-3 sm:grid-cols-3">
                {brief.wildcard_picks.map((w) => (
                  <div
                    key={w.player}
                    className="rounded-xl border border-pitch-500/30 bg-pitch-800 p-4"
                  >
                    <p className="font-medium text-white">{w.player}</p>
                    <p className="text-xs text-pitch-300">
                      {w.position} · fills {w.fills_gap}
                    </p>
                    <p className="mt-1 text-sm text-pitch-500">
                      {(w.similarity_to_opponent * 100).toFixed(0)}% style match to opponent
                      archetype
                    </p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {outlook && (
            <section className="rounded-2xl border border-pitch-700 bg-pitch-900/40 p-6">
              <h3 className="mb-2 text-lg font-medium text-white">Tournament Outlook</h3>
              <p className="mb-4 text-sm text-pitch-300/80">{outlook.outlook_summary}</p>
              {outlook.hardest_matchups.length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-medium uppercase tracking-wide text-pitch-300/60">
                    Hardest stylistic matchups
                  </p>
                  {outlook.hardest_matchups.slice(0, 3).map((m) => (
                    <div key={m.opponent} className="text-sm text-pitch-300">
                      <span className="text-white">{m.opponent}</span> — score {m.difficulty_score}
                      : {m.top_gap}
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}
        </>
      )}

      {!brief && !loading && !error && (
        <p className="text-pitch-300/70">
          Select two teams to generate a fixture brief — style DNA, structural gaps, and scouting
          adjustments.
        </p>
      )}
    </div>
  );
}
