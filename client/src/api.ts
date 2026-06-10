import type {
  FixtureBrief,
  SimilarResponse,
  TeamFingerprint,
  TournamentOutlook,
  UMAPPoint,
} from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || res.statusText);
  }
  return res.json() as Promise<T>;
}

export function searchSimilar(player: string, k = 5): Promise<SimilarResponse> {
  const params = new URLSearchParams({ player, k: String(k) });
  return fetchJson<SimilarResponse>(`/similar?${params}`);
}

export function fetchPlayers(q?: string): Promise<string[]> {
  const params = q ? `?q=${encodeURIComponent(q)}` : "";
  return fetchJson<string[]>(`/players${params}`);
}

export function fetchUMAP(): Promise<UMAPPoint[]> {
  return fetchJson<UMAPPoint[]>("/umap");
}

export function fetchTeams(q?: string): Promise<string[]> {
  const params = q ? `?q=${encodeURIComponent(q)}` : "";
  return fetchJson<string[]>(`/teams${params}`);
}

export function fetchTeamFingerprint(team: string): Promise<TeamFingerprint> {
  const params = new URLSearchParams({ team });
  return fetchJson<TeamFingerprint>(`/team-fingerprint?${params}`);
}

export function fetchFixtureBrief(teamA: string, teamB: string): Promise<FixtureBrief> {
  const params = new URLSearchParams({ team_a: teamA, team_b: teamB });
  return fetchJson<FixtureBrief>(`/fixture-brief?${params}`);
}

export function fetchTournamentOutlook(team: string): Promise<TournamentOutlook> {
  const params = new URLSearchParams({ team });
  return fetchJson<TournamentOutlook>(`/tournament-outlook?${params}`);
}
