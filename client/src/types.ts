export interface SimilarPlayer {
  player: string;
  team: string;
  nation: string;
  position: string;
  season: number;
  similarity: number;
  key_stats: Record<string, number>;
}

export interface SimilarResponse {
  query: string;
  results: SimilarPlayer[];
}

export interface UMAPPoint {
  id: number;
  player: string;
  team: string;
  nation: string;
  position: string;
  season: number;
  x: number;
  y: number;
}

export interface TeamFingerprint {
  team: string;
  squad_size: number;
  total_minutes: number;
  style_dna: Record<string, number>;
  archetype_mix: Record<string, number>;
  top_players: Array<{
    player: string;
    position: string;
    minutes: number;
    archetype: string;
  }>;
}

export interface StyleClash {
  dimension: string;
  team_a_value: number;
  team_b_value: number;
  delta_pct: number;
  description: string;
}

export interface StructuralGap {
  archetype: string;
  opponent_share: number;
  your_share: number;
  description: string;
}

export interface AdjustmentCard {
  category: string;
  title: string;
  detail: string;
  suggested_players: string[];
}

export interface WildcardPick {
  player: string;
  position: string;
  archetype: string;
  fills_gap: string;
  similarity_to_opponent: number;
}

export interface FixtureBrief {
  team_a: string;
  team_b: string;
  team_a_fingerprint: TeamFingerprint;
  team_b_fingerprint: TeamFingerprint;
  style_clashes: StyleClash[];
  structural_gaps: StructuralGap[];
  exploit_vectors: string[];
  adjustments: AdjustmentCard[];
  wildcard_picks: WildcardPick[];
  summary: string;
}

export interface TournamentOutlook {
  team: string;
  fingerprint: TeamFingerprint;
  hardest_matchups: Array<{
    opponent: string;
    difficulty_score: number;
    top_gap: string;
    key_adjustment: string | null;
  }>;
  squad_redundancy: string[];
  outlook_summary: string;
}
