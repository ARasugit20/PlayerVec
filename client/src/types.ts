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
