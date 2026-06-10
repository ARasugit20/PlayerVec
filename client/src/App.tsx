import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchUMAP, searchSimilar } from "./api";
import ClusterMap from "./components/ClusterMap";
import SearchBar from "./components/SearchBar";
import SimilarPlayers from "./components/SimilarPlayers";
import type { SimilarPlayer, UMAPPoint } from "./types";

export default function App() {
  const [query, setQuery] = useState<string | null>(null);
  const [results, setResults] = useState<SimilarPlayer[]>([]);
  const [umapPoints, setUmapPoints] = useState<UMAPPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchUMAP().then(setUmapPoints).catch(() => setUmapPoints([]));
  }, []);

  const handleSearch = useCallback(async (player: string) => {
    setQuery(player);
    setLoading(true);
    setError(null);
    try {
      const res = await searchSimilar(player, 5);
      setResults(res.results);
    } catch (err) {
      setResults([]);
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const highlightIds = useMemo(() => {
    if (!query || results.length === 0) return new Set<number>();
    const names = new Set([query, ...results.map((r) => r.player)]);
    return new Set(umapPoints.filter((p) => names.has(p.player)).map((p) => p.id));
  }, [query, results, umapPoints]);

  return (
    <div className="min-h-screen">
      <header className="border-b border-pitch-700/50 bg-pitch-900/80 backdrop-blur">
        <div className="mx-auto max-w-6xl px-4 py-8">
          <p className="mb-1 text-sm font-medium uppercase tracking-widest text-pitch-500">
            PlayerVec
          </p>
          <h1 className="mb-2 text-3xl font-bold text-white md:text-4xl">
            World Cup Player Style Search
          </h1>
          <p className="mb-6 max-w-2xl text-pitch-300/80">
            A PyTorch autoencoder compresses ~30 stats into 32-dim embeddings. Find players with
            similar styles — high-press midfielders, deep-lying playmakers, target strikers — without
            position labels.
          </p>
          <SearchBar onSearch={handleSearch} loading={loading} />
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-10 px-4 py-10">
        <SimilarPlayers query={query} results={results} loading={loading} error={error} />

        <ClusterMap
          points={umapPoints}
          selectedPlayer={query}
          onSelectPlayer={handleSearch}
          highlightIds={highlightIds}
        />
      </main>

      <footer className="border-t border-pitch-700/50 py-6 text-center text-sm text-pitch-300/40">
        PlayerVec · unsupervised style embeddings · FBref / StatsBomb
      </footer>
    </div>
  );
}
