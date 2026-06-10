"""Minute-weighted team style fingerprints from player embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from data.features import FEATURE_COLUMNS

DATA_DIR = Path(__file__).parent.parent / "data"

STYLE_DIMENSIONS: dict[str, list[str]] = {
    "press_intensity": ["pressures_per90", "tackles_per90", "interceptions_per90"],
    "progression": ["progressive_passes_per90", "progressive_carries_per90", "key_passes_per90"],
    "aerial_direct": ["aerials_won_per90", "crosses_per90", "passes_into_penalty_area_per90"],
    "chance_creation": ["shot_creating_actions_per90", "goal_creating_actions_per90", "xa_per90"],
    "finishing": ["goals_per90", "xg_per90", "shots_on_target_per90"],
    "width": ["progressive_carries_per90", "carries_into_penalty_area_per90", "dribbles_per90"],
}

ARCHETYPE_LABELS = [
    "high_presser",
    "deep_progressor",
    "aerial_target",
    "chance_creator",
    "finisher",
    "wide_carrier",
    "ball_winner",
    "balanced",
]

N_ARCHETYPES = len(ARCHETYPE_LABELS)


@dataclass
class PlayerRecord:
    id: int
    player: str
    team: str
    position: str
    minutes: float
    embedding: np.ndarray
    stats: dict[str, float]
    archetype: str = "balanced"


@dataclass
class TeamFingerprint:
    team: str
    squad_size: int
    total_minutes: float
    embedding_centroid: list[float]
    style_dna: dict[str, float]
    archetype_mix: dict[str, float]
    top_players: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "squad_size": self.squad_size,
            "total_minutes": self.total_minutes,
            "embedding_centroid": self.embedding_centroid,
            "style_dna": self.style_dna,
            "archetype_mix": self.archetype_mix,
            "top_players": self.top_players,
        }


class TeamFingerprintEngine:
    def __init__(
        self,
        players_path: Path = DATA_DIR / "players.parquet",
        embeddings_path: Path = DATA_DIR / "embeddings.npy",
        index_path: Path = DATA_DIR / "player_index.json",
    ):
        self.df = pd.read_parquet(players_path)
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)
        self.meta: list[dict] = json.loads(index_path.read_text())
        self._players = self._build_player_records()
        self._archetype_model = self._fit_archetypes()
        self._assign_archetypes()

    def _build_player_records(self) -> list[PlayerRecord]:
        records = []
        for i, row in self.df.iterrows():
            meta = self.meta[i] if i < len(self.meta) else {}
            records.append(
                PlayerRecord(
                    id=int(meta.get("id", i)),
                    player=str(row["player"]),
                    team=str(row["team"]),
                    position=str(row["position"]),
                    minutes=float(row["minutes_played"]),
                    embedding=self.embeddings[i],
                    stats={col: float(row[col]) for col in FEATURE_COLUMNS},
                )
            )
        return records

    def _fit_archetypes(self) -> KMeans:
        X = self.embeddings
        k = min(N_ARCHETYPES, len(X))
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        return model

    def _assign_archetypes(self) -> None:
        labels = self._archetype_model.labels_
        centroids = self._archetype_model.cluster_centers_

        cluster_to_archetype: dict[int, str] = {}
        for cluster_id, centroid in enumerate(centroids):
            cluster_players = [p for p, lbl in zip(self._players, labels) if lbl == cluster_id]
            if not cluster_players:
                cluster_to_archetype[cluster_id] = "balanced"
                continue
            avg_stats = {}
            for dim, cols in STYLE_DIMENSIONS.items():
                vals = []
                for p in cluster_players:
                    vals.extend(p.stats.get(c, 0) for c in cols if c in p.stats)
                avg_stats[dim] = np.mean(vals) if vals else 0.0
            best_dim = max(avg_stats, key=avg_stats.get)
            mapping = {
                "press_intensity": "high_presser",
                "progression": "deep_progressor",
                "aerial_direct": "aerial_target",
                "chance_creation": "chance_creator",
                "finishing": "finisher",
                "width": "wide_carrier",
            }
            cluster_to_archetype[cluster_id] = mapping.get(best_dim, "balanced")

        for player, label in zip(self._players, labels):
            player.archetype = cluster_to_archetype.get(int(label), "balanced")

    def all_teams(self) -> list[str]:
        return sorted({p.team for p in self._players})

    def team_players(self, team: str) -> list[PlayerRecord]:
        key = team.lower().strip()
        return [p for p in self._players if p.team.lower() == key or key in p.team.lower()]

    def resolve_team(self, team: str) -> str:
        key = team.lower().strip()
        for t in self.all_teams():
            if t.lower() == key or key in t.lower():
                return t
        raise KeyError(f"Team not found: {team}")

    def _weighted_centroid(self, players: list[PlayerRecord]) -> np.ndarray:
        if not players:
            return np.zeros(self.embeddings.shape[1])
        weights = np.array([max(p.minutes, 1.0) for p in players])
        weights /= weights.sum()
        return np.average([p.embedding for p in players], axis=0, weights=weights)

    def _style_dna(self, players: list[PlayerRecord]) -> dict[str, float]:
        if not players:
            return {dim: 0.0 for dim in STYLE_DIMENSIONS}
        weights = np.array([max(p.minutes, 1.0) for p in players])
        weights /= weights.sum()
        dna = {}
        for dim, cols in STYLE_DIMENSIONS.items():
            vals = []
            for p, w in zip(players, weights):
                feat_vals = [p.stats.get(c, 0.0) for c in cols]
                vals.append(w * np.mean(feat_vals))
            dna[dim] = float(np.sum(vals))
        return dna

    def _archetype_mix(self, players: list[PlayerRecord]) -> dict[str, float]:
        if not players:
            return {}
        weights = np.array([max(p.minutes, 1.0) for p in players])
        weights /= weights.sum()
        mix: dict[str, float] = {}
        for p, w in zip(players, weights):
            mix[p.archetype] = mix.get(p.archetype, 0.0) + w
        return {k: round(v, 4) for k, v in sorted(mix.items(), key=lambda x: -x[1])}

    def fingerprint(self, team: str) -> TeamFingerprint:
        resolved = self.resolve_team(team)
        players = self.team_players(resolved)
        if not players:
            raise KeyError(f"No players for team: {team}")

        centroid = self._weighted_centroid(players)
        top = sorted(players, key=lambda p: p.minutes, reverse=True)[:5]

        return TeamFingerprint(
            team=resolved,
            squad_size=len(players),
            total_minutes=sum(p.minutes for p in players),
            embedding_centroid=centroid.tolist(),
            style_dna=self._style_dna(players),
            archetype_mix=self._archetype_mix(players),
            top_players=[
                {
                    "player": p.player,
                    "position": p.position,
                    "minutes": p.minutes,
                    "archetype": p.archetype,
                }
                for p in top
            ],
        )

    def all_fingerprints(self) -> dict[str, TeamFingerprint]:
        return {t: self.fingerprint(t) for t in self.all_teams()}
