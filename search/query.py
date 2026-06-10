"""Similarity search over player embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
SEARCH_DIR = Path(__file__).parent


class PlayerSearch:
    def __init__(
        self,
        index_path: Path = SEARCH_DIR / "faiss.index",
        meta_path: Path = SEARCH_DIR / "faiss_meta.json",
        embeddings_path: Path = DATA_DIR / "embeddings.npy",
    ):
        self.index = faiss.read_index(str(index_path))
        self.meta: list[dict] = json.loads(meta_path.read_text())
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        faiss.normalize_L2(self.embeddings)
        self._name_to_ids: dict[str, list[int]] = {}
        for row in self.meta:
            key = row["player"].lower()
            self._name_to_ids.setdefault(key, []).append(row["id"])

    def _resolve_id(self, player: str) -> int:
        key = player.lower().strip()
        if key.isdigit():
            return int(key)
        matches = self._name_to_ids.get(key, [])
        if not matches:
            partial = [rid for name, ids in self._name_to_ids.items() if key in name for rid in ids]
            if not partial:
                raise KeyError(f"Player not found: {player}")
            return partial[0]
        return matches[0]

    def similar(self, player: str, k: int = 5) -> list[dict]:
        player_id = self._resolve_id(player)
        query = self.embeddings[player_id : player_id + 1].copy()
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, k + 1)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == player_id:
                continue
            row = dict(self.meta[idx])
            row["similarity"] = float(score)
            results.append(row)
            if len(results) >= k:
                break
        return results

    def all_players(self) -> list[str]:
        return sorted({row["player"] for row in self.meta})
