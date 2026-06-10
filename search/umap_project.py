"""Compute 2D UMAP projection for cluster visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import umap

DATA_DIR = Path(__file__).parent.parent / "data"


def compute_umap(
    embeddings_path: Path = DATA_DIR / "embeddings.npy",
    player_index_path: Path = DATA_DIR / "player_index.json",
    output_path: Path = DATA_DIR / "umap_coords.json",
    seed: int = 42,
) -> list[dict]:
    embeddings = np.load(embeddings_path)
    meta = json.loads(player_index_path.read_text())

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    coords = reducer.fit_transform(embeddings)

    payload = []
    for i, row in enumerate(meta):
        payload.append(
            {
                **row,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
            }
        )

    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved UMAP coords for {len(payload)} players -> {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=DATA_DIR / "embeddings.npy")
    args = parser.parse_args()
    compute_umap(args.embeddings)


if __name__ == "__main__":
    main()
