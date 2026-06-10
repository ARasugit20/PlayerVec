"""Build FAISS index from player embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
SEARCH_DIR = Path(__file__).parent


def build_index(
    embeddings_path: Path = DATA_DIR / "embeddings.npy",
    index_path: Path = SEARCH_DIR / "faiss.index",
    meta_path: Path = SEARCH_DIR / "faiss_meta.json",
    player_index_path: Path = DATA_DIR / "player_index.json",
) -> faiss.Index:
    embeddings = np.load(embeddings_path).astype(np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    meta = json.loads(player_index_path.read_text())
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Built FAISS index: {index.ntotal} vectors, dim={dim} -> {index_path}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=DATA_DIR / "embeddings.npy")
    args = parser.parse_args()
    build_index(args.embeddings)


if __name__ == "__main__":
    main()
