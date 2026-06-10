"""Generate player embeddings from a trained autoencoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.features import FEATURE_COLUMNS
from model.autoencoder import PlayerAutoencoder

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def load_scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    return np.array(payload["mean"]), np.array(payload["scale"])


def embed_players(
    data_path: Path = DATA_DIR / "players.parquet",
    checkpoint: Path = CHECKPOINT_DIR / "autoencoder.pt",
    scaler_path: Path = CHECKPOINT_DIR / "scaler.json",
    output_embeddings: Path = DATA_DIR / "embeddings.npy",
    output_index: Path = DATA_DIR / "player_index.json",
) -> tuple[np.ndarray, list[dict]]:
    df = pd.read_parquet(data_path)
    mean, scale = load_scaler(scaler_path)
    X = df[FEATURE_COLUMNS].astype(float).values
    X_scaled = (X - mean) / scale

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PlayerAutoencoder(
        input_dim=ckpt["input_dim"],
        embedding_dim=ckpt["embedding_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        embeddings = model.encode(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    meta = df[["player", "team", "nation", "position", "season", "source"]].to_dict("records")
    for i, row in enumerate(meta):
        row["id"] = i
        row["minutes_played"] = float(df.iloc[i]["minutes_played"])

    output_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_embeddings, embeddings)
    output_index.write_text(json.dumps(meta, indent=2))

    print(f"Saved {len(meta)} embeddings ({embeddings.shape[1]}d) to {output_embeddings}")
    return embeddings, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR / "players.parquet")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "autoencoder.pt")
    args = parser.parse_args()
    embed_players(args.data, args.checkpoint)


if __name__ == "__main__":
    main()
