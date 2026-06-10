"""Training loop for the PlayerVec autoencoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data.features import FEATURE_COLUMNS
from model.autoencoder import PlayerAutoencoder

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
DEFAULT_DATA = DATA_DIR / "players.parquet"


def load_and_scale(path: Path) -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    df = pd.read_parquet(path)
    X = df[FEATURE_COLUMNS].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, df


def train(
    data_path: Path = DEFAULT_DATA,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, scaler, _ = load_and_scale(data_path)
    X_train, X_val = train_test_split(X, test_size=0.15, random_state=seed)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlayerAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "autoencoder.pt"
    scaler_path = CHECKPOINT_DIR / "scaler.json"

    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            recon, _ = model(val_tensor.to(device))
            val_loss = criterion(recon, val_tensor.to(device)).item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": model.encoder.encoder[0].in_features,
                    "embedding_dim": model.encoder.encoder[-1].out_features,
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                ckpt_path,
            )

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")

    scaler_payload = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "features": FEATURE_COLUMNS,
    }
    scaler_path.write_text(json.dumps(scaler_payload, indent=2))
    (CHECKPOINT_DIR / "history.json").write_text(json.dumps(history, indent=2))

    print(f"Best val loss: {best_val_loss:.4f} -> {ckpt_path}")
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
