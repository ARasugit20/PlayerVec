"""PyTorch autoencoder for World Cup player style embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from data.features import EMBEDDING_DIM, INPUT_DIM


class PlayerEncoder(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PlayerDecoder(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class PlayerAutoencoder(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.encoder = PlayerEncoder(input_dim, embedding_dim)
        self.decoder = PlayerDecoder(input_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
