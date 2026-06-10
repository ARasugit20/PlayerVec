"""FastAPI backend for PlayerVec similarity search."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data.features import FEATURE_COLUMNS
from search.query import PlayerSearch

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


class SimilarPlayer(BaseModel):
    player: str
    team: str
    nation: str
    position: str
    season: int
    similarity: float
    key_stats: dict[str, float]


class SimilarResponse(BaseModel):
    query: str
    results: list[SimilarPlayer]


class UMAPPoint(BaseModel):
    id: int
    player: str
    team: str
    nation: str
    position: str
    season: int
    x: float
    y: float


app = FastAPI(title="PlayerVec API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_search() -> PlayerSearch:
    return PlayerSearch()


@lru_cache
def get_stats_lookup() -> dict[str, dict[str, float]]:
    path = DATA_DIR / "players.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    lookup = {}
    for _, row in df.iterrows():
        lookup[row["player"]] = {col: float(row[col]) for col in FEATURE_COLUMNS[:6]}
    return lookup


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/players")
def list_players(q: Optional[str] = Query(None, description="Filter by name substring")) -> list[str]:
    players = get_search().all_players()
    if q:
        q_lower = q.lower()
        players = [p for p in players if q_lower in p.lower()]
    return players[:50]


@app.get("/similar", response_model=SimilarResponse)
def similar(
    player: str = Query(..., description="Player name"),
    k: int = Query(5, ge=1, le=20),
) -> SimilarResponse:
    try:
        results = get_search().similar(player, k=k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    stats_lookup = get_stats_lookup()
    enriched = []
    for row in results:
        enriched.append(
            SimilarPlayer(
                player=row["player"],
                team=row["team"],
                nation=row["nation"],
                position=row["position"],
                season=row["season"],
                similarity=row["similarity"],
                key_stats=stats_lookup.get(row["player"], {}),
            )
        )
    return SimilarResponse(query=player, results=enriched)


@app.get("/umap", response_model=list[UMAPPoint])
def umap_coords() -> list[UMAPPoint]:
    path = DATA_DIR / "umap_coords.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="UMAP coordinates not generated yet")
    data = json.loads(path.read_text())
    return [UMAPPoint(**row) for row in data]
