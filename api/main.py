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
from team.diagnose import FixtureDiagnostician
from team.fingerprint import TeamFingerprintEngine

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


class TeamFingerprintResponse(BaseModel):
    team: str
    squad_size: int
    total_minutes: float
    style_dna: dict[str, float]
    archetype_mix: dict[str, float]
    top_players: list[dict]


class FixtureBriefResponse(BaseModel):
    team_a: str
    team_b: str
    team_a_fingerprint: dict
    team_b_fingerprint: dict
    style_clashes: list[dict]
    structural_gaps: list[dict]
    exploit_vectors: list[str]
    adjustments: list[dict]
    wildcard_picks: list[dict]
    summary: str


class TournamentOutlookResponse(BaseModel):
    team: str
    fingerprint: dict
    hardest_matchups: list[dict]
    squad_redundancy: list[str]
    outlook_summary: str


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
def get_fingerprint_engine() -> TeamFingerprintEngine:
    return TeamFingerprintEngine()


@lru_cache
def get_diagnostician() -> FixtureDiagnostician:
    return FixtureDiagnostician(get_fingerprint_engine())


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


@app.get("/teams")
def list_teams(q: Optional[str] = Query(None, description="Filter by team name")) -> list[str]:
    teams = get_fingerprint_engine().all_teams()
    if q:
        q_lower = q.lower()
        teams = [t for t in teams if q_lower in t.lower()]
    return teams


@app.get("/team-fingerprint", response_model=TeamFingerprintResponse)
def team_fingerprint(team: str = Query(..., description="Team name")) -> TeamFingerprintResponse:
    try:
        fp = get_fingerprint_engine().fingerprint(team)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TeamFingerprintResponse(
        team=fp.team,
        squad_size=fp.squad_size,
        total_minutes=fp.total_minutes,
        style_dna=fp.style_dna,
        archetype_mix=fp.archetype_mix,
        top_players=fp.top_players,
    )


@app.get("/fixture-brief", response_model=FixtureBriefResponse)
def fixture_brief(
    team_a: str = Query(..., description="Your team"),
    team_b: str = Query(..., description="Opponent"),
) -> FixtureBriefResponse:
    try:
        brief = get_diagnostician().diagnose(team_a, team_b)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FixtureBriefResponse(**brief.to_dict())


@app.get("/tournament-outlook", response_model=TournamentOutlookResponse)
def tournament_outlook(team: str = Query(..., description="Team name")) -> TournamentOutlookResponse:
    try:
        outlook = get_diagnostician().tournament_outlook(team)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TournamentOutlookResponse(**outlook)
