"""World Cup roster enrichment from StatsBomb lineups (jersey, nationality, position)."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd
import requests

STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
STATSBOMB_SEASON_FILES = {
    2018: 3,
    2022: 106,
}


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9 ]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def normalize_team(team: str) -> str:
    return normalize_name(team)


def _primary_position(positions: list[dict]) -> str:
    if not positions:
        return ""
    counts: Counter[str] = Counter()
    for entry in positions:
        pos = entry.get("position")
        if pos:
            counts[pos] += 1
    return counts.most_common(1)[0][0] if counts else ""


def fetch_statsbomb_roster(season: int) -> pd.DataFrame:
    """Aggregate player roster from all World Cup match lineups."""
    if season not in STATSBOMB_SEASON_FILES:
        raise ValueError(f"No StatsBomb season file for {season}")

    season_file = STATSBOMB_SEASON_FILES[season]
    matches = requests.get(
        f"{STATSBOMB_BASE}/matches/43/{season_file}.json", timeout=60
    ).json()

    rows: list[dict] = []
    for match in matches:
        match_id = match["match_id"]
        lineups = requests.get(
            f"{STATSBOMB_BASE}/lineups/{match_id}.json", timeout=60
        ).json()
        for team_block in lineups:
            team = team_block["team_name"]
            for player in team_block["lineup"]:
                country = player.get("country") or {}
                rows.append(
                    {
                        "player": player["player_name"],
                        "team": team,
                        "nation": country.get("name", ""),
                        "jersey_number": player.get("jersey_number"),
                        "position_detail": _primary_position(player.get("positions", [])),
                        "season": season,
                    }
                )

    if not rows:
        raise RuntimeError(f"No StatsBomb roster rows for season {season}")

    df = pd.DataFrame(rows)
    df["name_key"] = df["player"].map(normalize_name)
    df["team_key"] = df["team"].map(normalize_team)

    agg = (
        df.groupby(["name_key", "team_key", "season"], as_index=False)
        .agg(
            player=("player", "first"),
            team=("team", "first"),
            nation=("nation", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
            jersey_number=("jersey_number", lambda s: int(s.mode().iloc[0]) if len(s.mode()) else None),
            position_detail=("position_detail", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
        )
    )
    return agg
