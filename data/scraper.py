"""
Scrape World Cup player stats from FBref (soccerdata) and StatsBomb open data.

Usage:
    python -m data.scraper --source fbref --seasons 2018 2022
    python -m data.scraper --source statsbomb --seasons 2014 2018 2022
    python -m data.scraper --source sample  # offline demo data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from data.features import FEATURE_COLUMNS, METADATA_COLUMNS

DATA_DIR = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "players.parquet"

STATSBOMB_COMPETITIONS = {
    2014: 43,
    2018: 43,
    2022: 43,
}

FBREF_STAT_TYPES = [
    "standard",
    "shooting",
    "passing",
    "passing_types",
    "gca",
    "defense",
    "possession",
    "misc",
]


def _per90(value: pd.Series, minutes: pd.Series) -> pd.Series:
    return (value / minutes.replace(0, np.nan)) * 90


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col if c).strip("_") for col in df.columns]
    return df


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _pick_column(df: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return _safe_numeric(df[name], default)
    return pd.Series(default, index=df.index, dtype=float)


def _normalize_position(pos: str) -> str:
    if not isinstance(pos, str) or not pos.strip():
        return "UNK"
    primary = pos.split(",")[0].strip().upper()
    mapping = {
        "GK": "GK",
        "DF": "DF",
        "MF": "MF",
        "FW": "FW",
        "CB": "DF",
        "LB": "DF",
        "RB": "DF",
        "WB": "DF",
        "DM": "MF",
        "CM": "MF",
        "AM": "MF",
        "LW": "MF",
        "RW": "MF",
        "ST": "FW",
        "CF": "FW",
    }
    return mapping.get(primary, primary[:2] if len(primary) >= 2 else "UNK")


def build_feature_frame(raw: pd.DataFrame, source: str, season: int) -> pd.DataFrame:
    """Map heterogeneous source columns into the unified feature schema."""
    raw = raw.copy()
    raw = _flatten_columns(raw)

    minutes = _pick_column(
        raw,
        ["Playing Time_Min", "Min", "minutes", "minutes_played", "Minutes"],
        default=np.nan,
    )
    minutes = minutes.replace(0, np.nan)
    matches = _pick_column(raw, ["Playing Time_MP", "MP", "matches"], default=1).replace(0, 1)

    out = pd.DataFrame(index=raw.index)
    out["player"] = _pick_column(raw, ["player", "Player"], default="Unknown").astype(str)
    out["team"] = _pick_column(raw, ["team", "Team", "squad_name"], default="Unknown").astype(str)
    out["nation"] = _pick_column(raw, ["nation", "Nation", "country"], default="").astype(str)
    out["position"] = raw.get("pos", raw.get("position", "UNK")).astype(str).map(_normalize_position)
    out["season"] = season
    out["source"] = source
    out["minutes_played"] = minutes.fillna(0)

    out["goals_per90"] = _per90(_pick_column(raw, ["Performance_Gls", "Gls", "goals"]), minutes)
    out["xg_per90"] = _per90(_pick_column(raw, ["Expected_xG", "xG", "xg"]), minutes)
    out["assists_per90"] = _per90(_pick_column(raw, ["Performance_Ast", "Ast", "assists"]), minutes)
    out["xa_per90"] = _per90(_pick_column(raw, ["Expected_xAG", "xAG", "xa"]), minutes)
    out["key_passes_per90"] = _per90(_pick_column(raw, ["KP", "key_passes"]), minutes)
    out["progressive_passes_per90"] = _per90(
        _pick_column(raw, ["PrgP", "progressive_passes"]), minutes
    )
    out["progressive_carries_per90"] = _per90(
        _pick_column(raw, ["PrgC", "progressive_carries"]), minutes
    )
    out["dribbles_per90"] = _per90(
        _pick_column(raw, ["Succ", "dribbles_completed", "dribbles"]), minutes
    )
    out["pressures_per90"] = _per90(_pick_column(raw, ["Press", "pressures"]), minutes)
    out["tackles_per90"] = _per90(_pick_column(raw, ["Tkl", "tackles"]), minutes)
    out["interceptions_per90"] = _per90(_pick_column(raw, ["Int", "interceptions"]), minutes)
    out["blocks_per90"] = _per90(_pick_column(raw, ["Blocks", "blocks"]), minutes)
    out["aerials_won_per90"] = _per90(_pick_column(raw, ["Won", "aerials_won"]), minutes)
    out["pass_completion_pct"] = _pick_column(
        raw, ["Cmp%", "pass_completion_pct", "pass_completion"], default=0.0
    )
    out["touches_final_third_per90"] = _per90(
        _pick_column(raw, ["Touches_Att 3rd", "touches_final_third"]), minutes
    )
    out["shot_creating_actions_per90"] = _per90(
        _pick_column(raw, ["SCA", "shot_creating_actions"]), minutes
    )
    out["goal_creating_actions_per90"] = _per90(
        _pick_column(raw, ["GCA", "goal_creating_actions"]), minutes
    )
    out["shots_per90"] = _per90(_pick_column(raw, ["Standard_Sh", "Sh", "shots"]), minutes)
    out["shots_on_target_per90"] = _per90(
        _pick_column(raw, ["Standard_SoT", "SoT", "shots_on_target"]), minutes
    )
    out["passes_into_penalty_area_per90"] = _per90(
        _pick_column(raw, ["PPA", "passes_into_penalty_area"]), minutes
    )
    out["crosses_per90"] = _per90(_pick_column(raw, ["Crs", "crosses"]), minutes)
    out["fouls_per90"] = _per90(_pick_column(raw, ["Fls", "fouls"]), minutes)
    out["fouled_per90"] = _per90(_pick_column(raw, ["Fld", "fouled"]), minutes)
    out["yellow_cards_per90"] = _per90(_pick_column(raw, ["CrdY", "yellow_cards"]), minutes)
    out["dispossessed_per90"] = _per90(_pick_column(raw, ["Dis", "dispossessed"]), minutes)
    out["miscontrols_per90"] = _per90(_pick_column(raw, ["Mis", "miscontrols"]), minutes)
    out["passes_received_per90"] = _per90(
        _pick_column(raw, ["Rec", "passes_received"]), minutes
    )
    out["progressive_passes_received_per90"] = _per90(
        _pick_column(raw, ["PrgR", "progressive_passes_received"]), minutes
    )
    out["carries_into_penalty_area_per90"] = _per90(
        _pick_column(raw, ["CPA", "carries_into_penalty_area"]), minutes
    )
    out["minutes_per_match"] = minutes / matches

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out[out["minutes_played"] >= 90].copy()
    return out


def scrape_fbref(seasons: Iterable[int]) -> pd.DataFrame:
    import soccerdata as sd

    frames: list[pd.DataFrame] = []
    for season in seasons:
        fbref = sd.FBref(leagues="INT-World Cup", seasons=str(season))
        merged = None
        for stat_type in FBREF_STAT_TYPES:
            try:
                stats = fbref.read_player_season_stats(stat_type=stat_type)
                stats = stats.reset_index()
                stats = _flatten_columns(stats)
                if merged is None:
                    merged = stats
                else:
                    key_cols = [c for c in ["league", "season", "team", "player"] if c in stats.columns]
                    merged = merged.merge(stats, on=key_cols, how="outer", suffixes=("", f"_{stat_type}"))
            except Exception as exc:
                print(f"  [fbref] skipped {stat_type} for {season}: {exc}")

        if merged is not None and not merged.empty:
            frames.append(build_feature_frame(merged, "fbref", season))

    if not frames:
        raise RuntimeError("No FBref data retrieved. Check seasons and network connectivity.")
    return pd.concat(frames, ignore_index=True)


def _statsbomb_player_season_stats(season: int) -> pd.DataFrame:
    comp_id = STATSBOMB_COMPETITIONS[season]
    base = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
    matches = requests.get(f"{base}/matches/{comp_id}/{season}.json", timeout=60).json()
    match_ids = {m["match_id"] for m in matches if m.get("competition", {}).get("competition_name") == "FIFA World Cup"}

    lineups = requests.get(f"{base}/lineups/{comp_id}.json", timeout=60).json()
    events_cache: dict[int, list] = {}

    rows: list[dict] = []
    for lineup in lineups:
        match_id = lineup["match_id"]
        if match_id not in match_ids:
            continue
        if match_id not in events_cache:
            events_cache[match_id] = requests.get(
                f"{base}/events/{match_id}.json", timeout=60
            ).json()

        events = events_cache[match_id]
        for team_lineup in lineup["lineup"]:
            team = team_lineup["team_name"]
            for player_info in team_lineup["lineup"]:
                player = player_info["player_name"]
                pid = player_info["player_id"]
                pos = player_info.get("player_position", "UNK")
                player_events = [e for e in events if e.get("player", {}).get("id") == pid]
                minutes = sum(
                    1 for e in player_events if e.get("type", {}).get("name") in {"Pass", "Carry", "Pressure"}
                )
                minutes = max(minutes / 10, 1)

                def count_event(name: str) -> int:
                    return sum(1 for e in player_events if e.get("type", {}).get("name") == name)

                def count_shot_stat(key: str) -> float:
                    return sum(
                        e.get("shot", {}).get(key, 0) or 0
                        for e in player_events
                        if e.get("type", {}).get("name") == "Shot"
                    )

                rows.append(
                    {
                        "player": player,
                        "team": team,
                        "nation": team,
                        "pos": pos,
                        "minutes": minutes,
                        "goals": count_shot_stat("outcome_name") if False else count_event("Shot"),
                        "Gls": sum(
                            1
                            for e in player_events
                            if e.get("type", {}).get("name") == "Shot"
                            and e.get("shot", {}).get("outcome", {}).get("name") == "Goal"
                        ),
                        "Ast": sum(
                            1
                            for e in player_events
                            if e.get("pass", {}).get("goal_assist")
                        ),
                        "xG": count_shot_stat("statsbomb_xg"),
                        "KP": sum(
                            1
                            for e in player_events
                            if e.get("pass", {}).get("shot_assist")
                        ),
                        "PrgP": sum(
                            1
                            for e in player_events
                            if e.get("pass", {}).get("progressive_pass")
                        ),
                        "PrgC": sum(
                            1
                            for e in player_events
                            if e.get("carry", {}).get("progressive_carry")
                        ),
                        "Press": count_event("Pressure"),
                        "Tkl": count_event("Duel"),
                        "Int": count_event("Interception"),
                        "Sh": count_event("Shot"),
                        "SCA": sum(
                            1
                            for e in player_events
                            if e.get("pass", {}).get("shot_assist")
                            or e.get("dribble", {}).get("outcome", {}).get("name") == "Complete"
                        ),
                    }
                )

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError(f"No StatsBomb rows for season {season}")
    agg = (
        raw.groupby(["player", "team", "nation", "pos"], as_index=False)
        .sum(numeric_only=True)
        .rename(columns={"minutes": "Min"})
    )
    return build_feature_frame(agg, "statsbomb", season)


def scrape_statsbomb(seasons: Iterable[int]) -> pd.DataFrame:
    frames = [_statsbomb_player_season_stats(season) for season in seasons]
    return pd.concat(frames, ignore_index=True)


def generate_sample_data(n_players: int = 400, seed: int = 42) -> pd.DataFrame:
    """Synthetic World Cup-style dataset for offline dev and CI."""
    rng = np.random.default_rng(seed)
    positions = ["GK", "DF", "MF", "FW"]
    teams = [
        "Brazil", "France", "Germany", "Argentina", "Spain", "England",
        "Portugal", "Netherlands", "Belgium", "Croatia", "Morocco", "Japan",
    ]
    first = ["Jude", "Erling", "Kylian", "Luka", "Kevin", "Virgil", "Rodri", "Pedri", "Vinicius", "Bukayo"]
    last = ["Bellingham", "Haaland", "Mbappe", "Modric", "De Bruyne", "Van Dijk", "Hernandez", "Gonzalez", "Junior", "Saka"]

    profiles = {
        "GK": dict(goals_per90=0.0, xg_per90=0.0, pressures_per90=0.5, tackles_per90=0.1, pass_completion_pct=72),
        "DF": dict(goals_per90=0.05, xg_per90=0.08, pressures_per90=12, tackles_per90=2.5, pass_completion_pct=86),
        "MF": dict(goals_per90=0.15, xg_per90=0.18, pressures_per90=18, tackles_per90=2.0, pass_completion_pct=84),
        "FW": dict(goals_per90=0.55, xg_per90=0.52, pressures_per90=10, tackles_per90=0.6, pass_completion_pct=74),
    }

    rows = []
    for i in range(n_players):
        pos = rng.choice(positions, p=[0.08, 0.28, 0.38, 0.26])
        base = profiles[pos]
        player = f"{rng.choice(first)} {rng.choice(last)} {i}"
        minutes = rng.integers(180, 720)
        row = {col: max(0.0, rng.normal(0.5, 0.35)) for col in FEATURE_COLUMNS}
        for key, val in base.items():
            row[key] = max(0.0, val + rng.normal(0, val * 0.25))
        row.update(
            {
                "player": player,
                "team": rng.choice(teams),
                "nation": rng.choice(teams),
                "position": pos,
                "season": int(rng.choice([2018, 2022, 2026])),
                "source": "sample",
                "minutes_played": minutes,
                "minutes_per_match": minutes / rng.integers(3, 8),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def save_players(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = METADATA_COLUMNS + FEATURE_COLUMNS
    df[keep].to_parquet(path, index=False)
    print(f"Saved {len(df)} players to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape World Cup player stats")
    parser.add_argument(
        "--source",
        choices=["fbref", "statsbomb", "sample"],
        default="sample",
        help="Data source (default: sample for offline use)",
    )
    parser.add_argument("--seasons", nargs="+", type=int, default=[2018, 2022])
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--sample-size", type=int, default=400)
    args = parser.parse_args()

    if args.source == "fbref":
        df = scrape_fbref(args.seasons)
    elif args.source == "statsbomb":
        df = scrape_statsbomb(args.seasons)
    else:
        df = generate_sample_data(n_players=args.sample_size)

    save_players(df, args.output)


if __name__ == "__main__":
    main()
