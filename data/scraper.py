"""
Scrape World Cup player stats from FBref (soccerdata) and StatsBomb open data.

Usage:
    python -m data.scraper --source worldcup --seasons 2022
    python -m data.scraper --source fbref --seasons 2018 2022
    python -m data.scraper --source statsbomb --seasons 2022
    python -m data.scraper --source sample  # synthetic data for CI only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from data.features import FEATURE_COLUMNS, METADATA_COLUMNS
from data.roster import fetch_statsbomb_roster, normalize_name, normalize_team

DATA_DIR = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "players.parquet"

# Cache soccerdata inside the project (avoids home-dir permission issues)
os.environ.setdefault("SOCCERDATA_DIR", str(DATA_DIR.parent / "soccerdata"))

STATSBOMB_SEASON_FILES = {
    2018: 3,
    2022: 106,
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


def _pick_numeric(df: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return _safe_numeric(df[name], default)
    return pd.Series(default, index=df.index, dtype=float)


def _pick_text(df: pd.DataFrame, candidates: list[str], default: str = "") -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return df[name].astype(str).replace({"nan": default, "None": default}).fillna(default)
    return pd.Series(default, index=df.index, dtype=str)


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
        "GOALKEEPER": "GK",
        "DEFENDER": "DF",
        "MIDFIELDER": "MF",
        "FORWARD": "FW",
    }
    return mapping.get(primary, primary[:2] if len(primary) >= 2 else "UNK")


def _position_detail_to_group(detail: str) -> str:
    if not detail:
        return "UNK"
    d = detail.lower()
    if "goalkeeper" in d:
        return "GK"
    if "wing" in d or "full back" in d or "back" in d:
        return "DF" if "wing back" not in d or "back" in d else "MF"
    if "forward" in d or "striker" in d:
        return "FW"
    if "midfield" in d:
        return "MF"
    return "UNK"


def enrich_with_roster(stats: pd.DataFrame, season: int) -> pd.DataFrame:
    """Merge StatsBomb jersey numbers, nationality, and detailed positions."""
    try:
        roster = fetch_statsbomb_roster(season)
    except Exception as exc:
        print(f"  [roster] could not enrich season {season}: {exc}")
        stats["position_detail"] = stats.get("position_detail", "")
        stats["jersey_number"] = stats.get("jersey_number", np.nan)
        return stats

    stats = stats.copy()
    stats["name_key"] = stats["player"].map(normalize_name)
    stats["team_key"] = stats["team"].map(normalize_team)

    roster_cols = roster[
        ["name_key", "team_key", "season", "nation", "jersey_number", "position_detail"]
    ].rename(
        columns={
            "nation": "nation_roster",
            "jersey_number": "jersey_roster",
            "position_detail": "position_detail_roster",
        }
    )

    merged = stats.merge(roster_cols, on=["name_key", "team_key", "season"], how="left")

    merged["nation"] = merged["nation_roster"].fillna(merged["nation"]).fillna(merged["team"])
    merged["jersey_number"] = pd.to_numeric(merged["jersey_roster"], errors="coerce")
    merged["position_detail"] = merged["position_detail_roster"].fillna(merged["position_detail"]).fillna("")

    missing_pos = merged["position"].isin(["UNK", "", "nan"])
    merged.loc[missing_pos, "position"] = merged.loc[missing_pos, "position_detail"].map(
        _position_detail_to_group
    )

    drop_cols = ["name_key", "team_key", "nation_roster", "jersey_roster", "position_detail_roster"]
    return merged.drop(columns=[c for c in drop_cols if c in merged.columns])


def build_feature_frame(raw: pd.DataFrame, source: str, season: int) -> pd.DataFrame:
    """Map heterogeneous source columns into the unified feature schema."""
    raw = raw.copy()
    raw = _flatten_columns(raw)

    minutes = _pick_numeric(
        raw,
        ["Playing Time_Min", "Min", "minutes", "minutes_played", "Minutes"],
        default=np.nan,
    )
    minutes = minutes.replace(0, np.nan)
    matches = _pick_numeric(raw, ["Playing Time_MP", "MP", "matches"], default=1).replace(0, 1)

    out = pd.DataFrame(index=raw.index)
    out["player"] = _pick_text(raw, ["player", "Player"], default="Unknown")
    out["team"] = _pick_text(raw, ["team", "Team", "squad_name"], default="Unknown")
    out["nation"] = _pick_text(raw, ["nation", "Nation", "country"], default="")
    pos_raw = _pick_text(raw, ["pos", "position", "Position"], default="UNK")
    out["position"] = pos_raw.map(_normalize_position)
    out["position_detail"] = _pick_text(raw, ["position_detail"], default="")
    out["jersey_number"] = _pick_numeric(raw, ["jersey_number"], default=np.nan)
    out["season"] = season
    out["source"] = source
    out["minutes_played"] = minutes.fillna(0)

    out["goals_per90"] = _per90(_pick_numeric(raw, ["Performance_Gls", "Gls", "goals"]), minutes)
    out["xg_per90"] = _per90(_pick_numeric(raw, ["Expected_xG", "xG", "xg"]), minutes)
    out["assists_per90"] = _per90(_pick_numeric(raw, ["Performance_Ast", "Ast", "assists"]), minutes)
    out["xa_per90"] = _per90(_pick_numeric(raw, ["Expected_xAG", "xAG", "xa"]), minutes)
    out["key_passes_per90"] = _per90(_pick_numeric(raw, ["KP", "key_passes"]), minutes)
    out["progressive_passes_per90"] = _per90(
        _pick_numeric(raw, ["PrgP", "progressive_passes"]), minutes
    )
    out["progressive_carries_per90"] = _per90(
        _pick_numeric(raw, ["PrgC", "progressive_carries"]), minutes
    )
    out["dribbles_per90"] = _per90(
        _pick_numeric(raw, ["Succ", "dribbles_completed", "dribbles"]), minutes
    )
    out["pressures_per90"] = _per90(_pick_numeric(raw, ["Press", "pressures"]), minutes)
    out["tackles_per90"] = _per90(_pick_numeric(raw, ["Tkl", "tackles"]), minutes)
    out["interceptions_per90"] = _per90(_pick_numeric(raw, ["Int", "interceptions"]), minutes)
    out["blocks_per90"] = _per90(_pick_numeric(raw, ["Blocks", "blocks"]), minutes)
    out["aerials_won_per90"] = _per90(_pick_numeric(raw, ["Won", "aerials_won"]), minutes)
    out["pass_completion_pct"] = _pick_numeric(
        raw, ["Cmp%", "pass_completion_pct", "pass_completion"], default=0.0
    )
    out["touches_final_third_per90"] = _per90(
        _pick_numeric(raw, ["Touches_Att 3rd", "touches_final_third"]), minutes
    )
    out["shot_creating_actions_per90"] = _per90(
        _pick_numeric(raw, ["SCA", "shot_creating_actions"]), minutes
    )
    out["goal_creating_actions_per90"] = _per90(
        _pick_numeric(raw, ["GCA", "goal_creating_actions"]), minutes
    )
    out["shots_per90"] = _per90(_pick_numeric(raw, ["Standard_Sh", "Sh", "shots"]), minutes)
    out["shots_on_target_per90"] = _per90(
        _pick_numeric(raw, ["Standard_SoT", "SoT", "shots_on_target"]), minutes
    )
    out["passes_into_penalty_area_per90"] = _per90(
        _pick_numeric(raw, ["PPA", "passes_into_penalty_area"]), minutes
    )
    out["crosses_per90"] = _per90(_pick_numeric(raw, ["Crs", "crosses"]), minutes)
    out["fouls_per90"] = _per90(_pick_numeric(raw, ["Fls", "fouls"]), minutes)
    out["fouled_per90"] = _per90(_pick_numeric(raw, ["Fld", "fouled"]), minutes)
    out["yellow_cards_per90"] = _per90(_pick_numeric(raw, ["CrdY", "yellow_cards"]), minutes)
    out["dispossessed_per90"] = _per90(_pick_numeric(raw, ["Dis", "dispossessed"]), minutes)
    out["miscontrols_per90"] = _per90(_pick_numeric(raw, ["Mis", "miscontrols"]), minutes)
    out["passes_received_per90"] = _per90(
        _pick_numeric(raw, ["Rec", "passes_received"]), minutes
    )
    out["progressive_passes_received_per90"] = _per90(
        _pick_numeric(raw, ["PrgR", "progressive_passes_received"]), minutes
    )
    out["carries_into_penalty_area_per90"] = _per90(
        _pick_numeric(raw, ["CPA", "carries_into_penalty_area"]), minutes
    )
    out["minutes_per_match"] = minutes / matches

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out[out["minutes_played"] >= 90].copy()
    return out


def scrape_fbref(seasons: Iterable[int]) -> pd.DataFrame:
    import soccerdata as sd

    frames: list[pd.DataFrame] = []
    for season in seasons:
        print(f"  [fbref] fetching World Cup {season}...")
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
            frame = build_feature_frame(merged, "fbref", season)
            frames.append(frame)
            print(f"  [fbref] {season}: {len(frame)} players with 90+ minutes")

    if not frames:
        raise RuntimeError("No FBref data retrieved. Check seasons and network connectivity.")
    return pd.concat(frames, ignore_index=True)


def scrape_worldcup(seasons: Iterable[int]) -> pd.DataFrame:
    """FBref stats enriched with StatsBomb roster metadata (jersey, nationality, position)."""
    stats = scrape_fbref(seasons)
    frames = []
    for season in seasons:
        season_df = stats[stats["season"] == season].copy()
        frames.append(enrich_with_roster(season_df, season))
    result = pd.concat(frames, ignore_index=True)
    print(f"  [worldcup] total: {len(result)} players")
    print(result[["player", "team", "nation", "jersey_number", "position", "position_detail"]].head(8).to_string())
    return result


def _statsbomb_player_season_stats(season: int) -> pd.DataFrame:
    if season not in STATSBOMB_SEASON_FILES:
        raise ValueError(f"No StatsBomb data for season {season}")

    base = STATSBOMB_SEASON_FILES[season]
    matches = requests.get(
        f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/43/{base}.json",
        timeout=60,
    ).json()
    match_ids = [m["match_id"] for m in matches]
    events_cache: dict[int, list] = {}

    rows: list[dict] = []
    for match_id in match_ids:
        lineups = requests.get(
            f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/lineups/{match_id}.json",
            timeout=60,
        ).json()
        if match_id not in events_cache:
            events_cache[match_id] = requests.get(
                f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json",
                timeout=60,
            ).json()
        events = events_cache[match_id]

        for team_block in lineups:
            team = team_block["team_name"]
            for player_info in team_block["lineup"]:
                player = player_info["player_name"]
                pid = player_info["player_id"]
                country = (player_info.get("country") or {}).get("name", team)
                jersey = player_info.get("jersey_number")
                pos_detail = ""
                if player_info.get("positions"):
                    pos_detail = player_info["positions"][0].get("position", "")
                pos = _position_detail_to_group(pos_detail) if pos_detail else "UNK"

                player_events = [e for e in events if e.get("player", {}).get("id") == pid]
                minutes = max(
                    sum(
                        1
                        for e in player_events
                        if e.get("type", {}).get("name") in {"Pass", "Carry", "Pressure"}
                    )
                    / 10,
                    1,
                )

                def count_event(name: str) -> int:
                    return sum(1 for e in player_events if e.get("type", {}).get("name") == name)

                rows.append(
                    {
                        "player": player,
                        "team": team,
                        "nation": country,
                        "pos": pos,
                        "position_detail": pos_detail,
                        "jersey_number": jersey,
                        "Min": minutes,
                        "Gls": sum(
                            1
                            for e in player_events
                            if e.get("type", {}).get("name") == "Shot"
                            and e.get("shot", {}).get("outcome", {}).get("name") == "Goal"
                        ),
                        "Ast": sum(
                            1 for e in player_events if e.get("pass", {}).get("goal_assist")
                        ),
                        "xG": sum(
                            e.get("shot", {}).get("statsbomb_xg", 0) or 0
                            for e in player_events
                            if e.get("type", {}).get("name") == "Shot"
                        ),
                        "KP": sum(
                            1 for e in player_events if e.get("pass", {}).get("shot_assist")
                        ),
                        "PrgP": sum(
                            1 for e in player_events if e.get("pass", {}).get("progressive_pass")
                        ),
                        "PrgC": sum(
                            1 for e in player_events if e.get("carry", {}).get("progressive_carry")
                        ),
                        "Press": count_event("Pressure"),
                        "Tkl": count_event("Duel"),
                        "Int": count_event("Interception"),
                        "Sh": count_event("Shot"),
                    }
                )

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError(f"No StatsBomb rows for season {season}")
    agg = (
        raw.groupby(["player", "team", "nation", "pos", "position_detail"], as_index=False)
        .agg(
            jersey_number=("jersey_number", lambda s: int(s.mode().iloc[0]) if len(s.mode()) else np.nan),
            Min=("Min", "sum"),
            Gls=("Gls", "sum"),
            Ast=("Ast", "sum"),
            xG=("xG", "sum"),
            KP=("KP", "sum"),
            PrgP=("PrgP", "sum"),
            PrgC=("PrgC", "sum"),
            Press=("Press", "sum"),
            Tkl=("Tkl", "sum"),
            Int=("Int", "sum"),
            Sh=("Sh", "sum"),
        )
    )
    return build_feature_frame(agg, "statsbomb", season)


def scrape_statsbomb(seasons: Iterable[int]) -> pd.DataFrame:
    frames = [_statsbomb_player_season_stats(season) for season in seasons]
    return pd.concat(frames, ignore_index=True)


def generate_sample_data(n_players: int = 400, seed: int = 42) -> pd.DataFrame:
    """Synthetic dataset — CI only. Use --source worldcup for real players."""
    rng = np.random.default_rng(seed)
    positions = ["GK", "DF", "MF", "FW"]
    teams = [
        "Brazil", "France", "Germany", "Argentina", "Spain", "England",
        "Portugal", "Netherlands", "Belgium", "Croatia", "Morocco", "Japan",
    ]
    # Real 2022 World Cup players for recognizable names in CI smoke tests
    roster = [
        ("Lionel Messi", "Argentina", "ARG", 10, "FW", "Right Forward"),
        ("Kylian Mbappé", "France", "FRA", 10, "FW", "Left Forward"),
        ("Jude Bellingham", "England", "ENG", 22, "MF", "Central Midfield"),
        ("Luka Modrić", "Croatia", "CRO", 10, "MF", "Central Midfield"),
        ("Vinícius Júnior", "Brazil", "BRA", 20, "FW", "Left Wing"),
        ("Erling Haaland", "Norway", "NOR", 9, "FW", "Centre Forward"),
        ("Rodri", "Spain", "ESP", 16, "MF", "Defensive Midfield"),
        ("Virgil van Dijk", "Netherlands", "NED", 4, "DF", "Centre Back"),
        ("Emiliano Martínez", "Argentina", "ARG", 23, "GK", "Goalkeeper"),
        ("Harry Kane", "England", "ENG", 9, "FW", "Centre Forward"),
    ]

    profiles = {
        "GK": dict(goals_per90=0.0, xg_per90=0.0, pressures_per90=0.5, tackles_per90=0.1, pass_completion_pct=72),
        "DF": dict(goals_per90=0.05, xg_per90=0.08, pressures_per90=12, tackles_per90=2.5, pass_completion_pct=86),
        "MF": dict(goals_per90=0.15, xg_per90=0.18, pressures_per90=18, tackles_per90=2.0, pass_completion_pct=84),
        "FW": dict(goals_per90=0.55, xg_per90=0.52, pressures_per90=10, tackles_per90=0.6, pass_completion_pct=74),
    }

    rows = []
    for i in range(n_players):
        if i < len(roster):
            name, team, nation, jersey, pos, detail = roster[i]
        else:
            pos = rng.choice(positions, p=[0.08, 0.28, 0.38, 0.26])
            team = rng.choice(teams)
            nation = team
            name = f"Squad Player {i}"
            jersey = int(rng.integers(1, 24))
            detail = ""

        base = profiles[pos]
        minutes = int(rng.integers(180, 720))
        row = {col: max(0.0, rng.normal(0.5, 0.35)) for col in FEATURE_COLUMNS}
        for key, val in base.items():
            row[key] = max(0.0, val + rng.normal(0, val * 0.25))
        row.update(
            {
                "player": name,
                "team": team,
                "nation": nation,
                "position": pos,
                "position_detail": detail,
                "jersey_number": jersey,
                "season": 2022,
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
    for col in keep:
        if col not in df.columns:
            if col == "jersey_number":
                df[col] = np.nan
            else:
                df[col] = ""
    df[keep].to_parquet(path, index=False)
    print(f"Saved {len(df)} players to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape World Cup player stats")
    parser.add_argument(
        "--source",
        choices=["worldcup", "fbref", "statsbomb", "sample"],
        default="worldcup",
        help="Data source (default: worldcup = FBref stats + StatsBomb roster)",
    )
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022])
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--sample-size", type=int, default=400)
    args = parser.parse_args()

    if args.source == "worldcup":
        df = scrape_worldcup(args.seasons)
    elif args.source == "fbref":
        df = scrape_fbref(args.seasons)
    elif args.source == "statsbomb":
        df = scrape_statsbomb(args.seasons)
    else:
        df = generate_sample_data(n_players=args.sample_size)

    save_players(df, args.output)


if __name__ == "__main__":
    main()
