"""Shared feature definitions for PlayerVec."""

from __future__ import annotations

FEATURE_COLUMNS = [
    "goals_per90",
    "xg_per90",
    "assists_per90",
    "xa_per90",
    "key_passes_per90",
    "progressive_passes_per90",
    "progressive_carries_per90",
    "dribbles_per90",
    "pressures_per90",
    "tackles_per90",
    "interceptions_per90",
    "blocks_per90",
    "aerials_won_per90",
    "pass_completion_pct",
    "touches_final_third_per90",
    "shot_creating_actions_per90",
    "goal_creating_actions_per90",
    "shots_per90",
    "shots_on_target_per90",
    "passes_into_penalty_area_per90",
    "crosses_per90",
    "fouls_per90",
    "fouled_per90",
    "yellow_cards_per90",
    "dispossessed_per90",
    "miscontrols_per90",
    "passes_received_per90",
    "progressive_passes_received_per90",
    "carries_into_penalty_area_per90",
    "minutes_per_match",
]

METADATA_COLUMNS = [
    "player",
    "team",
    "nation",
    "position",
    "position_detail",
    "jersey_number",
    "season",
    "source",
    "minutes_played",
]

INPUT_DIM = len(FEATURE_COLUMNS)
EMBEDDING_DIM = 32
