"""Tests for team diagnostics dataclasses (see team/diagnose.py, team/fingerprint.py)."""

import numpy as np

from team.diagnose import (
    AdjustmentCard,
    FixtureBrief,
    StructuralGap,
    StyleGap,
    WildcardPick,
)
from team.fingerprint import PlayerRecord, TeamFingerprint


class TestStyleGap:
    def test_style_gap_creation(self):
        gap = StyleGap(
            dimension="press_intensity",
            team_a_value=7.5,
            team_b_value=4.2,
            delta_pct=44.0,
            description="Team A presses more",
        )
        assert gap.dimension == "press_intensity"
        assert gap.delta_pct == 44.0

    def test_style_gap_positive_delta(self):
        gap = StyleGap("progression", 10.0, 5.0, 50.0, "A leads")
        assert gap.delta_pct > 0

    def test_style_gap_negative_delta(self):
        gap = StyleGap("finishing", 2.0, 8.0, -75.0, "B leads")
        assert gap.delta_pct < 0


class TestStructuralGap:
    def test_structural_gap_creation(self):
        gap = StructuralGap(
            archetype="deep_progressor",
            opponent_share=0.35,
            your_share=0.10,
            description="Germany has more progressors",
        )
        assert gap.archetype == "deep_progressor"
        assert gap.opponent_share > gap.your_share

    def test_structural_gap_shares_bounded(self):
        gap = StructuralGap("high_presser", 0.25, 0.15, "Press gap")
        assert 0 <= gap.your_share <= 1
        assert 0 <= gap.opponent_share <= 1


class TestAdjustmentCard:
    def test_adjustment_card_creation(self):
        card = AdjustmentCard(
            category="lineup",
            title="Fill progressor gap",
            detail="Start a second deep-lying midfielder",
            suggested_players=["Rodri"],
        )
        assert card.category == "lineup"
        assert "Rodri" in card.suggested_players

    def test_adjustment_card_default_players(self):
        card = AdjustmentCard(category="style_lever", title="Press higher", detail="...")
        assert card.suggested_players == []


class TestWildcardPick:
    def test_wildcard_pick_creation(self):
        pick = WildcardPick(
            player="Eduardo Camavinga",
            position="MF",
            archetype="deep_progressor",
            fills_gap="deep_progressor",
            similarity_to_opponent=0.82,
        )
        assert pick.player == "Eduardo Camavinga"
        assert pick.similarity_to_opponent == 0.82

    def test_wildcard_pick_gap_field(self):
        pick = WildcardPick("Test", "ST", "finisher", "finisher", 0.7)
        assert pick.fills_gap == "finisher"


class TestFixtureBrief:
    def test_fixture_brief_creation(self):
        brief = FixtureBrief(
            team_a="France",
            team_b="Germany",
            team_a_fingerprint={"style_dna": {"press_intensity": 7.5}},
            team_b_fingerprint={"style_dna": {"press_intensity": 6.2}},
            style_clashes=[],
            structural_gaps=[],
            exploit_vectors=["aerial duels"],
            adjustments=[],
            wildcard_picks=[],
            summary="Even match.",
        )
        assert brief.team_a == "France"

    def test_fixture_brief_to_dict(self):
        brief = FixtureBrief(
            team_a="Spain",
            team_b="Italy",
            team_a_fingerprint={},
            team_b_fingerprint={},
            style_clashes=[],
            structural_gaps=[],
            exploit_vectors=[],
            adjustments=[],
            wildcard_picks=[],
            summary="Tactical test.",
        )
        result = brief.to_dict()
        assert result["team_a"] == "Spain"


class TestTeamFingerprint:
    def test_fingerprint_creation(self):
        fp = TeamFingerprint(
            team="Brazil",
            squad_size=23,
            total_minutes=8000.0,
            embedding_centroid=[0.1] * 32,
            style_dna={"press_intensity": 6.5},
            archetype_mix={"finisher": 0.4},
            top_players=[{"player": "Neymar", "position": "FW"}],
        )
        assert fp.team == "Brazil"
        assert len(fp.embedding_centroid) == 32

    def test_fingerprint_style_dna_numeric(self):
        fp = TeamFingerprint(
            team="Brazil",
            squad_size=23,
            total_minutes=8000.0,
            embedding_centroid=[0.0] * 32,
            style_dna={"press_intensity": 7.1, "progression": 6.8},
            archetype_mix={},
        )
        for value in fp.style_dna.values():
            assert isinstance(value, int | float)

    def test_player_record_fields(self):
        rec = PlayerRecord(
            id=1,
            player="Test",
            team="Brazil",
            nation="Brazil",
            position="FW",
            position_detail="Centre Forward",
            jersey_number=9,
            minutes=450.0,
            embedding=np.zeros(32),
            stats={"goals_per90": 0.5},
        )
        assert rec.jersey_number == 9


class TestTeamDiagnosticsIntegration:
    def test_fixture_brief_has_all_components(self):
        brief = FixtureBrief(
            team_a="Team A",
            team_b="Team B",
            team_a_fingerprint={},
            team_b_fingerprint={},
            style_clashes=[{"dimension": "press"}],
            structural_gaps=[{"archetype": "finisher"}],
            exploit_vectors=["vector1"],
            adjustments=[{"title": "Press"}],
            wildcard_picks=[{"player": "X"}],
            summary="Summary",
        )
        assert brief.style_clashes and brief.structural_gaps

    def test_fingerprint_quantifiable_comparison(self):
        fp1 = TeamFingerprint("T1", 23, 8000.0, [0.0] * 32, {"press_intensity": 7.0}, {})
        fp2 = TeamFingerprint("T2", 23, 8000.0, [0.0] * 32, {"press_intensity": 5.0}, {})
        gap = abs(fp1.style_dna["press_intensity"] - fp2.style_dna["press_intensity"])
        assert gap == 2.0
