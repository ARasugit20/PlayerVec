"""Tests for team diagnostics logic."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from team.diagnose import StyleGap, StructuralGap, AdjustmentCard, WildcardPick, FixtureBrief
from team.fingerprint import TeamFingerprint, PlayerRecord


class TestStyleGap:
    """Test style gap data structure."""

    def test_style_gap_creation(self):
        """StyleGap can be instantiated."""
        gap = StyleGap(dimension="press", team_a_value=7.5, team_b_value=4.2, gap=3.3)
        assert gap.dimension == "press"
        assert gap.team_a_value == 7.5
        assert gap.team_b_value == 4.2
        assert gap.gap == 3.3

    def test_style_gap_gap_positive(self):
        """Gap is positive when team_a > team_b."""
        gap = StyleGap(dimension="progression", team_a_value=10, team_b_value=5, gap=5)
        assert gap.gap > 0

    def test_style_gap_gap_negative(self):
        """Gap can be negative."""
        gap = StyleGap(dimension="finishing", team_a_value=2.0, team_b_value=8.0, gap=-6.0)
        assert gap.gap < 0


class TestStructuralGap:
    """Test structural gap data structure."""

    def test_structural_gap_creation(self):
        """StructuralGap can be instantiated."""
        gap = StructuralGap(description="Germany lacks aerial presence", severity="high")
        assert gap.description == "Germany lacks aerial presence"
        assert gap.severity == "high"

    def test_structural_gap_severity_levels(self):
        """Severity levels are valid."""
        for severity in ["low", "medium", "high"]:
            gap = StructuralGap(description="Test", severity=severity)
            assert gap.severity == severity


class TestAdjustmentCard:
    """Test adjustment card data structure."""

    def test_adjustment_card_creation(self):
        """AdjustmentCard can be instantiated."""
        card = AdjustmentCard(
            player="Antonio Rudiger",
            position="CB",
            recommendation="Start for physical presence",
            rationale="High pressure environment",
        )
        assert card.player == "Antonio Rudiger"
        assert card.position == "CB"

    def test_adjustment_card_rationale(self):
        """Adjustment cards have clear rationale."""
        card = AdjustmentCard(
            player="N'Golo Kanté",
            position="CM",
            recommendation="Bench rotation",
            rationale="Recovery week after heavy schedule",
        )
        assert len(card.rationale) > 0


class TestWildcardPick:
    """Test wildcard pick data structure."""

    def test_wildcard_pick_creation(self):
        """WildcardPick can be instantiated."""
        pick = WildcardPick(
            player="Eduardo Camavinga",
            position="CM",
            case="Younger legs to counter press",
            gap_exploited="Aging midfield",
        )
        assert pick.player == "Eduardo Camavinga"
        assert pick.position == "CM"

    def test_wildcard_pick_gap_exploitation(self):
        """Wildcard picks target specific gaps."""
        pick = WildcardPick(
            player="Test Player",
            position="ST",
            case="Clinical finishing",
            gap_exploited="Defensive vulnerability",
        )
        assert len(pick.gap_exploited) > 0


class TestFixtureBrief:
    """Test fixture brief data structure."""

    def test_fixture_brief_creation(self):
        """FixtureBrief can be instantiated."""
        brief = FixtureBrief(
            team_a="France",
            team_b="Germany",
            team_a_fingerprint={"press": 7.5, "progression": 6.8},
            team_b_fingerprint={"press": 6.2, "progression": 7.1},
            style_clashes=[],
            structural_gaps=[],
            exploit_vectors=["aerial duels"],
            adjustments=[],
            wildcard_picks=[],
            summary="Even match, small advantages.",
        )
        assert brief.team_a == "France"
        assert brief.team_b == "Germany"

    def test_fixture_brief_to_dict(self):
        """FixtureBrief.to_dict() returns dict."""
        brief = FixtureBrief(
            team_a="Spain",
            team_b="Italy",
            team_a_fingerprint={"press": 6.0},
            team_b_fingerprint={"press": 6.5},
            style_clashes=[],
            structural_gaps=[],
            exploit_vectors=[],
            adjustments=[],
            wildcard_picks=[],
            summary="Tactical test.",
        )
        result = brief.to_dict()
        assert isinstance(result, dict)
        assert result["team_a"] == "Spain"
        assert result["team_b"] == "Italy"


class TestTeamFingerprint:
    """Test team fingerprint data structure."""

    def test_fingerprint_creation(self):
        """TeamFingerprint can be instantiated."""
        players = [
            PlayerRecord(
                player="Test Player",
                team="Brazil",
                position="ST",
                minutes=450,
                stats={"goals": 10},
            )
        ]
        fp = TeamFingerprint(
            team="Brazil",
            squad_size=23,
            total_minutes=8000,
            players=players,
            style_dna={"press": 6.5, "progression": 7.2},
            archetype_mix={"finisher": 0.4, "presser": 0.3},
            top_players=[{"player": "Neymar", "role": "CAM"}],
        )
        assert fp.team == "Brazil"
        assert fp.squad_size == 23

    def test_fingerprint_style_dna_values(self):
        """Style DNA values are numeric."""
        fp = TeamFingerprint(
            team="Brazil",
            squad_size=23,
            total_minutes=8000,
            players=[],
            style_dna={"press": 7.1, "progression": 6.8, "finishing": 7.5},
            archetype_mix={},
            top_players=[],
        )
        for dimension, value in fp.style_dna.items():
            assert isinstance(value, (int, float))
            assert 0 <= value <= 10

    def test_fingerprint_to_dict(self):
        """TeamFingerprint.to_dict() returns dict."""
        fp = TeamFingerprint(
            team="Argentina",
            squad_size=23,
            total_minutes=8000,
            players=[],
            style_dna={"press": 6.0},
            archetype_mix={"midfielder": 0.5},
            top_players=[],
        )
        result = fp.to_dict()
        assert isinstance(result, dict)
        assert result["team"] == "Argentina"


class TestTeamDiagnosticsIntegration:
    """Integration tests for team diagnostics."""

    def test_fixture_brief_has_all_components(self):
        """Fixture brief contains all required components."""
        brief = FixtureBrief(
            team_a="Team A",
            team_b="Team B",
            team_a_fingerprint={"stat": 5.0},
            team_b_fingerprint={"stat": 5.0},
            style_clashes=[StyleGap("dimension", 5.0, 5.0, 0.0)],
            structural_gaps=[StructuralGap("gap", "high")],
            exploit_vectors=["vector1"],
            adjustments=[AdjustmentCard("Player", "Position", "Rec", "Rationale")],
            wildcard_picks=[WildcardPick("Player", "Position", "Case", "Gap")],
            summary="Summary",
        )
        assert brief.style_clashes
        assert brief.structural_gaps
        assert brief.exploit_vectors
        assert brief.adjustments
        assert brief.wildcard_picks

    def test_fingerprint_quantifiable_comparison(self):
        """Fingerprints can be compared quantitatively."""
        fp1 = TeamFingerprint(
            team="Team1",
            squad_size=23,
            total_minutes=8000,
            players=[],
            style_dna={"press": 7.0, "progression": 6.0},
            archetype_mix={},
            top_players=[],
        )
        fp2 = TeamFingerprint(
            team="Team2",
            squad_size=23,
            total_minutes=8000,
            players=[],
            style_dna={"press": 5.0, "progression": 6.0},
            archetype_mix={},
            top_players=[],
        )
        gap = abs(fp1.style_dna["press"] - fp2.style_dna["press"])
        assert gap == 2.0
