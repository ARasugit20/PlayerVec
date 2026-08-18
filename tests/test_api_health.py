"""Tests for API health and basic endpoint smoke tests."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from api.main import app


@pytest.fixture
def client():
    """Provide FastAPI test client."""
    return TestClient(app)


class TestAPIHealth:
    """Test API health check."""

    def test_health_endpoint_returns_ok(self, client):
        """Health endpoint returns ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_endpoint_json_response(self, client):
        """Health endpoint returns JSON."""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"


class TestPlayersEndpoint:
    """Test players list endpoint."""

    @patch("api.main.get_search")
    def test_list_players_returns_list(self, mock_get_search, client):
        """Players endpoint returns list."""
        mock_search = MagicMock()
        mock_search.all_players.return_value = ["Haaland", "Kane", "Lewandowski"]
        mock_get_search.return_value = mock_search
        
        response = client.get("/players")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @patch("api.main.get_search")
    def test_list_players_empty(self, mock_get_search, client):
        """Players endpoint handles empty list."""
        mock_search = MagicMock()
        mock_search.all_players.return_value = []
        mock_get_search.return_value = mock_search
        
        response = client.get("/players")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    @patch("api.main.get_search")
    def test_list_players_with_query(self, mock_get_search, client):
        """Players endpoint filters by query."""
        mock_search = MagicMock()
        mock_search.all_players.return_value = ["Haaland", "Kane"]
        mock_get_search.return_value = mock_search
        
        response = client.get("/players?q=Haal")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @patch("api.main.get_search")
    def test_list_players_respects_limit(self, mock_get_search, client):
        """Players endpoint respects 50-player limit."""
        mock_search = MagicMock()
        many_players = [f"Player{i}" for i in range(100)]
        mock_search.all_players.return_value = many_players
        mock_get_search.return_value = mock_search
        
        response = client.get("/players")
        data = response.json()
        assert len(data) <= 50


class TestSimilarPlayersEndpoint:
    """Test similar players endpoint."""

    @patch("api.main.get_search")
    @patch("api.main.get_stats_lookup")
    def test_similar_returns_response_model(self, mock_stats, mock_get_search, client):
        """Similar endpoint returns proper response."""
        mock_search = MagicMock()
        mock_search.similar.return_value = [
            {
                "player": "Kane",
                "team": "Tottenham",
                "nation": "England",
                "position": "ST",
                "position_detail": None,
                "jersey_number": None,
                "season": 2022,
                "similarity": 0.95,
            }
        ]
        mock_get_search.return_value = mock_search
        mock_stats.return_value = {}
        
        response = client.get("/similar?player=Haaland&k=5")
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert data["query"] == "Haaland"

    @patch("api.main.get_search")
    def test_similar_player_not_found(self, mock_get_search, client):
        """Similar endpoint handles missing player."""
        mock_search = MagicMock()
        mock_search.similar.side_effect = KeyError("Player not found")
        mock_get_search.return_value = mock_search
        
        response = client.get("/similar?player=NonexistentPlayer&k=5")
        assert response.status_code == 404

    @patch("api.main.get_search")
    @patch("api.main.get_stats_lookup")
    def test_similar_k_parameter(self, mock_stats, mock_get_search, client):
        """Similar endpoint respects k parameter."""
        mock_search = MagicMock()
        mock_search.similar.return_value = [
            {
                "player": f"Player{i}",
                "team": "Team",
                "nation": "Nation",
                "position": "ST",
                "position_detail": None,
                "jersey_number": None,
                "season": 2022,
                "similarity": 0.9 - (i * 0.01),
            }
            for i in range(3)
        ]
        mock_get_search.return_value = mock_search
        mock_stats.return_value = {}
        
        response = client.get("/similar?player=Haaland&k=3")
        mock_search.similar.assert_called_with("Haaland", k=3)


class TestTeamsEndpoint:
    """Test teams list endpoint."""

    @patch("api.main.get_fingerprint_engine")
    def test_list_teams_returns_list(self, mock_get_fp, client):
        """Teams endpoint returns list."""
        mock_fp = MagicMock()
        mock_fp.all_teams.return_value = ["Brazil", "France", "Germany"]
        mock_get_fp.return_value = mock_fp
        
        response = client.get("/teams")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @patch("api.main.get_fingerprint_engine")
    def test_list_teams_with_query(self, mock_get_fp, client):
        """Teams endpoint filters by query."""
        mock_fp = MagicMock()
        mock_fp.all_teams.return_value = ["Brazil", "France"]
        mock_get_fp.return_value = mock_fp
        
        response = client.get("/teams?q=Fra")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestTeamFingerprintEndpoint:
    """Test team fingerprint endpoint."""

    @patch("api.main.get_fingerprint_engine")
    def test_team_fingerprint_returns_response(self, mock_get_fp, client):
        """Team fingerprint endpoint returns proper response."""
        mock_fp_obj = MagicMock()
        mock_fp_obj.team = "Brazil"
        mock_fp_obj.squad_size = 23
        mock_fp_obj.total_minutes = 8000
        mock_fp_obj.style_dna = {"press": 6.5, "progression": 7.0}
        mock_fp_obj.archetype_mix = {"finisher": 0.4}
        mock_fp_obj.top_players = [{"player": "Neymar"}]
        
        mock_fp = MagicMock()
        mock_fp.fingerprint.return_value = mock_fp_obj
        mock_get_fp.return_value = mock_fp
        
        response = client.get("/team-fingerprint?team=Brazil")
        assert response.status_code == 200
        data = response.json()
        assert data["team"] == "Brazil"

    @patch("api.main.get_fingerprint_engine")
    def test_team_fingerprint_not_found(self, mock_get_fp, client):
        """Team fingerprint endpoint handles missing team."""
        mock_fp = MagicMock()
        mock_fp.fingerprint.side_effect = KeyError("Team not found")
        mock_get_fp.return_value = mock_fp
        
        response = client.get("/team-fingerprint?team=NonexistentTeam")
        assert response.status_code == 404


class TestFixtureBriefEndpoint:
    """Test fixture brief endpoint."""

    @patch("api.main.get_diagnostician")
    def test_fixture_brief_returns_response(self, mock_get_diag, client):
        """Fixture brief endpoint returns proper response."""
        mock_brief = MagicMock()
        mock_brief.to_dict.return_value = {
            "team_a": "France",
            "team_b": "Germany",
            "team_a_fingerprint": {"press": 6.5},
            "team_b_fingerprint": {"press": 6.2},
            "style_clashes": [],
            "structural_gaps": [],
            "exploit_vectors": [],
            "adjustments": [],
            "wildcard_picks": [],
            "summary": "Even match",
        }
        
        mock_diag = MagicMock()
        mock_diag.diagnose.return_value = mock_brief
        mock_get_diag.return_value = mock_diag
        
        response = client.get("/fixture-brief?team_a=France&team_b=Germany")
        assert response.status_code == 200
        data = response.json()
        assert data["team_a"] == "France"

    @patch("api.main.get_diagnostician")
    def test_fixture_brief_missing_team(self, mock_get_diag, client):
        """Fixture brief handles missing team."""
        mock_diag = MagicMock()
        mock_diag.diagnose.side_effect = KeyError("Team not found")
        mock_get_diag.return_value = mock_diag
        
        response = client.get("/fixture-brief?team_a=Unknown&team_b=Germany")
        assert response.status_code == 404
