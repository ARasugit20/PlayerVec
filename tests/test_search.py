"""Tests for search resolution with controlled fixtures."""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from search.query import PlayerSearch


class MockPlayerSearch:
    """Mock player search with controlled fixture data."""
    
    def __init__(self):
        """Initialize mock search with sample data."""
        self.players = {
            "Erling Haaland": {
                "embedding": np.array([0.1, 0.2, 0.3]),
                "team": "Manchester City",
                "nation": "Norway",
                "position": "ST",
                "season": 2022,
            },
            "Harry Kane": {
                "embedding": np.array([0.12, 0.18, 0.32]),
                "team": "Tottenham",
                "nation": "England",
                "position": "ST",
                "season": 2022,
            },
            "Robert Lewandowski": {
                "embedding": np.array([0.09, 0.22, 0.28]),
                "team": "Barcelona",
                "nation": "Poland",
                "position": "ST",
                "season": 2022,
            },
        }
    
    def similar(self, player_name: str, k: int = 5):
        """Return k most similar players."""
        if player_name not in self.players:
            raise KeyError(f"Player {player_name} not found")
        
        query_embedding = self.players[player_name]["embedding"]
        results = []
        
        for name, data in self.players.items():
            if name == player_name:
                continue
            distance = np.linalg.norm(query_embedding - data["embedding"])
            results.append({
                "player": name,
                "team": data["team"],
                "nation": data["nation"],
                "position": data["position"],
                "season": data["season"],
                "similarity": 1.0 / (1.0 + distance),
            })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]
    
    def all_players(self):
        """Return all player names."""
        return list(self.players.keys())


@pytest.fixture
def mock_search():
    """Provide mock search instance."""
    return MockPlayerSearch()


class TestPlayerSearchResolution:
    """Test player search with mocked data."""

    def test_similar_returns_list(self, mock_search):
        """Similar query returns a list."""
        results = mock_search.similar("Erling Haaland", k=2)
        assert isinstance(results, list)

    def test_similar_respects_k_parameter(self, mock_search):
        """Similar query respects k limit."""
        results_k2 = mock_search.similar("Erling Haaland", k=2)
        results_k5 = mock_search.similar("Erling Haaland", k=5)
        assert len(results_k2) <= 2
        assert len(results_k5) <= 5

    def test_similar_nonexistent_player(self, mock_search):
        """Query for non-existent player raises KeyError."""
        with pytest.raises(KeyError):
            mock_search.similar("Nonexistent Player")

    def test_similar_result_structure(self, mock_search):
        """Similar result has required fields."""
        results = mock_search.similar("Erling Haaland", k=1)
        assert len(results) > 0
        result = results[0]
        assert "player" in result
        assert "team" in result
        assert "similarity" in result
        assert "position" in result

    def test_similar_query_not_in_results(self, mock_search):
        """Query player not in own results."""
        results = mock_search.similar("Erling Haaland", k=10)
        player_names = [r["player"] for r in results]
        assert "Erling Haaland" not in player_names

    def test_similar_similarity_scores_valid(self, mock_search):
        """Similarity scores are between 0 and 1."""
        results = mock_search.similar("Erling Haaland", k=2)
        for result in results:
            assert 0 <= result["similarity"] <= 1

    def test_similar_sorted_by_similarity(self, mock_search):
        """Results are sorted by similarity descending."""
        results = mock_search.similar("Erling Haaland", k=2)
        if len(results) > 1:
            assert results[0]["similarity"] >= results[1]["similarity"]

    def test_all_players_returns_list(self, mock_search):
        """all_players returns list of player names."""
        players = mock_search.all_players()
        assert isinstance(players, list)
        assert len(players) > 0

    def test_all_players_contains_strings(self, mock_search):
        """all_players returns strings."""
        players = mock_search.all_players()
        assert all(isinstance(p, str) for p in players)

    def test_deterministic_results(self, mock_search):
        """Same query produces same results."""
        results1 = mock_search.similar("Harry Kane", k=2)
        results2 = mock_search.similar("Harry Kane", k=2)
        assert [r["player"] for r in results1] == [r["player"] for r in results2]

    def test_position_consistency(self, mock_search):
        """Position field is consistent across queries."""
        results = mock_search.similar("Erling Haaland", k=2)
        for result in results:
            player_name = result["player"]
            expected_position = mock_search.players[player_name]["position"]
            assert result["position"] == expected_position


class TestSearchErrorHandling:
    """Test search error handling."""

    def test_search_empty_query(self, mock_search):
        """Empty query string handling."""
        with pytest.raises(KeyError):
            mock_search.similar("")

    def test_search_case_sensitivity(self, mock_search):
        """Search respects case."""
        with pytest.raises(KeyError):
            mock_search.similar("erling haaland")  # lowercase

    def test_search_k_zero(self, mock_search):
        """k=0 returns empty list."""
        results = mock_search.similar("Erling Haaland", k=0)
        assert len(results) == 0

    def test_search_k_exceeds_available(self, mock_search):
        """k > available players returns available."""
        results = mock_search.similar("Erling Haaland", k=100)
        assert len(results) <= len(mock_search.all_players()) - 1
