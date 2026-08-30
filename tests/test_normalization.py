"""Tests for data normalization and transformation logic."""

import math

import pandas as pd

from data.roster import _primary_position, normalize_name, normalize_team
from data.scraper import _flatten_columns, _per90, _safe_numeric


class TestNameNormalization:
    """Test player name normalization (ASCII fold for matching keys)."""

    def test_normalize_name_lowercase(self):
        assert normalize_name("Erling HAALAND") == "erling haaland"

    def test_normalize_name_extra_spaces(self):
        assert normalize_name("  João   SILVA  ") == "joao silva"

    def test_normalize_name_diacritics_stripped(self):
        result = normalize_name("José María")
        assert result == "jose maria"

    def test_normalize_name_empty(self):
        assert normalize_name("") == ""

    def test_normalize_name_single_word(self):
        assert normalize_name("Mbappé") == "mbappe"


class TestTeamNormalization:
    """Test team name normalization."""

    def test_normalize_team_standard(self):
        assert normalize_team("FRANCE") == "france"

    def test_normalize_team_with_spaces(self):
        assert normalize_team("South Africa") == "south africa"

    def test_normalize_team_special_chars(self):
        result = normalize_team("Côte d'Ivoire")
        assert result == "cote divoire"

    def test_normalize_team_extra_whitespace(self):
        assert normalize_team("  BRAZIL  ") == "brazil"


class TestPrimaryPosition:
    """Test position extraction from StatsBomb lineup stints."""

    def test_primary_position_single_position(self):
        positions = [{"position": "Centre Back"}]
        assert _primary_position(positions) == "Centre Back"

    def test_primary_position_multiple_positions(self):
        positions = [
            {"position": "Centre Back"},
            {"position": "Centre Back"},
            {"position": "Right Back"},
        ]
        assert _primary_position(positions) == "Centre Back"

    def test_primary_position_empty_list(self):
        assert _primary_position([]) == ""

    def test_primary_position_equal_counts(self):
        positions = [
            {"position": "Central Midfield"},
            {"position": "Attacking Midfield"},
        ]
        result = _primary_position(positions)
        assert result in ["Central Midfield", "Attacking Midfield"]


class TestPer90Calculation:
    """Test per-90 minute statistics calculation."""

    def test_per90_basic_calculation(self):
        value = pd.Series([10, 20, 30])
        minutes = pd.Series([450, 900, 1800])
        result = _per90(value, minutes)
        expected = pd.Series([2.0, 2.0, 1.5])
        pd.testing.assert_series_equal(result, expected)

    def test_per90_zero_minutes(self):
        value = pd.Series([10])
        minutes = pd.Series([0])
        result = _per90(value, minutes)
        assert math.isnan(result.iloc[0])

    def test_per90_preserves_index(self):
        value = pd.Series([5, 10], index=["A", "B"])
        minutes = pd.Series([450, 450], index=["A", "B"])
        result = _per90(value, minutes)
        assert list(result.index) == ["A", "B"]


class TestSafeNumeric:
    """Test safe numeric conversion."""

    def test_safe_numeric_valid_numbers(self):
        series = pd.Series([1.0, 2.5, 3.0])
        result = _safe_numeric(series)
        pd.testing.assert_series_equal(result, series)

    def test_safe_numeric_strings_to_float(self):
        series = pd.Series(["1.5", "2.0", "3.5"])
        result = _safe_numeric(series)
        expected = pd.Series([1.5, 2.0, 3.5])
        pd.testing.assert_series_equal(result, expected)

    def test_safe_numeric_invalid_strings(self):
        series = pd.Series(["1.5", "invalid", "2.0"])
        result = _safe_numeric(series)
        assert result.iloc[1] == 0.0

    def test_safe_numeric_custom_default(self):
        series = pd.Series(["invalid"])
        result = _safe_numeric(series, default=-1.0)
        assert result.iloc[0] == -1.0

    def test_safe_numeric_nans(self):
        series = pd.Series([1.0, float("nan"), 3.0])
        result = _safe_numeric(series)
        assert result.iloc[1] == 0.0


class TestFlattenColumns:
    """Test DataFrame column flattening."""

    def test_flatten_columns_multiindex(self):
        df = pd.DataFrame(
            [[1, 2, 3]],
            columns=pd.MultiIndex.from_tuples([("A", "x"), ("A", "y"), ("B", "z")]),
        )
        result = _flatten_columns(df)
        assert "A_x" in result.columns

    def test_flatten_columns_single_level(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        result = _flatten_columns(df)
        assert list(result.columns) == ["A", "B"]

    def test_flatten_columns_preserves_data(self):
        df = pd.DataFrame(
            [[1, 2]],
            columns=pd.MultiIndex.from_tuples([("X", "a"), ("Y", "b")]),
        )
        result = _flatten_columns(df)
        assert result.values.tolist() == [[1, 2]]
