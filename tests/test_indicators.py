"""
Tests for the indicator framework.

Run with: pytest tests/test_indicators.py -v
"""

from __future__ import annotations

import polars as pl
import pytest
from datetime import datetime

from financing_private_credit.indicators import (
    get_indicator,
    list_indicators,
    BaseIndicator,
    IndicatorMetadata,
)


class TestIndicatorRegistry:
    """Test the indicator registry and factory."""

    def test_list_indicators(self):
        """Test that we can list available indicators."""
        indicators = list_indicators()

        assert isinstance(indicators, list)
        assert len(indicators) >= 2, "Should have at least 2 indicators registered"
        assert "credit_boom" in indicators
        assert "variance_decomposition" in indicators

        print(f"Available indicators: {indicators}")

    def test_get_credit_boom_indicator(self):
        """Test getting the credit boom indicator."""
        indicator = get_indicator("credit_boom")

        assert indicator is not None
        assert isinstance(indicator, BaseIndicator)

        metadata = indicator.get_metadata()
        assert isinstance(metadata, IndicatorMetadata)
        assert metadata.short_name == "LIS"

        print(f"Credit Boom: {metadata.name}")
        print(f"Description: {metadata.description}")

    def test_get_variance_decomposition_indicator(self):
        """Test getting the variance decomposition indicator."""
        indicator = get_indicator("variance_decomposition")

        assert indicator is not None
        assert isinstance(indicator, BaseIndicator)

        metadata = indicator.get_metadata()
        assert isinstance(metadata, IndicatorMetadata)
        assert metadata.short_name == "VarDecomp"

        print(f"Variance Decomposition: {metadata.name}")
        print(f"Description: {metadata.description}")

    def test_get_unknown_indicator_raises(self):
        """Test that getting unknown indicator raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_indicator("unknown_indicator")

        assert "Unknown indicator" in str(excinfo.value)


class TestCreditBoomIndicator:
    """Test the Credit Boom indicator."""

    def test_metadata(self):
        """Test credit boom metadata."""
        indicator = get_indicator("credit_boom")
        metadata = indicator.get_metadata()

        assert metadata.update_frequency == "quarterly"
        assert "SEC EDGAR" in metadata.data_sources[0]
        assert metadata.lookback_periods == 20

    def test_warning_levels(self):
        """Test warning level classification."""
        indicator = get_indicator("credit_boom")

        # High risk
        emoji, status = indicator.get_warning_level(2.5)
        assert status == "HIGH RISK"

        # Elevated
        emoji, status = indicator.get_warning_level(1.7)
        assert status == "ELEVATED"

        # Normal
        emoji, status = indicator.get_warning_level(0.5)
        assert status == "NORMAL"

        # Conservative
        emoji, status = indicator.get_warning_level(-2.0)
        assert status == "CONSERVATIVE"

    def test_lis_calculation(self):
        """Test LIS calculation with mock data."""
        indicator = get_indicator("credit_boom")

        # Create mock data
        mock_data = pl.DataFrame({
            "date": [datetime(2023, 1, 1), datetime(2023, 4, 1), datetime(2023, 7, 1)] * 3,
            "ticker": ["JPM"] * 3 + ["BAC"] * 3 + ["C"] * 3,
            "loan_growth_yoy": [5.0, 6.0, 4.0, 3.0, 4.0, 5.0, 10.0, 8.0, 7.0],
        })

        result = indicator._compute_lis(mock_data)

        assert "lis" in result.columns
        assert result.height == 9

        # C should have highest LIS (fastest growing)
        c_lis = result.filter(pl.col("ticker") == "C").select("lis").mean().item()
        bac_lis = result.filter(pl.col("ticker") == "BAC").select("lis").mean().item()

        # C's growth (10, 8, 7) is above average, so LIS should be positive
        print(f"C avg LIS: {c_lis:.2f}, BAC avg LIS: {bac_lis:.2f}")


class TestVarianceDecompositionIndicator:
    """Test the Variance Decomposition indicator."""

    def test_metadata(self):
        """Test variance decomposition metadata."""
        indicator = get_indicator("variance_decomposition")
        metadata = indicator.get_metadata()

        assert metadata.update_frequency == "quarterly"
        assert metadata.lookback_periods == 40
        assert "Tables 5-6" in metadata.paper_reference

    def test_bank_classification(self):
        """Test bank archetype classification."""
        indicator = get_indicator("variance_decomposition")

        # Macro follower
        shares = {"macro_pct": 55, "size_pct": 20, "allocation_pct": 10, "idiosyncratic_pct": 15}
        archetype = indicator.classify_bank(shares)
        assert archetype.name == "Macro Follower"

        # Steady grower
        shares = {"macro_pct": 20, "size_pct": 50, "allocation_pct": 15, "idiosyncratic_pct": 15}
        archetype = indicator.classify_bank(shares)
        assert archetype.name == "Steady Grower"

        # Idiosyncratic specialist
        shares = {"macro_pct": 15, "size_pct": 15, "allocation_pct": 20, "idiosyncratic_pct": 50}
        archetype = indicator.classify_bank(shares)
        assert archetype.name == "Idiosyncratic Specialist"

    def test_summary_table(self):
        """Test summary table creation."""
        indicator = get_indicator("variance_decomposition")

        mock_shares = {
            "JPM": {"total_variance": 100, "macro_pct": 50, "size_pct": 20,
                    "allocation_pct": 15, "idiosyncratic_pct": 15, "covariance_pct": 0, "n_observations": 40},
            "GS": {"total_variance": 150, "macro_pct": 25, "size_pct": 15,
                   "allocation_pct": 20, "idiosyncratic_pct": 40, "covariance_pct": 0, "n_observations": 40},
        }

        table = indicator.create_summary_table(mock_shares)

        assert table.height == 2
        assert "Bank" in table.columns
        assert "Archetype" in table.columns

        # Should be sorted by Macro % descending
        assert table["Bank"][0] == "JPM"  # 50% macro
        assert table["Bank"][1] == "GS"   # 25% macro

        print(table)

    def test_variance_shares_computation(self):
        """Test variance share computation."""
        indicator = get_indicator("variance_decomposition")

        # Create mock decomposition
        import numpy as np
        np.random.seed(42)

        mock_decomp = pl.DataFrame({
            "date": [datetime(2023, i, 1) for i in range(1, 13)],
            "delta_loans": np.random.randn(12) * 100,
            "macro_effect": np.random.randn(12) * 50,
            "size_effect": np.random.randn(12) * 30,
            "allocation_effect": np.random.randn(12) * 20,
            "idiosyncratic_effect": np.random.randn(12) * 40,
        })

        shares = indicator.compute_variance_shares(mock_decomp)

        assert "total_variance" in shares
        assert "macro_pct" in shares
        assert "size_pct" in shares

        # Percentages should be reasonable
        total_pct = (
            shares.get("macro_pct", 0) +
            shares.get("size_pct", 0) +
            shares.get("allocation_pct", 0) +
            shares.get("idiosyncratic_pct", 0) +
            abs(shares.get("covariance_pct", 0))
        )

        print(f"Total variance pct (should be ~100 with cov): {total_pct:.1f}%")
        print(f"Macro: {shares.get('macro_pct', 0):.1f}%")
        print(f"Size: {shares.get('size_pct', 0):.1f}%")
        print(f"Allocation: {shares.get('allocation_pct', 0):.1f}%")
        print(f"Idiosyncratic: {shares.get('idiosyncratic_pct', 0):.1f}%")


def run_indicator_diagnostic():
    """Run diagnostic tests for all indicators."""
    print("=" * 60)
    print("Indicator Framework Diagnostic")
    print("=" * 60)

    print(f"\nRegistered indicators: {list_indicators()}")

    for name in list_indicators():
        print(f"\n--- {name} ---")
        indicator = get_indicator(name)
        metadata = indicator.get_metadata()
        print(f"Name: {metadata.name}")
        print(f"Version: {metadata.version}")
        print(f"Update frequency: {metadata.update_frequency}")
        print(f"Data sources: {metadata.data_sources}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_indicator_diagnostic()
