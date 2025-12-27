"""
Tests for FRED data fetching and caching.

Run with: pytest tests/test_data_fetching.py -v
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from financing_private_credit.data import FREDDataFetcher, PRIVATE_CREDIT_SERIES
from financing_private_credit.cache import FREDCache


class TestFREDDataFetcher:
    """Test the FRED data fetcher."""

    def test_fetch_single_series(self):
        """Test fetching a single FRED series."""
        fetcher = FREDDataFetcher()

        # Fetch weekly bank credit data
        df = fetcher.fetch_series(
            "TOTLL",
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # Check structure
        assert "date" in df.columns, "DataFrame should have 'date' column"
        assert "value" in df.columns, "DataFrame should have 'value' column"

        # Check data exists
        assert df.height > 0, "Should have some data rows"

        # Check date range
        min_date = df["date"].min()
        max_date = df["date"].max()
        assert min_date >= datetime(2023, 1, 1), f"Min date {min_date} should be >= 2023-01-01"

        print(f"Fetched {df.height} rows from {min_date} to {max_date}")

    def test_fetch_multiple_series(self):
        """Test fetching multiple FRED series."""
        fetcher = FREDDataFetcher()

        series_ids = ["TOTLL", "BUSLOANS", "CONSUMER"]
        df = fetcher.fetch_multiple_series(
            series_ids,
            start_date="2023-01-01",
        )

        # Check structure
        assert "date" in df.columns, "DataFrame should have 'date' column"

        # Check all series are present
        for series_id in series_ids:
            assert series_id in df.columns, f"Missing series: {series_id}"

        print(f"Fetched {len(series_ids)} series with {df.height} rows")
        print(f"Columns: {df.columns}")

    def test_fetch_weekly_h8_data(self):
        """Test fetching weekly H.8 bank credit data specifically."""
        fetcher = FREDDataFetcher()

        # H.8 series used in the dashboard
        h8_series = ["TOTLL", "BUSLOANS", "CONSUMER", "REALLN"]

        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        df = fetcher.fetch_multiple_series(h8_series, start_date=start_date)

        print(f"\n=== Weekly H.8 Data Diagnostic ===")
        print(f"Start date requested: {start_date}")
        print(f"Rows returned: {df.height}")
        print(f"Columns returned: {df.columns}")

        if df.height > 0:
            print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")

            # Check each series
            for series in h8_series:
                if series in df.columns:
                    non_null = df.filter(pl.col(series).is_not_null()).height
                    latest_row = df.filter(pl.col(series).is_not_null()).tail(1)
                    if latest_row.height > 0:
                        latest_date = latest_row["date"][0]
                        latest_value = latest_row[series][0]
                        print(f"{series}: {non_null} non-null values, latest={latest_date}: {latest_value:,.0f}")
                    else:
                        print(f"{series}: No non-null values found")
                else:
                    print(f"{series}: MISSING from DataFrame")

            # Check data freshness
            print(f"\n=== Data Freshness Check ===")
            latest_date = df["date"].max()
            days_old = (datetime.now().date() - latest_date.date()).days
            print(f"Most recent data: {latest_date}")
            print(f"Days old: {days_old}")

            if days_old > 14:
                print("WARNING: Data is more than 2 weeks old!")
        else:
            print("ERROR: No data returned!")

        assert df.height > 0, "Should have H.8 data"

    def test_financial_conditions_data(self):
        """Test fetching financial conditions indicators."""
        fetcher = FREDDataFetcher()

        conditions_series = ["NFCI", "BAMLH0A0HYM2"]

        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        df = fetcher.fetch_multiple_series(conditions_series, start_date=start_date)

        print(f"\n=== Financial Conditions Diagnostic ===")
        print(f"Rows returned: {df.height}")

        if df.height > 0:
            for series in conditions_series:
                if series in df.columns:
                    non_null = df.filter(pl.col(series).is_not_null()).height
                    latest = df.filter(pl.col(series).is_not_null()).tail(1)
                    if latest.height > 0:
                        print(f"{series}: {non_null} values, latest={latest[series][0]:.3f}")
                else:
                    print(f"{series}: MISSING")

        assert df.height > 0, "Should have financial conditions data"

    def test_handles_invalid_series(self):
        """Test handling of invalid series ID."""
        fetcher = FREDDataFetcher()

        df = fetcher.fetch_series("INVALID_SERIES_XYZ_123")

        # Should return empty DataFrame, not raise
        assert df.height == 0, "Invalid series should return empty DataFrame"


class TestFREDCache:
    """Test the local file cache for FRED data."""

    def test_cache_write_and_read(self):
        """Test writing and reading from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FREDCache(cache_dir=tmpdir)

            # Create test data
            test_df = pl.DataFrame({
                "date": [datetime(2023, 1, 1), datetime(2023, 1, 8)],
                "TOTLL": [12000.0, 12100.0],
            })

            # Write to cache
            cache.save("TOTLL", test_df, start_date="2023-01-01", end_date="2023-01-31")

            # Read from cache
            cached_df = cache.load("TOTLL", start_date="2023-01-01", end_date="2023-01-31")

            assert cached_df is not None, "Should find cached data"
            assert cached_df.height == test_df.height, "Cached data should match original"

    def test_cache_expiry(self):
        """Test cache expiry behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FREDCache(cache_dir=tmpdir, max_age_hours=0)  # Immediately expired

            test_df = pl.DataFrame({
                "date": [datetime(2023, 1, 1)],
                "TOTLL": [12000.0],
            })

            cache.save("TOTLL", test_df, start_date="2023-01-01", end_date="2023-01-31")

            # Should return None because cache is expired
            cached_df = cache.load("TOTLL", start_date="2023-01-01", end_date="2023-01-31")
            assert cached_df is None, "Expired cache should return None"

    def test_cache_miss(self):
        """Test cache miss behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FREDCache(cache_dir=tmpdir)

            # Try to load non-existent data
            cached_df = cache.load("NONEXISTENT", start_date="2023-01-01", end_date="2023-01-31")
            assert cached_df is None, "Cache miss should return None"


class TestCachedFREDFetcher:
    """Test the cached FRED fetcher integration."""

    def test_fetcher_uses_cache(self):
        """Test that fetcher properly uses cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from financing_private_credit.cache import CachedFREDFetcher

            fetcher = CachedFREDFetcher(cache_dir=tmpdir)

            # First fetch should hit API
            df1 = fetcher.fetch_series("TOTLL", start_date="2023-06-01", end_date="2023-06-30")

            if df1.height > 0:
                # Second fetch should use cache
                df2 = fetcher.fetch_series("TOTLL", start_date="2023-06-01", end_date="2023-06-30")

                assert df1.height == df2.height, "Cached data should match original"
                print(f"Cached fetch returned {df2.height} rows")


def run_diagnostic():
    """Run diagnostic to identify data fetching issues."""
    print("=" * 60)
    print("FRED Data Fetching Diagnostic")
    print("=" * 60)

    test = TestFREDDataFetcher()

    print("\n1. Testing weekly H.8 data...")
    try:
        test.test_fetch_weekly_h8_data()
        print("   PASSED")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n2. Testing financial conditions...")
    try:
        test.test_financial_conditions_data()
        print("   PASSED")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n3. Testing single series fetch...")
    try:
        test.test_fetch_single_series()
        print("   PASSED")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n" + "=" * 60)
    print("Diagnostic complete")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnostic()
