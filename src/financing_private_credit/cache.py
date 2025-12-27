"""
Local file cache for FRED API data.

Reduces API load and improves performance by caching fetched data locally.
Cache files are stored as Parquet for efficient storage and fast reads.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import polars as pl

from .data import FREDDataFetcher


def get_default_cache_dir() -> Path:
    """Get the default cache directory."""
    # Use XDG_CACHE_HOME if available, otherwise ~/.cache
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(cache_home) / "financing_private_credit" / "fred_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class FREDCache:
    """
    Local file cache for FRED data.

    Stores fetched data as Parquet files with metadata for cache invalidation.
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        max_age_hours: int = 6,
    ):
        """
        Initialize the FRED cache.

        Args:
            cache_dir: Directory to store cache files. Defaults to ~/.cache/financing_private_credit/fred_cache
            max_age_hours: Maximum age of cached data in hours before refresh. Default 6 hours for H.8 weekly data.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else get_default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_hours = max_age_hours
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def _get_cache_key(self, series_id: str, start_date: str, end_date: str) -> str:
        """Generate a unique cache key for a query."""
        key_str = f"{series_id}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.parquet"

    def is_valid(self, series_id: str, start_date: str, end_date: str) -> bool:
        """
        Check if cached data is valid (exists and not expired).

        Args:
            series_id: FRED series ID
            start_date: Start date
            end_date: End date

        Returns:
            True if cache is valid, False otherwise
        """
        cache_key = self._get_cache_key(series_id, start_date, end_date)

        if cache_key not in self._metadata:
            return False

        meta = self._metadata[cache_key]
        cached_time = datetime.fromisoformat(meta["cached_at"])
        age = datetime.now() - cached_time

        if age > timedelta(hours=self.max_age_hours):
            return False

        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists()

    def load(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pl.DataFrame]:
        """
        Load data from cache if valid.

        Args:
            series_id: FRED series ID
            start_date: Start date
            end_date: End date

        Returns:
            Cached DataFrame or None if not valid
        """
        cache_key = self._get_cache_key(series_id, start_date, end_date)

        if not self.is_valid(series_id, start_date, end_date):
            return None

        cache_path = self._get_cache_path(cache_key)

        try:
            return pl.read_parquet(cache_path)
        except Exception as e:
            print(f"Warning: Failed to read cache file {cache_path}: {e}")
            return None

    def save(
        self,
        series_id: str,
        data: pl.DataFrame,
        start_date: str,
        end_date: str,
    ):
        """
        Save data to cache.

        Args:
            series_id: FRED series ID
            data: DataFrame to cache
            start_date: Start date
            end_date: End date
        """
        cache_key = self._get_cache_key(series_id, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)

        try:
            data.write_parquet(cache_path)

            self._metadata[cache_key] = {
                "series_id": series_id,
                "start_date": start_date,
                "end_date": end_date,
                "cached_at": datetime.now().isoformat(),
                "rows": data.height,
            }
            self._save_metadata()

        except Exception as e:
            print(f"Warning: Failed to save cache file {cache_path}: {e}")

    def clear(self, series_id: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            series_id: If provided, clear only entries for this series.
                       If None, clear all cache entries.
        """
        if series_id is None:
            # Clear all
            for key in list(self._metadata.keys()):
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
            self._metadata = {}
        else:
            # Clear specific series
            keys_to_remove = [
                key for key, meta in self._metadata.items()
                if meta.get("series_id") == series_id
            ]
            for key in keys_to_remove:
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
                del self._metadata[key]

        self._save_metadata()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_files = len(self._metadata)
        total_size = sum(
            self._get_cache_path(key).stat().st_size
            for key in self._metadata
            if self._get_cache_path(key).exists()
        )

        return {
            "cache_dir": str(self.cache_dir),
            "total_entries": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "max_age_hours": self.max_age_hours,
        }


class CachedFREDFetcher(FREDDataFetcher):
    """
    FRED data fetcher with local file caching.

    Extends FREDDataFetcher to add transparent caching of fetched data.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        max_age_hours: int = 6,
    ):
        """
        Initialize the cached FRED fetcher.

        Args:
            api_key: Optional FRED API key
            cache_dir: Directory to store cache files
            max_age_hours: Maximum age of cached data in hours
        """
        super().__init__(api_key)
        self.cache = FREDCache(cache_dir=cache_dir, max_age_hours=max_age_hours)

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch a single series from FRED, using cache if available.

        Args:
            series_id: The FRED series ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Polars DataFrame with columns [date, value, series_id]
        """
        # Normalize dates for cache key
        start = start_date or "1900-01-01"
        end = end_date or datetime.now().strftime("%Y-%m-%d")

        # Try cache first
        cached = self.cache.load(series_id, start, end)
        if cached is not None:
            return cached

        # Fetch from API
        df = super().fetch_series(series_id, start_date, end_date)

        # Save to cache if we got data
        if df.height > 0:
            self.cache.save(series_id, df, start, end)

        return df

    def fetch_multiple_series(
        self,
        series_ids: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch multiple series and join them into a single DataFrame.

        Uses per-series caching for efficiency.

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Polars DataFrame with date column and one column per series
        """
        dfs = []
        for series_id in series_ids:
            df = self.fetch_series(series_id, start_date, end_date)
            if df.height > 0:
                df = df.select([
                    "date",
                    pl.col("value").alias(series_id)
                ])
                dfs.append(df)

        if not dfs:
            return pl.DataFrame({"date": []})

        # Join all series on date using coalesce for outer joins
        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, on="date", how="outer_coalesce")

        return result.sort("date")

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self, series_id: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            series_id: If provided, clear only entries for this series.
        """
        self.cache.clear(series_id)


if __name__ == "__main__":
    # Test the cache
    fetcher = CachedFREDFetcher()

    print("Cache stats before fetch:")
    print(fetcher.get_cache_stats())

    print("\nFetching TOTLL...")
    df = fetcher.fetch_series("TOTLL", start_date="2023-01-01")
    print(f"Fetched {df.height} rows")

    print("\nCache stats after fetch:")
    print(fetcher.get_cache_stats())

    print("\nFetching again (should use cache)...")
    df2 = fetcher.fetch_series("TOTLL", start_date="2023-01-01")
    print(f"Fetched {df2.height} rows")
