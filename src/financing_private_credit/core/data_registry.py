"""
Data Registry - Centralized data management with smart caching.

Provides a single source of truth for fetching and caching data used across
all indicators. Uses Arrow IPC format (Feather) for fast, compressed local storage.

Features:
- Shared data sources (bank panels, FRED macro data) fetched once per session
- Smart cache invalidation based on data freshness
- Support for indicator-specific custom data sources
- Efficient Arrow/Feather format for local persistence

Usage:
    from financing_private_credit.core import DataRegistry

    # Get singleton instance
    registry = DataRegistry.get_instance()

    # Fetch shared data (cached automatically)
    bank_panel = registry.get_bank_panel("2015-01-01")
    macro_data = registry.get_macro_series(["FEDFUNDS", "DGS10"], "2015-01-01")

    # Register custom data source for an indicator
    registry.register_source("call_reports", my_fetcher_function)
    call_data = registry.get("call_reports", start_date="2015-01-01")
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import polars as pl


T = TypeVar("T")


def get_default_data_dir() -> Path:
    """Get the default data cache directory."""
    # Use XDG_DATA_HOME if available, otherwise ~/.local/share
    data_home = os.environ.get(
        "XDG_DATA_HOME",
        os.path.expanduser("~/.local/share")
    )
    data_dir = Path(data_home) / "financing_private_credit" / "data_cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@dataclass
class CacheEntry:
    """Metadata for a cached data entry."""

    key: str
    file_path: str
    created_at: datetime
    expires_at: datetime
    row_count: int
    source_type: str  # "bank_panel", "fred", "custom"
    params: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        return datetime.now() < self.expires_at

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "key": self.key,
            "file_path": self.file_path,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "row_count": self.row_count,
            "source_type": self.source_type,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            key=d["key"],
            file_path=d["file_path"],
            created_at=datetime.fromisoformat(d["created_at"]),
            expires_at=datetime.fromisoformat(d["expires_at"]),
            row_count=d["row_count"],
            source_type=d["source_type"],
            params=d.get("params", {}),
        )


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    # Default TTLs by source type (in hours)
    ttl_bank_panel: int = 24  # Bank data updates quarterly
    ttl_fred_daily: int = 6  # Daily FRED series
    ttl_fred_weekly: int = 24  # Weekly FRED series (H.8)
    ttl_fred_monthly: int = 48  # Monthly FRED series
    ttl_custom: int = 12  # Custom sources

    # Force refresh on next fetch
    force_refresh: bool = False


class DataCache:
    """
    Local file cache using Arrow IPC format (Feather).

    Provides fast, compressed storage for DataFrames with automatic
    expiration and cache invalidation.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        config: Optional[CacheConfig] = None,
    ):
        """
        Initialize the data cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.local/share/financing_private_credit/data_cache
            config: Cache configuration
        """
        self.cache_dir = cache_dir or get_default_data_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or CacheConfig()
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._entries: dict[str, CacheEntry] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                self._entries = {
                    k: CacheEntry.from_dict(v) for k, v in data.items()
                }
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"Warning: Failed to load cache metadata: {e}")
                self._entries = {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._entries.items()},
                    f,
                    indent=2,
                )
        except IOError as e:
            print(f"Warning: Failed to save cache metadata: {e}")

    def _get_cache_key(self, source_type: str, **params) -> str:
        """Generate a unique cache key from parameters."""
        # Sort params for consistent hashing
        param_str = json.dumps(params, sort_keys=True, default=str)
        hash_input = f"{source_type}:{param_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _get_file_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.arrow"

    def _get_ttl(self, source_type: str) -> int:
        """Get TTL in hours for a source type."""
        ttl_map = {
            "bank_panel": self.config.ttl_bank_panel,
            "fred_daily": self.config.ttl_fred_daily,
            "fred_weekly": self.config.ttl_fred_weekly,
            "fred_monthly": self.config.ttl_fred_monthly,
            "custom": self.config.ttl_custom,
        }
        return ttl_map.get(source_type, self.config.ttl_custom)

    def get(
        self,
        source_type: str,
        **params,
    ) -> Optional[pl.DataFrame]:
        """
        Get cached data if valid.

        Args:
            source_type: Type of data source
            **params: Parameters that identify the data

        Returns:
            Cached DataFrame or None if not valid
        """
        if self.config.force_refresh:
            return None

        cache_key = self._get_cache_key(source_type, **params)

        if cache_key not in self._entries:
            return None

        entry = self._entries[cache_key]

        if not entry.is_valid():
            # Cache expired, remove entry
            self._remove_entry(cache_key)
            return None

        file_path = Path(entry.file_path)
        if not file_path.exists():
            self._remove_entry(cache_key)
            return None

        try:
            # Read Arrow IPC file
            return pl.read_ipc(file_path)
        except Exception as e:
            print(f"Warning: Failed to read cache file {file_path}: {e}")
            self._remove_entry(cache_key)
            return None

    def set(
        self,
        data: pl.DataFrame,
        source_type: str,
        **params,
    ) -> None:
        """
        Store data in cache.

        Args:
            data: DataFrame to cache
            source_type: Type of data source
            **params: Parameters that identify the data
        """
        cache_key = self._get_cache_key(source_type, **params)
        file_path = self._get_file_path(cache_key)

        try:
            # Write as Arrow IPC (Feather v2) with compression
            data.write_ipc(file_path, compression="zstd")

            ttl_hours = self._get_ttl(source_type)
            now = datetime.now()

            self._entries[cache_key] = CacheEntry(
                key=cache_key,
                file_path=str(file_path),
                created_at=now,
                expires_at=now + timedelta(hours=ttl_hours),
                row_count=data.height,
                source_type=source_type,
                params=params,
            )
            self._save_metadata()

        except Exception as e:
            print(f"Warning: Failed to cache data: {e}")

    def _remove_entry(self, cache_key: str) -> None:
        """Remove a cache entry."""
        if cache_key in self._entries:
            entry = self._entries[cache_key]
            file_path = Path(entry.file_path)
            if file_path.exists():
                try:
                    file_path.unlink()
                except IOError:
                    pass
            del self._entries[cache_key]
            self._save_metadata()

    def invalidate(
        self,
        source_type: Optional[str] = None,
        **params,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            source_type: If provided, only invalidate this source type.
                        If None with no params, invalidate all.
            **params: If provided with source_type, invalidate specific entry

        Returns:
            Number of entries invalidated
        """
        if source_type and params:
            # Invalidate specific entry
            cache_key = self._get_cache_key(source_type, **params)
            if cache_key in self._entries:
                self._remove_entry(cache_key)
                return 1
            return 0

        # Invalidate by source type or all
        keys_to_remove = [
            k for k, v in self._entries.items()
            if source_type is None or v.source_type == source_type
        ]

        for key in keys_to_remove:
            self._remove_entry(key)

        return len(keys_to_remove)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(
            Path(e.file_path).stat().st_size
            for e in self._entries.values()
            if Path(e.file_path).exists()
        )

        by_source = {}
        for entry in self._entries.values():
            if entry.source_type not in by_source:
                by_source[entry.source_type] = {"count": 0, "rows": 0}
            by_source[entry.source_type]["count"] += 1
            by_source[entry.source_type]["rows"] += entry.row_count

        valid_count = sum(1 for e in self._entries.values() if e.is_valid())

        return {
            "cache_dir": str(self.cache_dir),
            "total_entries": len(self._entries),
            "valid_entries": valid_count,
            "expired_entries": len(self._entries) - valid_count,
            "total_size_mb": total_size / (1024 * 1024),
            "by_source": by_source,
        }


class DataRegistry:
    """
    Central registry for all data sources used by indicators.

    Provides:
    - Singleton access to shared data (bank panels, FRED macro)
    - Smart caching with configurable TTLs
    - Registration of custom data sources
    - Efficient data sharing across indicators in a single session

    Example:
        registry = DataRegistry.get_instance()

        # Shared data - fetched once, cached locally
        bank_panel = registry.get_bank_panel("2015-01-01")
        macro = registry.get_macro_series(["FEDFUNDS", "DGS10"], "2015-01-01")

        # Custom data source
        def fetch_call_reports(start_date: str) -> pl.DataFrame:
            # Custom fetching logic
            ...

        registry.register_source("call_reports", fetch_call_reports)
        call_data = registry.get("call_reports", start_date="2015-01-01")
    """

    _instance: Optional["DataRegistry"] = None

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        """
        Initialize the data registry.

        Args:
            cache_dir: Directory for cache files
            cache_config: Cache configuration
        """
        self._cache = DataCache(cache_dir, cache_config)
        self._custom_sources: dict[str, Callable[..., pl.DataFrame]] = {}
        self._session_cache: dict[str, pl.DataFrame] = {}  # In-memory for session

    @classmethod
    def get_instance(
        cls,
        cache_dir: Optional[Path] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> "DataRegistry":
        """
        Get the singleton DataRegistry instance.

        Args:
            cache_dir: Directory for cache files (only used on first call)
            cache_config: Cache configuration (only used on first call)

        Returns:
            The DataRegistry singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(cache_dir, cache_config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def get_bank_panel(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        compute_derived: bool = True,
    ) -> pl.DataFrame:
        """
        Get bank panel data from SEC EDGAR.

        This is the primary shared data source - fetched once and cached.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date (defaults to today)
            compute_derived: Whether to compute derived metrics

        Returns:
            Panel DataFrame with bank-level quarterly data
        """
        # Check session cache first (in-memory)
        session_key = f"bank_panel_{start_date}_{end_date}_{compute_derived}"
        if session_key in self._session_cache:
            return self._session_cache[session_key]

        # Check persistent cache
        cached = self._cache.get(
            "bank_panel",
            start_date=start_date,
            end_date=end_date,
            compute_derived=compute_derived,
        )
        if cached is not None:
            self._session_cache[session_key] = cached
            return cached

        # Fetch fresh data
        from ..bank_data import BankDataCollector

        print("Fetching bank panel data from SEC EDGAR...")
        collector = BankDataCollector(start_date=start_date)
        panel = collector.fetch_all_banks()

        if compute_derived and panel.height > 0:
            panel = collector.compute_derived_metrics(panel)

        # Cache the result
        if panel.height > 0:
            self._cache.set(
                panel,
                "bank_panel",
                start_date=start_date,
                end_date=end_date,
                compute_derived=compute_derived,
            )

        self._session_cache[session_key] = panel
        return panel

    def get_macro_series(
        self,
        series_ids: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Get FRED macro data series.

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date (defaults to today)

        Returns:
            DataFrame with date column and one column per series
        """
        # Sort series for consistent caching
        series_key = ",".join(sorted(series_ids))
        session_key = f"fred_{series_key}_{start_date}_{end_date}"

        if session_key in self._session_cache:
            return self._session_cache[session_key]

        # Determine source type based on series frequency
        # Common weekly series
        weekly_series = {"TOTLL", "BUSLOANS", "CONSUMER", "REALLN", "TOTBKCR"}
        if any(s in weekly_series for s in series_ids):
            source_type = "fred_weekly"
        else:
            source_type = "fred_daily"

        cached = self._cache.get(
            source_type,
            series_ids=series_key,
            start_date=start_date,
            end_date=end_date,
        )
        if cached is not None:
            self._session_cache[session_key] = cached
            return cached

        # Fetch fresh data
        from ..cache import CachedFREDFetcher

        print(f"Fetching FRED data: {series_ids}...")
        fetcher = CachedFREDFetcher(max_age_hours=6)
        data = fetcher.fetch_multiple_series(series_ids, start_date, end_date)

        if data.height > 0:
            self._cache.set(
                data,
                source_type,
                series_ids=series_key,
                start_date=start_date,
                end_date=end_date,
            )

        self._session_cache[session_key] = data
        return data

    def get_data_quality_summary(self) -> pl.DataFrame:
        """
        Get data quality summary for all banks.

        Returns:
            DataFrame with data quality metrics per bank
        """
        session_key = "data_quality_summary"
        if session_key in self._session_cache:
            return self._session_cache[session_key]

        from ..bank_data import BankDataCollector

        collector = BankDataCollector()
        summary = collector.get_data_quality_summary()
        self._session_cache[session_key] = summary
        return summary

    def register_source(
        self,
        name: str,
        fetcher: Callable[..., pl.DataFrame],
        ttl_hours: Optional[int] = None,
    ) -> None:
        """
        Register a custom data source.

        Args:
            name: Unique name for the data source
            fetcher: Function that returns a DataFrame. Should accept keyword arguments.
            ttl_hours: Cache TTL in hours (defaults to config.ttl_custom)

        Example:
            def fetch_call_reports(start_date: str, bank_id: str) -> pl.DataFrame:
                # Fetch call report data
                ...

            registry.register_source("call_reports", fetch_call_reports, ttl_hours=48)
        """
        self._custom_sources[name] = fetcher
        if ttl_hours is not None:
            # Update TTL for this source
            self._cache.config.ttl_custom = ttl_hours

    def get(
        self,
        source_name: str,
        use_cache: bool = True,
        **params,
    ) -> pl.DataFrame:
        """
        Get data from a registered source.

        Args:
            source_name: Name of the registered data source
            use_cache: Whether to use cached data if available
            **params: Parameters to pass to the fetcher function

        Returns:
            DataFrame from the data source

        Raises:
            ValueError: If source is not registered
        """
        if source_name not in self._custom_sources:
            raise ValueError(
                f"Unknown data source: {source_name}. "
                f"Registered sources: {list(self._custom_sources.keys())}"
            )

        # Check cache
        if use_cache:
            session_key = f"custom_{source_name}_{hash(frozenset(params.items()))}"
            if session_key in self._session_cache:
                return self._session_cache[session_key]

            cached = self._cache.get("custom", source=source_name, **params)
            if cached is not None:
                self._session_cache[session_key] = cached
                return cached

        # Fetch from source
        fetcher = self._custom_sources[source_name]
        data = fetcher(**params)

        # Cache result
        if data.height > 0 and use_cache:
            self._cache.set(data, "custom", source=source_name, **params)
            session_key = f"custom_{source_name}_{hash(frozenset(params.items()))}"
            self._session_cache[session_key] = data

        return data

    def list_sources(self) -> dict[str, list[str]]:
        """
        List all available data sources.

        Returns:
            Dictionary with 'builtin' and 'custom' source lists
        """
        return {
            "builtin": ["bank_panel", "macro_series", "data_quality_summary"],
            "custom": list(self._custom_sources.keys()),
        }

    def invalidate(
        self,
        source_type: Optional[str] = None,
        clear_session: bool = True,
    ) -> int:
        """
        Invalidate cached data.

        Args:
            source_type: If provided, only invalidate this source type.
                        Options: "bank_panel", "fred_daily", "fred_weekly", "custom", or None for all
            clear_session: Whether to also clear in-memory session cache

        Returns:
            Number of persistent cache entries invalidated
        """
        if clear_session:
            if source_type:
                # Clear matching session cache entries
                keys_to_remove = [
                    k for k in self._session_cache
                    if k.startswith(source_type) or source_type in k
                ]
                for key in keys_to_remove:
                    del self._session_cache[key]
            else:
                self._session_cache.clear()

        return self._cache.invalidate(source_type)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = self._cache.get_stats()
        stats["session_cache_entries"] = len(self._session_cache)
        stats["custom_sources_registered"] = len(self._custom_sources)
        return stats

    def force_refresh(self) -> None:
        """Force refresh on next fetch (bypass cache)."""
        self._cache.config.force_refresh = True
        self._session_cache.clear()

    def reset_force_refresh(self) -> None:
        """Reset force refresh flag."""
        self._cache.config.force_refresh = False
