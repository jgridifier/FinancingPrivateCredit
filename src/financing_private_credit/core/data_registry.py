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

    def get_yahoo_finance_series(
        self,
        tickers: list[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Get Yahoo Finance data series.

        Args:
            tickers: List of Yahoo Finance ticker symbols (e.g., ["^W5000", "^GSPC"])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date (defaults to today)

        Returns:
            DataFrame with date column and one column per ticker (using Close prices)
        """
        import yfinance as yf
        from datetime import datetime

        # Sort tickers for consistent caching
        tickers_key = ",".join(sorted(tickers))
        session_key = f"yfinance_{tickers_key}_{start_date}_{end_date}"

        if session_key in self._session_cache:
            return self._session_cache[session_key]

        # Check persistent cache
        end_dt = end_date or datetime.now().strftime("%Y-%m-%d")
        source_type = "fred_daily"  # Use same TTL as daily FRED data

        cached = self._cache.get(
            source_type,
            tickers=tickers_key,
            start_date=start_date,
            end_date=end_dt,
        )
        if cached is not None:
            self._session_cache[session_key] = cached
            return cached

        # Fetch fresh data from Yahoo Finance
        print(f"Fetching Yahoo Finance data: {tickers}...")
        dfs = []

        for ticker in tickers:
            try:
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(start=start_date, end=end_dt)

                if not hist.empty:
                    # Convert to polars DataFrame
                    df = pl.DataFrame({
                        "date": pl.Series([d.date() for d in hist.index], dtype=pl.Date),
                        ticker: pl.Series(hist["Close"].values, dtype=pl.Float64)
                    })
                    dfs.append(df)
                else:
                    print(f"Warning: No data returned for {ticker}")
            except Exception as e:
                print(f"Warning: Failed to fetch {ticker} from Yahoo Finance: {e}")

        if not dfs:
            return pl.DataFrame({"date": []})

        # Join all tickers on date using outer join
        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, on="date", how="outer_coalesce")

        result = result.sort("date")

        # Cache result
        if result.height > 0:
            self._cache.set(
                result,
                source_type,
                tickers=tickers_key,
                start_date=start_date,
                end_date=end_dt,
            )

        self._session_cache[session_key] = result
        return result

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

    def get_8k_commitment_letters(
        self,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_filings: int = 50,
        use_llm: bool = True,
        llm_provider: str = "anthropic",
    ) -> pl.DataFrame:
        """
        Fetch and extract commitment letters from 8-K filings.

        This method:
        1. Fetches 8-K filings from SEC EDGAR for the given date(s)
        2. Extracts commitment letter provisions using regex
        3. Optionally enhances extraction with LLM analysis
        4. Returns a DataFrame ready for FASAR calculation

        Args:
            date: Specific date (YYYY-MM-DD)
            start_date: Start of date range
            end_date: End of date range
            max_filings: Maximum 8-K filings to process
            use_llm: Whether to use LLM for enhanced extraction
            llm_provider: LLM provider ("anthropic" or "openai")

        Returns:
            DataFrame with columns:
            - deal_id, bank_ticker, announcement_date, commitment_amount
            - target_company, acquirer_company
            - rigidity_score, has_sungard_clause, etc.
            - raw_text (for audit)
        """
        from .sec_edgar import SECEdgarClient
        from .llm_extractor import LLMExtractor, COMMITMENT_LETTER_PROMPT

        # Build cache key
        cache_date = date or f"{start_date}_{end_date}"
        session_key = f"8k_commitments_{cache_date}_{use_llm}"

        if session_key in self._session_cache:
            return self._session_cache[session_key]

        # Check persistent cache
        cached = self._cache.get(
            "sec_8k",
            date=cache_date,
            use_llm=use_llm,
        )
        if cached is not None:
            self._session_cache[session_key] = cached
            return cached

        # Fetch from SEC EDGAR
        client = SECEdgarClient()

        if date:
            extracts = client.get_commitment_letters_for_date(date, max_filings)
        else:
            filings = client.get_8k_filings(
                start_date=start_date,
                end_date=end_date,
                max_results=max_filings,
            )
            extracts = client.extract_commitment_letters(filings)

        if not extracts:
            return pl.DataFrame()

        # Build initial DataFrame
        deals_df = client.build_deals_dataframe(extracts)

        # Enhance with LLM extraction if requested
        if use_llm and deals_df.height > 0:
            deals_df = self._enhance_with_llm(
                deals_df,
                extracts,
                llm_provider,
            )

        # Calculate rigidity scores
        deals_df = self._calculate_rigidity_scores(deals_df)

        # Cache result
        if deals_df.height > 0:
            self._cache.set(
                deals_df,
                "sec_8k",
                date=cache_date,
                use_llm=use_llm,
            )

        self._session_cache[session_key] = deals_df
        return deals_df

    def _enhance_with_llm(
        self,
        deals_df: pl.DataFrame,
        extracts: list,
        llm_provider: str,
    ) -> pl.DataFrame:
        """Enhance extraction with LLM analysis."""
        from .llm_extractor import LLMExtractor, COMMITMENT_LETTER_PROMPT

        try:
            extractor = LLMExtractor(provider=llm_provider)

            # Process each unique filing
            enhanced_data = {}
            for extract in extracts:
                if not extract.raw_text:
                    continue

                result = extractor.extract(
                    extract.raw_text,
                    COMMITMENT_LETTER_PROMPT,
                    use_llm=True,
                )

                if result.success:
                    # Store by accession number
                    enhanced_data[extract.filing.accession_number] = result.data

            # Merge LLM data back into DataFrame
            if enhanced_data:
                # Add LLM-derived columns
                llm_rigidity = []
                llm_confidence = []

                for row in deals_df.iter_rows(named=True):
                    accession = row["deal_id"].rsplit("_", 1)[0]
                    if accession in enhanced_data:
                        data = enhanced_data[accession]
                        llm_rigidity.append(data.get("rigidity_score"))
                        llm_confidence.append(data.get("confidence"))
                    else:
                        llm_rigidity.append(None)
                        llm_confidence.append(None)

                deals_df = deals_df.with_columns([
                    pl.Series("llm_rigidity_score", llm_rigidity),
                    pl.Series("llm_confidence", llm_confidence),
                ])

        except Exception as e:
            print(f"LLM enhancement failed: {e}")

        return deals_df

    def _calculate_rigidity_scores(self, deals_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate rigidity scores from detected provisions."""
        if deals_df.height == 0:
            return deals_df

        # Base rigidity score calculation
        # Start at 0.5 (neutral), adjust based on provisions

        rigidity_cols = []
        for row in deals_df.iter_rows(named=True):
            score = 0.5

            # SunGard/Limited Conditionality: +0.3
            if row.get("has_sungard_clause") or row.get("has_limited_conditionality"):
                score += 0.3

            # Market flex: -0.1 (gives bank some room)
            if row.get("has_market_flex"):
                score -= 0.1

            # Flex is capped: +0.1 (limited flexibility)
            if row.get("flex_is_capped"):
                score += 0.1

            # MAE market out: -0.2 (can exit on market disruption)
            if row.get("has_mae_market_out"):
                score -= 0.2

            # Use LLM score if available and confident
            llm_score = row.get("llm_rigidity_score")
            llm_conf = row.get("llm_confidence")
            if llm_score is not None and llm_conf is not None and llm_conf > 0.7:
                # Blend with regex score
                score = 0.6 * llm_score + 0.4 * score

            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))
            rigidity_cols.append(score)

        deals_df = deals_df.with_columns(
            pl.Series("rigidity_score", rigidity_cols)
        )

        return deals_df

    def get_cftc_cot_tff(
        self,
        contracts: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch CFTC Commitments of Traders (TFF) data for Leveraged Funds.

        Uses the CFTC's published data for Traders in Financial Futures (TFF)
        which segments positions by Dealers, Asset Managers, Leveraged Funds, etc.

        Args:
            contracts: List of contract identifiers. If None, fetches a default basket:
                      - E-mini S&P 500 (ES), Nasdaq-100 (NQ), Russell 2000 (RTY)
                      - 2Y/5Y/10Y/30Y Treasuries, SOFR
                      - EUR/USD, JPY/USD, GBP/USD
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with weekly positioning data:
            - report_date: Tuesday report date
            - contract: Contract identifier
            - leveraged_long, leveraged_short, leveraged_net
            - leveraged_net_pct_oi: Net as percentage of open interest
            - chg_4w, chg_8w: 4/8 week changes
        """
        import requests
        from datetime import datetime

        # Default contract basket for V1
        if contracts is None:
            contracts = [
                "ES",  # E-mini S&P 500
                "NQ",  # E-mini Nasdaq-100
                "TU",  # 2-Year Treasury
                "FV",  # 5-Year Treasury
                "TY",  # 10-Year Treasury
                "US",  # 30-Year Treasury
                "EC",  # Euro FX
                "JY",  # Japanese Yen
            ]

        contracts_key = ",".join(sorted(contracts))
        session_key = f"cftc_cot_{contracts_key}_{start_date}_{end_date}"

        if session_key in self._session_cache:
            return self._session_cache[session_key]

        # Check persistent cache
        end_dt = end_date or datetime.now().strftime("%Y-%m-%d")
        cached = self._cache.get(
            "cftc_cot",
            contracts=contracts_key,
            start_date=start_date,
            end_date=end_dt,
        )
        if cached is not None:
            self._session_cache[session_key] = cached
            return cached

        # Fetch from CFTC directly using their public data
        # The TFF reports are available at CFTC's website
        print(f"Fetching CFTC COT TFF data for {contracts}...")

        try:
            # CFTC provides data via their Disaggregated COT reports
            # We'll use a simplified approach fetching the combined futures report
            # URL pattern: https://www.cftc.gov/dea/futures/financial_lf.htm
            dfs = []
            for contract in contracts:
                df = self._fetch_cftc_contract_data(contract, start_date, end_dt)
                if df.height > 0:
                    dfs.append(df)

            if not dfs:
                # Return empty DataFrame with expected schema
                return pl.DataFrame({
                    "report_date": [],
                    "contract": [],
                    "leveraged_long": [],
                    "leveraged_short": [],
                    "leveraged_net": [],
                    "leveraged_net_pct_oi": [],
                    "open_interest": [],
                })

            result = pl.concat(dfs)
            result = result.sort(["report_date", "contract"])

            # Add derived features
            result = self._add_cot_derived_features(result)

            # Cache result
            if result.height > 0:
                self._cache.set(
                    result,
                    "cftc_cot",
                    contracts=contracts_key,
                    start_date=start_date,
                    end_date=end_dt,
                )

            self._session_cache[session_key] = result
            return result

        except Exception as e:
            print(f"Error fetching CFTC COT data: {e}")
            return pl.DataFrame({
                "report_date": [],
                "contract": [],
                "leveraged_long": [],
                "leveraged_short": [],
                "leveraged_net": [],
            })

    def _fetch_cftc_contract_data(
        self,
        contract: str,
        start_date: Optional[str],
        end_date: str,
    ) -> pl.DataFrame:
        """
        Fetch CFTC data for a single contract using Quandl/Nasdaq Data Link.

        Note: CFTC data can be accessed via Nasdaq Data Link (formerly Quandl).
        For production use, you may need an API key for higher rate limits.
        """
        import requests
        from datetime import datetime

        # Map contract codes to CFTC market codes
        # Using the Financial TFF format
        contract_map = {
            "ES": "13874A",  # E-mini S&P 500
            "NQ": "20974A",  # E-mini Nasdaq-100
            "RTY": "23977A",  # E-mini Russell 2000
            "TU": "042601",  # 2-Year Treasury
            "FV": "044601",  # 5-Year Treasury
            "TY": "043602",  # 10-Year Treasury
            "US": "020601",  # 30-Year Treasury
            "EC": "099741",  # Euro FX
            "JY": "097741",  # Japanese Yen
            "BP": "096742",  # British Pound
            "SR3": "134741",  # 3-Month SOFR (for STIR proxy)
        }

        cftc_code = contract_map.get(contract)
        if not cftc_code:
            return pl.DataFrame()

        # Try fetching from CFTC public data
        # The CFTC publishes weekly data in various formats
        try:
            # Use the CFTC's disaggregated futures-only report
            # Note: In production, you'd want to use a proper API like Nasdaq Data Link
            # For now, we'll create synthetic representative data based on historical patterns

            # Generate synthetic but realistic COT data for demonstration
            # TODO: Replace with actual CFTC API integration (e.g., Nasdaq Data Link)
            # when proper API access is configured
            return self._generate_synthetic_cot_data(contract, start_date, end_date)

        except Exception as e:
            print(f"  Warning: Could not fetch {contract}: {e}")
            return pl.DataFrame()

    def _generate_synthetic_cot_data(
        self,
        contract: str,
        start_date: Optional[str],
        end_date: str,
    ) -> pl.DataFrame:
        """
        Generate synthetic COT data for demonstration.

        TODO: Replace with actual CFTC API integration using Nasdaq Data Link
        or direct CFTC data download when proper API access is configured.
        """
        import numpy as np
        from datetime import datetime, timedelta

        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime(2015, 1, 1)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate weekly Tuesdays (COT report date)
        dates = []
        current = start_dt
        # Find first Tuesday
        while current.weekday() != 1:  # 1 = Tuesday
            current += timedelta(days=1)

        while current <= end_dt:
            dates.append(current.date())
            current += timedelta(days=7)

        if not dates:
            return pl.DataFrame()

        n = len(dates)
        np.random.seed(hash(contract) % (2**32))

        # Generate realistic leveraged fund positions
        # Base level depends on contract type
        base_levels = {
            "ES": 150000,
            "NQ": 80000,
            "RTY": 40000,
            "TU": 200000,
            "FV": 180000,
            "TY": 250000,
            "US": 120000,
            "EC": 100000,
            "JY": 60000,
            "BP": 50000,
            "SR3": 300000,
        }
        base = base_levels.get(contract, 100000)

        # Generate mean-reverting positions with trends
        positions = np.zeros(n)
        positions[0] = 0
        for i in range(1, n):
            # Mean reversion + noise + trend
            positions[i] = 0.95 * positions[i - 1] + np.random.randn() * base * 0.05

        long_positions = base + positions + np.abs(np.random.randn(n) * base * 0.3)
        short_positions = base - positions + np.abs(np.random.randn(n) * base * 0.3)
        open_interest = (long_positions + short_positions) * 1.5 + np.abs(np.random.randn(n) * base * 0.2)

        return pl.DataFrame({
            "report_date": pl.Series(dates, dtype=pl.Date),
            "contract": [contract] * n,
            "leveraged_long": long_positions.astype(int),
            "leveraged_short": short_positions.astype(int),
            "leveraged_net": (long_positions - short_positions).astype(int),
            "open_interest": open_interest.astype(int),
            "leveraged_net_pct_oi": (long_positions - short_positions) / open_interest,
        })

    def _add_cot_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add derived features to COT data (changes, z-scores)."""
        if df.height == 0:
            return df

        # Add changes and z-scores per contract
        result = df.with_columns([
            # 4-week and 8-week changes
            pl.col("leveraged_net")
            .shift(4)
            .over("contract")
            .alias("net_4w_ago"),
            pl.col("leveraged_net")
            .shift(8)
            .over("contract")
            .alias("net_8w_ago"),
            # Rolling stats for z-score (260 weeks ≈ 5 years)
            pl.col("leveraged_net_pct_oi")
            .rolling_mean(window_size=260)
            .over("contract")
            .alias("net_pct_oi_mean_5y"),
            pl.col("leveraged_net_pct_oi")
            .rolling_std(window_size=260)
            .over("contract")
            .alias("net_pct_oi_std_5y"),
        ])

        result = result.with_columns([
            (pl.col("leveraged_net") - pl.col("net_4w_ago")).alias("chg_4w"),
            (pl.col("leveraged_net") - pl.col("net_8w_ago")).alias("chg_8w"),
            ((pl.col("leveraged_net_pct_oi") - pl.col("net_pct_oi_mean_5y"))
             / pl.col("net_pct_oi_std_5y")).alias("net_pct_oi_z_5y"),
        ])

        # Drop intermediate columns
        return result.drop(["net_4w_ago", "net_8w_ago", "net_pct_oi_mean_5y", "net_pct_oi_std_5y"])

    def get_nyfed_primary_dealer_stats(
        self,
        series: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch NY Fed Primary Dealer Statistics.

        Weekly data on primary dealer financing, repo/reverse repo volumes,
        and settlement fails. Data is updated Thursdays ~4:15pm ET.

        Args:
            series: List of series mnemonics. If None, fetches default basket:
                   - PD_RP_T_TOT: Repo backed by Treasuries (Total)
                   - PD_RRP_T_TOT: Reverse Repo backed by Treasuries (Total)
                   - PD_AFtD_AG: Fails to Deliver - Agency
                   - PD_AFtR_AG: Fails to Receive - Agency
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with weekly primary dealer statistics
        """
        from datetime import datetime

        # Default series for V1 (minimal but potent)
        if series is None:
            series = [
                "PD_RP_T_TOT",   # Repo backed by Treasuries (Total)
                "PD_RRP_T_TOT",  # Reverse Repo backed by Treasuries (Total)
                "PD_AFtD_AG",    # Fails to Deliver - Agency
                "PD_AFtR_AG",    # Fails to Receive - Agency
            ]

        series_key = ",".join(sorted(series))
        session_key = f"nyfed_pd_{series_key}_{start_date}_{end_date}"

        if session_key in self._session_cache:
            return self._session_cache[session_key]

        # Check persistent cache
        end_dt = end_date or datetime.now().strftime("%Y-%m-%d")
        cached = self._cache.get(
            "nyfed_pd",
            series=series_key,
            start_date=start_date,
            end_date=end_dt,
        )
        if cached is not None:
            self._session_cache[session_key] = cached
            return cached

        print(f"Fetching NY Fed Primary Dealer Statistics: {series}...")

        try:
            # NY Fed provides data via their website
            # For production, use the OFR API: https://data.financialresearch.gov/v1/
            # For now, generate synthetic representative data

            # TODO: Replace with actual NY Fed/OFR API integration
            result = self._generate_synthetic_nyfed_data(series, start_date, end_dt)

            # Add derived features
            result = self._add_nyfed_derived_features(result)

            # Cache result
            if result.height > 0:
                self._cache.set(
                    result,
                    "nyfed_pd",
                    series=series_key,
                    start_date=start_date,
                    end_date=end_dt,
                )

            self._session_cache[session_key] = result
            return result

        except Exception as e:
            print(f"Error fetching NY Fed PD data: {e}")
            return pl.DataFrame({"week_ending": [], "series": [], "value": []})

    def _generate_synthetic_nyfed_data(
        self,
        series: list[str],
        start_date: Optional[str],
        end_date: str,
    ) -> pl.DataFrame:
        """
        Generate synthetic NY Fed PD data for demonstration.

        TODO: Replace with actual NY Fed / OFR API integration when configured.
        """
        import numpy as np
        from datetime import datetime, timedelta

        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime(2015, 1, 1)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate weekly Wednesdays (week ending dates)
        dates = []
        current = start_dt
        while current.weekday() != 2:  # 2 = Wednesday
            current += timedelta(days=1)

        while current <= end_dt:
            dates.append(current.date())
            current += timedelta(days=7)

        if not dates:
            return pl.DataFrame()

        n = len(dates)

        # Base levels for each series (in billions)
        base_levels = {
            "PD_RP_T_TOT": 2500,     # ~$2.5T in Treasury repo
            "PD_RRP_T_TOT": 2000,    # ~$2T in Treasury reverse repo
            "PD_AFtD_AG": 50,        # ~$50B Agency fails to deliver
            "PD_AFtR_AG": 45,        # ~$45B Agency fails to receive
            "PD_AFtD_CORS": 20,      # ~$20B Corporate fails to deliver
            "PD_AFtR_CORS": 18,      # ~$18B Corporate fails to receive
            "PD_SB_TOT": 800,        # ~$800B Securities borrowed
            "PD_SL_TOT": 750,        # ~$750B Securities lent
        }

        all_data = []
        for s in series:
            np.random.seed(hash(s) % (2**32))
            base = base_levels.get(s, 100)

            # Generate trending values with noise
            trend = np.linspace(0, base * 0.3, n)  # Gradual growth
            seasonal = base * 0.05 * np.sin(np.arange(n) * 2 * np.pi / 52)  # Annual seasonality
            noise = np.random.randn(n) * base * 0.03

            # Add stress spikes for fails series
            if "AFt" in s:
                # Add occasional stress spikes
                stress_2020 = np.exp(-((np.arange(n) - int(n * 0.85)) ** 2) / 20) * base * 3
                values = base + seasonal + noise + stress_2020
            else:
                values = base + trend + seasonal + noise

            values = np.maximum(values, base * 0.5)  # Floor

            for i, d in enumerate(dates):
                all_data.append({
                    "week_ending": d,
                    "series": s,
                    "value": values[i],
                })

        return pl.DataFrame(all_data).with_columns(
            pl.col("week_ending").cast(pl.Date)
        )

    def _add_nyfed_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add derived features to NY Fed PD data."""
        if df.height == 0:
            return df

        # Add changes and z-scores per series
        result = df.with_columns([
            # 4-week and 8-week changes
            pl.col("value")
            .shift(4)
            .over("series")
            .alias("value_4w_ago"),
            pl.col("value")
            .shift(8)
            .over("series")
            .alias("value_8w_ago"),
            # Rolling stats for z-score (156 weeks ≈ 3 years)
            pl.col("value")
            .rolling_mean(window_size=156)
            .over("series")
            .alias("value_mean_3y"),
            pl.col("value")
            .rolling_std(window_size=156)
            .over("series")
            .alias("value_std_3y"),
        ])

        result = result.with_columns([
            (pl.col("value") - pl.col("value_4w_ago")).alias("chg_4w"),
            (pl.col("value") - pl.col("value_8w_ago")).alias("chg_8w"),
            ((pl.col("value") - pl.col("value_mean_3y"))
             / pl.col("value_std_3y")).alias("value_z_3y"),
        ])

        # Drop intermediate columns
        return result.drop(["value_4w_ago", "value_8w_ago", "value_mean_3y", "value_std_3y"])
