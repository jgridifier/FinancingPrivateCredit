"""
Indicator Framework Base Classes

This module provides the abstract base classes and common interfaces for
implementing credit and financing indicators. Each indicator follows the
methodology from NY Fed Staff Report 1111 and extends it for specific use cases.

Indicators implemented:
1. Credit Boom Leading Indicator (LIS-based)
2. Cross-Bank Variance Decomposition
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

import polars as pl


# Type variable for indicator-specific results
T = TypeVar("T")


@dataclass
class IndicatorMetadata:
    """Metadata describing an indicator."""

    name: str
    short_name: str
    description: str
    version: str
    paper_reference: str
    data_sources: list[str]
    update_frequency: str  # "daily", "weekly", "monthly", "quarterly"
    lookback_periods: int  # Number of periods needed for calculation


@dataclass
class IndicatorResult(Generic[T]):
    """Generic container for indicator results."""

    indicator_name: str
    calculation_date: datetime
    data: T
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "indicator_name": self.indicator_name,
            "calculation_date": self.calculation_date.isoformat(),
            "metadata": self.metadata,
        }


class BaseIndicator(ABC):
    """
    Abstract base class for all credit/financing indicators.

    Each indicator must implement:
    - get_metadata(): Return indicator metadata
    - fetch_data(): Fetch required data from sources
    - calculate(): Perform the indicator calculation
    - nowcast(): Provide high-frequency nowcast estimates
    - get_dashboard_components(): Return Streamlit components for display
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        """
        Initialize the indicator.

        Args:
            config_path: Optional path to JSON configuration file
        """
        self._config = self._load_config(config_path) if config_path else {}
        self._data_cache: dict[str, pl.DataFrame] = {}

    def _load_config(self, path: str | Path) -> dict:
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    @abstractmethod
    def get_metadata(self) -> IndicatorMetadata:
        """Return metadata describing this indicator."""
        pass

    @abstractmethod
    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch all required data for the indicator.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date (defaults to today)

        Returns:
            Dictionary mapping data source names to DataFrames
        """
        pass

    @abstractmethod
    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """
        Calculate the indicator values.

        Args:
            data: Dictionary of DataFrames from fetch_data()
            **kwargs: Additional calculation parameters

        Returns:
            IndicatorResult containing the calculated values
        """
        pass

    @abstractmethod
    def nowcast(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """
        Generate high-frequency nowcast estimates.

        Args:
            data: Dictionary of DataFrames including high-frequency proxies
            **kwargs: Additional nowcasting parameters

        Returns:
            IndicatorResult containing nowcast estimates
        """
        pass

    @abstractmethod
    def get_dashboard_components(self) -> dict[str, Any]:
        """
        Return components for Streamlit dashboard display.

        Returns:
            Dictionary with:
            - 'tabs': List of tab configurations
            - 'charts': Chart generation functions
            - 'metrics': Metric display functions
        """
        pass

    def validate_data(self, data: dict[str, pl.DataFrame]) -> list[str]:
        """
        Validate that required data is present and well-formed.

        Args:
            data: Dictionary of DataFrames to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        for name, df in data.items():
            if df.height == 0:
                errors.append(f"Empty DataFrame: {name}")
            if "date" not in df.columns:
                errors.append(f"Missing 'date' column: {name}")
        return errors


class BaseDecomposition(BaseIndicator):
    """
    Base class for variance/growth decomposition indicators.

    Extends BaseIndicator with decomposition-specific methods.
    """

    @abstractmethod
    def decompose(
        self,
        data: dict[str, pl.DataFrame],
        entity: str,
    ) -> pl.DataFrame:
        """
        Decompose a variable into component contributions.

        Args:
            data: Input data
            entity: Entity to decompose (e.g., bank ticker)

        Returns:
            DataFrame with component contributions
        """
        pass

    @abstractmethod
    def compute_variance_shares(
        self,
        decomposition: pl.DataFrame,
    ) -> dict[str, float]:
        """
        Compute variance contribution shares.

        Args:
            decomposition: DataFrame from decompose()

        Returns:
            Dictionary mapping component names to variance shares
        """
        pass

    def aggregate_decomposition(
        self,
        decompositions: dict[str, pl.DataFrame],
        weights: Optional[dict[str, float]] = None,
    ) -> pl.DataFrame:
        """
        Aggregate decompositions across entities.

        Args:
            decompositions: Dictionary mapping entity to decomposition
            weights: Optional weights for aggregation (defaults to equal)

        Returns:
            Aggregated decomposition DataFrame
        """
        if not decompositions:
            return pl.DataFrame()

        if weights is None:
            weights = {k: 1.0 / len(decompositions) for k in decompositions}

        # Stack all decompositions
        dfs = []
        for entity, df in decompositions.items():
            df = df.with_columns(
                pl.lit(entity).alias("entity"),
                pl.lit(weights.get(entity, 0)).alias("weight"),
            )
            dfs.append(df)

        combined = pl.concat(dfs)

        # Weighted average by date
        numeric_cols = [c for c in combined.columns if c not in ["date", "entity", "weight"]]

        aggregated = (
            combined
            .group_by("date")
            .agg([
                (pl.col(c) * pl.col("weight")).sum().alias(c)
                for c in numeric_cols
            ])
            .sort("date")
        )

        return aggregated


class BaseForecastModel(ABC):
    """
    Base class for forecasting models used by indicators.

    Supports multiple model types (APLR, ARDL, VAR, etc.) with
    a consistent interface.
    """

    @abstractmethod
    def fit(
        self,
        data: pl.DataFrame,
        target: str,
        features: list[str],
    ) -> dict[str, Any]:
        """
        Fit the model to data.

        Args:
            data: Training data
            target: Target variable name
            features: List of feature variable names

        Returns:
            Dictionary with fit metrics
        """
        pass

    @abstractmethod
    def predict(
        self,
        data: pl.DataFrame,
        horizon: int,
    ) -> pl.DataFrame:
        """
        Generate predictions.

        Args:
            data: Data for prediction
            horizon: Forecast horizon

        Returns:
            DataFrame with predictions
        """
        pass

    @abstractmethod
    def get_coefficients(self) -> dict[str, float]:
        """Return model coefficients."""
        pass


# Registry for indicators
_INDICATOR_REGISTRY: dict[str, type[BaseIndicator]] = {}


def register_indicator(name: str):
    """
    Decorator to register an indicator class.

    Usage:
        @register_indicator("credit_boom")
        class CreditBoomIndicator(BaseIndicator):
            ...
    """
    def decorator(cls: type[BaseIndicator]):
        _INDICATOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_indicator(name: str, **kwargs) -> BaseIndicator:
    """
    Get an indicator instance by name.

    Args:
        name: Registered indicator name
        **kwargs: Arguments to pass to indicator constructor

    Returns:
        Indicator instance
    """
    if name not in _INDICATOR_REGISTRY:
        available = ", ".join(_INDICATOR_REGISTRY.keys())
        raise ValueError(f"Unknown indicator: {name}. Available: {available}")
    return _INDICATOR_REGISTRY[name](**kwargs)


def list_indicators() -> list[str]:
    """Return list of registered indicator names."""
    return list(_INDICATOR_REGISTRY.keys())
