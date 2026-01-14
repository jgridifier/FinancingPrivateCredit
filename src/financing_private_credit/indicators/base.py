"""
Indicator Framework Base Classes

This module provides the abstract base classes and common interfaces for
implementing credit and financing indicators. Each indicator follows the
methodology from NY Fed Staff Report 1111 and extends it for specific use cases.

Indicators implemented:
1. Credit Boom Leading Indicator (LIS-based)
2. Cross-Bank Variance Decomposition
3. Bank Macro Sensitivity
4. Duration Mismatch
5. Funding Stability
6. Variance Decomposition

Design principles:
- BaseIndicator requires only get_metadata(), fetch_data(), and calculate()
- nowcast() and get_dashboard_components() are optional with sensible defaults
- BaseForecastModel is model-agnostic (works with sklearn, statsmodels, custom, etc.)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar, Union

import polars as pl


# =============================================================================
# Data Quality & Error Handling
# =============================================================================


class DataQualitySeverity(Enum):
    """Severity levels for data quality issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DataQualityIssue:
    """Data quality issue description."""

    severity: DataQualitySeverity
    source: str
    message: str
    affected_records: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity.value,
            "source": self.source,
            "message": self.message,
            "affected_records": self.affected_records,
        }


# Type variable for indicator-specific results
T = TypeVar("T")
# Type variable for model types (sklearn, statsmodels, custom, etc.)
M = TypeVar("M")


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

    Required methods (must implement):
    - get_metadata(): Return indicator metadata
    - fetch_data(): Fetch required data from sources
    - calculate(): Perform the indicator calculation

    Optional methods (have sensible defaults):
    - nowcast(): Provide high-frequency nowcast estimates (default: not supported)
    - get_dashboard_components(): Return Streamlit components (default: empty config)
    - validate_data(): Validate input data (default: checks for empty DataFrames)

    Example:
        @register_indicator("my_indicator")
        class MyIndicator(BaseIndicator):
            def get_metadata(self) -> IndicatorMetadata:
                return IndicatorMetadata(...)

            def fetch_data(self, start_date, end_date=None) -> dict[str, pl.DataFrame]:
                return {"bank_panel": ..., "macro_data": ...}

            def calculate(self, data, **kwargs) -> IndicatorResult:
                return IndicatorResult(...)
    """

    # Class-level flag indicating if this indicator supports nowcasting
    supports_nowcast: bool = False

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

    def nowcast(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """
        Generate high-frequency nowcast estimates.

        Override this method if your indicator supports nowcasting.
        Set `supports_nowcast = True` at the class level when implementing.

        Args:
            data: Dictionary of DataFrames including high-frequency proxies
            **kwargs: Additional nowcasting parameters

        Returns:
            IndicatorResult containing nowcast estimates

        Raises:
            NotImplementedError: If nowcasting is not supported by this indicator
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support nowcasting. "
            "Override nowcast() and set supports_nowcast=True to enable."
        )

    def get_dashboard_components(self) -> dict[str, Any]:
        """
        Return components for Streamlit dashboard display.

        Override this method to provide custom dashboard configuration.
        The default returns a minimal configuration.

        Returns:
            Dictionary with:
            - 'tabs': List of tab configurations
            - 'primary_metric': Main metric to display
            - 'alert_fields': Fields that trigger alerts
        """
        metadata = self.get_metadata()
        return {
            "tabs": [{"name": metadata.short_name, "icon": "chart"}],
            "primary_metric": None,
            "alert_fields": [],
        }

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

    def get_required_data_sources(self) -> list[str]:
        """
        Return list of data source keys expected by fetch_data().

        Override to document what data sources this indicator needs.
        Used by DataRegistry for smart pre-fetching.

        Returns:
            List of data source names (e.g., ["bank_panel", "macro_data"])
        """
        return []

    # =========================================================================
    # Error Handling & Data Quality Methods
    # =========================================================================

    def fetch_data_with_fallback(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        strict: bool = False,
    ) -> tuple[dict[str, pl.DataFrame], list[DataQualityIssue]]:
        """
        Fetch data with graceful degradation and quality checking.

        This method wraps fetch_data() with error handling and data quality
        validation. Use this in production scenarios where robustness is
        more important than strict failure.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date
            strict: If True, raise on any errors; if False, attempt fallback

        Returns:
            Tuple of:
            - data: Dictionary of DataFrames (may be partial if not strict)
            - issues: List of DataQualityIssue objects describing any problems

        Example:
            indicator = get_indicator("my_indicator")
            data, issues = indicator.fetch_data_with_fallback("2015-01-01")

            for issue in issues:
                if issue.severity == DataQualitySeverity.ERROR:
                    print(f"ERROR: {issue.message}")
        """
        data: dict[str, pl.DataFrame] = {}
        issues: list[DataQualityIssue] = []

        try:
            data = self.fetch_data(start_date, end_date)
            issues.append(DataQualityIssue(
                severity=DataQualitySeverity.INFO,
                source="fetch_data",
                message=f"Successfully fetched {len(data)} data sources",
            ))
        except Exception as e:
            issue = DataQualityIssue(
                severity=DataQualitySeverity.ERROR,
                source="fetch_data",
                message=f"Data fetch failed: {e}",
            )
            issues.append(issue)

            if strict:
                raise

            # Attempt fallback
            data = self._get_fallback_data()
            if data:
                issues.append(DataQualityIssue(
                    severity=DataQualitySeverity.WARNING,
                    source="fetch_data",
                    message="Using fallback data due to fetch failure",
                ))

        # Run quality checks
        quality_issues = self._check_data_quality(data)
        issues.extend(quality_issues)

        return data, issues

    def _check_data_quality(
        self,
        data: dict[str, pl.DataFrame],
    ) -> list[DataQualityIssue]:
        """
        Check data quality and return issues.

        Performs the following checks:
        - Empty DataFrames
        - Missing required columns
        - Null value percentages
        - Date gaps in time series
        - Stale data detection

        Args:
            data: Dictionary of DataFrames to check

        Returns:
            List of DataQualityIssue objects
        """
        issues: list[DataQualityIssue] = []

        for source_name, df in data.items():
            # Check 1: Empty data
            if df.height == 0:
                issues.append(DataQualityIssue(
                    severity=DataQualitySeverity.ERROR,
                    source=source_name,
                    message="DataFrame is empty",
                ))
                continue

            # Check 2: Missing required columns
            if "date" not in df.columns:
                issues.append(DataQualityIssue(
                    severity=DataQualitySeverity.ERROR,
                    source=source_name,
                    message="Missing 'date' column",
                ))

            # Check 3: Null values
            null_counts = df.null_count()
            for col in null_counts.columns:
                null_count = null_counts[col][0]
                if null_count > 0:
                    null_pct = null_count / df.height
                    if null_pct > 0.2:  # >20% null
                        severity = DataQualitySeverity.ERROR
                    elif null_pct > 0.1:  # >10% null
                        severity = DataQualitySeverity.WARNING
                    else:
                        continue  # Skip low null percentages

                    issues.append(DataQualityIssue(
                        severity=severity,
                        source=source_name,
                        message=f"Column '{col}' has {null_pct:.1%} null values",
                        affected_records=null_count,
                    ))

            # Check 4: Date coverage
            if "date" in df.columns:
                try:
                    dates = df["date"].unique().sort()
                    if dates.height > 1:
                        expected_count = self._expected_observation_count(
                            dates[0], dates[-1]
                        )
                        if expected_count > 0 and dates.height < expected_count * 0.8:
                            issues.append(DataQualityIssue(
                                severity=DataQualitySeverity.WARNING,
                                source=source_name,
                                message=(
                                    f"Potential date gaps: {dates.height} observations, "
                                    f"expected ~{expected_count}"
                                ),
                            ))
                except Exception:
                    pass  # Skip if date parsing fails

            # Check 5: Stale data
            if "date" in df.columns:
                try:
                    latest_date = df["date"].max()
                    if latest_date is not None:
                        # Convert to date if needed
                        if hasattr(latest_date, "date"):
                            latest_date = latest_date.date()
                        elif isinstance(latest_date, str):
                            # Try parsing common formats
                            for fmt in ["%Y-%m-%d", "%Y-Q%q", "%Y-%m"]:
                                try:
                                    latest_date = datetime.strptime(
                                        latest_date[:10], "%Y-%m-%d"
                                    ).date()
                                    break
                                except ValueError:
                                    continue

                        if isinstance(latest_date, date):
                            age_days = (date.today() - latest_date).days

                            freq = self.get_metadata().update_frequency
                            max_age = {
                                "daily": 7,
                                "weekly": 14,
                                "monthly": 45,
                                "quarterly": 120,
                            }.get(freq, 30)

                            if age_days > max_age:
                                issues.append(DataQualityIssue(
                                    severity=DataQualitySeverity.WARNING,
                                    source=source_name,
                                    message=(
                                        f"Data is {age_days} days old "
                                        f"(expected <{max_age} days for {freq} data)"
                                    ),
                                ))
                except Exception:
                    pass  # Skip if date analysis fails

        return issues

    def _expected_observation_count(
        self,
        start_date: Any,
        end_date: Any,
    ) -> int:
        """
        Calculate expected number of observations between dates.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Expected observation count based on update_frequency
        """
        try:
            # Try to get days between dates
            if hasattr(start_date, "date"):
                start_date = start_date.date()
            if hasattr(end_date, "date"):
                end_date = end_date.date()

            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date[:10], "%Y-%m-%d").date()
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date[:10], "%Y-%m-%d").date()

            days = (end_date - start_date).days
            if days <= 0:
                return 0

            freq = self.get_metadata().update_frequency
            return {
                "daily": days,
                "weekly": days // 7,
                "monthly": days // 30,
                "quarterly": days // 90,
            }.get(freq, days // 30)
        except Exception:
            return 0

    def _get_fallback_data(self) -> dict[str, pl.DataFrame]:
        """
        Get fallback data when primary fetch fails.

        Override this method to provide fallback data sources such as:
        - Cached data from previous successful fetches
        - Sample/demo data for development
        - Stub data for testing

        Returns:
            Dictionary of fallback DataFrames (empty by default)
        """
        return {
            "bank_panel": pl.DataFrame(),
            "macro_data": pl.DataFrame(),
        }


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


@dataclass
class ForecastResult:
    """Container for forecast results."""

    target: str
    horizon: int
    predictions: pl.DataFrame
    confidence_intervals: Optional[pl.DataFrame] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "target": self.target,
            "horizon": self.horizon,
            "n_predictions": self.predictions.height,
            "metadata": self.metadata,
        }


class BaseForecastModel(ABC, Generic[M]):
    """
    Model-agnostic base class for forecasting.

    Works with any underlying model type: sklearn, statsmodels, PyTorch,
    custom implementations, etc. The generic type M represents the
    underlying model type.

    Design:
    - `fit()` and `predict()` are the only required methods
    - Optional methods for coefficients, feature importance, diagnostics
    - Supports both single-step and multi-horizon forecasting
    - Works with Polars DataFrames throughout

    Example with sklearn:
        class RandomForestForecaster(BaseForecastModel[RandomForestRegressor]):
            def __init__(self, **rf_params):
                self._model = RandomForestRegressor(**rf_params)

            def fit(self, data, target, features) -> dict[str, Any]:
                X = data.select(features).to_numpy()
                y = data[target].to_numpy()
                self._model.fit(X, y)
                return {"r2": self._model.score(X, y)}

            def predict(self, data, horizon) -> ForecastResult:
                # Implementation
                ...

    Example with statsmodels:
        class ARDLForecaster(BaseForecastModel[AutoReg]):
            def fit(self, data, target, features) -> dict[str, Any]:
                # Fit ARDL model
                ...

    Example with custom model:
        class CustomForecaster(BaseForecastModel[None]):
            # Use None for custom implementations without external model
            ...
    """

    def __init__(self):
        """Initialize the forecast model."""
        self._model: Optional[M] = None
        self._is_fitted: bool = False
        self._target: Optional[str] = None
        self._features: Optional[list[str]] = None
        self._fit_metadata: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted

    @property
    def model(self) -> Optional[M]:
        """Access the underlying model object."""
        return self._model

    @abstractmethod
    def fit(
        self,
        data: pl.DataFrame,
        target: str,
        features: list[str],
        **kwargs,
    ) -> dict[str, Any]:
        """
        Fit the model to training data.

        Args:
            data: Training data as Polars DataFrame
            target: Name of target variable column
            features: List of feature column names
            **kwargs: Model-specific parameters

        Returns:
            Dictionary with fit metrics (e.g., R², AIC, BIC, RMSE)
        """
        pass

    @abstractmethod
    def predict(
        self,
        data: pl.DataFrame,
        horizon: int = 1,
        **kwargs,
    ) -> ForecastResult:
        """
        Generate predictions for future periods.

        Args:
            data: Input data for prediction (may include exogenous forecasts)
            horizon: Number of periods to forecast
            **kwargs: Model-specific parameters (e.g., confidence level)

        Returns:
            ForecastResult containing predictions and optional confidence intervals
        """
        pass

    def get_coefficients(self) -> Optional[dict[str, float]]:
        """
        Return model coefficients if available.

        Not all models have interpretable coefficients (e.g., neural nets).
        Returns None if not applicable.

        Returns:
            Dictionary mapping feature names to coefficients, or None
        """
        return None

    def get_feature_importance(self) -> Optional[dict[str, float]]:
        """
        Return feature importance scores if available.

        For tree-based models, returns feature_importances_.
        For linear models, may return absolute coefficient values.
        Returns None if not applicable.

        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        return None

    def get_diagnostics(self) -> dict[str, Any]:
        """
        Return model diagnostics.

        Override to provide model-specific diagnostics such as:
        - Residual analysis
        - Autocorrelation tests
        - Stationarity tests
        - Cross-validation scores

        Returns:
            Dictionary with diagnostic information
        """
        return {
            "is_fitted": self._is_fitted,
            "target": self._target,
            "n_features": len(self._features) if self._features else 0,
        }

    def save(self, path: str | Path) -> None:
        """
        Save model to disk.

        Override for custom serialization. Default uses pickle if available.

        Args:
            path: File path to save model
        """
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BaseForecastModel":
        """
        Load model from disk.

        Args:
            path: File path to load model from

        Returns:
            Loaded model instance
        """
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def clone(self) -> "BaseForecastModel":
        """
        Create an unfitted clone of this model with same parameters.

        Useful for cross-validation and grid search.

        Returns:
            New unfitted instance with same configuration
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement clone(). "
            "Override to support cross-validation workflows."
        )


# =============================================================================
# Optional Pattern Base Classes
# =============================================================================
# These base classes standardize optional patterns (nowcast, backtest, viz)
# that are commonly used across indicators but not required.


@dataclass
class BacktestResult:
    """Container for backtest results."""

    spec_name: str
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    directional_accuracy: float  # Fraction of correct direction predictions
    n_predictions: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spec_name": self.spec_name,
            "mae": self.mae,
            "rmse": self.rmse,
            "directional_accuracy": self.directional_accuracy,
            "n_predictions": self.n_predictions,
            "metadata": self.metadata,
        }


class BaseNowcaster(ABC):
    """
    Base class for high-frequency nowcasting.

    Nowcasters update indicator estimates between quarterly releases
    using proxy variables (stock prices, CDS spreads, etc.).

    Temporal context:
        T-8 ←→ T-1         T (now)          T+1 ←→ T+4
        ───────────────    ────────         ────────────────
        calculate()        nowcast()         forecast()
        (historical)       (adjust for       (predict
                           current market)   future)

    Example:
        class MyNowcaster(BaseNowcaster):
            def nowcast(self, quarterly_data, proxy_data):
                # Use stock prices to adjust last quarterly value
                ...
                return IndicatorResult(...)
    """

    @abstractmethod
    def nowcast(
        self,
        quarterly_data: pl.DataFrame,
        proxy_data: dict[str, pl.DataFrame],
    ) -> IndicatorResult:
        """
        Produce nowcast using proxy variables.

        Args:
            quarterly_data: Latest quarterly indicator results
                           Expected columns: ticker, date, value
            proxy_data: High-frequency proxy variables
                       Common keys: "stock_data", "cds_data", "h8_data"

        Returns:
            IndicatorResult with nowcast values
        """
        pass

    def get_proxy_requirements(self) -> list[str]:
        """
        Return list of required proxy data sources.

        Override to document what proxy data this nowcaster needs.

        Returns:
            List of proxy data source names (e.g., ["stock_data", "cds_data"])
        """
        return []


class BaseBacktester(ABC):
    """
    Base class for model backtesting.

    Backtests validate forecasting performance using historical data
    with rolling-window or expanding-window methodologies.

    Example:
        class MyBacktester(BaseBacktester):
            def run_backtest(self, data, initial_window=20, step=1):
                # Rolling-window backtest
                for t in range(initial_window, len(data), step):
                    train = data[:t]
                    test = data[t:t+1]
                    # Fit and predict
                    ...
                return BacktestResult(...)
    """

    @abstractmethod
    def run_backtest(
        self,
        data: pl.DataFrame,
        initial_window: int = 20,
        step: int = 1,
    ) -> BacktestResult:
        """
        Run rolling-window backtest.

        Args:
            data: Historical data for backtesting
            initial_window: Initial training window size (in periods)
            step: Step size for rolling window

        Returns:
            BacktestResult with performance metrics
        """
        pass

    def get_metrics(self) -> dict[str, float]:
        """
        Get available performance metrics.

        Override to provide additional metrics beyond MAE/RMSE.

        Returns:
            Dictionary mapping metric names to values
        """
        return {}

    def compare_specs(
        self,
        data: pl.DataFrame,
        specs: list[Any],
        **kwargs,
    ) -> list[BacktestResult]:
        """
        Compare multiple specifications via backtesting.

        Args:
            data: Historical data
            specs: List of specification objects to compare
            **kwargs: Additional parameters for run_backtest

        Returns:
            List of BacktestResults, one per spec
        """
        results = []
        for spec in specs:
            result = self.run_backtest(data, **kwargs)
            result.spec_name = getattr(spec, "name", str(spec))
            results.append(result)
        return results


class BaseVisualizer(ABC):
    """
    Base class for indicator visualizations.

    Visualizers create numbered charts using Vega-Altair for storytelling.
    Chart numbering helps with reproducibility and references.

    Convention:
        - chart_1_*: Overview/summary chart
        - chart_2_*: Time series detail
        - chart_3_*: Cross-sectional comparison
        - chart_4+_*: Additional analysis

    Example:
        class MyVisualizer(BaseVisualizer):
            def chart_1_overview(self):
                return alt.Chart(self.data).mark_bar()...

            def chart_2_time_series(self, ticker):
                return alt.Chart(self.data.filter(...)).mark_line()...
    """

    def __init__(self, data: pl.DataFrame):
        """
        Initialize visualizer with data.

        Args:
            data: Indicator results to visualize
        """
        self.data = data

    @abstractmethod
    def chart_1_overview(self) -> Any:
        """
        Chart 1: High-level overview of indicator values.

        Typically shows indicator values across all banks or time periods.

        Returns:
            Altair Chart object
        """
        pass

    def get_all_charts(self) -> dict[str, Any]:
        """
        Get all available charts.

        Returns:
            Dictionary mapping chart names to Chart objects
        """
        import inspect

        charts = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("chart_"):
                try:
                    # Only call methods that take no additional required args
                    sig = inspect.signature(method)
                    required_params = [
                        p for p in sig.parameters.values()
                        if p.default == inspect.Parameter.empty
                        and p.name != "self"
                    ]
                    if not required_params:
                        charts[name] = method()
                except Exception:
                    pass  # Skip charts that fail
        return charts

    def save_all_charts(
        self,
        output_dir: Union[str, Path],
        format: str = "html",
    ) -> list[Path]:
        """
        Save all charts to files.

        Args:
            output_dir: Directory to save charts
            format: Output format ("html", "png", "svg", "pdf")

        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved = []
        for name, chart in self.get_all_charts().items():
            file_path = output_path / f"{name}.{format}"
            if hasattr(chart, "save"):
                chart.save(str(file_path))
                saved.append(file_path)
        return saved


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
