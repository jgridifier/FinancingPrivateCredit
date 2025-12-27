"""
Provision rate forecasting models using APLR (Automatic Piecewise Linear Regression).

APLR from interpretml provides interpretable, automatic feature engineering with
piecewise linear relationships - suitable for financial time series with non-linear
patterns and regime changes.

Key features:
- Automatic piecewise linear fitting
- Handles seasonality through explicit seasonal features
- Configurable via JSON specification files
- Built-in backtesting support
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

try:
    from interpret.glassbox import APLRRegressor
    APLR_AVAILABLE = True
except ImportError:
    APLR_AVAILABLE = False
    print("Warning: interpret-ml not installed. Install with: pip install interpret")


@dataclass
class ModelSpecification:
    """
    Specification for a provision rate forecasting model.

    Can be loaded from JSON for easy experimentation.
    """

    name: str
    description: str

    # Target variable
    target: str = "provision_rate"

    # Feature configuration
    ar_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    exogenous_vars: list[str] = field(default_factory=list)
    exogenous_lags: dict[str, list[int]] = field(default_factory=dict)

    # Seasonality
    include_seasonality: bool = True
    seasonality_period: int = 4  # Quarterly = 4, Monthly = 12

    # Model parameters
    max_bins: int = 10  # APLR max bins per feature
    min_observations_in_split: int = 5
    max_interaction_level: int = 1  # 0 = no interactions, 1 = pairwise

    # Training
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    holdout_periods: int = 4  # Quarters to hold out for validation

    # Bank-specific
    bank_fixed_effects: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelSpecification":
        """Load specification from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path):
        """Save specification to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class ForecastResult:
    """Results from a forecast model."""

    bank: str
    forecast_date: datetime
    horizon: int
    point_forecast: float
    lower_bound: float
    upper_bound: float
    model_name: str


@dataclass
class BacktestResult:
    """Results from backtesting a model specification."""

    spec_name: str
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    n_forecasts: int
    by_bank: dict[str, dict[str, float]] = field(default_factory=dict)
    by_horizon: dict[int, dict[str, float]] = field(default_factory=dict)


class SeasonalFeatureGenerator:
    """
    Generate seasonal features for time series.

    Supports multiple approaches to handling seasonality:
    1. Fourier features (sine/cosine at different frequencies)
    2. Quarter/month dummies
    3. Seasonal differencing indicators
    """

    def __init__(self, period: int = 4, method: str = "fourier"):
        """
        Initialize the seasonal feature generator.

        Args:
            period: Seasonality period (4 for quarterly, 12 for monthly)
            method: 'fourier', 'dummies', or 'both'
        """
        self.period = period
        self.method = method

    def generate(self, df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
        """
        Add seasonal features to DataFrame.

        Args:
            df: DataFrame with date column
            date_col: Name of date column

        Returns:
            DataFrame with seasonal features added
        """
        result = df.clone()

        # Extract time index
        if self.period == 4:
            # Quarterly: use quarter of year
            result = result.with_columns(
                pl.col(date_col).dt.quarter().alias("quarter"),
                (pl.col(date_col).dt.year() * 4 + pl.col(date_col).dt.quarter()).alias("time_idx")
            )
        else:
            # Monthly: use month of year
            result = result.with_columns(
                pl.col(date_col).dt.month().alias("month"),
                (pl.col(date_col).dt.year() * 12 + pl.col(date_col).dt.month()).alias("time_idx")
            )

        if self.method in ["fourier", "both"]:
            # Add Fourier features
            n_harmonics = min(2, self.period // 2)  # Up to 2 harmonics

            for k in range(1, n_harmonics + 1):
                if self.period == 4:
                    result = result.with_columns([
                        (2 * np.pi * k * pl.col("quarter") / self.period).sin().alias(f"sin_{k}"),
                        (2 * np.pi * k * pl.col("quarter") / self.period).cos().alias(f"cos_{k}"),
                    ])
                else:
                    result = result.with_columns([
                        (2 * np.pi * k * pl.col("month") / self.period).sin().alias(f"sin_{k}"),
                        (2 * np.pi * k * pl.col("month") / self.period).cos().alias(f"cos_{k}"),
                    ])

        if self.method in ["dummies", "both"]:
            # Add seasonal dummies (leaving one out to avoid collinearity)
            if self.period == 4:
                for q in range(1, 4):  # Q1, Q2, Q3 (Q4 is reference)
                    result = result.with_columns(
                        (pl.col("quarter") == q).cast(pl.Float64).alias(f"Q{q}")
                    )
            else:
                for m in range(1, 12):  # M1-M11 (M12 is reference)
                    result = result.with_columns(
                        (pl.col("month") == m).cast(pl.Float64).alias(f"M{m}")
                    )

        return result


class APLRForecaster:
    """
    Provision rate forecaster using APLR (Automatic Piecewise Linear Regression).

    APLR is from the interpretml package and provides:
    - Automatic piecewise linear fitting
    - Interpretable feature importance
    - Built-in regularization
    """

    def __init__(self, spec: ModelSpecification):
        """
        Initialize the APLR forecaster.

        Args:
            spec: Model specification
        """
        if not APLR_AVAILABLE:
            raise ImportError(
                "interpret-ml is required for APLRForecaster. "
                "Install with: pip install interpret"
            )

        self.spec = spec
        self.seasonal_gen = SeasonalFeatureGenerator(
            period=spec.seasonality_period,
            method="both" if spec.include_seasonality else "fourier"
        )
        self.models: dict[str, APLRRegressor] = {}
        self._feature_names: list[str] = []

    def _build_features(
        self,
        df: pl.DataFrame,
        bank: str,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Build feature matrix for a single bank.

        Args:
            df: Panel data with all banks
            bank: Bank ticker to build features for

        Returns:
            Tuple of (X, y, feature_names)
        """
        # Filter to bank
        bank_df = df.filter(pl.col("ticker") == bank).sort("date")

        if bank_df.height < 10:
            raise ValueError(f"Insufficient data for {bank}: {bank_df.height} rows")

        # Add seasonal features
        bank_df = self.seasonal_gen.generate(bank_df)

        # Build feature list
        features = []
        feature_names = []

        # AR lags of target
        for lag in self.spec.ar_lags:
            col_name = f"{self.spec.target}_lag{lag}"
            bank_df = bank_df.with_columns(
                pl.col(self.spec.target).shift(lag).alias(col_name)
            )
            features.append(col_name)
            feature_names.append(col_name)

        # Exogenous variables and their lags
        for var in self.spec.exogenous_vars:
            if var in bank_df.columns:
                # Current value
                features.append(var)
                feature_names.append(var)

                # Lags if specified
                lags = self.spec.exogenous_lags.get(var, [])
                for lag in lags:
                    col_name = f"{var}_lag{lag}"
                    bank_df = bank_df.with_columns(
                        pl.col(var).shift(lag).alias(col_name)
                    )
                    features.append(col_name)
                    feature_names.append(col_name)

        # Seasonal features
        if self.spec.include_seasonality:
            seasonal_cols = [c for c in bank_df.columns if c.startswith(("sin_", "cos_", "Q", "M"))]
            features.extend(seasonal_cols)
            feature_names.extend(seasonal_cols)

        # Drop rows with missing values
        bank_df = bank_df.drop_nulls(subset=features + [self.spec.target])

        if bank_df.height < 10:
            raise ValueError(f"Insufficient data after feature engineering for {bank}")

        # Extract arrays
        X = bank_df.select(features).to_numpy()
        y = bank_df.select(self.spec.target).to_numpy().flatten()

        return X, y, feature_names

    def fit(self, panel: pl.DataFrame) -> dict[str, float]:
        """
        Fit APLR models for each bank.

        Args:
            panel: Panel data with columns [date, ticker, provision_rate, ...]

        Returns:
            Dictionary of R-squared scores by bank
        """
        scores = {}

        for bank in panel["ticker"].unique().to_list():
            try:
                X, y, feature_names = self._build_features(panel, bank)
                self._feature_names = feature_names

                # Split into train/holdout
                n = len(y)
                train_end = n - self.spec.holdout_periods

                X_train, y_train = X[:train_end], y[:train_end]

                # Fit APLR
                model = APLRRegressor(
                    max_bins=self.spec.max_bins,
                    min_observations_in_split=self.spec.min_observations_in_split,
                    max_interaction_level=self.spec.max_interaction_level,
                )

                model.fit(X_train, y_train)
                self.models[bank] = model

                # Score on full data
                y_pred = model.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                scores[bank] = r2

            except Exception as e:
                print(f"Warning: Could not fit model for {bank}: {e}")
                scores[bank] = None

        return scores

    def predict(
        self,
        panel: pl.DataFrame,
        bank: str,
        horizon: int = 4,
    ) -> list[ForecastResult]:
        """
        Generate forecasts for a bank.

        Args:
            panel: Panel data
            bank: Bank ticker
            horizon: Number of periods to forecast

        Returns:
            List of ForecastResult objects
        """
        if bank not in self.models:
            raise ValueError(f"No model fitted for {bank}")

        model = self.models[bank]

        # Get latest data point
        bank_df = panel.filter(pl.col("ticker") == bank).sort("date")
        bank_df = self.seasonal_gen.generate(bank_df)

        # Build initial feature vector
        X_last, _, _ = self._build_features(panel, bank)
        x_current = X_last[-1:].copy()

        results = []
        last_date = bank_df["date"].max()

        for h in range(1, horizon + 1):
            # Predict
            y_pred = model.predict(x_current)[0]

            # Estimate prediction interval (simplified - uses historical residuals)
            # In practice, you'd want proper prediction intervals
            std_est = np.std(model.predict(X_last) - bank_df.select(self.spec.target).to_numpy().flatten()[-len(X_last):])

            results.append(ForecastResult(
                bank=bank,
                forecast_date=last_date,
                horizon=h,
                point_forecast=y_pred,
                lower_bound=y_pred - 1.96 * std_est,
                upper_bound=y_pred + 1.96 * std_est,
                model_name=self.spec.name,
            ))

            # Update feature vector for next step (shift AR lags)
            # This is a simplified version - proper implementation would
            # recursively update all features
            for i, lag in enumerate(self.spec.ar_lags):
                if i < len(self.spec.ar_lags) - 1:
                    x_current[0, i] = x_current[0, i + 1]
                else:
                    x_current[0, i] = y_pred

        return results

    def get_feature_importance(self, bank: str) -> dict[str, float]:
        """
        Get feature importance for a bank's model.

        Args:
            bank: Bank ticker

        Returns:
            Dictionary of feature name -> importance score
        """
        if bank not in self.models:
            raise ValueError(f"No model fitted for {bank}")

        model = self.models[bank]

        # APLR provides feature importance through explain_global
        try:
            explanation = model.explain_global()
            importances = dict(zip(self._feature_names, explanation.data()["scores"]))
            return importances
        except Exception:
            # Fallback to simple coefficient analysis
            return {}


class FallbackForecaster:
    """
    Simple mean-reversion forecaster when APLR is not available.

    Uses the same specification format for consistency.
    """

    def __init__(self, spec: ModelSpecification):
        self.spec = spec
        self.bank_means: dict[str, float] = {}
        self.bank_stds: dict[str, float] = {}

    def fit(self, panel: pl.DataFrame) -> dict[str, float]:
        """Fit simple mean models for each bank."""
        scores = {}

        for bank in panel["ticker"].unique().to_list():
            bank_df = panel.filter(pl.col("ticker") == bank)

            if self.spec.target in bank_df.columns:
                values = bank_df.select(self.spec.target).drop_nulls().to_numpy().flatten()
                if len(values) > 0:
                    self.bank_means[bank] = float(np.mean(values))
                    self.bank_stds[bank] = float(np.std(values))
                    scores[bank] = 0.0  # R-squared of mean model is 0

        return scores

    def predict(
        self,
        panel: pl.DataFrame,
        bank: str,
        horizon: int = 4,
    ) -> list[ForecastResult]:
        """Generate mean-reversion forecasts."""
        if bank not in self.bank_means:
            raise ValueError(f"No model fitted for {bank}")

        bank_df = panel.filter(pl.col("ticker") == bank).sort("date")
        last_row = bank_df.filter(pl.col(self.spec.target).is_not_null()).tail(1)

        if last_row.height == 0:
            raise ValueError(f"No valid data for {bank}")

        last_date = last_row["date"][0]
        last_value = last_row[self.spec.target][0]
        mean_value = self.bank_means[bank]
        std_value = self.bank_stds[bank]

        results = []
        decay = 0.9

        for h in range(1, horizon + 1):
            # Mean reversion with decay
            forecast = decay ** h * last_value + (1 - decay ** h) * mean_value

            results.append(ForecastResult(
                bank=bank,
                forecast_date=last_date,
                horizon=h,
                point_forecast=forecast,
                lower_bound=forecast - 1.96 * std_value,
                upper_bound=forecast + 1.96 * std_value,
                model_name=f"{self.spec.name}_fallback",
            ))

        return results


def get_forecaster(spec: ModelSpecification):
    """
    Get appropriate forecaster based on available packages.

    Args:
        spec: Model specification

    Returns:
        APLRForecaster if available, otherwise FallbackForecaster
    """
    if APLR_AVAILABLE:
        return APLRForecaster(spec)
    else:
        print("Warning: Using fallback forecaster. Install interpret-ml for APLR.")
        return FallbackForecaster(spec)


if __name__ == "__main__":
    # Test specification creation
    spec = ModelSpecification(
        name="baseline_aplr",
        description="Baseline APLR model with quarterly seasonality",
        target="provision_rate",
        ar_lags=[1, 2, 3, 4],
        exogenous_vars=["lis", "loan_growth_yoy"],
        exogenous_lags={"lis": [4, 8]},
        include_seasonality=True,
        seasonality_period=4,
    )

    # Save example spec
    spec.to_json("example_spec.json")
    print("Saved example specification to example_spec.json")

    # Test loading
    loaded_spec = ModelSpecification.from_json("example_spec.json")
    print(f"Loaded spec: {loaded_spec.name}")
