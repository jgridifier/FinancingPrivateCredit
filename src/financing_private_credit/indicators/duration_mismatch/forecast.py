"""
Duration Mismatch Forecasting with ARDL Models

Primary approach: ARDL (Autoregressive Distributed Lag)
- Well-suited for lead-lag relationships
- Handles mixed I(0)/I(1) variables
- Error correction mechanism captures adjustment dynamics

Secondary: APLR for non-linear sensitivity analysis

The key prediction:
    future_volatility = f(current_duration_exposure, yield_changes, lags)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import polars as pl
from scipy import stats

from .indicator import DurationMismatchSpec


@dataclass
class ARDLSpec:
    """Specification for ARDL model."""

    name: str

    # Dependent variable
    target: str = "earnings_volatility"  # or "stock_volatility"

    # AR lags for dependent variable
    ar_lags: list[int] = field(default_factory=lambda: [1, 2, 4])

    # Distributed lag structure for predictors
    predictors: list[str] = field(default_factory=lambda: [
        "predicted_impact_pct",
        "yield_change",
        "duration_zscore",
    ])

    predictor_lags: dict[str, list[int]] = field(default_factory=lambda: {
        "predicted_impact_pct": [0, 1, 2],
        "yield_change": [0, 1, 2, 4],
        "duration_zscore": [0, 1],
    })

    # Model settings
    include_constant: bool = True
    include_trend: bool = False
    use_robust_se: bool = True

    # Training
    min_observations: int = 20


@dataclass
class ARDLForecastResult:
    """Results from ARDL forecast."""

    ticker: str
    forecast_date: datetime
    horizon: int

    # Point forecasts and intervals
    point_forecast: float
    lower_bound: float  # 95% CI
    upper_bound: float

    # Model diagnostics
    r_squared: float
    durbin_watson: float
    residual_std: float

    # Long-run multipliers (cumulative effect)
    long_run_multipliers: dict[str, float]

    metadata: dict[str, Any] = field(default_factory=dict)


class ARDLModel:
    """
    Autoregressive Distributed Lag Model for volatility prediction.

    Model: y_t = c + Σ(φ_i * y_{t-i}) + Σ(β_j * x_{t-j}) + ε_t

    Key advantages:
    1. Captures lead-lag dynamics between duration exposure and volatility
    2. Error correction form reveals adjustment speed
    3. Works with non-stationary data under cointegration
    """

    def __init__(self, spec: ARDLSpec):
        self.spec = spec
        self.coefficients: Optional[np.ndarray] = None
        self.se: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
        self.feature_names: list[str] = []
        self._fitted = False
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def _build_features(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Build feature matrix with lagged variables.

        Returns:
            X: Feature matrix
            y: Target vector
            feature_names: Names for each feature
        """
        result = df.clone().sort("date")
        feature_names = []

        # Add AR lags of target
        for lag in self.spec.ar_lags:
            col_name = f"{self.spec.target}_lag{lag}"
            result = result.with_columns(
                pl.col(self.spec.target).shift(lag).alias(col_name)
            )
            feature_names.append(col_name)

        # Add distributed lags of predictors
        for predictor in self.spec.predictors:
            if predictor not in result.columns:
                continue

            lags = self.spec.predictor_lags.get(predictor, [0])
            for lag in lags:
                if lag == 0:
                    feature_names.append(predictor)
                else:
                    col_name = f"{predictor}_lag{lag}"
                    result = result.with_columns(
                        pl.col(predictor).shift(lag).alias(col_name)
                    )
                    feature_names.append(col_name)

        # Add constant
        if self.spec.include_constant:
            result = result.with_columns(pl.lit(1.0).alias("const"))
            feature_names = ["const"] + feature_names

        # Add trend
        if self.spec.include_trend:
            result = result.with_columns(
                pl.arange(0, result.height).alias("trend")
            )
            feature_names.append("trend")

        # Drop rows with NaN
        all_cols = feature_names + [self.spec.target]
        available_cols = [c for c in all_cols if c in result.columns]
        result = result.drop_nulls(subset=available_cols)

        if result.height < self.spec.min_observations:
            raise ValueError(
                f"Insufficient observations: {result.height} < {self.spec.min_observations}"
            )

        # Extract arrays
        available_features = [f for f in feature_names if f in result.columns]
        X = result.select(available_features).to_numpy()
        y = result.select(self.spec.target).to_numpy().flatten()

        return X, y, available_features

    def fit(self, df: pl.DataFrame) -> dict[str, float]:
        """
        Fit ARDL model using OLS.

        Args:
            df: DataFrame with target and predictor variables

        Returns:
            Dictionary with fit statistics
        """
        X, y, feature_names = self._build_features(df)
        self.feature_names = feature_names

        # Store for later prediction
        self._y_mean = np.mean(y)
        self._y_std = np.std(y)

        # OLS estimation
        try:
            # Normal equations: β = (X'X)^{-1} X'y
            XtX = X.T @ X
            Xty = X.T @ y

            # Add ridge regularization for stability
            ridge = 1e-6 * np.eye(XtX.shape[0])
            self.coefficients = np.linalg.solve(XtX + ridge, Xty)

            # Residuals and variance
            y_pred = X @ self.coefficients
            self.residuals = y - y_pred
            n, k = X.shape
            sigma2 = np.sum(self.residuals ** 2) / (n - k)

            # Standard errors
            if self.spec.use_robust_se:
                # Heteroskedasticity-robust (HC0)
                omega = np.diag(self.residuals ** 2)
                bread = np.linalg.inv(XtX + ridge)
                meat = X.T @ omega @ X
                var_matrix = bread @ meat @ bread
            else:
                var_matrix = sigma2 * np.linalg.inv(XtX + ridge)

            self.se = np.sqrt(np.diag(var_matrix))

            # R-squared
            ss_res = np.sum(self.residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Durbin-Watson statistic
            dw = np.sum(np.diff(self.residuals) ** 2) / ss_res if ss_res > 0 else 2.0

            self._fitted = True

            return {
                "r_squared": float(r_squared),
                "adj_r_squared": float(1 - (1 - r_squared) * (n - 1) / (n - k - 1)),
                "durbin_watson": float(dw),
                "residual_std": float(np.sqrt(sigma2)),
                "n_observations": n,
                "n_parameters": k,
            }

        except np.linalg.LinAlgError as e:
            raise ValueError(f"Model fitting failed: {e}")

    def predict(
        self,
        df: pl.DataFrame,
        horizon: int = 4,
    ) -> list[ARDLForecastResult]:
        """
        Generate multi-step ahead forecasts.

        Uses iterated forecasting for multi-step ahead.

        Args:
            df: DataFrame with current data
            horizon: Number of periods ahead

        Returns:
            List of forecast results
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get latest data point
        latest = df.sort("date").tail(1)
        ticker = latest["ticker"][0] if "ticker" in latest.columns else "UNKNOWN"
        last_date = latest["date"][0]

        results = []

        # Build initial state
        X_last, y_last, _ = self._build_features(df)
        if len(X_last) == 0:
            return results

        # Current prediction state
        y_history = y_last[-max(self.spec.ar_lags):].tolist()
        x_current = X_last[-1].copy()

        # Prediction error variance (for intervals)
        sigma = np.std(self.residuals) if self.residuals is not None else self._y_std

        for h in range(1, horizon + 1):
            # Point forecast
            y_pred = float(x_current @ self.coefficients)

            # Prediction interval (grows with horizon)
            interval_width = 1.96 * sigma * np.sqrt(h)

            # Compute long-run multipliers
            long_run = self._compute_long_run_multipliers()

            results.append(ARDLForecastResult(
                ticker=ticker,
                forecast_date=last_date,
                horizon=h,
                point_forecast=y_pred,
                lower_bound=y_pred - interval_width,
                upper_bound=y_pred + interval_width,
                r_squared=1 - np.sum(self.residuals ** 2) / np.sum((y_last - np.mean(y_last)) ** 2),
                durbin_watson=float(np.sum(np.diff(self.residuals) ** 2) / np.sum(self.residuals ** 2)),
                residual_std=sigma,
                long_run_multipliers=long_run,
            ))

            # Update for next iteration (shift AR lags)
            y_history.append(y_pred)
            y_history = y_history[-max(self.spec.ar_lags):]

            # Update x_current with new y lags
            for i, lag in enumerate(self.spec.ar_lags):
                lag_idx = len(y_history) - lag
                if lag_idx >= 0:
                    # Find position in feature array
                    feature_name = f"{self.spec.target}_lag{lag}"
                    if feature_name in self.feature_names:
                        pos = self.feature_names.index(feature_name)
                        x_current[pos] = y_history[lag_idx]

        return results

    def _compute_long_run_multipliers(self) -> dict[str, float]:
        """
        Compute long-run multipliers for each predictor.

        Long-run multiplier = Σ(β_j) / (1 - Σ(φ_i))
        """
        if self.coefficients is None:
            return {}

        # Sum of AR coefficients
        ar_sum = 0.0
        for lag in self.spec.ar_lags:
            name = f"{self.spec.target}_lag{lag}"
            if name in self.feature_names:
                idx = self.feature_names.index(name)
                ar_sum += self.coefficients[idx]

        denominator = 1 - ar_sum
        if abs(denominator) < 0.01:  # Near unit root
            denominator = 0.01

        # Long-run multipliers for each predictor
        multipliers = {}
        for predictor in self.spec.predictors:
            lags = self.spec.predictor_lags.get(predictor, [0])
            predictor_sum = 0.0

            for lag in lags:
                if lag == 0:
                    name = predictor
                else:
                    name = f"{predictor}_lag{lag}"

                if name in self.feature_names:
                    idx = self.feature_names.index(name)
                    predictor_sum += self.coefficients[idx]

            multipliers[predictor] = predictor_sum / denominator

        return multipliers

    def get_coefficients(self) -> pl.DataFrame:
        """Return coefficients with standard errors and t-stats."""
        if self.coefficients is None:
            return pl.DataFrame()

        t_stats = self.coefficients / self.se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(self.residuals) - len(self.coefficients)))

        return pl.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.coefficients,
            "std_error": self.se,
            "t_statistic": t_stats,
            "p_value": p_values,
        })


class DurationMismatchForecaster:
    """
    High-level forecaster for duration mismatch volatility prediction.

    Combines:
    1. ARDL for main prediction
    2. Scenario analysis for rate shocks
    3. Cross-bank comparison
    """

    # Predefined rate scenarios
    RATE_SCENARIOS = {
        "baseline": 0.0,        # No change
        "gradual_rise_25bp": 0.25,
        "gradual_rise_50bp": 0.50,
        "sharp_rise_100bp": 1.00,
        "rate_shock_200bp": 2.00,
        "gradual_fall_25bp": -0.25,
        "sharp_fall_100bp": -1.00,
    }

    def __init__(
        self,
        duration_data: pl.DataFrame,
        spec: Optional[DurationMismatchSpec] = None,
    ):
        """
        Initialize forecaster.

        Args:
            duration_data: DataFrame with duration exposure and volatility
            spec: Duration mismatch specification
        """
        self.duration_data = duration_data
        self.spec = spec or DurationMismatchSpec(name="default", description="Default")
        self.models: dict[str, ARDLModel] = {}
        self._fitted = False

    def fit(self, target: str = "earnings_volatility") -> dict[str, dict[str, float]]:
        """
        Fit ARDL models for each bank.

        Args:
            target: Target variable to predict

        Returns:
            Dictionary of fit statistics by bank
        """
        fit_stats = {}

        ardl_spec = ARDLSpec(
            name="duration_volatility",
            target=target,
        )

        for ticker in self.duration_data["ticker"].unique().to_list():
            bank_df = self.duration_data.filter(pl.col("ticker") == ticker).sort("date")

            if bank_df.height < ardl_spec.min_observations:
                print(f"Skipping {ticker}: insufficient data ({bank_df.height} obs)")
                continue

            if target not in bank_df.columns or bank_df[target].null_count() == bank_df.height:
                print(f"Skipping {ticker}: no {target} data")
                continue

            try:
                model = ARDLModel(ardl_spec)
                stats = model.fit(bank_df)
                self.models[ticker] = model
                fit_stats[ticker] = stats
            except Exception as e:
                print(f"Warning: Could not fit model for {ticker}: {e}")

        self._fitted = True
        return fit_stats

    def forecast(
        self,
        ticker: str,
        horizon: int = 4,
    ) -> list[ARDLForecastResult]:
        """
        Generate forecasts for a single bank.

        Args:
            ticker: Bank ticker
            horizon: Forecast horizon in quarters

        Returns:
            List of forecast results
        """
        if ticker not in self.models:
            raise ValueError(f"No model fitted for {ticker}")

        bank_df = self.duration_data.filter(pl.col("ticker") == ticker)
        return self.models[ticker].predict(bank_df, horizon)

    def scenario_analysis(
        self,
        rate_change: float,
        horizon: int = 4,
    ) -> pl.DataFrame:
        """
        Analyze impact of a rate shock on all banks.

        Args:
            rate_change: Change in rates (percentage points)
            horizon: Forecast horizon

        Returns:
            DataFrame with scenario impacts by bank
        """
        results = []

        for ticker, model in self.models.items():
            bank_df = self.duration_data.filter(pl.col("ticker") == ticker)

            if bank_df.height == 0:
                continue

            # Get current state
            latest = bank_df.sort("date").tail(1)

            # Baseline forecast
            baseline = model.predict(bank_df, horizon)

            # Shocked forecast (adjust predicted_impact)
            shocked_df = bank_df.clone()
            if "predicted_impact_pct" in shocked_df.columns:
                # Impact scales with rate change and duration
                duration = latest["estimated_duration"][0] if "estimated_duration" in latest.columns else 5.0
                shock_impact = -duration * rate_change / 100

                shocked_df = shocked_df.with_columns(
                    (pl.col("predicted_impact_pct") + shock_impact).alias("predicted_impact_pct")
                )

            # Refit temporarily with shocked data
            try:
                shocked_model = ARDLModel(model.spec)
                shocked_model.fit(shocked_df)
                shocked_forecast = shocked_model.predict(shocked_df, horizon)

                for h in range(horizon):
                    results.append({
                        "ticker": ticker,
                        "horizon": h + 1,
                        "baseline_vol": baseline[h].point_forecast,
                        "shocked_vol": shocked_forecast[h].point_forecast,
                        "vol_change": shocked_forecast[h].point_forecast - baseline[h].point_forecast,
                        "rate_change_bp": rate_change * 100,
                    })
            except Exception:
                continue

        return pl.DataFrame(results)

    def cross_bank_comparison(
        self,
        horizon: int = 4,
    ) -> pl.DataFrame:
        """
        Compare forecasts across all banks.

        Returns:
            DataFrame with forecasts and rankings
        """
        all_forecasts = []

        for ticker, model in self.models.items():
            bank_df = self.duration_data.filter(pl.col("ticker") == ticker)

            try:
                forecasts = model.predict(bank_df, horizon)

                # Get latest duration data
                latest = bank_df.sort("date").tail(1)
                duration = latest["estimated_duration"][0] if "estimated_duration" in latest.columns else None
                dv01 = latest["dv01"][0] if "dv01" in latest.columns else None

                for f in forecasts:
                    all_forecasts.append({
                        "ticker": ticker,
                        "horizon": f.horizon,
                        "forecast_volatility": f.point_forecast,
                        "lower_bound": f.lower_bound,
                        "upper_bound": f.upper_bound,
                        "duration": duration,
                        "dv01": dv01,
                        "r_squared": f.r_squared,
                    })

            except Exception:
                continue

        df = pl.DataFrame(all_forecasts)

        # Add rankings
        if df.height > 0:
            df = df.with_columns(
                pl.col("forecast_volatility")
                .rank(descending=True)
                .over("horizon")
                .alias("volatility_rank")
            )

        return df

    def long_run_effects(self) -> pl.DataFrame:
        """
        Summarize long-run multipliers across banks.

        Shows how each bank responds to sustained changes in predictors.
        """
        results = []

        for ticker, model in self.models.items():
            multipliers = model._compute_long_run_multipliers()

            results.append({
                "ticker": ticker,
                **multipliers,
            })

        return pl.DataFrame(results)
