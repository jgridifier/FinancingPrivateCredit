"""
Bank-Specific Macro Sensitivity Indicator

Measures heterogeneous bank elasticities to macroeconomic variables using APLR.

Model:
    NIM_i,t = f(rate_spread_t, output_gap_t, inflation_t, ...) + bank_effects_i + ε

Where f() is estimated using APLR to capture:
- Non-linear relationships (e.g., NIM compression at low rates)
- Regime-dependent effects (crisis vs. normal periods)
- Interaction effects between macro variables

Key outputs:
- β_i_rates: Bank i's sensitivity to rate spread
- γ_i_macro: Bank i's sensitivity to output gap
- Regime indicators: Which banks outperform in different environments
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

from ..base import (
    BaseIndicator,
    IndicatorMetadata,
    IndicatorResult,
    register_indicator,
)

try:
    from interpret.glassbox import APLRRegressor
    APLR_AVAILABLE = True
except ImportError:
    APLR_AVAILABLE = False


@dataclass
class MacroSensitivitySpec:
    """
    Specification for macro sensitivity model.

    Load from JSON for easy experimentation with different model configurations.
    """

    name: str
    description: str

    # Target variable
    target: str = "nim"  # Net Interest Margin

    # Macro features
    macro_features: list[str] = field(default_factory=lambda: [
        "rate_spread",      # Bond yield - loan rate proxy
        "output_gap",       # GDP - Potential GDP
        "inflation_yoy",    # YoY CPI
        "term_spread",      # 10Y - 2Y Treasury
        "credit_spread",    # Baa - Treasury spread
        "vix",              # Financial stress
    ])

    # Lags for each macro feature
    macro_lags: dict[str, list[int]] = field(default_factory=lambda: {
        "rate_spread": [0, 1, 2],
        "output_gap": [0, 1, 4],
        "inflation_yoy": [0, 1],
    })

    # APLR model parameters
    max_bins: int = 8           # Piecewise linear segments
    min_observations_in_split: int = 10
    max_interaction_level: int = 2  # Allow pairwise interactions

    # Include bank fixed effects
    bank_fixed_effects: bool = True

    # Training configuration
    min_observations_per_bank: int = 20
    holdout_periods: int = 4

    @classmethod
    def from_json(cls, path: str | Path) -> "MacroSensitivitySpec":
        """Load specification from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        """Save specification to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class BankSensitivity:
    """Container for a bank's macro sensitivities."""

    ticker: str
    name: str

    # Rate sensitivity (β coefficient)
    rate_sensitivity: float
    rate_sensitivity_se: Optional[float] = None

    # Output gap sensitivity (γ coefficient)
    output_gap_sensitivity: float
    output_gap_sensitivity_se: Optional[float] = None

    # Other sensitivities
    inflation_sensitivity: float = 0.0
    credit_spread_sensitivity: float = 0.0

    # Model quality metrics
    r_squared: float = 0.0
    n_observations: int = 0

    # Regime classification
    rate_regime_advantage: str = "neutral"  # "rising", "falling", "neutral"
    macro_regime_advantage: str = "neutral"  # "expansion", "recession", "neutral"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "rate_sensitivity": self.rate_sensitivity,
            "output_gap_sensitivity": self.output_gap_sensitivity,
            "inflation_sensitivity": self.inflation_sensitivity,
            "credit_spread_sensitivity": self.credit_spread_sensitivity,
            "r_squared": self.r_squared,
            "rate_regime_advantage": self.rate_regime_advantage,
            "macro_regime_advantage": self.macro_regime_advantage,
        }


class APLRSensitivityModel:
    """
    APLR-based model for estimating bank macro sensitivities.

    Uses Automatic Piecewise Linear Regression from interpretml to:
    1. Automatically find breakpoints in relationships
    2. Capture non-linear effects (e.g., NIM compression at ZLB)
    3. Model interactions between macro variables
    """

    def __init__(self, spec: MacroSensitivitySpec):
        if not APLR_AVAILABLE:
            raise ImportError(
                "interpret-ml is required for APLR models. "
                "Install with: pip install interpret"
            )

        self.spec = spec
        self.models: dict[str, APLRRegressor] = {}
        self.feature_names: list[str] = []
        self.sensitivities: dict[str, BankSensitivity] = {}
        self._is_fitted = False

    def _build_features(
        self,
        bank_df: pl.DataFrame,
        macro_df: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Build feature matrix by joining bank and macro data.

        Returns:
            Tuple of (X, y, feature_names)
        """
        # Merge bank data with macro data on date
        merged = bank_df.join(macro_df, on="date", how="inner")

        features = []
        feature_names = []

        # Add macro features with specified lags
        for feature in self.spec.macro_features:
            if feature in merged.columns:
                # Current value
                features.append(feature)
                feature_names.append(feature)

                # Lagged values
                lags = self.spec.macro_lags.get(feature, [])
                for lag in lags:
                    if lag > 0:
                        col_name = f"{feature}_lag{lag}"
                        merged = merged.with_columns(
                            pl.col(feature).shift(lag).alias(col_name)
                        )
                        features.append(col_name)
                        feature_names.append(col_name)

        # Drop rows with missing values
        valid_cols = features + [self.spec.target]
        available_cols = [c for c in valid_cols if c in merged.columns]
        merged = merged.drop_nulls(subset=available_cols)

        if merged.height < self.spec.min_observations_per_bank:
            raise ValueError(
                f"Insufficient data: {merged.height} < {self.spec.min_observations_per_bank}"
            )

        # Extract arrays
        available_features = [f for f in features if f in merged.columns]
        X = merged.select(available_features).to_numpy()
        y = merged.select(self.spec.target).to_numpy().flatten()

        return X, y, available_features

    def fit(
        self,
        bank_panel: pl.DataFrame,
        macro_data: pl.DataFrame,
    ) -> dict[str, float]:
        """
        Fit APLR models for each bank.

        Args:
            bank_panel: Panel with [date, ticker, nim, ...]
            macro_data: DataFrame with [date, rate_spread, output_gap, ...]

        Returns:
            Dictionary of R-squared scores by bank
        """
        from ...bank_data import TARGET_BANKS

        scores = {}

        for ticker in bank_panel["ticker"].unique().to_list():
            try:
                # Filter to bank
                bank_df = bank_panel.filter(pl.col("ticker") == ticker).sort("date")

                # Build features
                X, y, feature_names = self._build_features(bank_df, macro_data)
                self.feature_names = feature_names

                # Split train/holdout
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
                self.models[ticker] = model

                # Calculate R-squared
                y_pred = model.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                scores[ticker] = r2

                # Extract sensitivities
                sensitivity = self._extract_sensitivities(model, ticker, r2, n)
                self.sensitivities[ticker] = sensitivity

            except Exception as e:
                print(f"Warning: Could not fit model for {ticker}: {e}")
                scores[ticker] = None

        self._is_fitted = True
        return scores

    def _extract_sensitivities(
        self,
        model: APLRRegressor,
        ticker: str,
        r2: float,
        n_obs: int,
    ) -> BankSensitivity:
        """Extract marginal sensitivities from APLR model."""
        from ...bank_data import TARGET_BANKS

        bank_info = TARGET_BANKS.get(ticker)
        bank_name = bank_info.name if bank_info else ticker

        # Get feature importance from APLR
        try:
            explanation = model.explain_global()
            importances = explanation.data()
            importance_dict = dict(zip(
                importances.get("names", self.feature_names),
                importances.get("scores", [0] * len(self.feature_names))
            ))
        except Exception:
            importance_dict = {}

        # Approximate linear sensitivities by averaging local effects
        # APLR gives piecewise linear, so we take mean slope
        rate_sens = importance_dict.get("rate_spread", 0)
        output_sens = importance_dict.get("output_gap", 0)
        inflation_sens = importance_dict.get("inflation_yoy", 0)
        credit_sens = importance_dict.get("credit_spread", 0)

        # Classify regime advantage based on sensitivity signs/magnitudes
        rate_regime = "neutral"
        if rate_sens > 0.5:
            rate_regime = "rising"
        elif rate_sens < -0.5:
            rate_regime = "falling"

        macro_regime = "neutral"
        if output_sens > 0.3:
            macro_regime = "expansion"
        elif output_sens < -0.3:
            macro_regime = "recession"

        return BankSensitivity(
            ticker=ticker,
            name=bank_name,
            rate_sensitivity=float(rate_sens),
            output_gap_sensitivity=float(output_sens),
            inflation_sensitivity=float(inflation_sens),
            credit_spread_sensitivity=float(credit_sens),
            r_squared=float(r2),
            n_observations=n_obs,
            rate_regime_advantage=rate_regime,
            macro_regime_advantage=macro_regime,
        )

    def predict(
        self,
        ticker: str,
        macro_scenario: dict[str, float],
    ) -> float:
        """
        Predict NIM for a bank given a macro scenario.

        Args:
            ticker: Bank ticker
            macro_scenario: Dictionary of macro variable values

        Returns:
            Predicted NIM
        """
        if ticker not in self.models:
            raise ValueError(f"No model fitted for {ticker}")

        model = self.models[ticker]

        # Build feature vector from scenario
        X = np.zeros((1, len(self.feature_names)))
        for i, name in enumerate(self.feature_names):
            # Handle lagged features by using current value
            base_name = name.split("_lag")[0]
            X[0, i] = macro_scenario.get(base_name, macro_scenario.get(name, 0))

        return float(model.predict(X)[0])

    def get_sensitivity_ranking(
        self,
        by: str = "rate_sensitivity",
    ) -> pl.DataFrame:
        """
        Rank banks by a specific sensitivity measure.

        Args:
            by: Column to rank by

        Returns:
            DataFrame with banks ranked by sensitivity
        """
        if not self.sensitivities:
            return pl.DataFrame()

        records = [s.to_dict() for s in self.sensitivities.values()]
        df = pl.DataFrame(records)

        return df.sort(by, descending=True)


class FallbackLinearModel:
    """
    Fallback OLS-based model when APLR is not available.

    Uses simple linear regression to estimate sensitivities.
    """

    def __init__(self, spec: MacroSensitivitySpec):
        self.spec = spec
        self.coefficients: dict[str, dict[str, float]] = {}
        self.sensitivities: dict[str, BankSensitivity] = {}
        self._is_fitted = False

    def fit(
        self,
        bank_panel: pl.DataFrame,
        macro_data: pl.DataFrame,
    ) -> dict[str, float]:
        """Fit linear models for each bank."""
        from ...bank_data import TARGET_BANKS

        scores = {}

        for ticker in bank_panel["ticker"].unique().to_list():
            try:
                bank_df = bank_panel.filter(pl.col("ticker") == ticker).sort("date")
                merged = bank_df.join(macro_data, on="date", how="inner")

                # Get available macro features
                features = [f for f in self.spec.macro_features if f in merged.columns]
                merged = merged.drop_nulls(subset=features + [self.spec.target])

                if merged.height < self.spec.min_observations_per_bank:
                    continue

                # Simple OLS
                X = merged.select(features).to_numpy()
                y = merged.select(self.spec.target).to_numpy().flatten()

                # Add constant
                X_with_const = np.column_stack([np.ones(len(y)), X])

                # Solve normal equations
                try:
                    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue

                # Store coefficients
                self.coefficients[ticker] = {
                    "intercept": beta[0],
                    **{features[i]: beta[i + 1] for i in range(len(features))}
                }

                # Calculate R-squared
                y_pred = X_with_const @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                scores[ticker] = r2

                # Extract sensitivities
                bank_info = TARGET_BANKS.get(ticker)
                bank_name = bank_info.name if bank_info else ticker

                self.sensitivities[ticker] = BankSensitivity(
                    ticker=ticker,
                    name=bank_name,
                    rate_sensitivity=self.coefficients[ticker].get("rate_spread", 0),
                    output_gap_sensitivity=self.coefficients[ticker].get("output_gap", 0),
                    inflation_sensitivity=self.coefficients[ticker].get("inflation_yoy", 0),
                    credit_spread_sensitivity=self.coefficients[ticker].get("credit_spread", 0),
                    r_squared=r2,
                    n_observations=len(y),
                )

            except Exception as e:
                print(f"Warning: Could not fit model for {ticker}: {e}")

        self._is_fitted = True
        return scores

    def predict(
        self,
        ticker: str,
        macro_scenario: dict[str, float],
    ) -> float:
        """Predict NIM given macro scenario."""
        if ticker not in self.coefficients:
            raise ValueError(f"No model fitted for {ticker}")

        coef = self.coefficients[ticker]
        pred = coef.get("intercept", 0)

        for feature, value in macro_scenario.items():
            if feature in coef:
                pred += coef[feature] * value

        return pred

    def get_sensitivity_ranking(
        self,
        by: str = "rate_sensitivity",
    ) -> pl.DataFrame:
        """Rank banks by sensitivity."""
        if not self.sensitivities:
            return pl.DataFrame()

        records = [s.to_dict() for s in self.sensitivities.values()]
        return pl.DataFrame(records).sort(by, descending=True)


def get_sensitivity_model(spec: MacroSensitivitySpec):
    """Get appropriate model based on available packages."""
    if APLR_AVAILABLE:
        return APLRSensitivityModel(spec)
    else:
        print("Warning: interpret-ml not available, using linear fallback model")
        return FallbackLinearModel(spec)


@register_indicator("bank_macro_sensitivity")
class BankMacroSensitivityIndicator(BaseIndicator):
    """
    Bank-Specific Macro Sensitivity Indicator.

    Measures heterogeneous bank elasticities to macroeconomic variables
    to identify structurally advantaged banks in different macro regimes.

    Trading signal: Banks with high rate sensitivity outperform when rates rise.
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self._spec: Optional[MacroSensitivitySpec] = None
        self._model = None
        self._macro_data: Optional[pl.DataFrame] = None

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="Bank-Specific Macro Sensitivity",
            short_name="MacroSens",
            description=(
                "Measures bank-specific elasticities to macro variables (rates, output gap, "
                "inflation) using APLR models. Identifies banks structurally advantaged "
                "in different macro regimes for equity trading signals."
            ),
            version="1.0.0",
            paper_reference="Extension of NY Fed Staff Report 1111",
            data_sources=["SEC EDGAR (10-Q)", "FRED", "CBO"],
            update_frequency="quarterly",
            lookback_periods=20,
        )

    def _load_macro_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """Load and prepare macro data from FRED."""
        from ...macro import MacroDataFetcher

        fetcher = MacroDataFetcher(start_date=start_date)

        # Fetch and compute derived variables
        macro_df = fetcher.compute_derived_variables()

        # Aggregate to quarterly
        macro_df = fetcher.get_quarterly_macro()

        # Compute rate spread (10Y Treasury - Fed Funds as proxy)
        if "DGS10" in macro_df.columns and "DFF" in macro_df.columns:
            macro_df = macro_df.with_columns(
                (pl.col("DGS10") - pl.col("DFF")).alias("rate_spread")
            )

        # Rename columns to match spec
        renames = {
            "BAA10Y": "credit_spread",
            "VIXCLS": "vix",
        }
        for old, new in renames.items():
            if old in macro_df.columns:
                macro_df = macro_df.rename({old: new})

        self._macro_data = macro_df
        return macro_df

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch SEC bank data and FRED macro data."""
        from ...bank_data import BankDataCollector

        # Fetch bank-level data
        collector = BankDataCollector(start_date=start_date)
        bank_panel = collector.fetch_all_banks()
        bank_panel = collector.compute_derived_metrics(bank_panel)

        # Fetch macro data
        macro_data = self._load_macro_data(start_date, end_date)

        return {
            "bank_panel": bank_panel,
            "macro_data": macro_data,
            "data_quality": collector.get_data_quality_summary(),
        }

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        spec: Optional[MacroSensitivitySpec] = None,
        **kwargs,
    ) -> IndicatorResult:
        """
        Calculate macro sensitivity scores for all banks.

        Args:
            data: Dictionary with bank_panel and macro_data
            spec: Model specification (uses default if None)

        Returns:
            IndicatorResult with sensitivity rankings
        """
        bank_panel = data.get("bank_panel", pl.DataFrame())
        macro_data = data.get("macro_data", self._macro_data)

        if bank_panel.height == 0:
            return IndicatorResult(
                indicator_name="bank_macro_sensitivity",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No bank data available"},
            )

        if macro_data is None or macro_data.height == 0:
            return IndicatorResult(
                indicator_name="bank_macro_sensitivity",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No macro data available"},
            )

        # Load specification
        if spec is None:
            spec = MacroSensitivitySpec(
                name="default",
                description="Default macro sensitivity model",
            )
        self._spec = spec

        # Fit model
        self._model = get_sensitivity_model(spec)
        scores = self._model.fit(bank_panel, macro_data)

        # Get sensitivity rankings
        sensitivity_df = self._model.get_sensitivity_ranking()

        return IndicatorResult(
            indicator_name="bank_macro_sensitivity",
            calculation_date=datetime.now(),
            data=sensitivity_df,
            metadata={
                "n_banks": len(scores),
                "avg_r_squared": np.mean([v for v in scores.values() if v is not None]),
                "model_type": "APLR" if APLR_AVAILABLE else "OLS",
                "spec_name": spec.name,
            },
        )

    def nowcast(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """
        Nowcast current quarter sensitivities using latest macro data.

        Uses high-frequency data where available to estimate current
        quarter's macro environment and predict bank responses.
        """
        from .nowcast import MacroSensitivityNowcaster

        if self._model is None or not self._model._is_fitted:
            # Need to fit model first
            self.calculate(data)

        nowcaster = MacroSensitivityNowcaster(self._model, self._spec)

        macro_data = data.get("macro_data", self._macro_data)
        if macro_data is None:
            return IndicatorResult(
                indicator_name="bank_macro_sensitivity_nowcast",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No macro data for nowcast"},
            )

        nowcast_result = nowcaster.nowcast(macro_data)

        return IndicatorResult(
            indicator_name="bank_macro_sensitivity_nowcast",
            calculation_date=datetime.now(),
            data=nowcast_result,
            metadata={
                "methodology": "APLR prediction with current macro data",
            },
        )

    def get_regime_classification(self) -> pl.DataFrame:
        """
        Classify banks by their regime advantages.

        Returns DataFrame showing which banks are advantaged in:
        - Rising rate environment
        - Falling rate environment
        - Expansion
        - Recession
        """
        if self._model is None or not self._model.sensitivities:
            return pl.DataFrame()

        records = []
        for ticker, sens in self._model.sensitivities.items():
            records.append({
                "ticker": ticker,
                "name": sens.name,
                "rate_regime_advantage": sens.rate_regime_advantage,
                "macro_regime_advantage": sens.macro_regime_advantage,
                "rate_sensitivity": sens.rate_sensitivity,
                "output_gap_sensitivity": sens.output_gap_sensitivity,
            })

        return pl.DataFrame(records)

    def get_dashboard_components(self) -> dict[str, Any]:
        """Return dashboard configuration."""
        return {
            "tabs": [
                {"name": "Sensitivity Analysis", "icon": "chart_with_upwards_trend"},
                {"name": "Regime Advantages", "icon": "target"},
                {"name": "Forecasts", "icon": "crystal_ball"},
            ],
            "primary_metric": "rate_sensitivity",
            "ranking_metric": "r_squared",
        }
