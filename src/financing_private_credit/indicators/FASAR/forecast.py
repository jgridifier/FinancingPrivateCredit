"""
FASAR Forecast - M&A Market Share Prediction

Models the "Scar Tissue" effect: when a bank gets "hung" (FASAR spike),
they enter a "penalty box" where risk committees cut limits, leading to
decline in M&A market share 4 quarters later.

M&A Market Share_{t+4} = α - β × Max(FASAR)_{t→t-4}

Reference:
- Ivashina & Scharfstein (2010): Loan Syndication and Credit Cycles
- Kaplan & Strömberg (2009): Leveraged Buyouts and Private Equity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import polars as pl

from ..base import BaseForecastModel, ForecastResult


@dataclass
class ScarTissueSpec:
    """Specification for Scar Tissue forecast model."""

    name: str = "default"
    description: str = "Scar Tissue effect model for M&A market share"

    # Model parameters
    forecast_horizon_quarters: int = 4  # Predict 4 quarters ahead
    fasar_lookback_quarters: int = 4  # Look at max FASAR over past 4 quarters

    # Dampening for large banks
    apply_cet1_dampening: bool = True
    cet1_dampening_threshold: float = 0.12  # Banks above 12% CET1 get dampening

    # Regression defaults (can be estimated from data)
    default_alpha: float = 0.10  # Baseline market share (10%)
    default_beta: float = 0.02  # Market share decline per FASAR point

    @classmethod
    def from_dict(cls, d: dict) -> "ScarTissueSpec":
        """Create spec from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MarketShareForecast:
    """Forecast for a single bank's M&A market share."""

    ticker: str
    current_market_share: float
    forecast_market_share: float
    change_pct: float

    # Drivers
    max_fasar: float
    fasar_impact: float
    cet1_dampening: float

    # Metadata
    forecast_date: datetime
    horizon_quarters: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "current_market_share": self.current_market_share,
            "forecast_market_share": self.forecast_market_share,
            "change_pct": self.change_pct,
            "max_fasar": self.max_fasar,
            "fasar_impact": self.fasar_impact,
            "cet1_dampening": self.cet1_dampening,
            "forecast_date": self.forecast_date.isoformat(),
            "horizon_quarters": self.horizon_quarters,
        }


class ScarTissueForecaster(BaseForecastModel[None]):
    """
    Forecaster for M&A market share based on FASAR "Scar Tissue" effect.

    When a bank experiences a FASAR spike (hung loan event), their
    risk committees tighten limits, leading to reduced M&A activity
    in subsequent quarters.

    Model: M&A Share_{t+4} = α - β × Max(FASAR)_{t→t-4} × (1 - CET1_Dampening)
    """

    def __init__(self, spec: Optional[ScarTissueSpec] = None):
        """
        Initialize the forecaster.

        Args:
            spec: Model specification
        """
        super().__init__()
        self.spec = spec or ScarTissueSpec()
        self._alpha: Optional[float] = None
        self._beta: Optional[float] = None

    def fit(
        self,
        data: pl.DataFrame,
        target: str = "market_share",
        features: list[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Fit the scar tissue model.

        Args:
            data: Historical data with:
                - ticker: Bank identifier
                - date: Quarter date
                - fasar: FASAR score
                - market_share: M&A market share
                - cet1_ratio: Optional CET1 ratio
            target: Target column name
            features: Not used (model has fixed specification)

        Returns:
            Dictionary with fit metrics
        """
        self._target = target
        self._features = ["max_fasar_4q"]
        self._is_fitted = True

        if data.height == 0:
            self._alpha = self.spec.default_alpha
            self._beta = self.spec.default_beta
            return {"n_observations": 0, "using_defaults": True}

        # Calculate max FASAR over lookback period for each bank-quarter
        data_with_max = self._add_max_fasar(data)

        # Shift market share forward by forecast horizon
        data_with_target = data_with_max.with_columns(
            pl.col(target).shift(-self.spec.forecast_horizon_quarters).alias("future_share")
        ).drop_nulls(subset=["future_share", "max_fasar_4q"])

        if data_with_target.height < 10:
            # Insufficient data for regression
            self._alpha = self.spec.default_alpha
            self._beta = self.spec.default_beta
            return {
                "n_observations": data_with_target.height,
                "using_defaults": True,
                "reason": "Insufficient observations for regression",
            }

        # Simple OLS: market_share = alpha - beta * max_fasar
        y = data_with_target["future_share"].to_numpy()
        x = data_with_target["max_fasar_4q"].to_numpy()

        n = len(y)
        x_mean = x.mean()
        y_mean = y.mean()

        # Calculate beta
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator > 0:
            self._beta = -numerator / denominator  # Negative because higher FASAR = lower share
            self._alpha = y_mean + self._beta * x_mean
        else:
            self._alpha = self.spec.default_alpha
            self._beta = self.spec.default_beta

        # Calculate R²
        y_pred = [self._alpha - self._beta * xi for xi in x]
        ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        self._fit_metadata = {
            "alpha": self._alpha,
            "beta": self._beta,
            "n_observations": n,
            "r_squared": r_squared,
        }

        return self._fit_metadata

    def predict(
        self,
        data: pl.DataFrame,
        horizon: int = 4,
        **kwargs,
    ) -> ForecastResult:
        """
        Predict future M&A market share.

        Args:
            data: DataFrame with:
                - ticker: Bank identifier
                - fasar: Recent FASAR scores
                - current_market_share: Current market share
                - cet1_ratio: Optional CET1 ratio
            horizon: Forecast horizon in quarters

        Returns:
            ForecastResult with predictions
        """
        if not self._is_fitted:
            # Use defaults
            self._alpha = self.spec.default_alpha
            self._beta = self.spec.default_beta

        horizon = horizon or self.spec.forecast_horizon_quarters

        # Get max FASAR for each bank
        if "fasar" in data.columns:
            max_fasar = data.group_by("ticker").agg(
                pl.col("fasar").max().alias("max_fasar")
            )
        elif "effective_fasar" in data.columns:
            max_fasar = data.group_by("ticker").agg(
                pl.col("effective_fasar").max().alias("max_fasar")
            )
        else:
            # No FASAR data
            return ForecastResult(
                target="market_share",
                horizon=horizon,
                predictions=pl.DataFrame(),
                metadata={"error": "No FASAR data provided"},
            )

        # Get current market share if available
        if "current_market_share" in data.columns:
            current_share = data.group_by("ticker").agg(
                pl.col("current_market_share").last().alias("current_share")
            )
            max_fasar = max_fasar.join(current_share, on="ticker", how="left")
        else:
            max_fasar = max_fasar.with_columns(
                pl.lit(self.spec.default_alpha).alias("current_share")
            )

        # Get CET1 if available
        if "cet1_ratio" in data.columns:
            cet1 = data.group_by("ticker").agg(
                pl.col("cet1_ratio").last().alias("cet1_ratio")
            )
            max_fasar = max_fasar.join(cet1, on="ticker", how="left")
        else:
            max_fasar = max_fasar.with_columns(
                pl.lit(None).alias("cet1_ratio")
            )

        # Calculate forecasts
        predictions = []
        for row in max_fasar.iter_rows(named=True):
            forecast = self._predict_single(
                ticker=row["ticker"],
                max_fasar=row["max_fasar"],
                current_share=row.get("current_share", self.spec.default_alpha),
                cet1_ratio=row.get("cet1_ratio"),
                horizon=horizon,
            )
            predictions.append(forecast.to_dict())

        predictions_df = pl.DataFrame(predictions)

        return ForecastResult(
            target="market_share",
            horizon=horizon,
            predictions=predictions_df,
            metadata={
                "alpha": self._alpha,
                "beta": self._beta,
                "n_banks": len(predictions),
            },
        )

    def _predict_single(
        self,
        ticker: str,
        max_fasar: float,
        current_share: float,
        cet1_ratio: Optional[float],
        horizon: int,
    ) -> MarketShareForecast:
        """
        Predict market share for a single bank.
        """
        # Calculate FASAR impact
        fasar_impact = self._beta * max_fasar

        # Apply CET1 dampening for well-capitalized banks
        cet1_dampening = 1.0
        if (self.spec.apply_cet1_dampening and
            cet1_ratio is not None and
            cet1_ratio > self.spec.cet1_dampening_threshold):
            # Reduce impact for well-capitalized banks
            excess_capital = cet1_ratio - self.spec.cet1_dampening_threshold
            cet1_dampening = max(0.3, 1.0 - excess_capital * 5)  # Up to 70% reduction

        # Calculate forecast
        adjusted_impact = fasar_impact * cet1_dampening
        forecast_share = max(0, self._alpha - adjusted_impact)

        # Calculate change from current
        change_pct = ((forecast_share / current_share) - 1) * 100 if current_share > 0 else 0

        return MarketShareForecast(
            ticker=ticker,
            current_market_share=current_share,
            forecast_market_share=forecast_share,
            change_pct=change_pct,
            max_fasar=max_fasar,
            fasar_impact=fasar_impact,
            cet1_dampening=cet1_dampening,
            forecast_date=datetime.now(),
            horizon_quarters=horizon,
        )

    def _add_max_fasar(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add rolling max FASAR over lookback period."""
        lookback = self.spec.fasar_lookback_quarters

        fasar_col = "fasar" if "fasar" in data.columns else "effective_fasar"

        return data.sort(["ticker", "date"]).with_columns(
            pl.col(fasar_col)
            .rolling_max(window_size=lookback)
            .over("ticker")
            .alias("max_fasar_4q")
        )

    def get_coefficients(self) -> Optional[dict[str, float]]:
        """Return model coefficients."""
        if not self._is_fitted:
            return None
        return {
            "alpha": self._alpha,
            "beta": self._beta,
        }

    def get_diagnostics(self) -> dict[str, Any]:
        """Return model diagnostics."""
        base = super().get_diagnostics()
        base.update({
            "spec_name": self.spec.name,
            "forecast_horizon": self.spec.forecast_horizon_quarters,
            "fasar_lookback": self.spec.fasar_lookback_quarters,
            "cet1_dampening_enabled": self.spec.apply_cet1_dampening,
        })
        if self._fit_metadata:
            base.update(self._fit_metadata)
        return base


class FASARForecaster:
    """
    Convenience wrapper for FASAR forecasting.

    Provides methods for both:
    1. M&A market share prediction (Scar Tissue model)
    2. Future FASAR trajectory (if commitment pipeline data available)
    """

    def __init__(self, spec: Optional[ScarTissueSpec] = None):
        """
        Initialize the forecaster.

        Args:
            spec: Model specification
        """
        self.scar_tissue_model = ScarTissueForecaster(spec)

    def forecast_market_share(
        self,
        fasar_data: pl.DataFrame,
        market_share_data: Optional[pl.DataFrame] = None,
        horizon: int = 4,
    ) -> ForecastResult:
        """
        Forecast M&A market share based on FASAR history.

        Args:
            fasar_data: Historical FASAR scores
            market_share_data: Optional historical market share for fitting
            horizon: Forecast horizon in quarters

        Returns:
            ForecastResult with market share predictions
        """
        # Fit model if training data available
        if market_share_data is not None and market_share_data.height > 0:
            # Merge FASAR and market share data
            combined = fasar_data.join(
                market_share_data,
                on=["ticker", "date"],
                how="inner",
            )
            self.scar_tissue_model.fit(combined, target="market_share")
        else:
            # Use defaults
            self.scar_tissue_model._is_fitted = True
            self.scar_tissue_model._alpha = self.scar_tissue_model.spec.default_alpha
            self.scar_tissue_model._beta = self.scar_tissue_model.spec.default_beta

        return self.scar_tissue_model.predict(fasar_data, horizon)

    def identify_at_risk_banks(
        self,
        fasar_data: pl.DataFrame,
        fasar_threshold: float = 1.5,
    ) -> pl.DataFrame:
        """
        Identify banks at risk of market share decline.

        Args:
            fasar_data: Current FASAR scores
            fasar_threshold: FASAR level indicating elevated risk

        Returns:
            DataFrame with at-risk banks and projected impact
        """
        # Get predictions
        result = self.forecast_market_share(fasar_data)

        if result.predictions.height == 0:
            return pl.DataFrame()

        # Filter to at-risk banks
        at_risk = result.predictions.filter(
            (pl.col("max_fasar") > fasar_threshold) &
            (pl.col("change_pct") < -5)  # Projected >5% decline
        ).sort("change_pct")

        return at_risk
