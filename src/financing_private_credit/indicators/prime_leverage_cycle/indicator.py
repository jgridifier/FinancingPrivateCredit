"""
Prime Leverage Cycle Indicator

Measures hedge fund leverage cycle positioning using prime brokerage margin loans
and market valuations. Based on Adrian & Shin (2010) procyclical leverage theory
and BIS (2024) prime broker-hedge fund nexus research.

LCI = (HF_Margin_Loans / HF_AUM_Proxy) × (SP500 / MA_252(SP500))

Key signals:
- LCI > 75th percentile: Expansion regime (peak risk)
- LCI < 25th percentile: Contraction regime (recovery opportunity)
- LCI_ZScore > 2.0: Peak warning
- LCI_Velocity < -15%: Deleveraging event
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import polars as pl

from ..base import (
    BaseIndicator,
    IndicatorMetadata,
    IndicatorResult,
    register_indicator,
)


@dataclass
class LeverageCycleSpec:
    """Configuration for Leverage Cycle Index calculation."""

    name: str = "default"
    description: str = "Default leverage cycle specification"

    # AUM proxy calibration
    hf_share_of_equity: float = 0.035  # 3.5% hedge fund share of equity market
    historical_leverage: float = 1.75  # Historical average leverage ratio

    # Moving average window for market valuation
    ma_window_days: int = 252  # 1-year moving average

    # Z-score rolling window (quarters)
    zscore_window_quarters: int = 8

    # Regime thresholds (percentiles)
    expansion_percentile: float = 0.75
    contraction_percentile: float = 0.25

    # Signal thresholds
    peak_warning_zscore: float = 2.0
    deleveraging_velocity_threshold: float = -15.0  # percent
    recovery_zscore: float = -1.5
    recovery_velocity: float = 5.0  # percent

    @classmethod
    def from_dict(cls, d: dict) -> "LeverageCycleSpec":
        """Create spec from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@register_indicator("prime_leverage_cycle")
class PrimeLeverageCycleIndicator(BaseIndicator):
    """
    Prime Leverage Cycle Indicator.

    Measures hedge fund leverage cycle positioning to identify systemic risk
    and revenue cycle turning points for prime brokerage businesses.

    Core mechanism:
    - Expansion: Markets up -> Collateral values up -> More borrowing -> Leverage up
    - Contraction: Markets down -> Margin calls -> Forced deleveraging -> Liquidity spiral

    References:
    - Adrian, T., & Shin, H. S. (2010). "Liquidity and Leverage"
    - BIS (2024). "The prime broker-hedge fund nexus"
    - Brunnermeier, M. K., & Pedersen, L. H. (2009). "Market Liquidity and Funding Liquidity"
    """

    supports_nowcast: bool = False

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the indicator."""
        super().__init__(config_path)
        self._spec: Optional[LeverageCycleSpec] = None

    def get_metadata(self) -> IndicatorMetadata:
        """Return indicator metadata."""
        return IndicatorMetadata(
            name="Prime Leverage Cycle Index",
            short_name="LCI",
            description=(
                "Measures hedge fund leverage cycle positioning using prime brokerage "
                "margin loan data and market valuations. Identifies systemic risk buildup "
                "and revenue cycle turning points based on procyclical leverage theory."
            ),
            version="1.0.0",
            paper_reference="Adrian & Shin (2010); BIS (2024); Brunnermeier & Pedersen (2009)",
            data_sources=["FRED (BOGZ1FL624123035Q, SP500, WILL5000PRFC)"],
            update_frequency="quarterly",
            lookback_periods=40,  # 10 years of quarterly data
        )

    def get_required_data_sources(self) -> list[str]:
        """Document data sources needed."""
        return ["hf_margin_loans", "sp500", "wilshire5000"]

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch data required for Leverage Cycle Index calculation.

        Fetches from FRED:
        - BOGZ1FL624123035Q: Hedge Fund Margin Loans (quarterly)
        - SP500: S&P 500 Index (daily)
        - WILL5000PRFC: Wilshire 5000 Total Market Cap (weekly)
        """
        from ...core import DataRegistry

        registry = DataRegistry.get_instance()

        # Fetch FRED series
        fred_series = [
            "BOGZ1FL624123035Q",  # HF margin loans
            "SP500",  # S&P 500
            "WILL5000PRFC",  # Wilshire 5000 (for AUM proxy)
        ]

        try:
            macro_data = registry.get_macro_series(fred_series, start_date)
        except Exception:
            macro_data = pl.DataFrame({"date": []})

        return {"macro_data": macro_data}

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        spec: Optional[LeverageCycleSpec] = None,
        **kwargs,
    ) -> IndicatorResult:
        """
        Calculate the Leverage Cycle Index and derivative indicators.

        Args:
            data: Dictionary with macro_data containing FRED series
            spec: Optional specification override

        Returns:
            IndicatorResult with LCI, velocity, z-score, and regime classification
        """
        if spec is not None:
            self._spec = spec
        elif self._spec is None:
            self._spec = LeverageCycleSpec()

        macro_data = data.get("macro_data", pl.DataFrame())

        if macro_data.height == 0:
            return IndicatorResult(
                indicator_name="prime_leverage_cycle",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No macro data available"},
            )

        # Process the data
        result_df = self._calculate_lci(macro_data)

        if result_df.height == 0:
            return IndicatorResult(
                indicator_name="prime_leverage_cycle",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "Insufficient data for LCI calculation"},
            )

        # Calculate summary statistics
        summary = self._calculate_summary(result_df)

        return IndicatorResult(
            indicator_name="prime_leverage_cycle",
            calculation_date=datetime.now(),
            data=result_df,
            metadata={
                "spec": self._spec.name,
                "n_observations": result_df.height,
                "date_range": {
                    "start": str(result_df["date"].min()),
                    "end": str(result_df["date"].max()),
                },
                "summary": summary,
                "current_regime": (
                    result_df["regime"][-1] if result_df.height > 0 else None
                ),
                "current_signal": self._get_current_signal(result_df),
            },
        )

    def _calculate_lci(self, macro_data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Leverage Cycle Index from FRED data.

        Steps:
        1. Calculate market valuation (SP500 / MA_252)
        2. Construct HF AUM proxy (hybrid method)
        3. Calculate leverage ratio
        4. Compute LCI = leverage_ratio × market_valuation
        5. Add derivative indicators (velocity, z-score, regime)
        """
        # Check for required columns
        required = ["date"]
        has_margin = "BOGZ1FL624123035Q" in macro_data.columns
        has_sp500 = "SP500" in macro_data.columns
        has_wilshire = "WILL5000PRFC" in macro_data.columns

        if not (has_margin and has_sp500):
            return pl.DataFrame()

        # Ensure date column is proper datetime
        df = macro_data.with_columns(pl.col("date").cast(pl.Date))

        # Step 1: Calculate SP500 moving average and market valuation (daily)
        if has_sp500:
            df = df.sort("date").with_columns([
                pl.col("SP500")
                .rolling_mean(window_size=self._spec.ma_window_days)
                .alias("sp500_ma252"),
            ]).with_columns([
                (pl.col("SP500") / pl.col("sp500_ma252")).alias("market_valuation")
            ])

        # Step 2: Aggregate to quarterly
        df = df.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.quarter().alias("quarter"),
        ])

        # Group by quarter and get end-of-quarter values
        quarterly = df.group_by(["year", "quarter"]).agg([
            pl.col("date").max().alias("date"),
            pl.col("SP500").last().alias("sp500_qtr_end"),
            pl.col("market_valuation").last().alias("market_valuation"),
            pl.col("BOGZ1FL624123035Q").last().alias("hf_margin_loans"),
            pl.col("WILL5000PRFC").last().alias("total_equity_mcap") if has_wilshire else pl.lit(None).alias("total_equity_mcap"),
        ]).sort("date")

        # Filter out rows with null margin loan data
        quarterly = quarterly.filter(pl.col("hf_margin_loans").is_not_null())

        if quarterly.height == 0:
            return pl.DataFrame()

        # Step 3: Construct HF AUM proxy (hybrid method)
        quarterly = quarterly.with_columns([
            # Method 1: Market cap based
            (pl.col("total_equity_mcap") * self._spec.hf_share_of_equity)
            .alias("aum_proxy_method1"),
            # Method 2: Leverage ratio based
            (pl.col("hf_margin_loans") / self._spec.historical_leverage)
            .alias("aum_proxy_method2"),
        ])

        # Hybrid: average of both methods (handle nulls)
        quarterly = quarterly.with_columns([
            pl.when(pl.col("aum_proxy_method1").is_not_null())
            .then((pl.col("aum_proxy_method1") + pl.col("aum_proxy_method2")) / 2)
            .otherwise(pl.col("aum_proxy_method2"))
            .alias("hf_aum_proxy")
        ])

        # Step 4: Calculate leverage ratio
        quarterly = quarterly.with_columns([
            (pl.col("hf_margin_loans") / pl.col("hf_aum_proxy")).alias("leverage_ratio")
        ])

        # Step 5: Calculate LCI
        quarterly = quarterly.with_columns([
            (pl.col("leverage_ratio") * pl.col("market_valuation")).alias("lci")
        ])

        # Step 6: Calculate derivative indicators
        quarterly = quarterly.with_columns([
            # Velocity (QoQ % change)
            (pl.col("lci").pct_change() * 100).alias("lci_velocity"),
            # Rolling mean and std for Z-score
            pl.col("lci")
            .rolling_mean(window_size=self._spec.zscore_window_quarters)
            .alias("lci_rolling_mean"),
            pl.col("lci")
            .rolling_std(window_size=self._spec.zscore_window_quarters)
            .alias("lci_rolling_std"),
        ])

        # Calculate Z-score
        quarterly = quarterly.with_columns([
            ((pl.col("lci") - pl.col("lci_rolling_mean")) / pl.col("lci_rolling_std"))
            .alias("lci_zscore")
        ])

        # Step 7: Calculate regime classification using historical percentiles
        lci_values = quarterly.filter(pl.col("lci").is_not_null())["lci"]
        if lci_values.len() > 0:
            p25 = lci_values.quantile(self._spec.contraction_percentile)
            p75 = lci_values.quantile(self._spec.expansion_percentile)

            quarterly = quarterly.with_columns([
                pl.when(pl.col("lci") > p75)
                .then(pl.lit("Expansion"))
                .when(pl.col("lci") < p25)
                .then(pl.lit("Contraction"))
                .otherwise(pl.lit("Normal"))
                .alias("regime")
            ])
        else:
            quarterly = quarterly.with_columns([pl.lit("Unknown").alias("regime")])

        # Select final output columns
        return quarterly.select([
            "date",
            "hf_margin_loans",
            "hf_aum_proxy",
            "leverage_ratio",
            "sp500_qtr_end",
            "market_valuation",
            "lci",
            "lci_velocity",
            "lci_zscore",
            "regime",
        ])

    def _calculate_summary(self, df: pl.DataFrame) -> dict[str, Any]:
        """Calculate summary statistics for the LCI."""
        if df.height == 0:
            return {}

        lci_col = df["lci"].drop_nulls()
        if lci_col.len() == 0:
            return {}

        return {
            "lci_mean": float(lci_col.mean()),
            "lci_std": float(lci_col.std()),
            "lci_min": float(lci_col.min()),
            "lci_max": float(lci_col.max()),
            "lci_p25": float(lci_col.quantile(0.25)),
            "lci_p75": float(lci_col.quantile(0.75)),
            "pct_expansion": float((df["regime"] == "Expansion").mean()),
            "pct_normal": float((df["regime"] == "Normal").mean()),
            "pct_contraction": float((df["regime"] == "Contraction").mean()),
        }

    def _get_current_signal(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        Get current signal based on latest data.

        Returns signal: 1 (bullish), 0 (neutral), -1 (bearish)
        """
        if df.height == 0:
            return {"signal": 0, "reason": "No data"}

        latest = df.tail(1)
        lci_zscore = latest["lci_zscore"][0]
        lci_velocity = latest["lci_velocity"][0]

        if lci_zscore is None or lci_velocity is None:
            return {"signal": 0, "reason": "Insufficient history"}

        # Peak or deleveraging = bearish
        if (lci_zscore > self._spec.peak_warning_zscore or
                lci_velocity < self._spec.deleveraging_velocity_threshold):
            return {
                "signal": -1,
                "reason": "Peak warning or deleveraging event",
                "lci_zscore": float(lci_zscore),
                "lci_velocity": float(lci_velocity),
            }

        # Bottom and re-leveraging = bullish
        if (lci_zscore < self._spec.recovery_zscore and
                lci_velocity > self._spec.recovery_velocity):
            return {
                "signal": 1,
                "reason": "Recovery signal - bottom and re-leveraging",
                "lci_zscore": float(lci_zscore),
                "lci_velocity": float(lci_velocity),
            }

        # Neutral
        return {
            "signal": 0,
            "reason": "Normal range",
            "lci_zscore": float(lci_zscore),
            "lci_velocity": float(lci_velocity),
        }

    def get_regime_interpretation(self, regime: str) -> dict[str, str]:
        """
        Get interpretation and recommended actions for a regime.

        Args:
            regime: One of "Expansion", "Normal", "Contraction"

        Returns:
            Dictionary with interpretation and action guidance
        """
        interpretations = {
            "Expansion": {
                "description": "High leverage + elevated valuations",
                "revenue_outlook": "Revenue near peak",
                "risk": "Sudden reversal (2000, 2007, 2021 patterns)",
                "action": "Prepare for deleveraging",
            },
            "Normal": {
                "description": "Moderate leverage, moderate valuations",
                "revenue_outlook": "Stable revenue environment",
                "risk": "Standard market risk",
                "action": "Business as usual",
            },
            "Contraction": {
                "description": "Low leverage + depressed valuations",
                "revenue_outlook": "Revenue trough",
                "risk": "Recovery opportunity (2009, 2020 patterns)",
                "action": "Anticipate re-leveraging",
            },
        }
        return interpretations.get(regime, {"description": "Unknown regime"})
