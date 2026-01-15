"""
Prime Leverage Cycle Indicator

Measures hedge fund leverage cycle positioning using prime brokerage margin loans
and market valuations. Based on Adrian & Shin (2010) procyclical leverage theory
and BIS (2024) prime broker-hedge fund nexus research.

LCI = (HF_Margin_Loans / HF_AUM_Proxy) × (MarketIndex / MA_252(MarketIndex))

Market index can be configured via spec.market_index:
- "^W5000" (default): Wilshire 5000 - broad US equity market
- "^SPX": S&P 500 - large cap US equities

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

    # Market index selection
    market_index: str = "^SPX"  # Options: "^W5000" (Wilshire 5000) or "^SPX" (S&P 500)

    # AUM proxy calibration
    # TODO: Replace static hf_share_of_equity with data-driven approach using:
    # 1. BarclayHedge or HFR industry AUM reports (quarterly)
    # 2. 13F filings aggregation for institutional equity holdings
    # 3. SEC Form PF aggregate data when available
    # Current calibration: ~$5T HF AUM / ~$130T total equity market ≈ 3.5%
    hf_share_of_equity: float = 0.035  # 3.5% hedge fund share of equity market
    # TODO: Replace static historical_leverage with rolling estimate from:
    # 1. Prime broker leverage data from FR Y-9C filings
    # 2. Academic estimates (Ang et al. 2011 suggest 1.5-2.0x)
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
            data_sources=["FRED (BOGZ1FL624123035Q)", "Yahoo Finance (^GSPC, ^W5000)"],
            update_frequency="quarterly",
            lookback_periods=40,  # 10 years of quarterly data
        )

    def get_required_data_sources(self) -> list[str]:
        """Document data sources needed."""
        return ["hf_margin_loans", "^W5000", "wilshire5000"]

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch data required for Leverage Cycle Index calculation.

        Fetches from FRED:
        - BOGZ1FL624123035Q: Hedge Fund Margin Loans (quarterly)

        Fetches from Yahoo Finance:
        - ^GSPC: S&P 500 Index (daily)
        - ^W5000: Wilshire 5000 Total Market Index (FRED deprecated ^W5000)
        """
        from ...core import DataRegistry

        registry = DataRegistry.get_instance()

        # Fetch FRED series (only hedge fund margin loans)
        fred_series = [
            "BOGZ1FL624123035Q",  # HF margin loans
        ]

        try:
            macro_data = registry.get_macro_series(fred_series, start_date, end_date)
        except Exception:
            macro_data = pl.DataFrame({"date": []})

        # Fetch S&P 500 and Wilshire 5000 from Yahoo Finance
        try:
            yf_data = registry.get_yahoo_finance_series(["^SPX", "^W5000"], start_date, end_date)
        except Exception as e:
            print(f"Warning: Failed to fetch data from Yahoo Finance: {e}")
            yf_data = pl.DataFrame({"date": []})

        return {
            "fred_data": macro_data,
            "market_data": yf_data,
        }

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

        fred_data = data.get("fred_data", pl.DataFrame())
        market_data = data.get("market_data", pl.DataFrame())

        if fred_data.height == 0 or market_data.height == 0:
            return IndicatorResult(
                indicator_name="prime_leverage_cycle",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "Insufficient data (missing FRED or market data)"},
            )

        # Process the data
        result_df = self._calculate_lci(fred_data, market_data)

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

    def _calculate_lci(self, fred_data: pl.DataFrame, market_data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Leverage Cycle Index from separate FRED and market data.

        Steps:
        1. Calculate market valuation (MarketIndex / MA_252) on daily market data
        2. Resample market data to quarterly (end-of-quarter values)
        3. Join quarterly market data with FRED quarterly data
        4. Construct HF AUM proxy (hybrid method using selected market index)
        5. Calculate leverage ratio
        6. Compute LCI = leverage_ratio × market_valuation
        7. Add derivative indicators (velocity, z-score, regime)
        """
        # Check for required columns
        has_margin = "BOGZ1FL624123035Q" in fred_data.columns if fred_data.height > 0 else False
        market_idx = self._spec.market_index
        has_market_idx = market_idx in market_data.columns if market_data.height > 0 else False

        if not (has_margin and has_market_idx):
            return pl.DataFrame()

        # Step 1: Process market data - calculate rolling mean and market valuation
        market_df = market_data.with_columns(pl.col("date").cast(pl.Date)).sort("date")

        # Calculate rolling mean and market valuation for selected index
        market_df = market_df.with_columns([
            pl.col(market_idx)
            .rolling_mean(window_size=self._spec.ma_window_days)
            .alias(f"{market_idx}_ma252"),
        ]).with_columns([
            (pl.col(market_idx) / pl.col(f"{market_idx}_ma252")).alias("market_valuation")
        ]).with_columns([
            pl.col("market_valuation").forward_fill()
        ])

        # Step 2: Resample market data to quarterly (last trading day of each quarter)
        market_df = market_df.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.quarter().alias("quarter"),
        ])

        market_quarterly = market_df.group_by(["year", "quarter"]).agg([
            pl.col("date").max().alias("market_date"),
            pl.col(market_idx).last().alias("market_idx_qtr_end"),
            pl.col("market_valuation").last().alias("market_valuation"),
        ]).sort("market_date")

        # Step 3: Process FRED data - filter to valid margin loans (non-null and non-zero)
        fred_df = fred_data.with_columns(pl.col("date").cast(pl.Date)).sort("date")

        fred_df = fred_df.filter(
            (pl.col("BOGZ1FL624123035Q").is_not_null()) &
            (pl.col("BOGZ1FL624123035Q") > 0)
        )

        if fred_df.height == 0:
            return pl.DataFrame()

        fred_df = fred_df.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.quarter().alias("quarter"),
        ])

        # FRED reports on specific dates (often start of quarter), extract year/quarter
        fred_quarterly = fred_df.select([
            "year",
            "quarter",
            pl.col("date").alias("fred_date"),
            pl.col("BOGZ1FL624123035Q").alias("hf_margin_loans"),
        ])

        # Step 4: Join quarterly market data with FRED data on year/quarter
        quarterly = fred_quarterly.join(
            market_quarterly,
            on=["year", "quarter"],
            how="inner"  # Only keep quarters with both FRED and market data
        ).sort("market_date")

        # Use market_date as the primary date (end of quarter)
        quarterly = quarterly.with_columns([
            pl.col("market_date").alias("date")
        ]).drop(["market_date", "fred_date"])

        if quarterly.height == 0:
            return pl.DataFrame()

        # Step 5: Construct HF AUM proxy (hybrid method using selected market index)
        quarterly = quarterly.with_columns([
            # Method 1: Market cap based (using selected index as proxy for total market)
            (pl.col("market_idx_qtr_end") * self._spec.hf_share_of_equity)
            .alias("aum_proxy_method1"),
            # Method 2: Leverage ratio based
            (pl.col("hf_margin_loans") / self._spec.historical_leverage)
            .alias("aum_proxy_method2"),
        ])

        # Hybrid: average of both methods
        quarterly = quarterly.with_columns([
            ((pl.col("aum_proxy_method1") + pl.col("aum_proxy_method2")) / 2)
            .alias("hf_aum_proxy")
        ])

        # Step 6: Calculate leverage ratio
        quarterly = quarterly.with_columns([
            (pl.col("hf_margin_loans") / pl.col("hf_aum_proxy")).alias("leverage_ratio")
        ])

        # Step 7: Calculate LCI
        quarterly = quarterly.with_columns([
            (pl.col("leverage_ratio") * pl.col("market_valuation")).alias("lci")
        ])

        # Step 8: Calculate derivative indicators
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

        # Step 9: Calculate regime classification using historical percentiles
        lci_values = quarterly.filter(pl.col("lci").is_not_null())["lci"]
        if lci_values.len() > 0:
            p25 = lci_values.quantile(self._spec.contraction_percentile)
            p75 = lci_values.quantile(self._spec.expansion_percentile)

            # Calculate percentile rank for each observation (0-100 scale)
            quarterly = quarterly.with_columns([
                (pl.col("lci").rank(method="average") / lci_values.len() * 100)
                .alias("lci_percentile")
            ])

            quarterly = quarterly.with_columns([
                pl.when(pl.col("lci") > p75)
                .then(pl.lit("Expansion"))
                .when(pl.col("lci") < p25)
                .then(pl.lit("Contraction"))
                .otherwise(pl.lit("Normal"))
                .alias("regime")
            ])
        else:
            quarterly = quarterly.with_columns([
                pl.lit(None).alias("lci_percentile"),
                pl.lit("Unknown").alias("regime")
            ])

        # Rename market_idx_qtr_end to have the actual index name for clarity
        quarterly = quarterly.rename({"market_idx_qtr_end": f"{market_idx}_qtr_end"})

        # Select final output columns
        return quarterly.select([
            "date",
            "hf_margin_loans",
            "hf_aum_proxy",
            "leverage_ratio",
            f"{market_idx}_qtr_end",
            "market_valuation",
            "lci",
            "lci_velocity",
            "lci_zscore",
            "lci_percentile",
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
