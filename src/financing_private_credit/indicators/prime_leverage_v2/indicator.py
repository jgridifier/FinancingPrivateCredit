"""
Prime Brokerage Leverage Lead Indicator V2 (PB-LLI)

A public-data, implementation-ready indicator suite designed to forecast normal-times
prime brokerage / broker-dealer performance 1-2 quarters ahead by combining:
1. Hedge fund balance-sheet leverage demand (quarterly)
2. Dealer balance-sheet supply (quarterly)
3. Higher-frequency nowcast layer (weekly)

Key outputs:
- Quarterly Anchor Index (structural truth): PB_Intensity, Dealer_SupplyGrowth
- Weekly Nowcast Factor (timeliness): pb_intensity_g_nowcast
- Composite Forecast Signal: 1-2 quarter projections with confidence bands

Run-rate regimes (not stress regimes):
- Accelerating: Top tercile - improving prime balances/revenue momentum
- Stable: Middle tercile - neutral
- Decelerating: Bottom tercile - slowing balances/revenue momentum

References:
- Adrian, T., & Shin, H. S. (2010). "Liquidity and Leverage"
- Fed Z.1 Financial Accounts (hedge fund sector data)
- CFTC Traders in Financial Futures reports
- NY Fed Primary Dealer Statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import polars as pl

from ..base import (
    BaseIndicator,
    IndicatorMetadata,
    IndicatorResult,
    register_indicator,
)


@dataclass
class PBLLISpec:
    """Configuration for Prime Brokerage Leverage Lead Indicator."""

    name: str = "default"
    description: str = "Default PB-LLI specification"

    # Quarterly anchor FRED series
    # Hedge fund demand side
    hf_pb_borrowing_series: str = "BOGZ1FL624123035Q"  # HF margin loans
    hf_assets_series: str = "BOGZ1FL624090005Q"  # HF total financial assets
    hf_liabilities_series: str = "BOGZ1FL624190005Q"  # HF total financial liabilities
    hf_repo_liab_series: str = "BOGZ1FL622151005Q"  # HF repo liabilities (optional)

    # Dealer supply side
    bd_cust_recv_series: str = "BOGZ1FL663067003Q"  # B-D customer receivables

    # Weekly nowcast configuration
    cftc_contracts: list[str] = field(default_factory=lambda: [
        "ES", "NQ", "TU", "FV", "TY", "US", "EC", "JY"
    ])
    nyfed_series: list[str] = field(default_factory=lambda: [
        "PD_RP_T_TOT", "PD_RRP_T_TOT", "PD_AFtD_AG", "PD_AFtR_AG"
    ])

    # Bridge regression / nowcast model
    # TODO: Replace with fitted coefficients from historical backtest
    nowcast_weight: float = 0.4  # Weight on current quarter nowcast
    lagged_intensity_weight: float = 0.35  # Weight on prior quarter intensity
    dealer_supply_weight: float = 0.25  # Weight on dealer supply growth

    # Regime classification percentiles
    accelerating_percentile: float = 0.67  # Top tercile
    decelerating_percentile: float = 0.33  # Bottom tercile

    # Rolling window for z-scores and normalizations
    zscore_window_quarters: int = 20  # 5 years
    forecast_horizon_quarters: int = 2  # t+1 and t+2 forecasts

    # Stress overlay thresholds (secondary diagnostic)
    stress_pb_intensity_drop: float = -0.15  # 15% QoQ drop triggers stress flag
    stress_dealer_supply_drop: float = -0.10  # 10% QoQ drop

    @classmethod
    def from_dict(cls, d: dict) -> "PBLLISpec":
        """Create spec from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@register_indicator("prime_leverage_v2")
class PrimeLeverageV2Indicator(BaseIndicator):
    """
    Prime Brokerage Leverage Lead Indicator V2.

    Forecasts prime brokerage / broker-dealer performance 1-2 quarters ahead
    using a two-sided model: HF demand (leverage intensity) + Dealer supply.

    Architecture:
    1. Quarterly Anchor Index: High signal, low frequency structural truth
    2. Weekly Nowcast Factor: Higher noise, high frequency edge layer
    3. Composite Forecast Signal: Combined output for projections

    Normal-times focus: Core indicator is a run-rate forecaster;
    stress is handled as a secondary diagnostic overlay.
    """

    supports_nowcast: bool = True

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the indicator."""
        super().__init__(config_path)
        self._spec: Optional[PBLLISpec] = None

    def get_metadata(self) -> IndicatorMetadata:
        """Return indicator metadata."""
        return IndicatorMetadata(
            name="Prime Brokerage Leverage Lead Indicator V2",
            short_name="PB-LLI",
            description=(
                "Forecasts prime brokerage performance 1-2 quarters ahead using "
                "hedge fund leverage demand (Fed Z.1), dealer supply, and weekly "
                "nowcast layer (CFTC COT, NY Fed PD). Normal-times run-rate "
                "forecaster with stress overlay."
            ),
            version="2.0.0",
            paper_reference="Adrian & Shin (2010); Fed Z.1 Financial Accounts",
            data_sources=[
                "FRED (BOGZ1FL624123035Q, BOGZ1FL624090005Q, BOGZ1FL624190005Q, "
                "BOGZ1FL622151005Q, BOGZ1FL663067003Q)",
                "CFTC COT TFF (Leveraged Funds)",
                "NY Fed Primary Dealer Statistics",
            ],
            update_frequency="weekly (nowcast), quarterly (anchor)",
            lookback_periods=40,  # 10 years of quarterly data
        )

    def get_required_data_sources(self) -> list[str]:
        """Document data sources needed."""
        return [
            "hf_pb_borrowing", "hf_assets", "hf_liabilities", "hf_repo_liab",
            "bd_cust_recv", "cftc_cot_tff", "nyfed_pd_stats"
        ]

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch all data required for PB-LLI calculation.

        Returns:
            Dictionary with:
            - quarterly_data: Fed Z.1 quarterly series
            - weekly_cot: CFTC COT TFF data
            - weekly_pd: NY Fed Primary Dealer stats
        """
        from ...core import DataRegistry

        registry = DataRegistry.get_instance()
        spec = self._spec or PBLLISpec()

        # Fetch quarterly FRED series
        fred_series = [
            spec.hf_pb_borrowing_series,
            spec.hf_assets_series,
            spec.hf_liabilities_series,
            spec.hf_repo_liab_series,
            spec.bd_cust_recv_series,
        ]

        try:
            quarterly_data = registry.get_macro_series(fred_series, start_date, end_date)
        except Exception as e:
            print(f"Warning: Failed to fetch FRED data: {e}")
            quarterly_data = pl.DataFrame({"date": []})

        # Fetch weekly CFTC COT data
        try:
            weekly_cot = registry.get_cftc_cot_tff(
                contracts=spec.cftc_contracts,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            print(f"Warning: Failed to fetch CFTC COT data: {e}")
            weekly_cot = pl.DataFrame()

        # Fetch weekly NY Fed PD data
        try:
            weekly_pd = registry.get_nyfed_primary_dealer_stats(
                series=spec.nyfed_series,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            print(f"Warning: Failed to fetch NY Fed PD data: {e}")
            weekly_pd = pl.DataFrame()

        return {
            "quarterly_data": quarterly_data,
            "weekly_cot": weekly_cot,
            "weekly_pd": weekly_pd,
        }

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        spec: Optional[PBLLISpec] = None,
        **kwargs,
    ) -> IndicatorResult:
        """
        Calculate the PB-LLI composite indicator and forecasts.

        Returns IndicatorResult with:
        - Quarterly anchor series (pb_intensity, dealer_supply_g)
        - Weekly nowcast values
        - Composite PB_Lead and forecasts
        - Run-rate regime classification
        - Attribution decomposition
        """
        if spec is not None:
            self._spec = spec
        elif self._spec is None:
            self._spec = PBLLISpec()

        quarterly_data = data.get("quarterly_data", pl.DataFrame())
        weekly_cot = data.get("weekly_cot", pl.DataFrame())
        weekly_pd = data.get("weekly_pd", pl.DataFrame())

        if quarterly_data.height == 0:
            return IndicatorResult(
                indicator_name="prime_leverage_v2",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No quarterly data available"},
            )

        # Step 1: Calculate quarterly anchor index
        quarterly_anchor = self._calculate_quarterly_anchor(quarterly_data)

        if quarterly_anchor.height == 0:
            return IndicatorResult(
                indicator_name="prime_leverage_v2",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "Failed to calculate quarterly anchor"},
            )

        # Step 2: Calculate weekly nowcast
        weekly_nowcast = self._calculate_weekly_nowcast(
            quarterly_anchor, weekly_cot, weekly_pd
        )

        # Step 3: Calculate composite forecast signal
        result_df = self._calculate_composite_forecast(quarterly_anchor, weekly_nowcast)

        # Step 4: Add run-rate regime classification
        result_df = self._classify_run_rate_regime(result_df)

        # Step 5: Calculate attribution
        attribution = self._calculate_attribution(result_df)

        # Step 6: Generate forecasts for t+1 and t+2
        forecasts = self._generate_forecasts(result_df)

        # Build metadata
        summary = self._calculate_summary(result_df)
        current_signal = self._get_current_signal(result_df)

        return IndicatorResult(
            indicator_name="prime_leverage_v2",
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
                "current_regime": result_df["run_rate_regime"][-1] if result_df.height > 0 else None,
                "current_signal": current_signal,
                "attribution": attribution,
                "forecasts": forecasts,
                "weekly_nowcast_available": weekly_nowcast.height > 0,
            },
        )

    def _calculate_quarterly_anchor(self, quarterly_data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate quarterly anchor index from Fed Z.1 data.

        Computes:
        - hf_equity = hf_assets - hf_liabilities
        - pb_intensity = hf_pb_borrowing / hf_equity
        - pb_intensity_g = Δln(pb_intensity) (QoQ growth)
        - dealer_supply_g = Δln(bd_cust_recv) (QoQ growth)
        - repo_intensity (optional channel split)
        """
        spec = self._spec

        # Check required columns
        required_cols = [
            spec.hf_pb_borrowing_series,
            spec.hf_assets_series,
            spec.hf_liabilities_series,
            spec.bd_cust_recv_series,
        ]

        has_required = all(col in quarterly_data.columns for col in required_cols)
        if not has_required:
            print(f"Warning: Missing required columns. Have: {quarterly_data.columns}")
            return pl.DataFrame()

        # Filter to valid observations
        df = quarterly_data.with_columns(pl.col("date").cast(pl.Date)).sort("date")

        # Calculate HF equity buffer
        df = df.with_columns([
            (pl.col(spec.hf_assets_series) - pl.col(spec.hf_liabilities_series))
            .alias("hf_equity")
        ])

        # Validation: hf_equity must be positive
        df = df.filter(pl.col("hf_equity") > 0)

        if df.height == 0:
            return pl.DataFrame()

        # Calculate PB intensity (core quantity driver)
        df = df.with_columns([
            (pl.col(spec.hf_pb_borrowing_series) / pl.col("hf_equity"))
            .alias("pb_intensity")
        ])

        # Calculate growth rates (log changes for QoQ)
        df = df.with_columns([
            # PB intensity growth
            (pl.col("pb_intensity").log().diff()).alias("pb_intensity_g"),
            # Dealer supply growth
            (pl.col(spec.bd_cust_recv_series).log().diff()).alias("dealer_supply_g"),
        ])

        # Optional: channel split (repo vs PB)
        if spec.hf_repo_liab_series in quarterly_data.columns:
            df = df.with_columns([
                (pl.col(spec.hf_repo_liab_series) / pl.col("hf_equity"))
                .alias("repo_intensity"),
                (pl.col(spec.hf_repo_liab_series).log().diff())
                .alias("repo_intensity_g"),
            ])

        # Add year/quarter for alignment
        df = df.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.quarter().alias("quarter"),
        ])

        # Add rolling statistics for normalization
        df = df.with_columns([
            pl.col("pb_intensity_g")
            .rolling_mean(window_size=spec.zscore_window_quarters)
            .alias("pb_intensity_g_mean"),
            pl.col("pb_intensity_g")
            .rolling_std(window_size=spec.zscore_window_quarters)
            .alias("pb_intensity_g_std"),
            pl.col("dealer_supply_g")
            .rolling_mean(window_size=spec.zscore_window_quarters)
            .alias("dealer_supply_g_mean"),
            pl.col("dealer_supply_g")
            .rolling_std(window_size=spec.zscore_window_quarters)
            .alias("dealer_supply_g_std"),
        ])

        # Calculate z-scores
        df = df.with_columns([
            ((pl.col("pb_intensity_g") - pl.col("pb_intensity_g_mean"))
             / pl.col("pb_intensity_g_std")).alias("pb_intensity_g_z"),
            ((pl.col("dealer_supply_g") - pl.col("dealer_supply_g_mean"))
             / pl.col("dealer_supply_g_std")).alias("dealer_supply_g_z"),
        ])

        return df

    def _calculate_weekly_nowcast(
        self,
        quarterly_anchor: pl.DataFrame,
        weekly_cot: pl.DataFrame,
        weekly_pd: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calculate weekly nowcast of pb_intensity_g using bridge regression.

        V1 approach: Bridge regression mapping weekly features to quarterly outcome.
        1. Create weekly feature matrix (CFTC + NY Fed)
        2. Aggregate to quarterly means
        3. Fit bridge regression on historical data
        4. Nowcast current quarter using partial-quarter data
        """
        if weekly_cot.height == 0 and weekly_pd.height == 0:
            return pl.DataFrame()

        # Step 1: Create aggregate weekly features
        weekly_features = self._create_weekly_features(weekly_cot, weekly_pd)

        if weekly_features.height == 0:
            return pl.DataFrame()

        # Step 2: Aggregate weekly features to quarterly
        quarterly_features = self._aggregate_weekly_to_quarterly(weekly_features)

        if quarterly_features.height == 0:
            return pl.DataFrame()

        # Step 3: Fit bridge regression using historical quarterly data
        # (In practice, this would use sklearn/statsmodels with ridge regularization)
        # For now, use a simplified heuristic mapping
        nowcast_df = self._fit_bridge_nowcast(quarterly_anchor, quarterly_features)

        return nowcast_df

    def _create_weekly_features(
        self,
        weekly_cot: pl.DataFrame,
        weekly_pd: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Create standardized weekly feature set from COT and NY Fed PD data.

        Features:
        - equity_net_z: Mean z-score of equity index net positions (ES, NQ)
        - rates_net_z: Mean z-score of rates net positions (TU, FV, TY, US)
        - fx_net_z: Mean z-score of FX net positions (EC, JY)
        - dealer_financing_z: Mean z-score of repo volumes
        - settlement_friction_z: Mean z-score of fails
        """
        features_data = []

        def safe_mean(series: pl.Series) -> float | None:
            """Convert polars mean to Python float or None."""
            val = series.mean()
            if val is None or (hasattr(val, 'is_nan') and val.is_nan()):
                return None
            return float(val)

        # Process CFTC COT data
        if weekly_cot.height > 0:
            # Get unique dates
            dates = weekly_cot.select("report_date").unique().sort("report_date")

            for row in dates.iter_rows(named=True):
                report_date = row["report_date"]
                week_data = weekly_cot.filter(pl.col("report_date") == report_date)

                # Extract z-scores by asset class
                equity_contracts = ["ES", "NQ", "RTY"]
                rates_contracts = ["TU", "FV", "TY", "US", "SR3"]
                fx_contracts = ["EC", "JY", "BP"]

                equity_z = week_data.filter(pl.col("contract").is_in(equity_contracts))
                rates_z = week_data.filter(pl.col("contract").is_in(rates_contracts))
                fx_z = week_data.filter(pl.col("contract").is_in(fx_contracts))

                feature_row = {
                    "date": report_date,
                    "equity_net_z": (
                        safe_mean(equity_z["net_pct_oi_z_5y"])
                        if equity_z.height > 0 and "net_pct_oi_z_5y" in equity_z.columns
                        else None
                    ),
                    "rates_net_z": (
                        safe_mean(rates_z["net_pct_oi_z_5y"])
                        if rates_z.height > 0 and "net_pct_oi_z_5y" in rates_z.columns
                        else None
                    ),
                    "fx_net_z": (
                        safe_mean(fx_z["net_pct_oi_z_5y"])
                        if fx_z.height > 0 and "net_pct_oi_z_5y" in fx_z.columns
                        else None
                    ),
                }
                features_data.append(feature_row)

        # Process NY Fed PD data
        if weekly_pd.height > 0:
            # Get unique dates
            dates = weekly_pd.select("week_ending").unique().sort("week_ending")

            for row in dates.iter_rows(named=True):
                week_ending = row["week_ending"]
                week_data = weekly_pd.filter(pl.col("week_ending") == week_ending)

                # Extract z-scores by category
                repo_series = ["PD_RP_T_TOT", "PD_RRP_T_TOT"]
                fails_series = ["PD_AFtD_AG", "PD_AFtR_AG", "PD_AFtD_CORS", "PD_AFtR_CORS"]

                repo_z = week_data.filter(pl.col("series").is_in(repo_series))
                fails_z = week_data.filter(pl.col("series").is_in(fails_series))

                # Find matching feature row or create new
                matching_idx = None
                for i, f in enumerate(features_data):
                    # Match within 3 days (COT is Tuesday, PD is Wednesday)
                    date_diff = abs((f["date"] - week_ending).days) if f.get("date") else 999
                    if date_diff <= 3:
                        matching_idx = i
                        break

                if matching_idx is not None:
                    features_data[matching_idx]["dealer_financing_z"] = (
                        safe_mean(repo_z["value_z_3y"])
                        if repo_z.height > 0 and "value_z_3y" in repo_z.columns
                        else None
                    )
                    features_data[matching_idx]["settlement_friction_z"] = (
                        safe_mean(fails_z["value_z_3y"])
                        if fails_z.height > 0 and "value_z_3y" in fails_z.columns
                        else None
                    )
                else:
                    features_data.append({
                        "date": week_ending,
                        "equity_net_z": None,
                        "rates_net_z": None,
                        "fx_net_z": None,
                        "dealer_financing_z": (
                            safe_mean(repo_z["value_z_3y"])
                            if repo_z.height > 0 and "value_z_3y" in repo_z.columns
                            else None
                        ),
                        "settlement_friction_z": (
                            safe_mean(fails_z["value_z_3y"])
                            if fails_z.height > 0 and "value_z_3y" in fails_z.columns
                            else None
                        ),
                    })

        if not features_data:
            return pl.DataFrame()

        # Create DataFrame with explicit schema to handle None values
        schema = {
            "date": pl.Date,
            "equity_net_z": pl.Float64,
            "rates_net_z": pl.Float64,
            "fx_net_z": pl.Float64,
            "dealer_financing_z": pl.Float64,
            "settlement_friction_z": pl.Float64,
        }

        return pl.DataFrame(features_data, schema=schema).sort("date")

    def _aggregate_weekly_to_quarterly(self, weekly_features: pl.DataFrame) -> pl.DataFrame:
        """Aggregate weekly features to quarterly using within-quarter mean."""
        if weekly_features.height == 0:
            return pl.DataFrame()

        # Add year/quarter columns
        df = weekly_features.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.quarter().alias("quarter"),
        ])

        # Aggregate by quarter
        numeric_cols = [c for c in df.columns if c not in ["date", "year", "quarter"]]

        quarterly = df.group_by(["year", "quarter"]).agg([
            pl.col("date").max().alias("quarter_end_date"),
            *[pl.col(c).mean().alias(f"{c}_qtr") for c in numeric_cols],
            pl.col("date").count().alias("weeks_in_quarter"),
        ]).sort(["year", "quarter"])

        return quarterly

    def _fit_bridge_nowcast(
        self,
        quarterly_anchor: pl.DataFrame,
        quarterly_features: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Fit bridge regression and produce nowcast.

        Simplified V1: Use heuristic weights based on feature correlations.
        TODO: Replace with proper ridge regression using statsmodels/sklearn.
        """
        if quarterly_anchor.height == 0 or quarterly_features.height == 0:
            return pl.DataFrame()

        # Join quarterly features with anchor
        merged = quarterly_anchor.join(
            quarterly_features,
            on=["year", "quarter"],
            how="inner",
        )

        if merged.height == 0:
            return pl.DataFrame()

        # Simplified nowcast: weighted combination of weekly z-scores
        # These weights should be fit from data; using heuristics for V1
        # Equity positioning is most correlated with HF leverage appetite
        # TODO: Calibrate via historical backtest on training sample

        feature_cols = [
            "equity_net_z_qtr",
            "rates_net_z_qtr",
            "fx_net_z_qtr",
            "dealer_financing_z_qtr",
        ]

        # Check which features are available
        available_features = [c for c in feature_cols if c in merged.columns]

        if not available_features:
            # No weekly features, use quarterly anchor only
            return merged.select([
                "date", "year", "quarter",
                "pb_intensity_g",
                pl.lit(None).cast(pl.Float64).alias("pb_intensity_g_nowcast"),
            ])

        # Heuristic weights (equity positioning is primary signal)
        weights = {
            "equity_net_z_qtr": 0.40,
            "rates_net_z_qtr": 0.20,
            "fx_net_z_qtr": 0.15,
            "dealer_financing_z_qtr": 0.25,
        }

        # Calculate nowcast as weighted sum, scaled to match pb_intensity_g distribution
        pb_g_std = merged["pb_intensity_g"].drop_nulls().std() or 0.05
        pb_g_mean = merged["pb_intensity_g"].drop_nulls().mean() or 0.0

        nowcast_expr = pl.lit(0.0)
        total_weight = 0.0

        for col in available_features:
            w = weights.get(col, 0.1)
            nowcast_expr = nowcast_expr + pl.col(col).fill_null(0) * w
            total_weight += w

        if total_weight > 0:
            nowcast_expr = nowcast_expr / total_weight

        # Scale to match pb_intensity_g distribution
        merged = merged.with_columns([
            (nowcast_expr * pb_g_std + pb_g_mean).alias("pb_intensity_g_nowcast")
        ])

        return merged

    def _calculate_composite_forecast(
        self,
        quarterly_anchor: pl.DataFrame,
        weekly_nowcast: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calculate composite PB_Lead forecast signal.

        PB_Lead = a * pb_intensity_g_nowcast + b * pb_intensity_g(t-1) + c * dealer_supply_g

        This is the main output for 1-2 quarter forecasting.
        """
        spec = self._spec

        if weekly_nowcast.height > 0:
            # Use merged data with nowcast
            df = weekly_nowcast.clone()
        else:
            # Fall back to quarterly anchor only
            df = quarterly_anchor.clone()
            df = df.with_columns([
                pl.lit(None).cast(pl.Float64).alias("pb_intensity_g_nowcast")
            ])

        # Add lagged pb_intensity_g
        df = df.with_columns([
            pl.col("pb_intensity_g").shift(1).alias("pb_intensity_g_lag1")
        ])

        # Calculate composite PB_Lead
        # Use nowcast if available, else fall back to current quarter value
        nowcast_term = pl.when(pl.col("pb_intensity_g_nowcast").is_not_null()).then(
            pl.col("pb_intensity_g_nowcast") * spec.nowcast_weight
        ).otherwise(
            pl.col("pb_intensity_g") * spec.nowcast_weight
        )

        lagged_term = pl.col("pb_intensity_g_lag1") * spec.lagged_intensity_weight
        dealer_term = pl.col("dealer_supply_g") * spec.dealer_supply_weight

        df = df.with_columns([
            (nowcast_term + lagged_term + dealer_term).alias("pb_lead")
        ])

        # Add rolling statistics for pb_lead
        df = df.with_columns([
            pl.col("pb_lead")
            .rolling_mean(window_size=spec.zscore_window_quarters)
            .alias("pb_lead_mean"),
            pl.col("pb_lead")
            .rolling_std(window_size=spec.zscore_window_quarters)
            .alias("pb_lead_std"),
        ])

        df = df.with_columns([
            ((pl.col("pb_lead") - pl.col("pb_lead_mean"))
             / pl.col("pb_lead_std")).alias("pb_lead_z")
        ])

        return df

    def _classify_run_rate_regime(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Classify run-rate regimes using rolling percentiles.

        Regimes (based on PB_Lead percentile):
        - Accelerating: Top tercile (>67th percentile)
        - Stable: Middle tercile
        - Decelerating: Bottom tercile (<33rd percentile)

        This is intentionally NOT a stress classifier; it's a run-rate classifier.
        """
        spec = self._spec

        if df.height == 0:
            return df

        # Calculate rolling percentile rank
        pb_lead_values = df.filter(pl.col("pb_lead").is_not_null())["pb_lead"]

        if pb_lead_values.len() == 0:
            return df.with_columns([
                pl.lit("Unknown").alias("run_rate_regime"),
                pl.lit(None).cast(pl.Float64).alias("pb_lead_percentile"),
            ])

        p33 = pb_lead_values.quantile(spec.decelerating_percentile)
        p67 = pb_lead_values.quantile(spec.accelerating_percentile)

        # Calculate percentile rank
        df = df.with_columns([
            (pl.col("pb_lead").rank(method="average") / pb_lead_values.len() * 100)
            .alias("pb_lead_percentile")
        ])

        # Classify regime
        df = df.with_columns([
            pl.when(pl.col("pb_lead") > p67)
            .then(pl.lit("Accelerating"))
            .when(pl.col("pb_lead") < p33)
            .then(pl.lit("Decelerating"))
            .otherwise(pl.lit("Stable"))
            .alias("run_rate_regime")
        ])

        # Add stress flag (secondary diagnostic)
        df = df.with_columns([
            ((pl.col("pb_intensity_g") < spec.stress_pb_intensity_drop) |
             (pl.col("dealer_supply_g") < spec.stress_dealer_supply_drop))
            .alias("stress_flag")
        ])

        return df

    def _calculate_attribution(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        Calculate attribution decomposition for the latest period.

        Shows contribution from:
        1. HF demand (pb_intensity_g nowcast)
        2. Lagged intensity
        3. Dealer supply growth
        """
        spec = self._spec

        if df.height == 0:
            return {}

        latest = df.tail(1)

        nowcast_val = latest["pb_intensity_g_nowcast"][0]
        if nowcast_val is None:
            nowcast_val = latest["pb_intensity_g"][0] or 0.0

        lagged_val = latest["pb_intensity_g_lag1"][0] or 0.0
        dealer_val = latest["dealer_supply_g"][0] or 0.0
        pb_lead = latest["pb_lead"][0] or 0.0

        # Calculate contributions
        nowcast_contrib = nowcast_val * spec.nowcast_weight
        lagged_contrib = lagged_val * spec.lagged_intensity_weight
        dealer_contrib = dealer_val * spec.dealer_supply_weight

        total = abs(nowcast_contrib) + abs(lagged_contrib) + abs(dealer_contrib)

        return {
            "hf_demand_contribution": float(nowcast_contrib),
            "hf_demand_pct": float(abs(nowcast_contrib) / total * 100) if total > 0 else 0,
            "lagged_intensity_contribution": float(lagged_contrib),
            "lagged_intensity_pct": float(abs(lagged_contrib) / total * 100) if total > 0 else 0,
            "dealer_supply_contribution": float(dealer_contrib),
            "dealer_supply_pct": float(abs(dealer_contrib) / total * 100) if total > 0 else 0,
            "pb_lead_total": float(pb_lead),
        }

    def _generate_forecasts(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        Generate t+1 and t+2 forecasts with confidence bands.

        Simple AR-style forecast using pb_lead as predictor of dealer_supply_g.
        Confidence bands from historical nowcast error.
        """
        if df.height < 4:
            return {}

        # Use simple persistence + mean reversion for forecasting
        latest = df.tail(4)  # Last 4 quarters

        pb_lead_current = latest["pb_lead"][-1]
        pb_lead_mean = df["pb_lead"].mean()
        pb_lead_std = df["pb_lead"].std()

        if pb_lead_current is None or pb_lead_mean is None:
            return {}

        # Simple mean-reversion forecast
        # t+1: weighted average of current and mean
        # t+2: more weight on mean (further reversion)
        t1_forecast = 0.7 * pb_lead_current + 0.3 * pb_lead_mean
        t2_forecast = 0.5 * pb_lead_current + 0.5 * pb_lead_mean

        # Confidence bands (1 std historical error)
        conf_band = pb_lead_std * 1.0 if pb_lead_std else 0.02

        return {
            "t_plus_1": {
                "forecast": float(t1_forecast),
                "lower_bound": float(t1_forecast - conf_band),
                "upper_bound": float(t1_forecast + conf_band),
            },
            "t_plus_2": {
                "forecast": float(t2_forecast),
                "lower_bound": float(t2_forecast - 1.5 * conf_band),
                "upper_bound": float(t2_forecast + 1.5 * conf_band),
            },
        }

    def _calculate_summary(self, df: pl.DataFrame) -> dict[str, Any]:
        """Calculate summary statistics."""
        if df.height == 0:
            return {}

        pb_lead = df["pb_lead"].drop_nulls()
        pb_intensity_g = df["pb_intensity_g"].drop_nulls()
        dealer_supply_g = df["dealer_supply_g"].drop_nulls()

        return {
            "pb_lead_mean": float(pb_lead.mean()) if pb_lead.len() > 0 else None,
            "pb_lead_std": float(pb_lead.std()) if pb_lead.len() > 0 else None,
            "pb_intensity_g_mean": float(pb_intensity_g.mean()) if pb_intensity_g.len() > 0 else None,
            "pb_intensity_g_std": float(pb_intensity_g.std()) if pb_intensity_g.len() > 0 else None,
            "dealer_supply_g_mean": float(dealer_supply_g.mean()) if dealer_supply_g.len() > 0 else None,
            "dealer_supply_g_std": float(dealer_supply_g.std()) if dealer_supply_g.len() > 0 else None,
            "pct_accelerating": float((df["run_rate_regime"] == "Accelerating").mean()),
            "pct_stable": float((df["run_rate_regime"] == "Stable").mean()),
            "pct_decelerating": float((df["run_rate_regime"] == "Decelerating").mean()),
        }

    def _get_current_signal(self, df: pl.DataFrame) -> dict[str, Any]:
        """Get current signal based on latest data."""
        if df.height == 0:
            return {"signal": 0, "reason": "No data"}

        latest = df.tail(1)

        regime = latest["run_rate_regime"][0]
        pb_lead_z = latest["pb_lead_z"][0]
        stress_flag = latest["stress_flag"][0] if "stress_flag" in latest.columns else False

        # Map regime to signal
        signal_map = {
            "Accelerating": 1,
            "Stable": 0,
            "Decelerating": -1,
        }

        signal = signal_map.get(regime, 0)

        # Override with stress if flagged
        if stress_flag:
            return {
                "signal": -1,
                "reason": "Stress flag triggered (sharp drop in PB intensity or dealer supply)",
                "regime": regime,
                "pb_lead_z": float(pb_lead_z) if pb_lead_z else None,
                "stress_flag": True,
            }

        reasons = {
            "Accelerating": "Improving prime balances/revenue momentum likely 1-2 quarters ahead",
            "Stable": "Neutral run-rate environment",
            "Decelerating": "Slowing balances/revenue momentum likely 1-2 quarters ahead",
        }

        return {
            "signal": signal,
            "reason": reasons.get(regime, "Unknown regime"),
            "regime": regime,
            "pb_lead_z": float(pb_lead_z) if pb_lead_z else None,
            "stress_flag": False,
        }

    def get_regime_interpretation(self, regime: str) -> dict[str, str]:
        """Get interpretation and recommended actions for a run-rate regime."""
        interpretations = {
            "Accelerating": {
                "description": "Top tercile of PB_Lead composite signal",
                "revenue_outlook": "Improving prime brokerage balances and revenue momentum",
                "expected_impact": "1-2 quarters of positive run-rate tailwind",
                "action": "Position for revenue growth; monitor for regime peak",
            },
            "Stable": {
                "description": "Middle tercile of PB_Lead composite signal",
                "revenue_outlook": "Neutral run-rate environment",
                "expected_impact": "Continuation of current trend",
                "action": "Business as usual; watch for regime transitions",
            },
            "Decelerating": {
                "description": "Bottom tercile of PB_Lead composite signal",
                "revenue_outlook": "Slowing prime brokerage balances and revenue momentum",
                "expected_impact": "1-2 quarters of negative run-rate headwind",
                "action": "Prepare for lower revenues; look for recovery signals",
            },
        }
        return interpretations.get(regime, {"description": "Unknown regime"})

    def nowcast(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """
        Generate weekly nowcast update for current quarter.

        This provides a higher-frequency update to the quarterly anchor,
        useful for tracking intra-quarter developments.
        """
        return self.calculate(data, **kwargs)

    def calculate_shadow_extension(
        self,
        data: dict[str, pl.DataFrame],
        result: IndicatorResult,
        as_of_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Calculate shadow/nowcast extension beyond official Z.1 data.

        Implements the three-layer publication system:
        1. Official layer: through last Z.1 release (e.g., Q2'25)
        2. Shadow backfill: completed but unreported quarters (e.g., Q3'25)
        3. Progressive nowcast: current quarter (e.g., Q4'25)

        Args:
            data: Data dictionary from fetch_data
            result: IndicatorResult from calculate()
            as_of_date: As-of date for nowcast (defaults to today)

        Returns:
            Dictionary with shadow estimates and forecasts
        """
        from .shadow_nowcast import (
            ShadowNowcaster,
            ShadowNowcastConfig,
            get_quarter_from_date,
            format_quarter_label,
        )
        from datetime import date

        as_of = as_of_date.date() if as_of_date else date.today()

        # Get weekly data
        weekly_cot = data.get("weekly_cot", pl.DataFrame())
        weekly_pd = data.get("weekly_pd", pl.DataFrame())

        # Initialize shadow nowcaster
        nowcaster = ShadowNowcaster()

        # Build weekly factor (may be empty if APIs unavailable)
        weekly_factor = nowcaster.build_weekly_factor(weekly_cot, weekly_pd)

        # Even without weekly data, we can estimate using dealer supply and mean-reversion
        has_weekly_data = weekly_factor.height > 0

        # Fit shadow model on historical data
        quarterly_anchor = result.data

        model_fit = nowcaster.fit_shadow_model(
            quarterly_anchor,
            weekly_factor,
        )

        if not model_fit.get("success"):
            return {
                "success": False,
                "reason": model_fit.get("reason", "Model fitting failed"),
            }

        # Determine quarters to estimate
        # Last official quarter from Z.1 data
        last_official_date = result.data["date"].max()
        last_official_year = last_official_date.year
        last_official_quarter = (last_official_date.month - 1) // 3 + 1

        # Current quarter
        current_year, current_quarter = get_quarter_from_date(as_of)

        # Last official PB intensity level
        last_official_intensity = float(result.data["pb_intensity"][-1])

        # Generate shadow estimates
        shadow_estimates = []

        # Shadow backfill for quarters between last official and current
        q_year, q_num = last_official_year, last_official_quarter

        # Track last pb_intensity_g for AR(1) estimation
        last_pb_intensity_g = float(result.data["pb_intensity_g"][-1]) if result.data["pb_intensity_g"][-1] is not None else None
        quarters_from_official = 0

        while True:
            # Move to next quarter
            q_num += 1
            if q_num > 4:
                q_num = 1
                q_year += 1

            quarters_from_official += 1

            # Stop if we've passed current quarter
            if (q_year > current_year) or (q_year == current_year and q_num > current_quarter):
                break

            # Get dealer supply if available (use last known value with mean-reversion)
            dealer_supply_g = None
            if "dealer_supply_g" in result.data.columns:
                last_dealer_g = result.data["dealer_supply_g"][-1]
                if last_dealer_g is not None:
                    # Mean-revert dealer supply toward historical average
                    dealer_mean = float(result.data["dealer_supply_g"].mean())
                    dealer_supply_g = float(0.7 ** quarters_from_official * last_dealer_g +
                                           (1 - 0.7 ** quarters_from_official) * dealer_mean)

            # Determine if shadow (complete) or nowcast (partial)
            if q_year < current_year or (q_year == current_year and q_num < current_quarter):
                # Completed quarter - shadow estimate
                estimate = nowcaster.estimate_shadow_quarter(
                    q_year, q_num, weekly_factor,
                    last_official_intensity,
                    dealer_supply_g=dealer_supply_g,
                    completeness_disclosure=0.0,  # TODO: Add disclosure pulse
                    last_pb_intensity_g=last_pb_intensity_g,
                    quarters_from_official=quarters_from_official,
                )
            else:
                # Current quarter - progressive nowcast
                estimate = nowcaster.estimate_shadow_quarter(
                    q_year, q_num, weekly_factor,
                    last_official_intensity,
                    dealer_supply_g=dealer_supply_g,
                    as_of_date=as_of,
                    completeness_disclosure=0.0,
                    last_pb_intensity_g=last_pb_intensity_g,
                    quarters_from_official=quarters_from_official,
                )

            shadow_estimates.append(estimate)
            last_official_intensity = estimate.pb_intensity_level
            last_pb_intensity_g = estimate.pb_intensity_g  # Chain for next estimate

        # Chain levels forward
        if shadow_estimates:
            base_level = float(result.data["pb_intensity"][-1])
            shadow_estimates = nowcaster.chain_intensity_levels(shadow_estimates, base_level)

        # Generate forecasts for next 2 quarters beyond current
        forecast_estimates = []
        forecast_year, forecast_quarter = current_year, current_quarter

        for _ in range(2):
            forecast_quarter += 1
            if forecast_quarter > 4:
                forecast_quarter = 1
                forecast_year += 1

            # Use mean-reversion forecast
            pb_lead_mean = result.metadata.get("summary", {}).get("pb_lead_mean", 0.01)
            pb_lead_std = result.metadata.get("summary", {}).get("pb_lead_std", 0.04)

            # Current pb_lead if available
            current_pb_lead = result.data["pb_lead"][-1] if "pb_lead" in result.data.columns else pb_lead_mean

            # Mean reversion
            reversion_speed = 0.3  # Revert 30% per quarter
            forecast_pb_lead = current_pb_lead * (1 - reversion_speed) + pb_lead_mean * reversion_speed

            forecast_estimates.append({
                "quarter": f"{forecast_year}Q{forecast_quarter}",
                "quarter_label": format_quarter_label(forecast_year, forecast_quarter),
                "pb_lead_forecast": float(forecast_pb_lead),
                "confidence_lower": float(forecast_pb_lead - 1.96 * pb_lead_std),
                "confidence_upper": float(forecast_pb_lead + 1.96 * pb_lead_std),
            })

            current_pb_lead = forecast_pb_lead

        # Build output
        shadow_output = []
        for est in shadow_estimates:
            shadow_output.append({
                "quarter": est.quarter,
                "quarter_label": format_quarter_label(
                    int(est.quarter[:4]),
                    int(est.quarter[-1])
                ),
                "estimate_type": est.estimate_type,
                "pb_intensity_g": est.pb_intensity_g,
                "pb_intensity_g_pct": est.pb_intensity_g * 100,
                "pb_intensity_level": est.pb_intensity_level,
                "uncertainty_std": est.uncertainty_std,
                "confidence_lower": est.confidence_lower,
                "confidence_upper": est.confidence_upper,
                "confidence_lower_pct": est.confidence_lower * 100,
                "confidence_upper_pct": est.confidence_upper * 100,
                "completeness_weekly": est.completeness_weekly,
                "completeness_disclosure": est.completeness_disclosure,
                "as_of_date": str(est.as_of_date),
                "components": est.components,
            })

        return {
            "success": True,
            "as_of_date": str(as_of),
            "last_official_quarter": f"{last_official_year}Q{last_official_quarter}",
            "last_official_quarter_label": format_quarter_label(last_official_year, last_official_quarter),
            "model_fit": model_fit,
            "shadow_estimates": shadow_output,
            "forecasts": forecast_estimates,
            "methodology_note": (
                "Shadow estimates use ridge regression on weekly leverage appetite factor. "
                "Uncertainty bands widen for quarters with less weekly data. "
                "Estimates will be replaced with official Z.1 values when released."
            ),
        }
