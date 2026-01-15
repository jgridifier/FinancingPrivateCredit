"""
Prime Rehypothecation Liquidity Indicator

Measures liquidity creation through collateral rehypothecation in the prime
brokerage system. Based on haircut differentials between client and third-party
financing.

RLI = (Margin_Receivables + Repo_Liabilities) × Haircut_Spread

Key signals:
- RLI_Velocity < -20%: Severe liquidity contraction
- Haircut_Spread > 6%: Extreme stress regime
- RLI_Normalized < 1%: Below crisis levels

References:
- Eren, E. (2014). "Intermediary Funding Liquidity and Rehypothecation"
- Singh, M., & Aitken, J. (2010). "The (Sizable) Role of Rehypothecation in Shadow Banking"
- Infante, S. (2015). "Liquidity Windfalls: The Consequences of Repo Rehypothecation"
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
class RehypoLiquiditySpec:
    """Configuration for Rehypothecation Liquidity Index calculation."""

    name: str = "default"
    description: str = "Default rehypothecation liquidity specification"

    # Haircut spread calibration - Low volatility regime (VIX < 20)
    # TODO: Replace static calibration with market-derived haircut spreads using:
    # 1. DTCC repo data for actual repo haircuts by collateral type
    # 2. Prime broker disclosures from 10-K/10-Q filings
    # 3. Fed repo facility rates as benchmark
    # Current calibration from Singh & Aitken (2010), Eren (2014)
    alpha_low: float = 0.02  # Base spread 2% in low vol
    beta_low: float = 0.0005  # Spread widens 0.5 bps per VIX point

    # Haircut spread calibration - High volatility regime (VIX >= 20)
    alpha_high: float = 0.03  # Base spread 3% in high vol
    beta_high: float = 0.002  # Spread widens 2 bps per VIX point

    # VIX threshold for regime switching
    vix_regime_threshold: float = 20.0

    # Critical thresholds for signals
    severe_contraction_velocity: float = -20.0  # percent QoQ
    extreme_stress_haircut: float = 0.06  # 6%
    crisis_normalized_rli: float = 0.01  # 1% of total assets

    @classmethod
    def from_dict(cls, d: dict) -> "RehypoLiquiditySpec":
        """Create spec from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@register_indicator("prime_rehypo_liquidity")
class PrimeRehypoLiquidityIndicator(BaseIndicator):
    """
    Prime Rehypothecation Liquidity Indicator.

    Measures liquidity creation through collateral rehypothecation in the
    prime brokerage system.

    Core mechanism:
    1. Hedge fund pledges collateral worth $100, receives $90 cash (10% haircut)
    2. Prime broker repledges same collateral to MMF for $95 (5% haircut)
    3. Prime broker retains $5 = liquidity creation through haircut differential

    Risk: During stress, third-party haircuts increase faster than client
    haircuts, causing liquidity to evaporate.

    References:
    - Eren, E. (2014). "Intermediary Funding Liquidity and Rehypothecation"
    - Singh, M., & Aitken, J. (2010). "The (Sizable) Role of Rehypothecation"
    - Kirk et al. (2014). "Matching Collateral Supply and Financing Demands"
    """

    supports_nowcast: bool = False

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the indicator."""
        super().__init__(config_path)
        self._spec: Optional[RehypoLiquiditySpec] = None

    def get_metadata(self) -> IndicatorMetadata:
        """Return indicator metadata."""
        return IndicatorMetadata(
            name="Prime Rehypothecation Liquidity Index",
            short_name="RLI",
            description=(
                "Measures liquidity creation through collateral rehypothecation using "
                "broker-dealer margin receivables, repo liabilities, and VIX-based "
                "haircut spread estimation. Identifies funding liquidity stress in "
                "the prime brokerage system."
            ),
            version="1.0.0",
            paper_reference="Eren (2014); Singh & Aitken (2010); Infante (2015)",
            data_sources=[
                "FRED (BOGZ1FL663067003Q, BOGZ1FL662151003Q, BOGZ1FL664090005Q, VIXCLS)"
            ],
            update_frequency="quarterly",
            lookback_periods=40,  # 10 years of quarterly data
        )

    def get_required_data_sources(self) -> list[str]:
        """Document data sources needed."""
        return ["margin_receivables", "repo_liabilities", "total_bd_assets", "vix"]

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch data required for Rehypothecation Liquidity Index calculation.

        Fetches from FRED:
        - BOGZ1FL663067003Q: Broker-Dealer Margin Receivables (quarterly)
        - BOGZ1FL662151003Q: Broker-Dealer Repo Liabilities (quarterly)
        - BOGZ1FL664090005Q: Broker-Dealer Total Assets (quarterly)
        - VIXCLS: CBOE Volatility Index (daily)
        """
        from ...core import DataRegistry

        registry = DataRegistry.get_instance()

        # Fetch FRED series
        fred_series = [
            "BOGZ1FL663067003Q",  # Margin receivables
            "BOGZ1FL662151003Q",  # Repo liabilities
            "BOGZ1FL664090005Q",  # Total B-D assets
            "VIXCLS",  # VIX daily
        ]

        try:
            macro_data = registry.get_macro_series(fred_series, start_date)
        except Exception:
            macro_data = pl.DataFrame({"date": []})

        return {"macro_data": macro_data}

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        spec: Optional[RehypoLiquiditySpec] = None,
        **kwargs,
    ) -> IndicatorResult:
        """
        Calculate the Rehypothecation Liquidity Index and derivative indicators.

        Args:
            data: Dictionary with macro_data containing FRED series
            spec: Optional specification override

        Returns:
            IndicatorResult with RLI, normalized RLI, velocity, and regime info
        """
        if spec is not None:
            self._spec = spec
        elif self._spec is None:
            self._spec = RehypoLiquiditySpec()

        macro_data = data.get("macro_data", pl.DataFrame())

        if macro_data.height == 0:
            return IndicatorResult(
                indicator_name="prime_rehypo_liquidity",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No macro data available"},
            )

        # Process the data
        result_df = self._calculate_rli(macro_data)

        if result_df.height == 0:
            return IndicatorResult(
                indicator_name="prime_rehypo_liquidity",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "Insufficient data for RLI calculation"},
            )

        # Calculate summary statistics
        summary = self._calculate_summary(result_df)

        return IndicatorResult(
            indicator_name="prime_rehypo_liquidity",
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
                "current_regime": self._get_current_regime(result_df),
                "current_signal": self._get_current_signal(result_df),
            },
        )

    def _calculate_haircut_spread(self, vix: float) -> float:
        """
        Calculate haircut spread based on VIX regime.

        Low volatility (VIX < 20): α_low + β_low × VIX
        High volatility (VIX >= 20): α_high + β_high × VIX
        """
        if vix < self._spec.vix_regime_threshold:
            return self._spec.alpha_low + self._spec.beta_low * vix
        else:
            return self._spec.alpha_high + self._spec.beta_high * vix

    def _calculate_rli(self, macro_data: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Rehypothecation Liquidity Index from FRED data.

        Steps:
        1. Aggregate VIX to quarterly average
        2. Merge quarterly broker-dealer data
        3. Calculate haircut spread based on VIX regime
        4. Compute RLI = collateral_base × haircut_spread
        5. Add normalized RLI and velocity
        """
        # Check for required columns
        has_margin = "BOGZ1FL663067003Q" in macro_data.columns
        has_repo = "BOGZ1FL662151003Q" in macro_data.columns
        has_assets = "BOGZ1FL664090005Q" in macro_data.columns
        has_vix = "VIXCLS" in macro_data.columns

        if not (has_margin and has_repo and has_vix):
            return pl.DataFrame()

        # Ensure date column is proper datetime
        df = macro_data.with_columns(pl.col("date").cast(pl.Date))

        # Step 1: Calculate quarterly VIX average
        df = df.sort("date").with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.quarter().alias("quarter"),
        ])

        # Aggregate VIX to quarterly
        vix_quarterly = df.filter(pl.col("VIXCLS").is_not_null()).group_by(
            ["year", "quarter"]
        ).agg([
            pl.col("VIXCLS").mean().alias("vix_avg"),
        ])

        # Get quarterly broker-dealer data (last observation per quarter)
        bd_quarterly = df.group_by(["year", "quarter"]).agg([
            pl.col("date").max().alias("date"),
            pl.col("BOGZ1FL663067003Q").last().alias("margin_receivables"),
            pl.col("BOGZ1FL662151003Q").last().alias("repo_liabilities"),
            pl.col("BOGZ1FL664090005Q").last().alias("total_bd_assets") if has_assets else pl.lit(None).alias("total_bd_assets"),
        ])

        # Step 2: Merge quarterly data
        quarterly = bd_quarterly.join(
            vix_quarterly, on=["year", "quarter"], how="inner"
        ).sort("date")

        # Filter out rows with null data
        quarterly = quarterly.filter(
            pl.col("margin_receivables").is_not_null() &
            pl.col("repo_liabilities").is_not_null() &
            pl.col("vix_avg").is_not_null()
        )

        if quarterly.height == 0:
            return pl.DataFrame()

        # Step 3: Calculate haircut spread based on VIX regime
        # TODO: Enhance with actual market haircut data from:
        # 1. DTCC GCF repo index for Treasury/Agency haircuts
        # 2. Prime broker 10-K disclosures for client haircut schedules
        # 3. Fed RRP/ON RRP rates as floor for third-party financing
        quarterly = quarterly.with_columns([
            pl.when(pl.col("vix_avg") < self._spec.vix_regime_threshold)
            .then(self._spec.alpha_low + self._spec.beta_low * pl.col("vix_avg"))
            .otherwise(self._spec.alpha_high + self._spec.beta_high * pl.col("vix_avg"))
            .alias("haircut_spread"),
            # Volatility regime classification
            pl.when(pl.col("vix_avg") < self._spec.vix_regime_threshold)
            .then(pl.lit("Low"))
            .otherwise(pl.lit("High"))
            .alias("vol_regime"),
        ])

        # Step 4: Calculate RLI components
        quarterly = quarterly.with_columns([
            # Total collateral base (millions USD)
            (pl.col("margin_receivables") + pl.col("repo_liabilities"))
            .alias("collateral_base"),
        ])

        # Calculate RLI (millions USD)
        quarterly = quarterly.with_columns([
            (pl.col("collateral_base") * pl.col("haircut_spread")).alias("rli"),
        ])

        # Step 5: Calculate normalized RLI and velocity
        quarterly = quarterly.with_columns([
            # Normalized RLI (as fraction of total B-D assets)
            pl.when(pl.col("total_bd_assets").is_not_null() & (pl.col("total_bd_assets") > 0))
            .then(pl.col("rli") / pl.col("total_bd_assets"))
            .otherwise(pl.lit(None))
            .alias("rli_normalized"),
            # RLI velocity (QoQ percent change)
            (pl.col("rli").pct_change() * 100).alias("rli_velocity"),
        ])

        # Select final output columns
        return quarterly.select([
            "date",
            "margin_receivables",
            "repo_liabilities",
            "collateral_base",
            "total_bd_assets",
            "vix_avg",
            "vol_regime",
            "haircut_spread",
            "rli",
            "rli_normalized",
            "rli_velocity",
        ])

    def _calculate_summary(self, df: pl.DataFrame) -> dict[str, Any]:
        """Calculate summary statistics for the RLI."""
        if df.height == 0:
            return {}

        rli_col = df["rli"].drop_nulls()
        if rli_col.len() == 0:
            return {}

        velocity_col = df["rli_velocity"].drop_nulls()

        return {
            "rli_mean": float(rli_col.mean()),
            "rli_std": float(rli_col.std()),
            "rli_min": float(rli_col.min()),
            "rli_max": float(rli_col.max()),
            "rli_p10": float(rli_col.quantile(0.10)),
            "rli_p90": float(rli_col.quantile(0.90)),
            "velocity_mean": float(velocity_col.mean()) if velocity_col.len() > 0 else None,
            "velocity_std": float(velocity_col.std()) if velocity_col.len() > 0 else None,
            "pct_high_vol_regime": float((df["vol_regime"] == "High").mean()),
        }

    def _get_current_regime(self, df: pl.DataFrame) -> dict[str, Any]:
        """Get current volatility regime and haircut spread."""
        if df.height == 0:
            return {"regime": "Unknown"}

        latest = df.tail(1)
        return {
            "vol_regime": latest["vol_regime"][0],
            "vix_avg": float(latest["vix_avg"][0]) if latest["vix_avg"][0] is not None else None,
            "haircut_spread": float(latest["haircut_spread"][0]) if latest["haircut_spread"][0] is not None else None,
            "haircut_spread_pct": f"{latest['haircut_spread'][0] * 100:.1f}%" if latest["haircut_spread"][0] is not None else None,
        }

    def _get_current_signal(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        Get current signal based on latest data.

        Critical thresholds:
        - RLI_Velocity < -20%: Severe liquidity contraction
        - Haircut_Spread > 6%: Extreme stress regime
        - RLI_Normalized < 1%: Below crisis levels
        """
        if df.height == 0:
            return {"signal": 0, "reason": "No data"}

        latest = df.tail(1)
        rli_velocity = latest["rli_velocity"][0]
        haircut_spread = latest["haircut_spread"][0]
        rli_normalized = latest["rli_normalized"][0]

        signals = []

        # Check severe contraction
        if rli_velocity is not None and rli_velocity < self._spec.severe_contraction_velocity:
            signals.append({
                "type": "severe_contraction",
                "message": f"RLI velocity {rli_velocity:.1f}% < {self._spec.severe_contraction_velocity}%",
                "severity": "high",
            })

        # Check extreme stress haircut
        if haircut_spread is not None and haircut_spread > self._spec.extreme_stress_haircut:
            signals.append({
                "type": "extreme_stress",
                "message": f"Haircut spread {haircut_spread:.1%} > {self._spec.extreme_stress_haircut:.0%}",
                "severity": "high",
            })

        # Check crisis level normalized RLI
        if rli_normalized is not None and rli_normalized < self._spec.crisis_normalized_rli:
            signals.append({
                "type": "crisis_level",
                "message": f"Normalized RLI {rli_normalized:.2%} < {self._spec.crisis_normalized_rli:.0%}",
                "severity": "critical",
            })

        if signals:
            # Return most severe signal
            severity_order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
            signals.sort(key=lambda x: severity_order.get(x["severity"], 0), reverse=True)
            return {
                "signal": -1,
                "reason": signals[0]["message"],
                "all_signals": signals,
                "rli_velocity": float(rli_velocity) if rli_velocity is not None else None,
                "haircut_spread": float(haircut_spread) if haircut_spread is not None else None,
            }

        return {
            "signal": 0,
            "reason": "Normal conditions",
            "rli_velocity": float(rli_velocity) if rli_velocity is not None else None,
            "haircut_spread": float(haircut_spread) if haircut_spread is not None else None,
        }

    def get_stress_interpretation(
        self,
        rli_velocity: float,
        haircut_spread: float,
    ) -> dict[str, str]:
        """
        Get interpretation of stress levels.

        Args:
            rli_velocity: Quarter-over-quarter percent change
            haircut_spread: Current haircut spread (decimal)

        Returns:
            Dictionary with interpretation and expected revenue impact
        """
        # Velocity interpretation
        if rli_velocity < -20:
            velocity_interp = "Severe liquidity contraction - expect 10-15% revenue decline"
        elif rli_velocity < -10:
            velocity_interp = "Moderate contraction - monitor closely"
        elif rli_velocity > 10:
            velocity_interp = "Strong liquidity expansion - supportive of revenue"
        else:
            velocity_interp = "Normal velocity range"

        # Haircut spread interpretation
        if haircut_spread > 0.06:
            spread_interp = "Extreme stress - historical median -25% revenue impact"
        elif haircut_spread > 0.04:
            spread_interp = "Elevated stress - increased funding costs"
        elif haircut_spread < 0.025:
            spread_interp = "Favorable conditions - low funding costs"
        else:
            spread_interp = "Normal spread environment"

        return {
            "velocity_interpretation": velocity_interp,
            "spread_interpretation": spread_interp,
            "overall": (
                "STRESS" if rli_velocity < -20 or haircut_spread > 0.06
                else "WATCH" if rli_velocity < -10 or haircut_spread > 0.04
                else "NORMAL"
            ),
        }

    def get_historical_benchmarks(self) -> dict[str, dict[str, Any]]:
        """
        Return historical benchmark values for key stress events.

        These can be used to contextualize current readings.
        """
        return {
            "lehman_crisis_q4_2008": {
                "description": "Lehman Crisis peak stress",
                "expected_rli_velocity": -35.0,
                "expected_haircut_spread": 0.08,
                "expected_rli_normalized": 0.008,
            },
            "covid_crash_q1_2020": {
                "description": "COVID-19 market crash",
                "expected_rli_velocity": -25.0,
                "expected_haircut_spread": 0.065,
                "expected_rli_normalized": 0.012,
            },
            "archegos_q1_2021": {
                "description": "Archegos collapse",
                "expected_rli_velocity": -8.0,
                "expected_haircut_spread": 0.035,
                "notes": "Localized event, limited systemic impact",
            },
            "normal_conditions": {
                "description": "Typical non-stress period",
                "expected_rli_velocity_range": (-5.0, 5.0),
                "expected_haircut_spread_range": (0.02, 0.035),
                "expected_rli_normalized_range": (0.015, 0.025),
            },
        }
