"""
Funding Stability Score Indicator

Novel Insight 3: Creates a "funding resilience" score that predicts which banks
will cut lending most aggressively in downturns based on the paper's finding
that banks respond more procyclically to macro conditions.

Reformulated Score Components:
1. Stability Factors (positive contribution):
   - Core Deposit Funding Ratio
   - Duration Match Score (assets vs liabilities)

2. Risk Factors (inverted - lower is better):
   - Uninsured Deposit Ratio (run risk - the "SVB" variable)
   - FHLB Advance Utilization (desperation signal)
   - Brokered Deposit Ratio (hot money)
   - AOCI Impact Ratio (trapped capital from unrealized losses)
   - Wholesale Funding Ratio (non-core dependence)
   - Deposit Rate Beta (rate sensitivity)

Formula:
    Funding_Resilience_Score =
        w1 * (deposit_funding_ratio) +
        w2 * (1 - wholesale_funding_ratio) +
        w3 * (1 - uninsured_deposit_ratio) +
        w4 * (1 - fhlb_advance_ratio) +
        w5 * (1 - brokered_deposit_ratio) +
        w6 * (1 - aoci_impact_ratio) +
        w7 * (duration_match_score) +
        w8 * (1 - rate_beta_deposits)

Data Sources:
- FFIEC Call Reports: RC-O, RC-M, RC-E, RC-B, RC-R schedules
- SEC EDGAR: 10-K/10-Q filings (fallback)
- FRED: Fed funds rate, deposit rates
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


@dataclass
class FundingStabilitySpec:
    """Specification for Funding Stability Score calculation."""

    name: str
    description: str

    # Component weights (default: equal weighting normalized)
    weight_deposit_funding: float = 0.125
    weight_wholesale_funding: float = 0.125
    weight_uninsured_deposits: float = 0.150  # Higher weight - SVB variable
    weight_fhlb_advances: float = 0.150  # Higher weight - desperation signal
    weight_brokered_deposits: float = 0.100
    weight_aoci_impact: float = 0.125  # Confidence multiplier
    weight_duration_match: float = 0.100
    weight_rate_beta: float = 0.125

    # Thresholds for risk classification
    uninsured_threshold_high: float = 0.50  # >50% uninsured = high risk
    uninsured_threshold_moderate: float = 0.30
    fhlb_threshold_high: float = 0.10  # >10% FHLB = desperation
    fhlb_threshold_moderate: float = 0.05
    brokered_threshold_high: float = 0.15
    aoci_threshold_high: float = 0.50  # >50% of TCE = critical

    # Score scaling
    score_min: float = 0.0
    score_max: float = 100.0

    # Rolling windows
    rate_beta_window: int = 12  # Quarters for beta calculation
    volatility_window: int = 8

    @classmethod
    def from_json(cls, path: str | Path) -> "FundingStabilitySpec":
        """Load specification from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        """Save specification to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    def validate_weights(self) -> bool:
        """Check that weights sum to approximately 1.0."""
        total = (
            self.weight_deposit_funding +
            self.weight_wholesale_funding +
            self.weight_uninsured_deposits +
            self.weight_fhlb_advances +
            self.weight_brokered_deposits +
            self.weight_aoci_impact +
            self.weight_duration_match +
            self.weight_rate_beta
        )
        return abs(total - 1.0) < 0.01


@dataclass
class BankFundingProfile:
    """Funding stability profile for a single bank."""

    ticker: str
    name: str
    as_of_date: datetime

    # Raw metrics
    total_deposits: float
    uninsured_deposits: float
    brokered_deposits: float
    fhlb_advances: float
    total_liabilities: float
    wholesale_funding: float
    tangible_common_equity: float
    unrealized_securities_loss: float

    # Computed ratios (0-1 scale)
    deposit_funding_ratio: float
    wholesale_funding_ratio: float
    uninsured_deposit_ratio: float
    fhlb_advance_ratio: float
    brokered_deposit_ratio: float
    aoci_impact_ratio: float
    duration_match_score: float
    rate_beta: float

    # Composite score
    funding_resilience_score: float
    resilience_rank: int
    risk_tier: str  # "Low", "Moderate", "High", "Critical"

    # Stress indicators
    is_fhlb_dependent: bool
    is_run_vulnerable: bool
    has_aoci_stress: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            # Raw metrics
            "total_deposits": self.total_deposits,
            "uninsured_deposits": self.uninsured_deposits,
            "brokered_deposits": self.brokered_deposits,
            "fhlb_advances": self.fhlb_advances,
            "wholesale_funding": self.wholesale_funding,
            "tangible_common_equity": self.tangible_common_equity,
            "unrealized_securities_loss": self.unrealized_securities_loss,
            # Ratios
            "deposit_funding_ratio": self.deposit_funding_ratio,
            "wholesale_funding_ratio": self.wholesale_funding_ratio,
            "uninsured_deposit_ratio": self.uninsured_deposit_ratio,
            "fhlb_advance_ratio": self.fhlb_advance_ratio,
            "brokered_deposit_ratio": self.brokered_deposit_ratio,
            "aoci_impact_ratio": self.aoci_impact_ratio,
            "duration_match_score": self.duration_match_score,
            "rate_beta": self.rate_beta,
            # Score
            "funding_resilience_score": self.funding_resilience_score,
            "resilience_rank": self.resilience_rank,
            "risk_tier": self.risk_tier,
            # Flags
            "is_fhlb_dependent": self.is_fhlb_dependent,
            "is_run_vulnerable": self.is_run_vulnerable,
            "has_aoci_stress": self.has_aoci_stress,
        }


class DepositRateBetaCalculator:
    """
    Calculate deposit rate beta (sensitivity to Fed funds rate).

    Rate Beta = Δ(deposit_rate) / Δ(fed_funds_rate)

    Lower beta = stickier deposits = more stable funding.
    """

    def __init__(self, window: int = 12):
        """
        Initialize calculator.

        Args:
            window: Rolling window in quarters for beta calculation
        """
        self.window = window

    def calculate_beta(
        self,
        deposit_rates: pl.DataFrame,
        fed_funds: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calculate rolling deposit rate beta.

        Args:
            deposit_rates: DataFrame with date, ticker, deposit_rate
            fed_funds: DataFrame with date, fed_funds_rate

        Returns:
            DataFrame with date, ticker, rate_beta
        """
        if deposit_rates.height == 0 or fed_funds.height == 0:
            return pl.DataFrame()

        # Join data
        merged = deposit_rates.join(fed_funds, on="date", how="left")

        # Calculate changes
        merged = merged.sort(["ticker", "date"])
        merged = merged.with_columns([
            (pl.col("deposit_rate") - pl.col("deposit_rate").shift(1).over("ticker"))
            .alias("deposit_rate_change"),
            (pl.col("fed_funds_rate") - pl.col("fed_funds_rate").shift(1))
            .alias("fed_funds_change"),
        ])

        # Rolling beta (covariance / variance)
        # Using a simplified approach: ratio of cumulative changes
        result = merged.with_columns([
            (
                pl.col("deposit_rate_change").rolling_sum(window_size=self.window).over("ticker") /
                pl.col("fed_funds_change").rolling_sum(window_size=self.window).clip(lower_bound=0.01)
            ).alias("rate_beta")
        ])

        # Clip beta to reasonable range [0, 1.5]
        result = result.with_columns(
            pl.col("rate_beta").clip(lower_bound=0.0, upper_bound=1.5)
        )

        return result.select(["date", "ticker", "rate_beta"])

    def estimate_beta_from_nim(
        self,
        bank_data: pl.DataFrame,
        fed_funds: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Estimate rate beta from NIM changes (when direct deposit rates unavailable).

        Higher NIM sensitivity to rates = higher effective beta.

        Args:
            bank_data: DataFrame with date, ticker, nim
            fed_funds: DataFrame with date, fed_funds_rate

        Returns:
            DataFrame with estimated rate_beta
        """
        if bank_data.height == 0 or "nim" not in bank_data.columns:
            return pl.DataFrame()

        merged = bank_data.join(fed_funds, on="date", how="left")

        # NIM typically moves inversely with rate beta
        # Higher beta = faster liability repricing = NIM compression when rates rise
        merged = merged.sort(["ticker", "date"])
        merged = merged.with_columns([
            (pl.col("nim") - pl.col("nim").shift(4).over("ticker")).alias("nim_change_yoy"),
            (pl.col("fed_funds_rate") - pl.col("fed_funds_rate").shift(4)).alias("rate_change_yoy"),
        ])

        # Negative relationship: NIM falls when rates rise = high beta
        merged = merged.with_columns(
            pl.when(pl.col("rate_change_yoy").abs() > 0.1)  # At least 10bp change
            .then(-pl.col("nim_change_yoy") / pl.col("rate_change_yoy"))
            .otherwise(pl.lit(0.5))  # Default to moderate beta
            .clip(lower_bound=0.0, upper_bound=1.5)
            .alias("rate_beta")
        )

        return merged.select(["date", "ticker", "rate_beta"])


class FundingResilienceScorer:
    """
    Calculate the Funding Resilience Score from component metrics.
    """

    def __init__(self, spec: Optional[FundingStabilitySpec] = None):
        """
        Initialize scorer.

        Args:
            spec: Scoring specification with weights and thresholds
        """
        self.spec = spec or FundingStabilitySpec(
            name="default",
            description="Default funding stability specification"
        )

    def calculate_score(
        self,
        metrics_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calculate Funding Resilience Score for each bank-date.

        Args:
            metrics_df: DataFrame with all component metrics

        Returns:
            DataFrame with scores and risk classifications
        """
        result = metrics_df.clone()

        # Ensure all required columns exist with defaults
        required_cols = {
            "deposit_funding_ratio": 0.5,
            "wholesale_funding_ratio": 0.3,
            "uninsured_deposit_ratio": 0.4,
            "fhlb_advance_ratio": 0.05,
            "brokered_deposit_ratio": 0.1,
            "aoci_impact_ratio": 0.2,
            "duration_match_score": 0.5,
            "rate_beta": 0.5,
        }

        for col, default in required_cols.items():
            if col not in result.columns:
                result = result.with_columns(pl.lit(default).alias(col))

        # Calculate composite score
        # Stability factors (higher is better)
        stability_component = (
            self.spec.weight_deposit_funding * pl.col("deposit_funding_ratio") +
            self.spec.weight_duration_match * pl.col("duration_match_score")
        )

        # Risk factors (inverted - lower raw value is better)
        risk_component = (
            self.spec.weight_wholesale_funding * (1 - pl.col("wholesale_funding_ratio")) +
            self.spec.weight_uninsured_deposits * (1 - pl.col("uninsured_deposit_ratio")) +
            self.spec.weight_fhlb_advances * (1 - pl.col("fhlb_advance_ratio").clip(upper_bound=0.3) / 0.3) +
            self.spec.weight_brokered_deposits * (1 - pl.col("brokered_deposit_ratio")) +
            self.spec.weight_aoci_impact * (1 - pl.col("aoci_impact_ratio").clip(upper_bound=1.0)) +
            self.spec.weight_rate_beta * (1 - pl.col("rate_beta"))
        )

        # Raw score (0-1)
        raw_score = stability_component + risk_component

        # Scale to configured range
        result = result.with_columns(
            (raw_score * (self.spec.score_max - self.spec.score_min) + self.spec.score_min)
            .alias("funding_resilience_score")
        )

        # Add stress flags
        result = result.with_columns([
            (pl.col("fhlb_advance_ratio") > self.spec.fhlb_threshold_high)
            .alias("is_fhlb_dependent"),
            (pl.col("uninsured_deposit_ratio") > self.spec.uninsured_threshold_high)
            .alias("is_run_vulnerable"),
            (pl.col("aoci_impact_ratio") > self.spec.aoci_threshold_high)
            .alias("has_aoci_stress"),
        ])

        # Classify risk tier
        result = result.with_columns(
            pl.when(pl.col("funding_resilience_score") >= 75)
            .then(pl.lit("Low"))
            .when(pl.col("funding_resilience_score") >= 50)
            .then(pl.lit("Moderate"))
            .when(pl.col("funding_resilience_score") >= 25)
            .then(pl.lit("High"))
            .otherwise(pl.lit("Critical"))
            .alias("risk_tier")
        )

        # Rank banks at each date
        result = result.with_columns(
            pl.col("funding_resilience_score")
            .rank(descending=True)
            .over("date")
            .alias("resilience_rank")
        )

        return result

    def calculate_component_contributions(
        self,
        metrics_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calculate how much each component contributes to the final score.

        Useful for understanding which factors are driving a bank's score.

        Args:
            metrics_df: DataFrame with all component metrics

        Returns:
            DataFrame with contribution columns for each component
        """
        result = metrics_df.clone()

        # Calculate individual contributions
        result = result.with_columns([
            (self.spec.weight_deposit_funding * pl.col("deposit_funding_ratio") * 100)
            .alias("contrib_deposit_funding"),
            (self.spec.weight_wholesale_funding * (1 - pl.col("wholesale_funding_ratio")) * 100)
            .alias("contrib_wholesale_funding"),
            (self.spec.weight_uninsured_deposits * (1 - pl.col("uninsured_deposit_ratio")) * 100)
            .alias("contrib_uninsured_deposits"),
            (self.spec.weight_fhlb_advances * (1 - pl.col("fhlb_advance_ratio").clip(upper_bound=0.3) / 0.3) * 100)
            .alias("contrib_fhlb_advances"),
            (self.spec.weight_brokered_deposits * (1 - pl.col("brokered_deposit_ratio")) * 100)
            .alias("contrib_brokered_deposits"),
            (self.spec.weight_aoci_impact * (1 - pl.col("aoci_impact_ratio").clip(upper_bound=1.0)) * 100)
            .alias("contrib_aoci_impact"),
            (self.spec.weight_duration_match * pl.col("duration_match_score") * 100)
            .alias("contrib_duration_match"),
            (self.spec.weight_rate_beta * (1 - pl.col("rate_beta")) * 100)
            .alias("contrib_rate_beta"),
        ])

        return result


@register_indicator("funding_stability")
class FundingStabilityIndicator(BaseIndicator):
    """
    Funding Stability Indicator.

    Combines multiple funding metrics to create a composite score
    that predicts which banks will constrain credit in downturns.

    Key insight: Banks with lower scores will:
    - Cut lending more aggressively in downturns
    - Have higher loan loss provisions
    - Underperform in credit boom periods
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self._spec: Optional[FundingStabilitySpec] = None
        self._scorer = FundingResilienceScorer()
        self._beta_calculator = DepositRateBetaCalculator()
        self._funding_data: Optional[pl.DataFrame] = None

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="Funding Stability Score",
            short_name="FundingStability",
            description=(
                "Composite score measuring bank funding resilience based on "
                "deposit structure, wholesale funding, FHLB dependence, uninsured deposits, "
                "and unrealized securities losses. Predicts procyclical lending behavior."
            ),
            version="1.0.0",
            paper_reference="Extension of NY Fed Staff Report 1111",
            data_sources=[
                "FFIEC Call Reports (RC-O, RC-M, RC-E, RC-B, RC-R)",
                "SEC EDGAR 10-K/10-Q",
                "FRED",
            ],
            update_frequency="quarterly",
            lookback_periods=20,
        )

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch all data needed for funding stability calculation.

        Sources:
        1. Call Report schedules (RC-O, RC-M, RC-E, RC-B, RC-R)
        2. SEC EDGAR (fallback)
        3. FRED (Fed funds rate, deposit rates)
        4. Bank fundamentals (for NIM-based beta)
        """
        from ...bank_data import TARGET_BANKS, BankDataCollector
        from ...cache import CachedFREDFetcher
        from .call_report_fetcher import SECFundingDataExtractor

        # 1. Fetch SEC funding data (fallback source)
        sec_extractor = SECFundingDataExtractor()
        sec_dfs = []

        for ticker, bank_info in TARGET_BANKS.items():
            print(f"Fetching funding data for {ticker}...")
            try:
                sec_df = sec_extractor.extract_funding_data(bank_info.cik, ticker)
                if sec_df.height > 0:
                    sec_dfs.append(sec_df)
            except Exception as e:
                print(f"  Warning: Could not fetch SEC data for {ticker}: {e}")

        sec_panel = (
            pl.concat(sec_dfs, how="diagonal").sort(["ticker", "date"])
            if sec_dfs else pl.DataFrame()
        )

        # 2. Fetch bank fundamentals (for NIM, assets, deposits)
        collector = BankDataCollector(start_date=start_date)
        bank_panel = collector.fetch_all_banks()
        bank_panel = collector.compute_derived_metrics(bank_panel)

        # 3. Fetch FRED data (Fed funds rate for beta calculation)
        fred = CachedFREDFetcher(max_age_hours=6)
        fred_series = [
            "FEDFUNDS",  # Federal funds rate
            "WDDNS",  # Weekly deposit rates (if available)
            "DPRIME",  # Prime rate
        ]
        macro_data = fred.fetch_multiple_series(fred_series, start_date=start_date)

        # 4. Compute rate beta from NIM
        rate_beta_df = pl.DataFrame()
        if bank_panel.height > 0 and macro_data is not None and not macro_data.is_empty():
            # Prepare fed funds data
            fed_funds = macro_data.select([
                "date",
                pl.col("FEDFUNDS").alias("fed_funds_rate"),
            ]).drop_nulls()

            rate_beta_df = self._beta_calculator.estimate_beta_from_nim(
                bank_panel, fed_funds
            )

        return {
            "sec_panel": sec_panel,
            "bank_panel": bank_panel,
            "macro_data": macro_data,
            "rate_beta": rate_beta_df,
            "data_quality": collector.get_data_quality_summary(),
        }

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        spec: Optional[FundingStabilitySpec] = None,
        **kwargs,
    ) -> IndicatorResult:
        """
        Calculate Funding Resilience Score for all banks.

        Steps:
        1. Merge funding metrics from multiple sources
        2. Calculate component ratios
        3. Apply scoring formula with weights
        4. Classify risk tiers
        """
        sec_panel = data.get("sec_panel", pl.DataFrame())
        bank_panel = data.get("bank_panel", pl.DataFrame())
        rate_beta_df = data.get("rate_beta", pl.DataFrame())

        if spec is None:
            spec = FundingStabilitySpec(
                name="default",
                description="Default funding stability specification"
            )
        self._spec = spec
        self._scorer = FundingResilienceScorer(spec)

        # Start with bank fundamentals
        if bank_panel.height == 0:
            return IndicatorResult(
                indicator_name="funding_stability",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No bank data available"},
            )

        # Select relevant columns from bank panel
        base_cols = ["date", "ticker"]
        available_cols = [c for c in ["total_assets", "total_deposits", "total_liabilities",
                                      "net_interest_income", "nim", "interest_expense"]
                         if c in bank_panel.columns]

        result = bank_panel.select(base_cols + available_cols)

        # Merge SEC funding data if available
        if sec_panel.height > 0:
            result = result.join(
                sec_panel.select([c for c in sec_panel.columns if c != "date" or c == "date"]),
                on=["date", "ticker"],
                how="left"
            )

        # Merge rate beta
        if rate_beta_df.height > 0:
            result = result.join(rate_beta_df, on=["date", "ticker"], how="left")

        # Calculate funding ratios
        result = self._calculate_funding_ratios(result)

        # Calculate resilience score
        result = self._scorer.calculate_score(result)

        # Calculate component contributions
        result = self._scorer.calculate_component_contributions(result)

        self._funding_data = result

        return IndicatorResult(
            indicator_name="funding_stability",
            calculation_date=datetime.now(),
            data=result,
            metadata={
                "n_banks": result["ticker"].n_unique() if result.height > 0 else 0,
                "avg_score": float(result["funding_resilience_score"].mean())
                if "funding_resilience_score" in result.columns else None,
                "spec_name": spec.name,
                "weights_valid": spec.validate_weights(),
            },
        )

    def _calculate_funding_ratios(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate all funding stability ratios.

        Handles missing data with reasonable defaults.
        """
        result = df.clone()

        # Deposit funding ratio (core deposits / total liabilities)
        if "total_deposits" in result.columns and "total_liabilities" in result.columns:
            result = result.with_columns(
                (pl.col("total_deposits") / pl.col("total_liabilities").clip(lower_bound=1))
                .clip(upper_bound=1.0)
                .alias("deposit_funding_ratio")
            )
        else:
            result = result.with_columns(pl.lit(0.6).alias("deposit_funding_ratio"))

        # Wholesale funding ratio (estimate from deposits/liabilities gap)
        if "total_deposits" in result.columns and "total_liabilities" in result.columns:
            result = result.with_columns(
                (1 - pl.col("total_deposits") / pl.col("total_liabilities").clip(lower_bound=1))
                .clip(lower_bound=0.0, upper_bound=1.0)
                .alias("wholesale_funding_ratio")
            )
        else:
            result = result.with_columns(pl.lit(0.3).alias("wholesale_funding_ratio"))

        # FHLB advance ratio (default to low - will be updated from Call Reports)
        if "fhlb_advances" not in result.columns:
            result = result.with_columns(pl.lit(0.03).alias("fhlb_advance_ratio"))

        # Uninsured deposit ratio (default to moderate - critical to update)
        if "uninsured_deposit_ratio" not in result.columns:
            result = result.with_columns(pl.lit(0.35).alias("uninsured_deposit_ratio"))

        # Brokered deposit ratio
        if "brokered_deposit_ratio" not in result.columns:
            result = result.with_columns(pl.lit(0.05).alias("brokered_deposit_ratio"))

        # AOCI impact ratio
        if "aoci_impact_ratio" not in result.columns:
            result = result.with_columns(pl.lit(0.15).alias("aoci_impact_ratio"))

        # Duration match score (would integrate with duration_mismatch indicator)
        if "duration_match_score" not in result.columns:
            result = result.with_columns(pl.lit(0.5).alias("duration_match_score"))

        # Rate beta
        if "rate_beta" not in result.columns:
            result = result.with_columns(pl.lit(0.5).alias("rate_beta"))

        return result

    def get_resilience_ranking(self) -> pl.DataFrame:
        """Get current funding resilience rankings."""
        if self._funding_data is None:
            return pl.DataFrame()

        # Get latest date for each bank
        latest = self._funding_data.group_by("ticker").agg(
            pl.all().sort_by("date").last()
        )

        return latest.select([
            "ticker", "funding_resilience_score", "resilience_rank", "risk_tier",
            "is_fhlb_dependent", "is_run_vulnerable", "has_aoci_stress",
            "uninsured_deposit_ratio", "fhlb_advance_ratio", "aoci_impact_ratio",
        ]).sort("resilience_rank")

    def get_stress_indicators(self) -> pl.DataFrame:
        """Get banks with active stress indicators."""
        if self._funding_data is None:
            return pl.DataFrame()

        # Get latest and filter for stressed banks
        latest = self._funding_data.group_by("ticker").agg(
            pl.all().sort_by("date").last()
        )

        stressed = latest.filter(
            pl.col("is_fhlb_dependent") |
            pl.col("is_run_vulnerable") |
            pl.col("has_aoci_stress")
        )

        return stressed.select([
            "ticker", "funding_resilience_score", "risk_tier",
            "is_fhlb_dependent", "is_run_vulnerable", "has_aoci_stress",
        ]).sort("funding_resilience_score")

    def get_dashboard_components(self) -> dict[str, Any]:
        """Return dashboard configuration."""
        return {
            "tabs": [
                {"name": "Resilience Scores", "icon": "shield"},
                {"name": "Risk Factors", "icon": "warning"},
                {"name": "Stress Indicators", "icon": "alert"},
                {"name": "Component Analysis", "icon": "pie_chart"},
            ],
            "primary_metric": "funding_resilience_score",
            "ranking_metric": "resilience_rank",
            "alert_fields": [
                "is_fhlb_dependent",
                "is_run_vulnerable",
                "has_aoci_stress",
            ],
        }
