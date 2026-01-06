"""
FASAR Indicator - Flex-Adjusted Syndicate Absorption Ratio

Measures "trapped volume" - commitments that banks cannot escape weighted
by rigidity, divided by CLO market absorption capacity.

FASAR = Σ(Volume × Rigidity) / CLO Velocity

- FASAR > 2.0: Bank has strict commitments but CLO exit is closed ("Hung Loan" risk)
- FASAR < 1.0: Bank has flexibility or market is absorbing debt normally
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import polars as pl

from ..base import (
    BaseIndicator,
    IndicatorMetadata,
    IndicatorResult,
    register_indicator,
)


@dataclass
class FASARSpec:
    """Configuration for FASAR calculation."""

    name: str = "default"
    description: str = "Default FASAR specification"

    # Thresholds for risk classification
    high_risk_threshold: float = 2.0  # FASAR > 2.0 = "Trapped"
    elevated_threshold: float = 1.5  # FASAR > 1.5 = "Elevated"
    normal_threshold: float = 1.0  # FASAR > 1.0 = "Watch"

    # CLO velocity calculation
    clo_velocity_lookback_weeks: int = 4
    min_clo_velocity: float = 0.01  # Prevent division by zero

    # Base syndication capacity (in millions USD)
    # This is the "normal" weekly syndication capacity that velocity=1.0 represents
    # Typical weekly CLO issuance is ~$5-10B, use $5B as base
    base_syndication_capacity: float = 5000.0  # $5B per week

    # Rigidity score defaults for missing data
    default_rigidity_redacted: float = 0.5  # When fee letter is redacted
    sungard_rigidity: float = 1.0  # SunGard/Limited Conditionality
    market_outs_rigidity: float = 0.0  # Full market outs

    # Capital adjustment (CET1)
    apply_cet1_adjustment: bool = True
    regulatory_minimum_cet1: float = 0.045  # 4.5% regulatory minimum

    @classmethod
    def from_dict(cls, d: dict) -> "FASARSpec":
        """Create spec from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: str | Path) -> None:
        """Save specification to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class CommitmentDeal:
    """A single bridge commitment deal."""

    deal_id: str
    bank_ticker: str
    announcement_date: datetime
    commitment_amount: float  # In millions USD
    target_company: str
    acquirer_company: str

    # Rigidity components
    rigidity_score: float  # 0.0 to 1.0
    has_sungard_clause: bool = False
    has_limited_conditionality: bool = False
    has_market_flex: bool = True
    flex_is_capped: bool = False
    successful_syndication_condition: bool = False

    # Status
    is_syndicated: bool = False
    is_hung: bool = False
    syndication_date: Optional[datetime] = None

    # Source
    filing_url: Optional[str] = None
    cik: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deal_id": self.deal_id,
            "bank_ticker": self.bank_ticker,
            "announcement_date": self.announcement_date.isoformat(),
            "commitment_amount": self.commitment_amount,
            "target_company": self.target_company,
            "acquirer_company": self.acquirer_company,
            "rigidity_score": self.rigidity_score,
            "has_sungard_clause": self.has_sungard_clause,
            "has_limited_conditionality": self.has_limited_conditionality,
            "has_market_flex": self.has_market_flex,
            "flex_is_capped": self.flex_is_capped,
            "successful_syndication_condition": self.successful_syndication_condition,
            "is_syndicated": self.is_syndicated,
            "is_hung": self.is_hung,
        }


@dataclass
class RigidityResult:
    """Result of rigidity classification for a deal."""

    deal_id: str
    rigidity_score: float
    confidence: float
    classification: str  # "trapped", "flexible", "unknown"

    # Evidence
    sungard_detected: bool = False
    limited_conditionality_detected: bool = False
    successful_syndication_detected: bool = False
    market_flex_detected: bool = False
    flex_cap_detected: bool = False
    mae_excludes_market: bool = False

    # Raw text snippets for audit
    evidence_snippets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deal_id": self.deal_id,
            "rigidity_score": self.rigidity_score,
            "confidence": self.confidence,
            "classification": self.classification,
            "sungard_detected": self.sungard_detected,
            "limited_conditionality_detected": self.limited_conditionality_detected,
            "successful_syndication_detected": self.successful_syndication_detected,
            "evidence_snippets": self.evidence_snippets[:3],  # Limit for storage
        }


@register_indicator("fasar")
class FASARIndicator(BaseIndicator):
    """
    Flex-Adjusted Syndicate Absorption Ratio (FASAR) Indicator.

    Measures the mismatch between a bank's contractual inability to escape
    a deal and the market's inability to absorb that deal.

    FASAR = Σ(Volume × Rigidity) / CLO Velocity

    Key signals:
    - FASAR > 2.0: "Trapped" - Bank must fund but cannot syndicate
    - FASAR 1.5-2.0: "Elevated" - Syndication stress emerging
    - FASAR < 1.0: "Normal" - Market absorbing debt

    Reference:
    - Ivashina & Scharfstein (2010): Loan Syndication and Credit Cycles
    """

    supports_nowcast: bool = True

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the FASAR indicator."""
        super().__init__(config_path)
        self._spec: Optional[FASARSpec] = None
        self._rigidity_scorer = None
        self._clo_calculator = None

    def get_metadata(self) -> IndicatorMetadata:
        """Return indicator metadata."""
        return IndicatorMetadata(
            name="Flex-Adjusted Syndicate Absorption Ratio",
            short_name="FASAR",
            description=(
                "Measures trapped syndicated loan volume by combining NLP-derived "
                "rigidity scores from 8-K filings with CLO market absorption capacity. "
                "High FASAR indicates banks are contractually committed to deals "
                "they cannot syndicate."
            ),
            version="1.0.0",
            paper_reference="Ivashina & Scharfstein (2010); Gatev & Strahan (2006)",
            data_sources=[
                "SEC EDGAR (8-K filings, Item 1.01)",
                "FRED (BAMLC0A0CM - Corp Spreads)",
                "ETF Data (JBBB, BKLN, HYG)",
            ],
            update_frequency="weekly",
            lookback_periods=52,  # 1 year of weekly data
        )

    def get_required_data_sources(self) -> list[str]:
        """Document data sources needed."""
        return [
            "commitment_deals",  # 8-K extracted deal data
            "clo_velocity",  # CLO market flow data
            "bank_cet1",  # Bank capital ratios (optional)
        ]

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch data required for FASAR calculation.

        Note: In production, this would fetch from SEC EDGAR and market data APIs.
        For now, we provide the structure and expect data to be passed in.
        """
        from ...core import DataRegistry

        registry = DataRegistry.get_instance()

        # Get macro data for CLO velocity proxies
        clo_series = [
            "BAMLC0A0CM",  # US Corp Master OAS
            "BAMLH0A0HYM2",  # US High Yield Master II OAS
        ]

        try:
            macro_data = registry.get_macro_series(clo_series, start_date)
        except Exception:
            # FRED may not have these exact series, create placeholder
            macro_data = pl.DataFrame({"date": []})

        # Bank capital data (from bank panel if available)
        try:
            bank_panel = registry.get_bank_panel(start_date)
        except Exception:
            bank_panel = pl.DataFrame()

        return {
            "macro_data": macro_data,
            "bank_panel": bank_panel,
            "commitment_deals": pl.DataFrame(),  # To be populated externally
            "etf_flows": pl.DataFrame(),  # To be populated externally
        }

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        spec: Optional[FASARSpec] = None,
        **kwargs,
    ) -> IndicatorResult:
        """
        Calculate FASAR for all banks.

        Args:
            data: Dictionary with:
                - commitment_deals: DataFrame with deal data including rigidity
                - clo_velocity: DataFrame with CLO market flow data
                - bank_cet1: Optional DataFrame with bank capital ratios

        Returns:
            IndicatorResult with FASAR scores per bank
        """
        if spec is not None:
            self._spec = spec
        elif self._spec is None:
            self._spec = FASARSpec()

        commitment_deals = data.get("commitment_deals", pl.DataFrame())
        clo_velocity_data = data.get("clo_velocity", pl.DataFrame())
        bank_cet1 = data.get("bank_cet1", pl.DataFrame())

        if commitment_deals.height == 0:
            return IndicatorResult(
                indicator_name="fasar",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No commitment deal data provided"},
            )

        # Calculate CLO velocity (market absorption capacity)
        clo_velocity = self._calculate_clo_velocity(clo_velocity_data)

        # Calculate FASAR per bank
        results = self._calculate_fasar_by_bank(
            commitment_deals,
            clo_velocity,
            bank_cet1,
        )

        return IndicatorResult(
            indicator_name="fasar",
            calculation_date=datetime.now(),
            data=results,
            metadata={
                "spec": self._spec.name,
                "clo_velocity": clo_velocity,
                "n_banks": results["ticker"].n_unique() if results.height > 0 else 0,
                "n_deals": commitment_deals.height,
                "high_risk_count": (
                    results.filter(pl.col("risk_level") == "HIGH_RISK").height
                    if results.height > 0 else 0
                ),
            },
        )

    def _calculate_clo_velocity(
        self,
        clo_data: pl.DataFrame,
        as_of_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate CLO velocity (market absorption capacity).

        Uses ETF flows and spread data as proxies for CLO formation rate.

        Returns:
            CLO velocity score (higher = more absorption capacity)
        """
        if clo_data.height == 0:
            # Default to moderate velocity when no data
            return 1.0

        # If we have ETF flow data
        if "jbbb_flow" in clo_data.columns:
            recent = clo_data.tail(self._spec.clo_velocity_lookback_weeks)
            avg_flow = recent["jbbb_flow"].mean()

            # Normalize to 0-2 scale (1.0 = normal, >1 = strong, <1 = weak)
            if avg_flow is not None:
                # Positive flows = good absorption
                velocity = 1.0 + (avg_flow / 100)  # Scale factor
                return max(self._spec.min_clo_velocity, velocity)

        # If we have spread data (inverse relationship)
        if "spread" in clo_data.columns:
            recent = clo_data.tail(self._spec.clo_velocity_lookback_weeks)
            avg_spread = recent["spread"].mean()

            if avg_spread is not None:
                # Higher spreads = lower velocity (harder to syndicate)
                # Normalize: 400bps = velocity 0.5, 200bps = velocity 1.5
                velocity = 2.0 - (avg_spread / 400)
                return max(self._spec.min_clo_velocity, velocity)

        return 1.0

    def _calculate_fasar_by_bank(
        self,
        deals: pl.DataFrame,
        clo_velocity: float,
        bank_cet1: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calculate FASAR for each bank.

        FASAR = Σ(Volume × Rigidity) / CLO Velocity
        """
        # Ensure required columns exist
        required_cols = ["bank_ticker", "commitment_amount", "rigidity_score"]
        for col in required_cols:
            if col not in deals.columns:
                return pl.DataFrame()

        # Filter to active (non-syndicated) deals
        if "is_syndicated" in deals.columns:
            active_deals = deals.filter(pl.col("is_syndicated") == False)
        else:
            active_deals = deals

        if active_deals.height == 0:
            return pl.DataFrame()

        # Calculate weighted commitment (trapped volume)
        active_deals = active_deals.with_columns(
            (pl.col("commitment_amount") * pl.col("rigidity_score")).alias(
                "trapped_volume"
            )
        )

        # Aggregate by bank
        bank_fasar = active_deals.group_by("bank_ticker").agg([
            pl.col("trapped_volume").sum().alias("total_trapped_volume"),
            pl.col("commitment_amount").sum().alias("total_commitment"),
            pl.col("rigidity_score").mean().alias("avg_rigidity"),
            pl.len().alias("n_active_deals"),
        ])

        # Calculate raw FASAR
        # FASAR = Trapped Volume / Effective Syndication Capacity
        # Effective Capacity = Base Capacity * CLO Velocity
        # This normalizes to ~1.0 for normal market conditions
        effective_capacity = self._spec.base_syndication_capacity * clo_velocity
        bank_fasar = bank_fasar.with_columns(
            (pl.col("total_trapped_volume") / effective_capacity).alias("raw_fasar")
        )

        # Apply CET1 adjustment if available and enabled
        if self._spec.apply_cet1_adjustment and bank_cet1.height > 0:
            if "ticker" in bank_cet1.columns and "cet1_ratio" in bank_cet1.columns:
                bank_fasar = bank_fasar.join(
                    bank_cet1.select(["ticker", "cet1_ratio"]),
                    left_on="bank_ticker",
                    right_on="ticker",
                    how="left",
                )

                # Effective FASAR = Raw FASAR / (CET1 - Reg Min)
                bank_fasar = bank_fasar.with_columns(
                    pl.when(pl.col("cet1_ratio").is_not_null())
                    .then(
                        pl.col("raw_fasar") / (
                            pl.col("cet1_ratio") - self._spec.regulatory_minimum_cet1
                        ).clip(lower_bound=0.01)
                    )
                    .otherwise(pl.col("raw_fasar"))
                    .alias("effective_fasar")
                )
            else:
                bank_fasar = bank_fasar.with_columns(
                    pl.col("raw_fasar").alias("effective_fasar")
                )
        else:
            bank_fasar = bank_fasar.with_columns(
                pl.col("raw_fasar").alias("effective_fasar")
            )

        # Classify risk level
        bank_fasar = bank_fasar.with_columns(
            pl.when(pl.col("effective_fasar") > self._spec.high_risk_threshold)
            .then(pl.lit("HIGH_RISK"))
            .when(pl.col("effective_fasar") > self._spec.elevated_threshold)
            .then(pl.lit("ELEVATED"))
            .when(pl.col("effective_fasar") > self._spec.normal_threshold)
            .then(pl.lit("WATCH"))
            .otherwise(pl.lit("NORMAL"))
            .alias("risk_level")
        )

        # Rename for consistency
        bank_fasar = bank_fasar.rename({"bank_ticker": "ticker"})

        return bank_fasar.select([
            "ticker",
            "raw_fasar",
            "effective_fasar",
            "risk_level",
            "total_trapped_volume",
            "total_commitment",
            "avg_rigidity",
            "n_active_deals",
        ])

    def get_warning_level(self, fasar_value: float) -> tuple[str, str]:
        """
        Get warning emoji and status for a FASAR value.

        Args:
            fasar_value: The FASAR score

        Returns:
            Tuple of (emoji, status_text)
        """
        if fasar_value > self._spec.high_risk_threshold:
            return ("🔴", "TRAPPED - Hung Loan Risk")
        elif fasar_value > self._spec.elevated_threshold:
            return ("🟠", "ELEVATED - Syndication Stress")
        elif fasar_value > self._spec.normal_threshold:
            return ("🟡", "WATCH - Monitor Closely")
        else:
            return ("🟢", "NORMAL - Market Absorbing")

    def calculate_from_deals(
        self,
        deals: list[CommitmentDeal],
        clo_velocity: float = 1.0,
        bank_cet1: Optional[dict[str, float]] = None,
    ) -> pl.DataFrame:
        """
        Convenience method to calculate FASAR from a list of deal objects.

        Args:
            deals: List of CommitmentDeal objects
            clo_velocity: CLO market velocity (default 1.0 = normal)
            bank_cet1: Optional dict mapping ticker -> CET1 ratio

        Returns:
            DataFrame with FASAR per bank
        """
        if not deals:
            return pl.DataFrame()

        # Ensure spec is initialized
        if self._spec is None:
            self._spec = FASARSpec()

        deals_df = pl.DataFrame([d.to_dict() for d in deals])

        cet1_df = pl.DataFrame()
        if bank_cet1:
            cet1_df = pl.DataFrame({
                "ticker": list(bank_cet1.keys()),
                "cet1_ratio": list(bank_cet1.values()),
            })

        return self._calculate_fasar_by_bank(deals_df, clo_velocity, cet1_df)

    def nowcast(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """
        Generate high-frequency nowcast using CLO spread proxies.

        Uses:
        - AAA CLO spread widening signal
        - JBBB vs HYG divergence ("Choke" signal)
        """
        from .nowcast import FASARNowcaster

        nowcaster = FASARNowcaster(self._spec or FASARSpec())

        # Get latest FASAR calculation as base
        base_result = self.calculate(data)

        # Apply nowcast adjustments
        return nowcaster.nowcast(
            base_fasar=base_result.data,
            etf_data=data.get("etf_data", pl.DataFrame()),
            spread_data=data.get("spread_data", pl.DataFrame()),
        )
