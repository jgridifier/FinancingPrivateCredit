"""
FASAR Nowcasting - Daily Market Pulse

Provides high-frequency updates to FASAR using:
1. AAA CLO Spread Monitor - widening indicates CLO creation stopping
2. JBBB vs HYG Spread Monitor - divergence indicates "choke"

When signals trigger, the CLO velocity denominator is adjusted downward,
increasing effective FASAR and signaling stress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import polars as pl

from ..base import IndicatorResult
from .indicator import FASARSpec


@dataclass
class NowcastSignal:
    """A nowcast adjustment signal."""

    signal_type: str  # "aaa_spread_widening", "clo_hy_divergence"
    triggered: bool
    magnitude: float  # 0-1, severity of signal
    velocity_multiplier: float  # Applied to CLO velocity
    description: str
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_type": self.signal_type,
            "triggered": self.triggered,
            "magnitude": self.magnitude,
            "velocity_multiplier": self.velocity_multiplier,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class WarehouseStressIndicator:
    """
    Warehouse Stress Indicator from AAA CLO spreads.

    When AAA CLO spreads widen >10bps in a week, CLO arbitrage math breaks
    and creation stops. This means the "exit door" for syndicated loans closes.
    """

    current_spread: float
    week_ago_spread: float
    spread_change_bps: float
    is_stressed: bool
    stress_level: str  # "normal", "elevated", "critical"

    @classmethod
    def calculate(
        cls,
        spread_data: pl.DataFrame,
        as_of_date: Optional[datetime] = None,
    ) -> Optional["WarehouseStressIndicator"]:
        """
        Calculate warehouse stress from AAA CLO spread data.

        Args:
            spread_data: DataFrame with date and aaa_clo_spread columns
            as_of_date: Date to calculate as of

        Returns:
            WarehouseStressIndicator or None if insufficient data
        """
        if spread_data.height == 0:
            return None

        as_of = as_of_date or datetime.now()
        week_ago = as_of - timedelta(days=7)

        # Find spread column
        spread_col = None
        for col in ["aaa_clo_spread", "aaa_spread", "spread"]:
            if col in spread_data.columns:
                spread_col = col
                break

        if spread_col is None:
            return None

        # Get current and week-ago spreads
        current = spread_data.filter(pl.col("date") <= as_of).sort("date").tail(1)
        past = spread_data.filter(pl.col("date") <= week_ago).sort("date").tail(1)

        if current.height == 0 or past.height == 0:
            return None

        current_spread = float(current[spread_col][0])
        week_ago_spread = float(past[spread_col][0])
        change = current_spread - week_ago_spread

        # Determine stress level
        # From spec: >10bps widening in a week = stress
        if change > 20:
            stress_level = "critical"
            is_stressed = True
        elif change > 10:
            stress_level = "elevated"
            is_stressed = True
        else:
            stress_level = "normal"
            is_stressed = False

        return cls(
            current_spread=current_spread,
            week_ago_spread=week_ago_spread,
            spread_change_bps=change,
            is_stressed=is_stressed,
            stress_level=stress_level,
        )


class FASARNowcaster:
    """
    High-frequency nowcaster for FASAR.

    Monitors:
    1. AAA CLO spread widening (warehouse stress)
    2. JBBB vs HYG divergence (CLO market "choke")

    When signals trigger, adjusts CLO velocity downward.
    """

    def __init__(self, spec: Optional[FASARSpec] = None):
        """
        Initialize the nowcaster.

        Args:
            spec: FASAR specification for thresholds
        """
        self.spec = spec or FASARSpec()

        # Signal thresholds
        self.aaa_spread_threshold_bps = 10.0  # >10bps widening = stress
        self.divergence_threshold_pct = 1.0  # >1% divergence = choke
        self.divergence_lookback_days = 2

    def nowcast(
        self,
        base_fasar: pl.DataFrame,
        etf_data: Optional[pl.DataFrame] = None,
        spread_data: Optional[pl.DataFrame] = None,
        as_of_date: Optional[datetime] = None,
    ) -> IndicatorResult:
        """
        Generate nowcast adjustment to FASAR.

        Args:
            base_fasar: Base FASAR scores from calculate()
            etf_data: DataFrame with ticker, date, close columns
            spread_data: DataFrame with date and spread columns
            as_of_date: Date to nowcast as of

        Returns:
            IndicatorResult with adjusted FASAR scores
        """
        as_of = as_of_date or datetime.now()
        signals = []

        # Check AAA spread widening signal
        if spread_data is not None and spread_data.height > 0:
            aaa_signal = self._check_aaa_spread_signal(spread_data, as_of)
            signals.append(aaa_signal)

        # Check JBBB vs HYG divergence
        if etf_data is not None and etf_data.height > 0:
            divergence_signal = self._check_divergence_signal(etf_data, as_of)
            signals.append(divergence_signal)

        # Calculate combined velocity multiplier
        velocity_multiplier = 1.0
        for signal in signals:
            if signal.triggered:
                velocity_multiplier *= signal.velocity_multiplier

        # Adjust FASAR scores
        if base_fasar.height == 0:
            adjusted_fasar = base_fasar
        else:
            adjusted_fasar = self._adjust_fasar(base_fasar, velocity_multiplier)

        return IndicatorResult(
            indicator_name="fasar_nowcast",
            calculation_date=as_of,
            data=adjusted_fasar,
            metadata={
                "as_of_date": as_of.isoformat(),
                "velocity_multiplier": velocity_multiplier,
                "signals": [s.to_dict() for s in signals],
                "n_signals_triggered": sum(1 for s in signals if s.triggered),
            },
        )

    def _check_aaa_spread_signal(
        self,
        spread_data: pl.DataFrame,
        as_of: datetime,
    ) -> NowcastSignal:
        """
        Check AAA CLO spread widening signal.

        From spec: If AAA CLO spreads widen >10bps in a week,
        assume denominator of FASAR is ZERO (instant stress).
        """
        indicator = WarehouseStressIndicator.calculate(spread_data, as_of)

        if indicator is None:
            return NowcastSignal(
                signal_type="aaa_spread_widening",
                triggered=False,
                magnitude=0.0,
                velocity_multiplier=1.0,
                description="Insufficient spread data",
            )

        if indicator.is_stressed:
            # Calculate magnitude based on spread change
            magnitude = min(1.0, indicator.spread_change_bps / 30.0)

            # Velocity multiplier: at 10bps = 0.5, at 20bps+ = 0.1
            if indicator.stress_level == "critical":
                velocity_multiplier = 0.1
            else:
                velocity_multiplier = 0.5

            return NowcastSignal(
                signal_type="aaa_spread_widening",
                triggered=True,
                magnitude=magnitude,
                velocity_multiplier=velocity_multiplier,
                description=(
                    f"AAA CLO spreads widened {indicator.spread_change_bps:.0f}bps "
                    f"in past week ({indicator.stress_level})"
                ),
            )

        return NowcastSignal(
            signal_type="aaa_spread_widening",
            triggered=False,
            magnitude=0.0,
            velocity_multiplier=1.0,
            description=(
                f"AAA CLO spreads changed {indicator.spread_change_bps:.0f}bps "
                "(within normal range)"
            ),
        )

    def _check_divergence_signal(
        self,
        etf_data: pl.DataFrame,
        as_of: datetime,
    ) -> NowcastSignal:
        """
        Check JBBB vs HYG divergence signal.

        From spec: If HYG is flat/up but JBBB drops >1% in 2 days,
        this is the "Choke" signal - investors rejecting CLOs specifically.
        """
        if "ticker" not in etf_data.columns or "close" not in etf_data.columns:
            return NowcastSignal(
                signal_type="clo_hy_divergence",
                triggered=False,
                magnitude=0.0,
                velocity_multiplier=1.0,
                description="Insufficient ETF data",
            )

        lookback_start = as_of - timedelta(days=self.divergence_lookback_days)

        # Get JBBB data
        jbbb = etf_data.filter(
            (pl.col("ticker") == "JBBB") &
            (pl.col("date") >= lookback_start) &
            (pl.col("date") <= as_of)
        ).sort("date")

        # Get HYG data
        hyg = etf_data.filter(
            (pl.col("ticker") == "HYG") &
            (pl.col("date") >= lookback_start) &
            (pl.col("date") <= as_of)
        ).sort("date")

        if jbbb.height < 2 or hyg.height < 2:
            return NowcastSignal(
                signal_type="clo_hy_divergence",
                triggered=False,
                magnitude=0.0,
                velocity_multiplier=1.0,
                description="Insufficient ETF price history",
            )

        # Calculate returns
        jbbb_return = (jbbb["close"][-1] / jbbb["close"][0]) - 1
        hyg_return = (hyg["close"][-1] / hyg["close"][0]) - 1

        # Check for divergence: JBBB down significantly, HYG flat/up
        jbbb_threshold = -self.divergence_threshold_pct / 100
        hyg_threshold = -0.005  # HYG can be slightly down

        if jbbb_return < jbbb_threshold and hyg_return >= hyg_threshold:
            divergence = hyg_return - jbbb_return
            magnitude = min(1.0, abs(divergence) * 50)  # Scale to 0-1

            return NowcastSignal(
                signal_type="clo_hy_divergence",
                triggered=True,
                magnitude=magnitude,
                velocity_multiplier=0.5,  # From spec: multiply by 0.5
                description=(
                    f"CLO market choke: JBBB {jbbb_return*100:.1f}% vs "
                    f"HYG {hyg_return*100:.1f}% ({self.divergence_lookback_days}d)"
                ),
            )

        return NowcastSignal(
            signal_type="clo_hy_divergence",
            triggered=False,
            magnitude=0.0,
            velocity_multiplier=1.0,
            description=(
                f"Normal correlation: JBBB {jbbb_return*100:.1f}% vs "
                f"HYG {hyg_return*100:.1f}%"
            ),
        )

    def _adjust_fasar(
        self,
        base_fasar: pl.DataFrame,
        velocity_multiplier: float,
    ) -> pl.DataFrame:
        """
        Adjust FASAR scores based on nowcast signals.

        Lower velocity multiplier = higher adjusted FASAR.
        """
        if velocity_multiplier >= 1.0:
            # No adjustment needed
            return base_fasar.with_columns(
                pl.col("effective_fasar").alias("nowcast_fasar")
            )

        # Adjust: effective_fasar / velocity_multiplier
        adjusted = base_fasar.with_columns(
            (pl.col("effective_fasar") / velocity_multiplier).alias("nowcast_fasar")
        )

        # Re-classify risk levels with adjusted scores
        adjusted = adjusted.with_columns(
            pl.when(pl.col("nowcast_fasar") > self.spec.high_risk_threshold)
            .then(pl.lit("HIGH_RISK"))
            .when(pl.col("nowcast_fasar") > self.spec.elevated_threshold)
            .then(pl.lit("ELEVATED"))
            .when(pl.col("nowcast_fasar") > self.spec.normal_threshold)
            .then(pl.lit("WATCH"))
            .otherwise(pl.lit("NORMAL"))
            .alias("nowcast_risk_level")
        )

        return adjusted

    def get_market_pulse(
        self,
        etf_data: pl.DataFrame,
        spread_data: pl.DataFrame,
        as_of_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Get current market pulse summary.

        Returns:
            Dictionary with current market conditions
        """
        as_of = as_of_date or datetime.now()

        # Check signals
        aaa_signal = self._check_aaa_spread_signal(spread_data, as_of)
        divergence_signal = self._check_divergence_signal(etf_data, as_of)

        # Calculate warehouse stress
        warehouse = WarehouseStressIndicator.calculate(spread_data, as_of)

        return {
            "as_of": as_of.isoformat(),
            "overall_stress": "HIGH" if any([
                aaa_signal.triggered,
                divergence_signal.triggered
            ]) else "NORMAL",
            "warehouse_stress": warehouse.stress_level if warehouse else "unknown",
            "aaa_spread_change_bps": warehouse.spread_change_bps if warehouse else None,
            "clo_choke_detected": divergence_signal.triggered,
            "velocity_adjustment": min(
                aaa_signal.velocity_multiplier,
                divergence_signal.velocity_multiplier
            ),
            "signals": {
                "aaa_spread": aaa_signal.to_dict(),
                "divergence": divergence_signal.to_dict(),
            },
        }
