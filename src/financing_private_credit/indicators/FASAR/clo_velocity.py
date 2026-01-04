"""
CLO Velocity Calculator

Calculates the CLO market absorption capacity using:
- FRED credit spread data (BAMLC0A0CM)
- ETF flow data (JBBB, BKLN, HYG)

Higher velocity = market is absorbing syndicated loans
Lower velocity = exit door is closing for banks
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import polars as pl


@dataclass
class CLOVelocityResult:
    """Result of CLO velocity calculation."""

    velocity: float  # 0-2 scale (1.0 = normal)
    confidence: float  # 0-1

    # Component signals
    spread_signal: Optional[float] = None  # From credit spreads
    flow_signal: Optional[float] = None  # From ETF flows
    choke_signal: bool = False  # JBBB vs HYG divergence

    # Metadata
    as_of_date: Optional[datetime] = None
    data_sources_used: list[str] = None

    def __post_init__(self):
        if self.data_sources_used is None:
            self.data_sources_used = []


class CLOVelocityCalculator:
    """
    Calculate CLO market velocity (absorption capacity).

    Uses multiple proxies:
    1. Credit spreads (inverse relationship - wider = lower velocity)
    2. ETF flows (positive correlation - inflows = higher velocity)
    3. JBBB vs HYG divergence (choke signal)
    """

    # FRED series for credit spreads
    SPREAD_SERIES = {
        "BAMLC0A0CM": "US Corp Master OAS",
        "BAMLH0A0HYM2": "US High Yield Master II OAS",
    }

    # ETFs for CLO/Loan market flows
    LOAN_ETFS = ["BKLN", "SRLN", "FTSL"]  # Senior loan ETFs
    CLO_ETFS = ["JBBB", "CLOI", "AAA"]  # CLO ETFs
    HY_ETFS = ["HYG", "JNK", "USHY"]  # High yield comparison

    def __init__(
        self,
        lookback_weeks: int = 4,
        spread_baseline: float = 300.0,  # bps - "normal" spread level
        choke_threshold: float = 0.01,  # 1% divergence triggers choke
    ):
        """
        Initialize the CLO velocity calculator.

        Args:
            lookback_weeks: Weeks to look back for averaging
            spread_baseline: Baseline spread level (bps) for normalization
            choke_threshold: JBBB vs HYG divergence threshold
        """
        self.lookback_weeks = lookback_weeks
        self.spread_baseline = spread_baseline
        self.choke_threshold = choke_threshold

    def calculate(
        self,
        spread_data: Optional[pl.DataFrame] = None,
        etf_data: Optional[pl.DataFrame] = None,
        as_of_date: Optional[datetime] = None,
    ) -> CLOVelocityResult:
        """
        Calculate CLO velocity from available data.

        Args:
            spread_data: DataFrame with date and spread columns
            etf_data: DataFrame with date, ticker, close, flow columns
            as_of_date: Date to calculate as of (default: latest)

        Returns:
            CLOVelocityResult with velocity score and components
        """
        as_of = as_of_date or datetime.now()
        data_sources = []

        spread_signal = None
        flow_signal = None
        choke_signal = False

        # Calculate spread-based velocity
        if spread_data is not None and spread_data.height > 0:
            spread_signal = self._calculate_spread_velocity(spread_data, as_of)
            if spread_signal is not None:
                data_sources.append("credit_spreads")

        # Calculate flow-based velocity
        if etf_data is not None and etf_data.height > 0:
            flow_signal = self._calculate_flow_velocity(etf_data, as_of)
            if flow_signal is not None:
                data_sources.append("etf_flows")

            # Check for choke signal
            choke_signal = self._detect_choke_signal(etf_data, as_of)
            if choke_signal:
                data_sources.append("choke_detection")

        # Combine signals
        velocity = self._combine_signals(spread_signal, flow_signal, choke_signal)

        # Calculate confidence based on data availability
        confidence = len(data_sources) / 3.0  # Max 3 sources

        return CLOVelocityResult(
            velocity=velocity,
            confidence=confidence,
            spread_signal=spread_signal,
            flow_signal=flow_signal,
            choke_signal=choke_signal,
            as_of_date=as_of,
            data_sources_used=data_sources,
        )

    def _calculate_spread_velocity(
        self,
        spread_data: pl.DataFrame,
        as_of: datetime,
    ) -> Optional[float]:
        """
        Calculate velocity from credit spreads.

        Higher spreads = harder to syndicate = lower velocity
        """
        # Filter to lookback period
        lookback_start = as_of - timedelta(weeks=self.lookback_weeks)

        if "date" not in spread_data.columns:
            return None

        recent = spread_data.filter(
            (pl.col("date") >= lookback_start) & (pl.col("date") <= as_of)
        )

        if recent.height == 0:
            return None

        # Find spread column
        spread_col = None
        for col in ["spread", "BAMLC0A0CM", "BAMLH0A0HYM2", "oas"]:
            if col in recent.columns:
                spread_col = col
                break

        if spread_col is None:
            return None

        avg_spread = recent[spread_col].mean()
        if avg_spread is None:
            return None

        # Convert to velocity (inverse relationship)
        # 200bps = velocity 1.5 (strong)
        # 300bps = velocity 1.0 (normal)
        # 500bps = velocity 0.4 (weak)
        velocity = 2.0 - (float(avg_spread) / self.spread_baseline)
        return max(0.1, min(2.0, velocity))

    def _calculate_flow_velocity(
        self,
        etf_data: pl.DataFrame,
        as_of: datetime,
    ) -> Optional[float]:
        """
        Calculate velocity from ETF flows.

        Positive flows into CLO/Loan ETFs = higher velocity
        """
        lookback_start = as_of - timedelta(weeks=self.lookback_weeks)

        if "date" not in etf_data.columns:
            return None

        recent = etf_data.filter(
            (pl.col("date") >= lookback_start) & (pl.col("date") <= as_of)
        )

        if recent.height == 0:
            return None

        # Check for flow column
        flow_col = None
        for col in ["flow", "net_flow", "flows"]:
            if col in recent.columns:
                flow_col = col
                break

        if flow_col is None:
            return None

        # Filter to CLO/Loan ETFs if ticker column exists
        if "ticker" in recent.columns:
            target_etfs = self.CLO_ETFS + self.LOAN_ETFS
            recent = recent.filter(pl.col("ticker").is_in(target_etfs))

        if recent.height == 0:
            return None

        avg_flow = recent[flow_col].mean()
        if avg_flow is None:
            return None

        # Normalize flows to velocity
        # Assume flows in millions, scale to 0-2 range
        # +$500M/week = velocity 1.5
        # $0 = velocity 1.0
        # -$500M/week = velocity 0.5
        velocity = 1.0 + (float(avg_flow) / 500.0)
        return max(0.1, min(2.0, velocity))

    def _detect_choke_signal(
        self,
        etf_data: pl.DataFrame,
        as_of: datetime,
    ) -> bool:
        """
        Detect CLO market "choke" - when CLO ETFs diverge from HY ETFs.

        Signal: HYG flat/up but JBBB drops >1% in 2 days
        """
        lookback_days = 5  # Short lookback for divergence

        if "date" not in etf_data.columns or "ticker" not in etf_data.columns:
            return False

        lookback_start = as_of - timedelta(days=lookback_days)

        recent = etf_data.filter(
            (pl.col("date") >= lookback_start) & (pl.col("date") <= as_of)
        )

        if recent.height == 0:
            return False

        # Get returns for CLO ETF (JBBB) and HY ETF (HYG)
        clo_etf = recent.filter(pl.col("ticker") == "JBBB")
        hy_etf = recent.filter(pl.col("ticker") == "HYG")

        if clo_etf.height < 2 or hy_etf.height < 2:
            return False

        # Calculate returns
        if "close" not in clo_etf.columns:
            return False

        clo_etf = clo_etf.sort("date")
        hy_etf = hy_etf.sort("date")

        clo_return = (
            clo_etf["close"][-1] / clo_etf["close"][0] - 1
        )
        hy_return = (
            hy_etf["close"][-1] / hy_etf["close"][0] - 1
        )

        # Choke signal: JBBB drops significantly while HYG is flat/up
        if clo_return < -self.choke_threshold and hy_return >= -0.005:
            return True

        return False

    def _combine_signals(
        self,
        spread_signal: Optional[float],
        flow_signal: Optional[float],
        choke_signal: bool,
    ) -> float:
        """
        Combine component signals into overall velocity.
        """
        signals = []
        weights = []

        if spread_signal is not None:
            signals.append(spread_signal)
            weights.append(0.4)  # 40% weight to spreads

        if flow_signal is not None:
            signals.append(flow_signal)
            weights.append(0.4)  # 40% weight to flows

        if not signals:
            # No data - assume normal velocity
            base_velocity = 1.0
        else:
            # Weighted average
            total_weight = sum(weights)
            base_velocity = sum(s * w for s, w in zip(signals, weights)) / total_weight

        # Apply choke signal penalty
        if choke_signal:
            # From spec: multiply by 0.5 when choke detected
            base_velocity *= 0.5

        return max(0.01, base_velocity)  # Minimum velocity to prevent division issues

    def calculate_from_fred(
        self,
        fred_data: pl.DataFrame,
        as_of_date: Optional[datetime] = None,
    ) -> CLOVelocityResult:
        """
        Convenience method to calculate from FRED data directly.

        Args:
            fred_data: DataFrame from CachedFREDFetcher with date and series columns
            as_of_date: Date to calculate as of

        Returns:
            CLOVelocityResult
        """
        # FRED data typically has series as columns
        spread_data = fred_data.clone()

        # Rename FRED series to standard column
        for series in self.SPREAD_SERIES:
            if series in spread_data.columns:
                spread_data = spread_data.with_columns(
                    pl.col(series).alias("spread")
                )
                break

        return self.calculate(
            spread_data=spread_data,
            as_of_date=as_of_date,
        )
