"""
Nowcasting for Funding Stability Score

Updates the Funding Resilience Score using high-frequency proxy data
available between quarterly Call Report releases.

High-Frequency Proxies:
1. Stock price movements (especially relative to peers)
2. CDS spreads (if available)
3. Fed funds borrowing (weekly data)
4. Bank stock volatility
5. Deposit rate changes (from rate aggregators)

The nowcast adjusts the latest quarterly score based on changes
in these proxy indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl


@dataclass
class NowcastResult:
    """Result of a nowcast update."""

    ticker: str
    nowcast_date: datetime
    base_date: datetime  # Last quarterly observation

    # Base quarterly score
    base_score: float

    # Nowcast adjustments
    stock_adjustment: float
    volatility_adjustment: float
    rate_adjustment: float
    peer_relative_adjustment: float

    # Final nowcast
    nowcast_score: float
    confidence: float  # 0-1, lower as time from base increases

    # Change indicators
    score_change: float
    direction: str  # "improving", "stable", "deteriorating"


class HighFrequencyProxyCalculator:
    """
    Calculate high-frequency proxies for funding stability.
    """

    # Sensitivity parameters (estimated from historical data)
    STOCK_SENSITIVITY = -0.10  # 1% stock decline = 0.1 point score decrease
    VOLATILITY_SENSITIVITY = -0.05  # 1% increase in vol = 0.05 point decrease
    RATE_SENSITIVITY = -0.02  # 1bp deposit rate increase = 0.02 point decrease

    def __init__(self, decay_half_life: int = 30):
        """
        Initialize calculator.

        Args:
            decay_half_life: Days until proxy signal strength halves
        """
        self.decay_half_life = decay_half_life

    def calculate_stock_proxy(
        self,
        stock_data: pl.DataFrame,
        base_date: datetime,
        nowcast_date: datetime,
    ) -> pl.DataFrame:
        """
        Calculate stock-based proxy for funding stability changes.

        Sharp stock declines often precede funding stress.

        Args:
            stock_data: DataFrame with ticker, date, returns, volume
            base_date: Last quarterly observation date
            nowcast_date: Current nowcast date

        Returns:
            DataFrame with stock-based adjustments
        """
        if stock_data.height == 0:
            return pl.DataFrame()

        # Filter to relevant period
        period_data = stock_data.filter(
            (pl.col("date") > base_date) &
            (pl.col("date") <= nowcast_date)
        )

        if period_data.height == 0:
            return pl.DataFrame()

        # Calculate cumulative return since base date
        result = period_data.group_by("ticker").agg([
            (pl.col("daily_return") + 1).product().alias("cumulative_return"),
            pl.col("volatility_20d").last().alias("current_volatility"),
            pl.col("volume").mean().alias("avg_volume"),
        ])

        # Convert to score adjustments
        result = result.with_columns([
            ((pl.col("cumulative_return") - 1) * 100 * self.STOCK_SENSITIVITY)
            .alias("stock_adjustment"),
            # Higher volatility = more stress
            (pl.col("current_volatility") * 100 * self.VOLATILITY_SENSITIVITY)
            .alias("volatility_adjustment"),
        ])

        return result

    def calculate_peer_relative_proxy(
        self,
        stock_data: pl.DataFrame,
        base_date: datetime,
        nowcast_date: datetime,
    ) -> pl.DataFrame:
        """
        Calculate peer-relative performance proxy.

        Banks underperforming peers are likely experiencing
        idiosyncratic funding stress.

        Args:
            stock_data: DataFrame with ticker, date, returns
            base_date: Last quarterly observation date
            nowcast_date: Current nowcast date

        Returns:
            DataFrame with peer-relative adjustments
        """
        if stock_data.height == 0:
            return pl.DataFrame()

        # Filter to relevant period
        period_data = stock_data.filter(
            (pl.col("date") > base_date) &
            (pl.col("date") <= nowcast_date)
        )

        # Calculate cumulative return per ticker
        returns = period_data.group_by("ticker").agg(
            (pl.col("daily_return") + 1).product().alias("cumulative_return")
        )

        # Calculate peer median
        peer_median = returns["cumulative_return"].median()

        # Relative performance
        returns = returns.with_columns(
            ((pl.col("cumulative_return") - peer_median) * 100 * self.STOCK_SENSITIVITY)
            .alias("peer_relative_adjustment")
        )

        return returns

    def calculate_time_decay(
        self,
        base_date: datetime,
        nowcast_date: datetime,
    ) -> float:
        """
        Calculate time decay for confidence weighting.

        Confidence in nowcast decreases as time from last
        quarterly observation increases.

        Args:
            base_date: Last quarterly observation date
            nowcast_date: Current nowcast date

        Returns:
            Confidence weight (0-1)
        """
        days_elapsed = (nowcast_date - base_date).days
        decay = 0.5 ** (days_elapsed / self.decay_half_life)
        return max(0.1, min(1.0, decay))  # Clamp to [0.1, 1.0]


class FundingStabilityNowcaster:
    """
    Nowcaster for Funding Stability Score.

    Updates scores between quarterly observations using
    high-frequency market data.
    """

    def __init__(
        self,
        funding_data: pl.DataFrame,
        spec: Optional["FundingStabilitySpec"] = None,
    ):
        """
        Initialize nowcaster.

        Args:
            funding_data: Historical funding stability data
            spec: Model specification
        """
        from .indicator import FundingStabilitySpec
        self.funding_data = funding_data
        self.spec = spec or FundingStabilitySpec(
            name="default",
            description="Default spec"
        )
        self._proxy_calculator = HighFrequencyProxyCalculator()

    def nowcast(
        self,
        stock_data: pl.DataFrame,
        nowcast_date: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Generate nowcast for all banks.

        Args:
            stock_data: Recent stock market data
            nowcast_date: Date to nowcast (default: today)

        Returns:
            DataFrame with nowcast scores
        """
        if self.funding_data.height == 0:
            return pl.DataFrame()

        if nowcast_date is None:
            nowcast_date = datetime.now()

        # Get latest quarterly observation for each bank
        latest_quarterly = self.funding_data.group_by("ticker").agg(
            pl.all().sort_by("date").last()
        )

        if latest_quarterly.height == 0:
            return pl.DataFrame()

        # Get base date (latest quarterly date)
        base_date = latest_quarterly["date"].max()

        # Calculate stock-based adjustments
        stock_adj = self._proxy_calculator.calculate_stock_proxy(
            stock_data, base_date, nowcast_date
        )

        # Calculate peer-relative adjustments
        peer_adj = self._proxy_calculator.calculate_peer_relative_proxy(
            stock_data, base_date, nowcast_date
        )

        # Merge adjustments
        result = latest_quarterly.select([
            "ticker",
            "date",
            "funding_resilience_score",
        ]).rename({"date": "base_date", "funding_resilience_score": "base_score"})

        if stock_adj.height > 0:
            result = result.join(
                stock_adj.select(["ticker", "stock_adjustment", "volatility_adjustment"]),
                on="ticker",
                how="left"
            )

        if peer_adj.height > 0:
            result = result.join(
                peer_adj.select(["ticker", "peer_relative_adjustment"]),
                on="ticker",
                how="left"
            )

        # Fill nulls
        for col in ["stock_adjustment", "volatility_adjustment", "peer_relative_adjustment"]:
            if col not in result.columns:
                result = result.with_columns(pl.lit(0.0).alias(col))
            else:
                result = result.with_columns(pl.col(col).fill_null(0.0))

        # Calculate nowcast score
        confidence = self._proxy_calculator.calculate_time_decay(base_date, nowcast_date)

        result = result.with_columns([
            pl.lit(nowcast_date).alias("nowcast_date"),
            pl.lit(confidence).alias("confidence"),
            (
                pl.col("base_score") +
                pl.col("stock_adjustment") +
                pl.col("volatility_adjustment") +
                pl.col("peer_relative_adjustment")
            ).clip(lower_bound=0, upper_bound=100).alias("nowcast_score"),
        ])

        # Calculate change and direction
        result = result.with_columns([
            (pl.col("nowcast_score") - pl.col("base_score")).alias("score_change"),
        ])

        result = result.with_columns(
            pl.when(pl.col("score_change") > 2)
            .then(pl.lit("improving"))
            .when(pl.col("score_change") < -2)
            .then(pl.lit("deteriorating"))
            .otherwise(pl.lit("stable"))
            .alias("direction")
        )

        return result

    def get_alerts(
        self,
        nowcast_df: pl.DataFrame,
        threshold: float = -5.0,
    ) -> pl.DataFrame:
        """
        Get banks with significant score deterioration.

        Args:
            nowcast_df: Nowcast results
            threshold: Score change threshold for alert

        Returns:
            DataFrame with alerted banks
        """
        return nowcast_df.filter(
            pl.col("score_change") < threshold
        ).sort("score_change")
