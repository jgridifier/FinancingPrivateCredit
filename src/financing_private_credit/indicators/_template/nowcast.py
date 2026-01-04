"""
Template Nowcaster

This template shows how to implement nowcasting for your indicator.
Nowcasting provides high-frequency updates between quarterly data releases
using proxy variables that are available more frequently.

Temporal relationship:
    T-8 ←→ T-1         T (now)        T+1 ←→ T+4
    ──────────────     ───────        ─────────────
    calculate()        nowcast()       forecast()
    (historical        (adjust         (predict
     quarterly)        for today)      future)

Common proxy variables:
- Stock prices (daily)
- CDS spreads (daily)
- Fed H.8 weekly bank credit data
- Interest rate movements
- Volatility indices

Copy this file and adapt for your indicator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import polars as pl

from ..base import IndicatorResult


@dataclass
class NowcastSpec:
    """Configuration for nowcasting."""

    name: str = "default"
    description: str = "Default nowcast specification"

    # Proxy weights for combining signals
    weight_stock_return: float = 0.3
    weight_volatility: float = 0.2
    weight_peer_relative: float = 0.2
    weight_macro_surprise: float = 0.3

    # Time decay - confidence decreases as time passes since last quarterly data
    time_decay_rate: float = 0.1  # Decay per week
    min_confidence: float = 0.5  # Minimum confidence level

    # Adjustment bounds
    max_adjustment_pct: float = 0.20  # Maximum adjustment to base estimate

    @classmethod
    def from_dict(cls, d: dict) -> "NowcastSpec":
        """Create spec from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class NowcastResult:
    """Result container for a single bank's nowcast."""

    ticker: str
    base_value: float  # Last quarterly value
    nowcast_value: float  # Adjusted current estimate
    adjustment: float  # Change from base
    confidence: float  # Confidence in nowcast (0-1)
    as_of_date: datetime
    proxy_signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "base_value": self.base_value,
            "nowcast_value": self.nowcast_value,
            "adjustment": self.adjustment,
            "adjustment_pct": self.adjustment / self.base_value if self.base_value else 0,
            "confidence": self.confidence,
            "as_of_date": self.as_of_date.isoformat(),
            "proxy_signals": self.proxy_signals,
        }


class HighFrequencyProxyCalculator:
    """
    Calculate high-frequency proxy signals for nowcasting.

    This class demonstrates common proxy patterns. Customize for your indicator.
    """

    def __init__(self, spec: Optional[NowcastSpec] = None):
        """
        Initialize the proxy calculator.

        Args:
            spec: Nowcast specification
        """
        self.spec = spec or NowcastSpec()

    def calculate_stock_signal(
        self,
        stock_data: pl.DataFrame,
        ticker: str,
        lookback_days: int = 30,
    ) -> float:
        """
        Calculate signal from recent stock performance.

        A negative stock return suggests deteriorating conditions,
        which may indicate the indicator value should be adjusted down.

        Args:
            stock_data: DataFrame with date, ticker, close columns
            ticker: Bank ticker
            lookback_days: Days to look back

        Returns:
            Signal value (-1 to 1, where negative = deterioration)
        """
        if stock_data.height == 0 or ticker not in stock_data["ticker"].unique().to_list():
            return 0.0

        bank_data = stock_data.filter(
            (pl.col("ticker") == ticker) &
            (pl.col("date") >= datetime.now() - timedelta(days=lookback_days))
        ).sort("date")

        if bank_data.height < 2:
            return 0.0

        # Calculate return over lookback period
        first_price = bank_data["close"].first()
        last_price = bank_data["close"].last()

        if first_price is None or first_price == 0:
            return 0.0

        return_pct = (last_price - first_price) / first_price

        # Normalize to -1 to 1 range (assuming ±20% is extreme)
        return max(-1.0, min(1.0, return_pct / 0.20))

    def calculate_volatility_signal(
        self,
        stock_data: pl.DataFrame,
        ticker: str,
        lookback_days: int = 30,
    ) -> float:
        """
        Calculate signal from stock volatility.

        Higher volatility suggests uncertainty, which may indicate
        lower confidence in current estimates.

        Args:
            stock_data: DataFrame with date, ticker, close columns
            ticker: Bank ticker
            lookback_days: Days to look back

        Returns:
            Signal value (0 to 1, where higher = more volatile)
        """
        if stock_data.height == 0:
            return 0.0

        bank_data = stock_data.filter(
            (pl.col("ticker") == ticker) &
            (pl.col("date") >= datetime.now() - timedelta(days=lookback_days))
        ).sort("date")

        if bank_data.height < 5:
            return 0.0

        # Calculate daily returns
        returns = bank_data.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("return")
        )["return"].drop_nulls()

        if returns.len() == 0:
            return 0.0

        # Annualized volatility
        vol = returns.std() * (252 ** 0.5)

        # Normalize (assuming 30% annualized vol is high)
        return min(1.0, vol / 0.30) if vol is not None else 0.0

    def calculate_peer_relative_signal(
        self,
        stock_data: pl.DataFrame,
        ticker: str,
        peer_tickers: list[str],
        lookback_days: int = 30,
    ) -> float:
        """
        Calculate signal from performance relative to peers.

        Underperformance relative to peers suggests bank-specific issues.

        Args:
            stock_data: DataFrame with date, ticker, close columns
            ticker: Bank ticker
            peer_tickers: List of peer bank tickers
            lookback_days: Days to look back

        Returns:
            Signal value (-1 to 1, where negative = underperforming peers)
        """
        bank_signal = self.calculate_stock_signal(stock_data, ticker, lookback_days)

        peer_signals = [
            self.calculate_stock_signal(stock_data, peer, lookback_days)
            for peer in peer_tickers
            if peer != ticker
        ]

        if not peer_signals:
            return 0.0

        peer_avg = sum(peer_signals) / len(peer_signals)
        return bank_signal - peer_avg

    def calculate_time_decay_confidence(
        self,
        last_quarterly_date: datetime,
        as_of_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate confidence based on time since last quarterly data.

        Confidence decreases as we get further from the last data point.

        Args:
            last_quarterly_date: Date of last quarterly data
            as_of_date: Current date (defaults to now)

        Returns:
            Confidence level (0 to 1)
        """
        as_of = as_of_date or datetime.now()
        days_elapsed = (as_of - last_quarterly_date).days
        weeks_elapsed = days_elapsed / 7

        # Exponential decay
        confidence = 1.0 - (self.spec.time_decay_rate * weeks_elapsed)
        return max(self.spec.min_confidence, min(1.0, confidence))


class TemplateNowcaster:
    """
    Template nowcaster for high-frequency indicator updates.

    Combines multiple proxy signals to adjust the last quarterly value
    for current conditions.
    """

    def __init__(self, spec: Optional[NowcastSpec] = None):
        """
        Initialize the nowcaster.

        Args:
            spec: Nowcast specification
        """
        self.spec = spec or NowcastSpec()
        self.proxy_calculator = HighFrequencyProxyCalculator(spec)

    def nowcast(
        self,
        quarterly_data: pl.DataFrame,
        proxy_data: dict[str, pl.DataFrame],
        as_of_date: Optional[datetime] = None,
    ) -> IndicatorResult:
        """
        Generate nowcast for all banks.

        Args:
            quarterly_data: Most recent quarterly indicator values
                           Expected columns: ticker, date, value (your indicator)
            proxy_data: Dictionary of high-frequency proxy DataFrames
                       Expected keys: "stock_data" (with date, ticker, close)
            as_of_date: Date to nowcast as of (defaults to today)

        Returns:
            IndicatorResult with nowcast values
        """
        as_of = as_of_date or datetime.now()
        stock_data = proxy_data.get("stock_data", pl.DataFrame())

        # Get all peer tickers for relative signals
        all_tickers = quarterly_data["ticker"].unique().to_list()

        results = []

        for ticker in all_tickers:
            # Get last quarterly value for this bank
            bank_quarterly = quarterly_data.filter(
                pl.col("ticker") == ticker
            ).sort("date", descending=True)

            if bank_quarterly.height == 0:
                continue

            last_row = bank_quarterly.row(0, named=True)
            base_value = last_row.get("value", 0)
            last_date = last_row.get("date")

            if base_value is None or base_value == 0:
                continue

            # Calculate proxy signals
            stock_signal = self.proxy_calculator.calculate_stock_signal(
                stock_data, ticker
            )
            vol_signal = self.proxy_calculator.calculate_volatility_signal(
                stock_data, ticker
            )
            peer_signal = self.proxy_calculator.calculate_peer_relative_signal(
                stock_data, ticker, all_tickers
            )

            # Combine signals into adjustment
            weighted_signal = (
                self.spec.weight_stock_return * stock_signal +
                self.spec.weight_volatility * (-vol_signal) +  # Higher vol = negative
                self.spec.weight_peer_relative * peer_signal
            )

            # Bound the adjustment
            adjustment_pct = max(
                -self.spec.max_adjustment_pct,
                min(self.spec.max_adjustment_pct, weighted_signal * 0.10)
            )
            adjustment = base_value * adjustment_pct
            nowcast_value = base_value + adjustment

            # Calculate confidence
            last_date_dt = (
                datetime.combine(last_date, datetime.min.time())
                if hasattr(last_date, 'year') and not isinstance(last_date, datetime)
                else last_date
            )
            confidence = self.proxy_calculator.calculate_time_decay_confidence(
                last_date_dt, as_of
            )

            results.append(NowcastResult(
                ticker=ticker,
                base_value=float(base_value),
                nowcast_value=float(nowcast_value),
                adjustment=float(adjustment),
                confidence=float(confidence),
                as_of_date=as_of,
                proxy_signals={
                    "stock_return": float(stock_signal),
                    "volatility": float(vol_signal),
                    "peer_relative": float(peer_signal),
                },
            ))

        # Convert to DataFrame
        if not results:
            result_df = pl.DataFrame()
        else:
            result_df = pl.DataFrame([r.to_dict() for r in results])

        return IndicatorResult(
            indicator_name="template_nowcast",
            calculation_date=as_of,
            data=result_df,
            metadata={
                "n_banks": len(results),
                "as_of_date": as_of.isoformat(),
                "spec_name": self.spec.name,
                "avg_confidence": sum(r.confidence for r in results) / len(results) if results else 0,
            },
        )

    def nowcast_single(
        self,
        ticker: str,
        base_value: float,
        last_quarterly_date: datetime,
        stock_data: pl.DataFrame,
        peer_tickers: list[str],
        as_of_date: Optional[datetime] = None,
    ) -> NowcastResult:
        """
        Generate nowcast for a single bank.

        Args:
            ticker: Bank ticker
            base_value: Last quarterly indicator value
            last_quarterly_date: Date of last quarterly data
            stock_data: Stock price data
            peer_tickers: List of peer bank tickers
            as_of_date: Date to nowcast as of

        Returns:
            NowcastResult for this bank
        """
        as_of = as_of_date or datetime.now()

        # Calculate signals
        stock_signal = self.proxy_calculator.calculate_stock_signal(
            stock_data, ticker
        )
        vol_signal = self.proxy_calculator.calculate_volatility_signal(
            stock_data, ticker
        )
        peer_signal = self.proxy_calculator.calculate_peer_relative_signal(
            stock_data, ticker, peer_tickers
        )

        # Combine signals
        weighted_signal = (
            self.spec.weight_stock_return * stock_signal +
            self.spec.weight_volatility * (-vol_signal) +
            self.spec.weight_peer_relative * peer_signal
        )

        adjustment_pct = max(
            -self.spec.max_adjustment_pct,
            min(self.spec.max_adjustment_pct, weighted_signal * 0.10)
        )
        adjustment = base_value * adjustment_pct
        nowcast_value = base_value + adjustment

        confidence = self.proxy_calculator.calculate_time_decay_confidence(
            last_quarterly_date, as_of
        )

        return NowcastResult(
            ticker=ticker,
            base_value=base_value,
            nowcast_value=nowcast_value,
            adjustment=adjustment,
            confidence=confidence,
            as_of_date=as_of,
            proxy_signals={
                "stock_return": stock_signal,
                "volatility": vol_signal,
                "peer_relative": peer_signal,
            },
        )


# Integration with BaseIndicator
# ------------------------------
# To use nowcasting in your indicator, override the nowcast() method:
#
# class MyIndicator(BaseIndicator):
#     supports_nowcast = True  # Enable nowcasting
#
#     def __init__(self, ...):
#         super().__init__(...)
#         self._nowcaster = TemplateNowcaster()
#
#     def nowcast(self, data: dict[str, pl.DataFrame], **kwargs) -> IndicatorResult:
#         quarterly_data = data.get("quarterly_results")
#         stock_data = data.get("stock_data")
#
#         return self._nowcaster.nowcast(
#             quarterly_data,
#             {"stock_data": stock_data},
#         )
