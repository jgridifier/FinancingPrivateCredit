"""
Lending Intensity Score (LIS) Calculation.

Implements the core LIS methodology from NY Fed Staff Report 1111:
LIS = (Bank Loan Growth - System Loan Growth) / σ(System Loan Growth)

Key Insight:
- Banks that expand credit aggressively (high LIS) today will have
  higher provision rates 3-4 years later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl


@dataclass
class LISResult:
    """Lending Intensity Score result for a bank."""

    ticker: str
    date: datetime
    lis: float  # Standard deviations from system mean
    bank_growth: float
    system_growth: float
    system_std: float
    percentile: float  # Rank among peers


class LendingIntensityScore:
    """
    Compute Lending Intensity Score (LIS) for each bank.

    LIS = (Bank Loan Growth - System Loan Growth) / σ(System Loan Growth)

    Interpretation:
    - LIS > 0: Bank is lending more aggressively than peers
    - LIS > 1: Bank is > 1 SD above system average
    - LIS > 2: Warning sign of aggressive lending
    """

    def __init__(
        self,
        bank_data: pl.DataFrame,
        system_data: pl.DataFrame,
        growth_col: str = "loan_growth_yoy",
    ):
        """
        Initialize LIS calculator.

        Args:
            bank_data: Panel of bank-level data with ticker, date, growth
            system_data: System-wide growth data with date and growth
            growth_col: Name of growth column
        """
        self.bank_data = bank_data
        self.system_data = system_data
        self.growth_col = growth_col

    def compute_lis(self) -> pl.DataFrame:
        """
        Compute LIS for all banks at all dates.

        Returns:
            DataFrame with ticker, date, lis, and components
        """
        # Merge bank and system data
        df = self.bank_data.join(
            self.system_data.select(["date", self.growth_col]).rename(
                {self.growth_col: "system_growth"}
            ),
            on="date",
            how="left",
        )

        # Rename bank growth column
        df = df.with_columns(
            pl.col(self.growth_col).alias("bank_growth")
        )

        # Compute rolling system statistics (20-quarter window)
        df = df.with_columns([
            pl.col("system_growth").rolling_mean(window_size=20).alias("system_mean"),
            pl.col("system_growth").rolling_std(window_size=20).alias("system_std"),
        ])

        # Compute LIS
        df = df.with_columns(
            ((pl.col("bank_growth") - pl.col("system_growth")) / pl.col("system_std"))
            .alias("lis")
        )

        # Compute cumulative LIS (sum over 12 quarters)
        df = df.with_columns(
            pl.col("lis").rolling_sum(window_size=12).over("ticker").alias("lis_cumulative_12q")
        )

        # Compute percentile rank among peers at each date
        df = df.with_columns(
            pl.col("bank_growth").rank().over("date").alias("growth_rank")
        )
        df = df.with_columns(
            (pl.col("growth_rank") / pl.col("growth_rank").max().over("date") * 100)
            .alias("growth_percentile")
        )

        return df

    def get_current_signals(self, threshold: float = 1.0) -> pl.DataFrame:
        """
        Get current LIS signals for all banks.

        Args:
            threshold: LIS threshold for flagging (default: 1 SD)

        Returns:
            DataFrame with current signals sorted by risk
        """
        df = self.compute_lis()

        # Get most recent date for each bank
        latest = df.group_by("ticker").agg(
            pl.col("date").max().alias("date")
        )

        # Filter to latest observations
        current = df.join(latest, on=["ticker", "date"], how="inner")

        # Add risk flags
        current = current.with_columns([
            (pl.col("lis") > threshold).alias("elevated_lis"),
            (pl.col("lis_cumulative_12q") > threshold * 4).alias("sustained_elevation"),
        ])

        return current.sort("lis", descending=True)

    def compute_cross_sectional_lis(self, panel: pl.DataFrame) -> pl.DataFrame:
        """
        Compute LIS using cross-sectional standardization at each date.

        This is an alternative to time-series standardization that
        compares each bank to contemporaneous peers.
        """
        if panel.height == 0 or "loan_growth_yoy" not in panel.columns:
            return panel

        # Filter to rows with valid loan growth
        panel_valid = panel.filter(pl.col("loan_growth_yoy").is_not_null())

        if panel_valid.height == 0:
            return panel

        # Compute system-wide statistics at each date (cross-sectional)
        system_stats = (
            panel_valid
            .group_by("date")
            .agg([
                pl.col("loan_growth_yoy").mean().alias("system_growth"),
                pl.col("loan_growth_yoy").std().alias("system_std"),
                pl.col("loan_growth_yoy").count().alias("n_banks"),
            ])
            .sort("date")
        )

        # Join system stats back to panel
        result = panel.join(system_stats, on="date", how="left")

        # Compute LIS = (bank_growth - system_growth) / system_std
        result = result.with_columns(
            pl.when(pl.col("system_std") > 0.01)
            .then((pl.col("loan_growth_yoy") - pl.col("system_growth")) / pl.col("system_std"))
            .otherwise(0.0)
            .alias("lis")
        )

        # Compute cumulative LIS (rolling 12-quarter sum)
        result = result.sort(["ticker", "date"]).with_columns(
            pl.col("lis")
            .rolling_sum(window_size=12, min_periods=1)
            .over("ticker")
            .alias("lis_cumulative_12q")
        )

        return result
