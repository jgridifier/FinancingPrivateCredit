"""
Credit Boom Leading Indicator

Implements the Lending Intensity Score (LIS) methodology from NY Fed Staff Report 1111.
This indicator identifies banks with aggressive lending behavior that may precede
credit losses.

Key Metrics:
- LIS: (Bank Loan Growth - System Average) / System Std Dev
- Cumulative LIS: Rolling 12-quarter sum of LIS
- Provision Rate Forecasts: APLR-based predictions

Data Sources:
- SEC EDGAR: Bank-level loan and provision data
- FRED H.8: System-wide bank credit data
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import polars as pl

from .base import (
    BaseIndicator,
    IndicatorMetadata,
    IndicatorResult,
    register_indicator,
)


@register_indicator("credit_boom")
class CreditBoomIndicator(BaseIndicator):
    """
    Credit Boom Leading Indicator based on Lending Intensity Scores.

    Identifies banks with aggressive lending patterns that historically
    precede elevated credit losses.
    """

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="Credit Boom Leading Indicator",
            short_name="LIS",
            description=(
                "Measures relative lending intensity across banks using "
                "cross-sectional standardization. High LIS indicates aggressive "
                "lending relative to peers."
            ),
            version="1.0.0",
            paper_reference="NY Fed Staff Report 1111: Financing Private Credit",
            data_sources=["SEC EDGAR (10-K/10-Q)", "FRED H.8"],
            update_frequency="quarterly",
            lookback_periods=20,  # 5 years of quarterly data
        )

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch SEC EDGAR and FRED data."""
        from ..bank_data import BankDataCollector
        from ..cache import CachedFREDFetcher

        # Fetch bank-level data from SEC EDGAR
        collector = BankDataCollector(start_date=start_date)
        bank_panel = collector.fetch_all_banks()
        bank_panel = collector.compute_derived_metrics(bank_panel)

        # Fetch system-wide H.8 data
        fetcher = CachedFREDFetcher(max_age_hours=6)
        h8_series = ["TOTLL", "BUSLOANS", "CONSUMER", "REALLN"]
        system_data = fetcher.fetch_multiple_series(h8_series, start_date=start_date)

        # Get data quality summary
        quality = collector.get_data_quality_summary()

        return {
            "bank_panel": bank_panel,
            "system_h8": system_data,
            "data_quality": quality,
        }

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """Calculate LIS scores for all banks."""
        bank_panel = data.get("bank_panel", pl.DataFrame())

        if bank_panel.height == 0:
            return IndicatorResult(
                indicator_name="credit_boom",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No bank data available"},
            )

        # Compute LIS scores
        lis_panel = self._compute_lis(bank_panel)

        return IndicatorResult(
            indicator_name="credit_boom",
            calculation_date=datetime.now(),
            data=lis_panel,
            metadata={
                "n_banks": lis_panel["ticker"].n_unique(),
                "date_range": (
                    str(lis_panel["date"].min()),
                    str(lis_panel["date"].max()),
                ),
            },
        )

    def _compute_lis(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Compute Lending Intensity Scores."""
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

    def nowcast(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """
        Nowcast current quarter LIS using weekly H.8 data.

        Uses weekly bank credit growth as proxy for quarterly LIS.
        """
        system_h8 = data.get("system_h8", pl.DataFrame())

        if system_h8.height == 0 or "TOTLL" not in system_h8.columns:
            return IndicatorResult(
                indicator_name="credit_boom_nowcast",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No H.8 data available"},
            )

        # Compute weekly growth as proxy
        weekly_growth = system_h8.with_columns(
            ((pl.col("TOTLL") / pl.col("TOTLL").shift(52) - 1) * 100)
            .alias("total_credit_growth_yoy")
        )

        # Latest reading
        latest = weekly_growth.filter(
            pl.col("total_credit_growth_yoy").is_not_null()
        ).tail(1)

        return IndicatorResult(
            indicator_name="credit_boom_nowcast",
            calculation_date=datetime.now(),
            data=weekly_growth,
            metadata={
                "latest_growth": float(latest["total_credit_growth_yoy"][0]) if latest.height > 0 else None,
                "latest_date": str(latest["date"][0]) if latest.height > 0 else None,
                "methodology": "Weekly H.8 credit growth as LIS proxy",
            },
        )

    def get_dashboard_components(self) -> dict[str, Any]:
        """Return dashboard configuration for this indicator."""
        return {
            "tabs": [
                {"name": "LIS Analysis", "icon": "chart_with_upwards_trend"},
                {"name": "Early Warnings", "icon": "warning"},
                {"name": "Forecasts", "icon": "crystal_ball"},
            ],
            "primary_metric": "lis",
            "warning_thresholds": {
                "high_risk": 2.0,
                "elevated": 1.5,
                "moderate": 1.0,
                "conservative": -1.5,
            },
        }

    def get_warning_level(self, lis_value: float) -> tuple[str, str]:
        """
        Get warning level for a given LIS value.

        Returns:
            Tuple of (emoji, status_text)
        """
        if lis_value > 2.0:
            return ("🔴", "HIGH RISK")
        elif lis_value > 1.5:
            return ("🟠", "ELEVATED")
        elif lis_value > 1.0:
            return ("🟡", "MODERATE")
        elif lis_value < -1.5:
            return ("🟢", "CONSERVATIVE")
        else:
            return ("⚪", "NORMAL")
