"""
Nowcasting for Duration Mismatch Indicator

Provides real-time estimates of:
1. Current rate exposure using daily yield data
2. Implied volatility from options markets
3. Predicted earnings impact for current quarter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import polars as pl

from .indicator import DurationMismatchSpec


@dataclass
class DurationNowcastResult:
    """Result of duration mismatch nowcast."""

    nowcast_date: datetime
    quarter: str

    # Bank-level nowcasts
    bank_nowcasts: pl.DataFrame

    # Industry summary
    industry_avg_duration: float
    industry_total_dv01: float
    current_yield_change_qtd: float  # Quarter-to-date yield change

    # Risk assessment
    most_vulnerable_banks: list[str]
    least_vulnerable_banks: list[str]

    metadata: dict[str, Any] = field(default_factory=dict)


class YieldCurveNowcaster:
    """
    Nowcast yield curve movements using daily FRED data.

    Provides:
    - Quarter-to-date yield changes
    - Implied rate volatility
    - Term structure shifts
    """

    def __init__(self):
        self._yield_data: Optional[pl.DataFrame] = None

    def fetch_current_yields(self) -> pl.DataFrame:
        """Fetch latest yield curve data from FRED."""
        from ...cache import CachedFREDFetcher

        fetcher = CachedFREDFetcher(max_age_hours=1)

        # Key yield series
        series = ["DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]

        # Fetch last 6 months for context
        from datetime import datetime, timedelta
        start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

        yields = fetcher.fetch_multiple_series(series, start_date=start)
        self._yield_data = yields
        return yields

    def compute_qtd_changes(self) -> dict[str, float]:
        """Compute quarter-to-date yield changes."""
        if self._yield_data is None:
            self.fetch_current_yields()

        if self._yield_data is None or self._yield_data.height == 0:
            return {}

        # Find quarter start
        today = datetime.now()
        quarter_start_month = ((today.month - 1) // 3) * 3 + 1
        quarter_start = datetime(today.year, quarter_start_month, 1).date()

        # Filter to quarter
        qtd = self._yield_data.filter(pl.col("date") >= quarter_start)

        if qtd.height < 2:
            return {}

        changes = {}
        for col in ["DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]:
            if col in qtd.columns:
                first_val = qtd.filter(pl.col(col).is_not_null()).head(1)
                last_val = qtd.filter(pl.col(col).is_not_null()).tail(1)

                if first_val.height > 0 and last_val.height > 0:
                    changes[col] = float(last_val[col][0] - first_val[col][0])

        return changes

    def compute_yield_volatility(self, window: int = 20) -> dict[str, float]:
        """Compute realized volatility of yields."""
        if self._yield_data is None:
            self.fetch_current_yields()

        if self._yield_data is None or self._yield_data.height < window:
            return {}

        vols = {}
        for col in ["DGS2", "DGS10", "DGS30"]:
            if col in self._yield_data.columns:
                changes = self._yield_data[col].diff().drop_nulls()
                if len(changes) >= window:
                    recent = changes.tail(window)
                    vols[f"{col}_vol"] = float(recent.std() * np.sqrt(252))  # Annualized

        return vols


class DurationMismatchNowcaster:
    """
    Nowcast duration exposure and predicted impact.

    Uses:
    1. Latest bank duration estimates
    2. Real-time yield changes
    3. Options-implied volatility (if available)
    """

    def __init__(
        self,
        duration_data: pl.DataFrame,
        spec: Optional[DurationMismatchSpec] = None,
    ):
        self.duration_data = duration_data
        self.spec = spec
        self.yield_nowcaster = YieldCurveNowcaster()

    def nowcast(
        self,
        yields: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        Generate nowcast for current period.

        Args:
            yields: Optional pre-fetched yield data

        Returns:
            DataFrame with bank nowcasts
        """
        # Get latest duration for each bank
        latest_duration = self._get_latest_duration()

        if latest_duration.height == 0:
            return pl.DataFrame()

        # Get current yield changes
        if yields is not None and yields.height > 0:
            self.yield_nowcaster._yield_data = yields

        qtd_changes = self.yield_nowcaster.compute_qtd_changes()
        yield_change_10y = qtd_changes.get("DGS10", 0.0)

        # Compute predicted impact for current quarter
        result = latest_duration.with_columns([
            pl.lit(yield_change_10y).alias("yield_change_qtd"),
            (
                -pl.col("estimated_duration") * yield_change_10y / 100 *
                pl.col("total_securities").fill_null(0)
            ).alias("predicted_impact_qtd"),
        ])

        # Normalize impact
        total_securities_sum = result["total_securities"].sum()
        if total_securities_sum and total_securities_sum > 0:
            result = result.with_columns(
                (pl.col("predicted_impact_qtd") / total_securities_sum * 100)
                .alias("predicted_impact_pct_qtd")
            )

        # Current vulnerability score
        if "stock_volatility" in result.columns:
            result = result.with_columns(
                (
                    pl.col("estimated_duration").abs() *
                    pl.col("stock_volatility").fill_null(0.02) *
                    (1 + abs(yield_change_10y) / 0.5)  # Scale by rate move
                ).alias("current_vulnerability")
            )
        else:
            result = result.with_columns(
                (pl.col("estimated_duration").abs() * 0.02 * (1 + abs(yield_change_10y) / 0.5))
                .alias("current_vulnerability")
            )

        # Rank by vulnerability
        result = result.with_columns(
            pl.col("current_vulnerability").rank(descending=True).alias("vulnerability_rank")
        )

        return result.sort("vulnerability_rank")

    def _get_latest_duration(self) -> pl.DataFrame:
        """Get most recent duration data for each bank."""
        if self.duration_data.height == 0:
            return pl.DataFrame()

        # Get latest row for each bank
        latest = (
            self.duration_data
            .sort("date", descending=True)
            .group_by("ticker")
            .head(1)
        )

        return latest

    def get_full_nowcast(self) -> DurationNowcastResult:
        """Generate comprehensive nowcast result."""
        bank_nowcasts = self.nowcast()

        if bank_nowcasts.height == 0:
            return DurationNowcastResult(
                nowcast_date=datetime.now(),
                quarter=self._get_current_quarter(),
                bank_nowcasts=pl.DataFrame(),
                industry_avg_duration=0.0,
                industry_total_dv01=0.0,
                current_yield_change_qtd=0.0,
                most_vulnerable_banks=[],
                least_vulnerable_banks=[],
            )

        # Industry aggregates
        avg_duration = float(bank_nowcasts["estimated_duration"].mean())
        total_dv01 = float(bank_nowcasts["dv01"].sum()) if "dv01" in bank_nowcasts.columns else 0.0

        # Current yield change
        qtd_changes = self.yield_nowcaster.compute_qtd_changes()
        yield_change = qtd_changes.get("DGS10", 0.0)

        # Most/least vulnerable
        n_show = min(3, bank_nowcasts.height)
        most_vulnerable = bank_nowcasts.head(n_show)["ticker"].to_list()
        least_vulnerable = bank_nowcasts.tail(n_show)["ticker"].to_list()

        return DurationNowcastResult(
            nowcast_date=datetime.now(),
            quarter=self._get_current_quarter(),
            bank_nowcasts=bank_nowcasts,
            industry_avg_duration=avg_duration,
            industry_total_dv01=total_dv01,
            current_yield_change_qtd=yield_change,
            most_vulnerable_banks=most_vulnerable,
            least_vulnerable_banks=least_vulnerable,
            metadata={
                "yield_vols": self.yield_nowcaster.compute_yield_volatility(),
                "qtd_changes": qtd_changes,
            },
        )

    def _get_current_quarter(self) -> str:
        """Get current quarter string."""
        today = datetime.now()
        quarter = (today.month - 1) // 3 + 1
        return f"{today.year}Q{quarter}"

    def stress_test(
        self,
        rate_shocks: list[float] = [-1.0, -0.5, 0.5, 1.0, 2.0],
    ) -> pl.DataFrame:
        """
        Stress test banks under various rate shock scenarios.

        Args:
            rate_shocks: List of rate changes in percentage points

        Returns:
            DataFrame with stress test results
        """
        latest = self._get_latest_duration()

        if latest.height == 0:
            return pl.DataFrame()

        results = []

        for shock in rate_shocks:
            for row in latest.iter_rows(named=True):
                duration = row.get("estimated_duration", 5.0)
                total_sec = row.get("total_securities", 0)

                # Predicted impact = -duration × rate_change × portfolio
                impact = -duration * shock / 100 * total_sec

                # Impact as % of securities portfolio
                impact_pct = impact / total_sec * 100 if total_sec > 0 else 0

                results.append({
                    "ticker": row["ticker"],
                    "rate_shock_bp": shock * 100,
                    "duration": duration,
                    "predicted_impact_millions": impact / 1e6,
                    "impact_pct_portfolio": impact_pct,
                })

        df = pl.DataFrame(results)

        # Rank within each shock scenario
        df = df.with_columns(
            pl.col("impact_pct_portfolio").abs()
            .rank(descending=True)
            .over("rate_shock_bp")
            .alias("vulnerability_rank")
        )

        return df.sort(["rate_shock_bp", "vulnerability_rank"])
