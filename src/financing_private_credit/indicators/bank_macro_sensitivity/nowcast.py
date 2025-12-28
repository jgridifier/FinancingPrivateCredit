"""
Nowcasting for Bank Macro Sensitivity

Provides real-time estimates of bank NIM responses using:
1. High-frequency macro proxies (daily/weekly data from FRED)
2. Current quarter projections from the fitted models
3. Uncertainty quantification for incomplete quarters

Key data sources for nowcasting:
- Daily: Fed Funds Rate (DFF), Treasury yields (DGS10, DGS2), VIX
- Weekly: H.8 bank credit data, initial claims
- Monthly: CPI, employment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import polars as pl

from .indicator import MacroSensitivitySpec


@dataclass
class NowcastResult:
    """Result of a nowcast estimation."""

    nowcast_date: datetime
    quarter: str  # e.g., "2024Q4"

    # Current macro environment estimate
    current_macro: dict[str, float]

    # Bank-level nowcasts
    bank_nowcasts: pl.DataFrame  # [ticker, nowcast_nim, uncertainty]

    # Industry nowcast
    industry_nowcast: float
    industry_uncertainty: float

    # Data freshness
    data_as_of: dict[str, datetime]

    # Quality indicators
    macro_data_completeness: float  # 0-1, how much of quarter we have
    estimation_quality: str  # "high", "medium", "low"

    metadata: dict[str, Any] = field(default_factory=dict)


class MacroNowcaster:
    """
    Nowcast current macro environment using high-frequency data.

    Aggregates daily/weekly data to estimate current quarter's macro values.
    """

    # FRED series and their frequencies for nowcasting
    HF_SERIES = {
        # Daily rates
        "DFF": ("daily", "rate_spread"),  # Fed funds -> rate spread component
        "DGS10": ("daily", "rate_spread"),  # 10Y Treasury
        "DGS2": ("daily", "term_spread"),  # 2Y Treasury
        "VIXCLS": ("daily", "vix"),

        # Weekly credit data
        "TOTLL": ("weekly", "credit_growth"),

        # Monthly inflation
        "CPIAUCSL": ("monthly", "inflation_yoy"),
    }

    def __init__(self, start_date: str = "2020-01-01"):
        self.start_date = start_date
        self._hf_data: dict[str, pl.DataFrame] = {}

    def fetch_high_frequency_data(self) -> dict[str, pl.DataFrame]:
        """
        Fetch latest high-frequency data from FRED.

        Returns:
            Dictionary mapping series to DataFrames
        """
        from ...cache import CachedFREDFetcher

        fetcher = CachedFREDFetcher(max_age_hours=1)  # Fresh data for nowcasting

        for series_id in self.HF_SERIES.keys():
            try:
                df = fetcher.fetch_series(series_id, start_date=self.start_date)
                if df is not None and not df.is_empty():
                    self._hf_data[series_id] = df
            except Exception as e:
                print(f"Warning: Could not fetch {series_id}: {e}")

        return self._hf_data

    def estimate_current_quarter(self) -> tuple[dict[str, float], dict[str, datetime]]:
        """
        Estimate current quarter's macro values from available HF data.

        Returns:
            Tuple of (macro_estimates, data_as_of_dates)
        """
        if not self._hf_data:
            self.fetch_high_frequency_data()

        estimates = {}
        data_as_of = {}

        # Current quarter date range
        today = datetime.now()
        quarter_start = datetime(today.year, ((today.month - 1) // 3) * 3 + 1, 1)

        # Rate spread: 10Y - Fed Funds
        if "DGS10" in self._hf_data and "DFF" in self._hf_data:
            dgs10 = self._hf_data["DGS10"].filter(pl.col("date") >= quarter_start)
            dff = self._hf_data["DFF"].filter(pl.col("date") >= quarter_start)

            if dgs10.height > 0 and dff.height > 0:
                avg_10y = dgs10["DGS10"].mean()
                avg_ff = dff["DFF"].mean()
                estimates["rate_spread"] = float(avg_10y - avg_ff) if avg_10y and avg_ff else 0

                data_as_of["rate_spread"] = max(
                    dgs10["date"].max(),
                    dff["date"].max()
                )

        # Term spread: 10Y - 2Y
        if "DGS10" in self._hf_data and "DGS2" in self._hf_data:
            dgs10 = self._hf_data["DGS10"].filter(pl.col("date") >= quarter_start)
            dgs2 = self._hf_data["DGS2"].filter(pl.col("date") >= quarter_start)

            if dgs10.height > 0 and dgs2.height > 0:
                avg_10y = dgs10["DGS10"].mean()
                avg_2y = dgs2["DGS2"].mean()
                estimates["term_spread"] = float(avg_10y - avg_2y) if avg_10y and avg_2y else 0
                data_as_of["term_spread"] = max(dgs10["date"].max(), dgs2["date"].max())

        # VIX
        if "VIXCLS" in self._hf_data:
            vix = self._hf_data["VIXCLS"].filter(pl.col("date") >= quarter_start)
            if vix.height > 0:
                estimates["vix"] = float(vix["VIXCLS"].mean())
                data_as_of["vix"] = vix["date"].max()

        # Inflation (use latest YoY)
        if "CPIAUCSL" in self._hf_data:
            cpi = self._hf_data["CPIAUCSL"].sort("date")
            if cpi.height >= 12:
                latest = cpi["CPIAUCSL"][-1]
                year_ago = cpi["CPIAUCSL"][-12]
                if latest and year_ago:
                    estimates["inflation_yoy"] = float((latest / year_ago - 1) * 100)
                    data_as_of["inflation_yoy"] = cpi["date"].max()

        # Output gap proxy: use capacity utilization or other leading indicators
        # For simplicity, use trailing values or external forecasts
        # Default to slight positive gap
        if "output_gap" not in estimates:
            estimates["output_gap"] = 0.0
            data_as_of["output_gap"] = quarter_start

        # Credit spread default
        if "credit_spread" not in estimates:
            estimates["credit_spread"] = 1.5

        return estimates, data_as_of


class MacroSensitivityNowcaster:
    """
    Nowcast bank NIM using current macro environment.

    Combines:
    1. High-frequency macro data aggregation
    2. Fitted sensitivity models for prediction
    3. Uncertainty from partial-quarter data
    """

    def __init__(
        self,
        model,  # Fitted APLRSensitivityModel or FallbackLinearModel
        spec: Optional[MacroSensitivitySpec] = None,
    ):
        self.model = model
        self.spec = spec or MacroSensitivitySpec(name="nowcast", description="Nowcast spec")
        self.macro_nowcaster = MacroNowcaster()

    def nowcast(
        self,
        macro_data: Optional[pl.DataFrame] = None,
        use_high_frequency: bool = True,
    ) -> pl.DataFrame:
        """
        Generate nowcast for current quarter.

        Args:
            macro_data: Historical macro data (for comparison)
            use_high_frequency: Whether to use HF data for current quarter

        Returns:
            DataFrame with bank nowcasts
        """
        # Get current macro estimates
        if use_high_frequency:
            current_macro, data_as_of = self.macro_nowcaster.estimate_current_quarter()
        else:
            # Use latest from historical data
            if macro_data is not None and macro_data.height > 0:
                latest = macro_data.sort("date").tail(1)
                current_macro = {
                    col: float(latest[col][0]) if latest[col][0] is not None else 0
                    for col in self.spec.macro_features
                    if col in latest.columns
                }
                data_as_of = {"historical": latest["date"][0]}
            else:
                current_macro = {}
                data_as_of = {}

        # Generate predictions for each bank
        banks = list(self.model.models.keys()) if hasattr(self.model, 'models') else list(self.model.coefficients.keys())

        nowcasts = []
        for bank in banks:
            try:
                nim_pred = self.model.predict(bank, current_macro)

                # Estimate uncertainty based on data completeness
                # More of quarter elapsed = lower uncertainty
                today = datetime.now()
                quarter_progress = self._get_quarter_progress(today)
                base_std = 0.3  # Base NIM uncertainty in percentage points
                uncertainty = base_std * (1 - quarter_progress * 0.5)

                nowcasts.append({
                    "ticker": bank,
                    "nowcast_nim": nim_pred,
                    "uncertainty": uncertainty,
                    "lower_bound": nim_pred - 1.96 * uncertainty,
                    "upper_bound": nim_pred + 1.96 * uncertainty,
                })

            except Exception as e:
                nowcasts.append({
                    "ticker": bank,
                    "nowcast_nim": None,
                    "uncertainty": None,
                    "lower_bound": None,
                    "upper_bound": None,
                    "error": str(e),
                })

        return pl.DataFrame(nowcasts)

    def _get_quarter_progress(self, date: datetime) -> float:
        """Calculate how far through the current quarter we are (0-1)."""
        quarter_start_month = ((date.month - 1) // 3) * 3 + 1
        quarter_start = datetime(date.year, quarter_start_month, 1)

        if quarter_start_month == 10:
            quarter_end = datetime(date.year + 1, 1, 1)
        else:
            quarter_end = datetime(date.year, quarter_start_month + 3, 1)

        total_days = (quarter_end - quarter_start).days
        elapsed_days = (date - quarter_start).days

        return min(1.0, elapsed_days / total_days)

    def get_full_nowcast(
        self,
        macro_data: Optional[pl.DataFrame] = None,
    ) -> NowcastResult:
        """
        Generate comprehensive nowcast result.

        Args:
            macro_data: Historical macro data

        Returns:
            NowcastResult with all details
        """
        # Get current macro
        current_macro, data_as_of = self.macro_nowcaster.estimate_current_quarter()

        # Get bank nowcasts
        bank_nowcasts = self.nowcast(macro_data)

        # Calculate industry aggregate
        valid_nowcasts = bank_nowcasts.filter(pl.col("nowcast_nim").is_not_null())

        if valid_nowcasts.height > 0:
            industry_nowcast = float(valid_nowcasts["nowcast_nim"].mean())
            industry_uncertainty = float(valid_nowcasts["uncertainty"].mean())
        else:
            industry_nowcast = 0.0
            industry_uncertainty = 1.0

        # Determine data quality
        today = datetime.now()
        quarter_progress = self._get_quarter_progress(today)

        if quarter_progress > 0.8 and len(current_macro) >= 4:
            quality = "high"
        elif quarter_progress > 0.4 and len(current_macro) >= 2:
            quality = "medium"
        else:
            quality = "low"

        # Format quarter string
        quarter_num = (today.month - 1) // 3 + 1
        quarter_str = f"{today.year}Q{quarter_num}"

        return NowcastResult(
            nowcast_date=today,
            quarter=quarter_str,
            current_macro=current_macro,
            bank_nowcasts=bank_nowcasts,
            industry_nowcast=industry_nowcast,
            industry_uncertainty=industry_uncertainty,
            data_as_of={k: v for k, v in data_as_of.items() if isinstance(v, datetime)},
            macro_data_completeness=quarter_progress,
            estimation_quality=quality,
            metadata={
                "n_banks": valid_nowcasts.height,
                "macro_vars_available": list(current_macro.keys()),
            },
        )

    def compare_to_last_quarter(
        self,
        macro_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Compare nowcast to last quarter's actual values.

        Args:
            macro_data: Historical macro data with actual values

        Returns:
            DataFrame comparing nowcast to historical
        """
        nowcast_df = self.nowcast(macro_data)

        # Get last quarter's values from historical data
        # This would require joining with actual NIM data

        return nowcast_df.with_columns(
            pl.lit("Current Q Nowcast").alias("period_type")
        )

    def sensitivity_to_macro_surprise(
        self,
        surprise_variable: str,
        surprise_magnitude: float,
    ) -> pl.DataFrame:
        """
        Estimate bank response to a macro surprise.

        Args:
            surprise_variable: Which macro variable surprises
            surprise_magnitude: Size of surprise (in standard deviations)

        Returns:
            DataFrame with bank responses to the surprise
        """
        # Get baseline nowcast
        current_macro, _ = self.macro_nowcaster.estimate_current_quarter()

        # Get bank sensitivities
        banks = list(self.model.models.keys()) if hasattr(self.model, 'models') else list(self.model.coefficients.keys())

        responses = []
        for bank in banks:
            try:
                # Baseline prediction
                baseline = self.model.predict(bank, current_macro)

                # Shocked prediction
                shocked_macro = current_macro.copy()
                if surprise_variable in shocked_macro:
                    shocked_macro[surprise_variable] += surprise_magnitude
                else:
                    shocked_macro[surprise_variable] = surprise_magnitude

                shocked = self.model.predict(bank, shocked_macro)

                responses.append({
                    "ticker": bank,
                    "baseline_nim": baseline,
                    "shocked_nim": shocked,
                    "response": shocked - baseline,
                    "relative_response_pct": (shocked - baseline) / baseline * 100 if baseline != 0 else 0,
                })

            except Exception:
                pass

        return pl.DataFrame(responses).sort("response", descending=True)
