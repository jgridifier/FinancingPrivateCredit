"""
Macroeconomic data module for credit boom leading indicator model.

Fetches comprehensive macro controls from FRED matching the paper's methodology:
- Output gap (GDP vs potential)
- Inflation
- Credit spreads
- Financial conditions
- Banking system aggregates (H.8 release)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl

from .data import FREDDataFetcher


@dataclass
class MacroSeriesInfo:
    """Metadata for a macro series."""

    series_id: str
    description: str
    transformation: str  # 'level', 'yoy_pct', 'diff', 'log_diff'
    category: str


# Macro series from the paper's methodology
MACRO_SERIES = {
    # Output and Growth
    "GDPC1": MacroSeriesInfo(
        series_id="GDPC1",
        description="Real Gross Domestic Product",
        transformation="yoy_pct",
        category="output",
    ),
    "GDPPOT": MacroSeriesInfo(
        series_id="GDPPOT",
        description="Real Potential GDP",
        transformation="level",
        category="output",
    ),
    # Inflation
    "CPIAUCSL": MacroSeriesInfo(
        series_id="CPIAUCSL",
        description="Consumer Price Index (All Urban)",
        transformation="yoy_pct",
        category="inflation",
    ),
    "PCEPI": MacroSeriesInfo(
        series_id="PCEPI",
        description="PCE Price Index",
        transformation="yoy_pct",
        category="inflation",
    ),
    # Interest Rates
    "DFF": MacroSeriesInfo(
        series_id="DFF",
        description="Federal Funds Effective Rate",
        transformation="level",
        category="rates",
    ),
    "DGS10": MacroSeriesInfo(
        series_id="DGS10",
        description="10-Year Treasury Constant Maturity Rate",
        transformation="level",
        category="rates",
    ),
    "DGS2": MacroSeriesInfo(
        series_id="DGS2",
        description="2-Year Treasury Constant Maturity Rate",
        transformation="level",
        category="rates",
    ),
    # Credit Spreads
    "BAA10Y": MacroSeriesInfo(
        series_id="BAA10Y",
        description="Moody's Baa Corporate Bond Spread over 10Y Treasury",
        transformation="level",
        category="spreads",
    ),
    "AAA10Y": MacroSeriesInfo(
        series_id="AAA10Y",
        description="Moody's Aaa Corporate Bond Spread over 10Y Treasury",
        transformation="level",
        category="spreads",
    ),
    "BAMLH0A0HYM2": MacroSeriesInfo(
        series_id="BAMLH0A0HYM2",
        description="ICE BofA US High Yield Index OAS",
        transformation="level",
        category="spreads",
    ),
    # Financial Conditions
    "NFCI": MacroSeriesInfo(
        series_id="NFCI",
        description="Chicago Fed National Financial Conditions Index",
        transformation="level",
        category="conditions",
    ),
    "STLFSI4": MacroSeriesInfo(
        series_id="STLFSI4",
        description="St. Louis Fed Financial Stress Index",
        transformation="level",
        category="conditions",
    ),
    "VIXCLS": MacroSeriesInfo(
        series_id="VIXCLS",
        description="CBOE Volatility Index (VIX)",
        transformation="level",
        category="conditions",
    ),
    # Labor Market
    "UNRATE": MacroSeriesInfo(
        series_id="UNRATE",
        description="Unemployment Rate",
        transformation="level",
        category="labor",
    ),
    # Housing
    "CSUSHPISA": MacroSeriesInfo(
        series_id="CSUSHPISA",
        description="S&P/Case-Shiller U.S. National Home Price Index",
        transformation="yoy_pct",
        category="housing",
    ),
}

# H.8 Bank Credit Series (System-wide aggregates)
H8_SERIES = {
    # Total Bank Credit
    "TOTBKCR": MacroSeriesInfo(
        series_id="TOTBKCR",
        description="Bank Credit, All Commercial Banks",
        transformation="yoy_pct",
        category="bank_credit",
    ),
    # Loans and Leases
    "TOTLL": MacroSeriesInfo(
        series_id="TOTLL",
        description="Loans and Leases in Bank Credit, All Commercial Banks",
        transformation="yoy_pct",
        category="bank_credit",
    ),
    # Commercial and Industrial
    "BUSLOANS": MacroSeriesInfo(
        series_id="BUSLOANS",
        description="Commercial and Industrial Loans, All Commercial Banks",
        transformation="yoy_pct",
        category="bank_credit",
    ),
    # Real Estate
    "REALLN": MacroSeriesInfo(
        series_id="REALLN",
        description="Real Estate Loans, All Commercial Banks",
        transformation="yoy_pct",
        category="bank_credit",
    ),
    # Consumer
    "CONSUMER": MacroSeriesInfo(
        series_id="CONSUMER",
        description="Consumer Loans, All Commercial Banks",
        transformation="yoy_pct",
        category="bank_credit",
    ),
    # Deposits
    "DPSACBW027SBOG": MacroSeriesInfo(
        series_id="DPSACBW027SBOG",
        description="Deposits, All Commercial Banks",
        transformation="yoy_pct",
        category="bank_credit",
    ),
}


class MacroDataFetcher:
    """
    Fetch and process macroeconomic data for the leading indicator model.

    Implements the paper's variable construction:
    - Output gap = (Real GDP - Potential GDP) / Potential GDP
    - Inflation = YoY % change in CPI
    - Credit spread = Baa yield - 10Y Treasury
    - Real policy rate = Fed Funds - Inflation
    """

    def __init__(self, start_date: str = "2000-01-01"):
        """
        Initialize the macro data fetcher.

        Args:
            start_date: Start date for data (2000 onward for bank-level analysis)
        """
        self.start_date = start_date
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.fetcher = FREDDataFetcher()
        self._data: Optional[pl.DataFrame] = None

    def fetch_macro_data(self) -> pl.DataFrame:
        """
        Fetch all macro series from FRED.

        Returns:
            DataFrame with date and macro variables
        """
        all_series = {**MACRO_SERIES, **H8_SERIES}
        series_ids = list(all_series.keys())

        print(f"Fetching {len(series_ids)} macro series from FRED...")
        self._data = self.fetcher.fetch_multiple_series(
            series_ids,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        print(f"Fetched {self._data.height} observations")
        return self._data

    def compute_derived_variables(self) -> pl.DataFrame:
        """
        Compute derived macro variables matching the paper's methodology.

        Returns:
            DataFrame with computed variables
        """
        if self._data is None:
            self.fetch_macro_data()

        df = self._data.clone()

        # 1. Output Gap: (Real GDP - Potential GDP) / Potential GDP * 100
        if "GDPC1" in df.columns and "GDPPOT" in df.columns:
            df = df.with_columns(
                ((pl.col("GDPC1") - pl.col("GDPPOT")) / pl.col("GDPPOT") * 100)
                .alias("output_gap")
            )

        # 2. GDP Growth (YoY)
        if "GDPC1" in df.columns:
            df = df.with_columns(
                ((pl.col("GDPC1") / pl.col("GDPC1").shift(4) - 1) * 100)
                .alias("gdp_growth_yoy")
            )

        # 3. Inflation (YoY CPI)
        if "CPIAUCSL" in df.columns:
            df = df.with_columns(
                ((pl.col("CPIAUCSL") / pl.col("CPIAUCSL").shift(12) - 1) * 100)
                .alias("inflation_yoy")
            )

        # 4. Real Policy Rate: Fed Funds - Inflation
        if "DFF" in df.columns and "inflation_yoy" in df.columns:
            df = df.with_columns(
                (pl.col("DFF") - pl.col("inflation_yoy")).alias("real_policy_rate")
            )

        # 5. Term Spread: 10Y - 2Y (yield curve slope)
        if "DGS10" in df.columns and "DGS2" in df.columns:
            df = df.with_columns(
                (pl.col("DGS10") - pl.col("DGS2")).alias("term_spread")
            )

        # 6. Bank Credit Growth (YoY for H.8 series)
        for series_id in H8_SERIES:
            if series_id in df.columns:
                df = df.with_columns(
                    ((pl.col(series_id) / pl.col(series_id).shift(52) - 1) * 100)
                    .alias(f"{series_id}_growth_yoy")
                )

        # 7. Housing Price Growth
        if "CSUSHPISA" in df.columns:
            df = df.with_columns(
                ((pl.col("CSUSHPISA") / pl.col("CSUSHPISA").shift(12) - 1) * 100)
                .alias("house_price_growth_yoy")
            )

        return df

    def get_quarterly_macro(self) -> pl.DataFrame:
        """
        Aggregate macro data to quarterly frequency.

        Returns:
            Quarterly macro data
        """
        df = self.compute_derived_variables()

        # Add quarter column
        df = df.with_columns(
            pl.col("date").dt.truncate("1q").alias("quarter")
        )

        # Define aggregation: use last observation for levels, mean for rates
        level_cols = [c for c in df.columns if c not in ["date", "quarter"]]

        # Aggregate by quarter (last value in quarter)
        quarterly = df.group_by("quarter").agg(
            [pl.col(c).drop_nulls().last().alias(c) for c in level_cols]
        ).sort("quarter")

        return quarterly.rename({"quarter": "date"})

    def compute_3yr_changes(self) -> pl.DataFrame:
        """
        Compute 3-year changes in key variables (matching paper's Table 1).

        The paper uses 3-year changes in credit-to-GDP as the key predictor.

        Returns:
            DataFrame with 3-year change variables
        """
        quarterly = self.get_quarterly_macro()

        # 3-year = 12 quarters
        change_vars = ["output_gap", "inflation_yoy", "BAA10Y", "TOTLL", "BUSLOANS"]
        available = [v for v in change_vars if v in quarterly.columns]

        for var in available:
            quarterly = quarterly.with_columns(
                (pl.col(var) - pl.col(var).shift(12)).alias(f"{var}_3yr_change")
            )

        return quarterly


class BankSystemData:
    """
    Aggregate banking system data for comparison to individual banks.

    Used to compute Lending Intensity Score (LIS) as deviation from system average.
    """

    def __init__(self, start_date: str = "2000-01-01"):
        self.start_date = start_date
        self.fetcher = FREDDataFetcher()
        self._data: Optional[pl.DataFrame] = None

    def fetch_system_data(self) -> pl.DataFrame:
        """
        Fetch aggregate banking system data from H.8.

        Returns:
            DataFrame with system-wide bank credit data
        """
        series_ids = list(H8_SERIES.keys())

        print(f"Fetching {len(series_ids)} H.8 series...")
        self._data = self.fetcher.fetch_multiple_series(
            series_ids,
            start_date=self.start_date,
        )

        return self._data

    def get_quarterly_system_growth(self) -> pl.DataFrame:
        """
        Compute quarterly system-wide loan growth rates.

        Returns:
            DataFrame with system growth rates and rolling statistics
        """
        if self._data is None:
            self.fetch_system_data()

        df = self._data.clone()

        # Aggregate to quarterly (end of quarter values)
        df = df.with_columns(
            pl.col("date").dt.truncate("1q").alias("quarter")
        )

        quarterly = df.group_by("quarter").agg(
            [pl.col(c).drop_nulls().last().alias(c) for c in H8_SERIES.keys() if c in df.columns]
        ).sort("quarter")

        # Compute YoY growth rates
        for series_id in H8_SERIES:
            if series_id in quarterly.columns:
                quarterly = quarterly.with_columns([
                    # YoY growth
                    ((pl.col(series_id) / pl.col(series_id).shift(4) - 1) * 100)
                    .alias(f"{series_id}_growth"),
                    # Rolling mean of growth (for LIS denominator)
                    ((pl.col(series_id) / pl.col(series_id).shift(4) - 1) * 100)
                    .rolling_mean(window_size=20)
                    .alias(f"{series_id}_growth_mean"),
                    # Rolling std of growth (for LIS normalization)
                    ((pl.col(series_id) / pl.col(series_id).shift(4) - 1) * 100)
                    .rolling_std(window_size=20)
                    .alias(f"{series_id}_growth_std"),
                ])

        return quarterly.rename({"quarter": "date"})


if __name__ == "__main__":
    # Test macro data
    macro = MacroDataFetcher(start_date="2000-01-01")
    quarterly = macro.get_quarterly_macro()
    print("Quarterly macro data:")
    print(quarterly.tail(10))

    # Test 3-year changes
    changes = macro.compute_3yr_changes()
    print("\n3-year changes:")
    change_cols = [c for c in changes.columns if "3yr" in c]
    print(changes.select(["date"] + change_cols).tail(10))

    # Test system data
    system = BankSystemData(start_date="2000-01-01")
    sys_growth = system.get_quarterly_system_growth()
    print("\nSystem growth rates:")
    growth_cols = [c for c in sys_growth.columns if "growth" in c][:5]
    print(sys_growth.select(["date"] + growth_cols).tail(10))
