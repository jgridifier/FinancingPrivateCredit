"""
Data fetching module for private credit analysis.

Uses FRED API to fetch Z.1 Financial Accounts data for:
- Total private credit to nonfinancial sector
- Credit decomposition by lender type (banks vs nonbanks)
- Macroeconomic indicators for normalization
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import polars as pl
import requests


# FRED API base URL (no API key required for basic access via web scraping approach)
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@dataclass
class SeriesMetadata:
    """Metadata for a FRED series."""

    series_id: str
    description: str
    units: str
    frequency: str
    category: str


# Key Z.1 Financial Accounts series for private credit analysis
# Based on the methodology in Boyarchenko & Elias (2024)
PRIVATE_CREDIT_SERIES = {
    # Total Credit to Private Nonfinancial Sector
    "CRDQUSAPABIS": SeriesMetadata(
        series_id="CRDQUSAPABIS",
        description="Total Credit to Private Non-Financial Sector (BIS)",
        units="Billions USD",
        frequency="Q",
        category="total_credit",
    ),
    # Household sector debt
    "CMDEBT": SeriesMetadata(
        series_id="CMDEBT",
        description="Households and Nonprofit Orgs; Debt Securities and Loans; Liability",
        units="Billions USD",
        frequency="Q",
        category="household_debt",
    ),
    # Nonfinancial business debt - total
    "TBSDODNS": SeriesMetadata(
        series_id="TBSDODNS",
        description="Nonfinancial Business; Debt Securities and Loans; Liability",
        units="Billions USD",
        frequency="Q",
        category="business_debt",
    ),
    # Nonfinancial corporate business debt
    "BCNSDODNS": SeriesMetadata(
        series_id="BCNSDODNS",
        description="Nonfinancial Corporate Business; Debt Securities and Loans; Liability",
        units="Billions USD",
        frequency="Q",
        category="corporate_debt",
    ),
    # Bank credit to private sector (Commercial Bank Assets)
    "TOTLL": SeriesMetadata(
        series_id="TOTLL",
        description="Loans and Leases in Bank Credit, All Commercial Banks",
        units="Billions USD",
        frequency="W",
        category="bank_credit",
    ),
    # Commercial and Industrial Loans
    "BUSLOANS": SeriesMetadata(
        series_id="BUSLOANS",
        description="Commercial and Industrial Loans, All Commercial Banks",
        units="Billions USD",
        frequency="W",
        category="bank_credit",
    ),
    # Consumer Loans at Commercial Banks
    "CONSUMER": SeriesMetadata(
        series_id="CONSUMER",
        description="Consumer Loans at All Commercial Banks",
        units="Billions USD",
        frequency="W",
        category="bank_credit",
    ),
    # Real Estate Loans at Commercial Banks
    "REALLN": SeriesMetadata(
        series_id="REALLN",
        description="Real Estate Loans, All Commercial Banks",
        units="Billions USD",
        frequency="W",
        category="bank_credit",
    ),
    # GDP for normalization
    "GDP": SeriesMetadata(
        series_id="GDP",
        description="Gross Domestic Product",
        units="Billions USD",
        frequency="Q",
        category="macro",
    ),
    # Shadow Bank Proxies - Money Market Funds
    "MMMFFAQ027S": SeriesMetadata(
        series_id="MMMFFAQ027S",
        description="Money Market Funds; Total Financial Assets",
        units="Billions USD",
        frequency="Q",
        category="shadow_bank",
    ),
    # Security Brokers and Dealers
    "BOGZ1FL664090005Q": SeriesMetadata(
        series_id="BOGZ1FL664090005Q",
        description="Security Brokers and Dealers; Total Financial Assets",
        units="Billions USD",
        frequency="Q",
        category="shadow_bank",
    ),
    # Note: ABS Issuer series discontinued; using remaining shadow bank proxies
    # Finance Companies
    "BOGZ1FL614090005Q": SeriesMetadata(
        series_id="BOGZ1FL614090005Q",
        description="Finance Companies; Total Financial Assets",
        units="Billions USD",
        frequency="Q",
        category="shadow_bank",
    ),
    # Life Insurance Companies (credit intermediation)
    "BOGZ1FL544090005Q": SeriesMetadata(
        series_id="BOGZ1FL544090005Q",
        description="Life Insurance Companies; Total Financial Assets",
        units="Billions USD",
        frequency="Q",
        category="insurance",
    ),
    # Private Pension Funds
    "BOGZ1FL574090005Q": SeriesMetadata(
        series_id="BOGZ1FL574090005Q",
        description="Private Pension Funds; Total Financial Assets",
        units="Billions USD",
        frequency="Q",
        category="pension",
    ),
}


class FREDDataFetcher:
    """
    Fetch data from FRED (Federal Reserve Economic Data).

    Uses direct CSV download (no API key required) for simplicity.
    For heavy usage, consider using fredapi with an API key.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FRED data fetcher.

        Args:
            api_key: Optional FRED API key. If not provided, uses direct CSV download.
        """
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self._cache: dict[str, pl.DataFrame] = {}

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch a single series from FRED.

        Args:
            series_id: The FRED series ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Polars DataFrame with columns [date, {series_id}]
        """
        cache_key = f"{series_id}_{start_date}_{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build URL for direct CSV download
        params = {"id": series_id}
        if start_date:
            params["cosd"] = start_date
        if end_date:
            params["coed"] = end_date

        url = FRED_BASE_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse CSV with Polars
            df = pl.read_csv(
                response.text.encode(),
                try_parse_dates=True,
            )

            # Rename columns for consistency (FRED uses 'observation_date')
            if "observation_date" in df.columns:
                df = df.rename({"observation_date": "date"})
            elif "DATE" in df.columns:
                df = df.rename({"DATE": "date"})

            # Rename value column
            if series_id in df.columns:
                df = df.rename({series_id: "value"})

            # Ensure value column is Float64 (Polars auto-parses, handles "." as null)
            if "value" in df.columns:
                # If string type, handle "." as missing and convert
                if df["value"].dtype == pl.String or df["value"].dtype == pl.Utf8:
                    df = df.with_columns(
                        pl.when(pl.col("value") == ".")
                        .then(None)
                        .otherwise(pl.col("value"))
                        .cast(pl.Float64)
                        .alias("value")
                    )
                # If already numeric, just ensure Float64
                elif df["value"].dtype != pl.Float64:
                    df = df.with_columns(pl.col("value").cast(pl.Float64))

            # Add series_id column
            df = df.with_columns(pl.lit(series_id).alias("series_id"))

            self._cache[cache_key] = df
            return df

        except Exception as e:
            print(f"Warning: Failed to fetch {series_id}: {e}")
            return pl.DataFrame({
                "date": [],
                "value": [],
                "series_id": [],
            })

    def fetch_multiple_series(
        self,
        series_ids: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch multiple series and join them into a single DataFrame.

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Polars DataFrame with date column and one column per series
        """
        dfs = []
        for series_id in series_ids:
            df = self.fetch_series(series_id, start_date, end_date)
            if df.height > 0:
                df = df.select([
                    "date",
                    pl.col("value").alias(series_id)
                ])
                dfs.append(df)

        if not dfs:
            return pl.DataFrame({"date": []})

        # Join all series on date using coalesce for outer joins
        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, on="date", how="outer_coalesce")

        return result.sort("date")


class PrivateCreditData:
    """
    Comprehensive private credit dataset following Boyarchenko & Elias (2024) methodology.

    Fetches and processes data to decompose private credit by:
    - Borrower type (households vs. nonfinancial businesses)
    - Lender type (banks vs. nonbanks/shadow banks)
    """

    def __init__(self, start_date: str = "1952-01-01", end_date: Optional[str] = None):
        """
        Initialize the private credit dataset.

        Args:
            start_date: Start date for data (default matches Z.1 historical start)
            end_date: End date for data (default is latest available)
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.fetcher = FREDDataFetcher()
        self._data: Optional[pl.DataFrame] = None

    def fetch_all(self) -> pl.DataFrame:
        """
        Fetch all private credit series from FRED.

        Returns:
            Polars DataFrame with all series joined by date
        """
        series_ids = list(PRIVATE_CREDIT_SERIES.keys())

        print(f"Fetching {len(series_ids)} series from FRED...")
        self._data = self.fetcher.fetch_multiple_series(
            series_ids,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        print(f"Fetched data from {self._data['date'].min()} to {self._data['date'].max()}")

        return self._data

    def get_quarterly_data(self) -> pl.DataFrame:
        """
        Get data aggregated to quarterly frequency.

        Aggregates weekly bank credit data (TOTLL, etc.) to quarterly frequency
        using end-of-quarter values, aligning with the Z.1 quarterly releases.

        Returns:
            DataFrame with quarterly data
        """
        if self._data is None:
            self.fetch_all()

        df = self._data.clone()

        # Identify series by frequency from metadata
        weekly_series = [
            sid for sid, meta in PRIVATE_CREDIT_SERIES.items()
            if meta.frequency == "W" and sid in df.columns
        ]
        quarterly_series = [
            sid for sid, meta in PRIVATE_CREDIT_SERIES.items()
            if meta.frequency == "Q" and sid in df.columns
        ]

        # Add quarter column
        df = df.with_columns(
            pl.col("date").dt.truncate("1q").alias("quarter")
        )

        # Aggregate weekly series to quarterly (using last value in quarter)
        if weekly_series:
            weekly_df = df.select(["quarter"] + weekly_series).group_by("quarter").agg(
                [pl.col(s).drop_nulls().last().alias(s) for s in weekly_series]
            )
        else:
            weekly_df = None

        # Get quarterly series (already at correct frequency)
        if quarterly_series:
            quarterly_df = df.filter(
                df["date"].is_in(df["quarter"])  # First day of quarter
            ).select(["quarter"] + quarterly_series)
        else:
            quarterly_df = None

        # Combine weekly and quarterly
        if weekly_df is not None and quarterly_df is not None:
            result = weekly_df.join(quarterly_df, on="quarter", how="outer_coalesce")
        elif weekly_df is not None:
            result = weekly_df
        elif quarterly_df is not None:
            result = quarterly_df
        else:
            return pl.DataFrame({"quarter": []})

        return result.sort("quarter").rename({"quarter": "date"})

    def compute_credit_decomposition(self) -> pl.DataFrame:
        """
        Compute the bank vs nonbank credit decomposition.

        Following Boyarchenko & Elias (2024), we decompose total private credit
        into credit provided by:
        1. Banks (commercial banks and credit unions)
        2. Nonbanks (shadow banks, insurance, pensions, etc.)

        Returns:
            DataFrame with bank_credit, nonbank_credit, and total_credit columns
        """
        if self._data is None:
            self.fetch_all()

        # Use quarterly aggregated data for proper frequency alignment
        df = self.get_quarterly_data()

        # Bank credit = sum of commercial bank lending categories
        bank_credit_cols = ["TOTLL"]  # Total loans and leases

        # Shadow bank credit approximation from Z.1 components
        shadow_bank_cols = [
            "MMMFFAQ027S",       # Money Market Funds
            "BOGZ1FL664090005Q", # Security Brokers/Dealers
            "BOGZ1FL614090005Q", # Finance Companies
        ]

        # Insurance and pension credit intermediation
        insurance_pension_cols = [
            "BOGZ1FL544090005Q",  # Life Insurance
            "BOGZ1FL574090005Q",  # Private Pensions
        ]

        # Compute aggregates (handle missing columns gracefully)
        available_bank_cols = [c for c in bank_credit_cols if c in df.columns]
        available_shadow_cols = [c for c in shadow_bank_cols if c in df.columns]
        available_ins_cols = [c for c in insurance_pension_cols if c in df.columns]

        # Start with original data and add computed columns
        result = df.clone()

        if available_bank_cols:
            result = result.with_columns(
                pl.sum_horizontal([pl.col(c).fill_null(0) for c in available_bank_cols])
                .alias("bank_credit")
            )

        if available_shadow_cols:
            result = result.with_columns(
                pl.sum_horizontal([pl.col(c).fill_null(0) for c in available_shadow_cols])
                .alias("shadow_bank_credit")
            )

        if available_ins_cols:
            result = result.with_columns(
                pl.sum_horizontal([pl.col(c).fill_null(0) for c in available_ins_cols])
                .alias("insurance_pension_credit")
            )

        # Total nonbank = shadow banks + insurance/pensions
        if "shadow_bank_credit" in result.columns and "insurance_pension_credit" in result.columns:
            result = result.with_columns(
                (pl.col("shadow_bank_credit") + pl.col("insurance_pension_credit"))
                .alias("nonbank_credit")
            )

        # Add total private credit from BIS series if available
        if "CRDQUSAPABIS" in df.columns:
            result = result.with_columns(
                df.select("CRDQUSAPABIS").to_series().alias("total_private_credit_bis")
            )

        # Add GDP for normalization
        if "GDP" in df.columns:
            result = result.with_columns(
                df.select("GDP").to_series().alias("gdp")
            )

        return result

    def compute_credit_to_gdp(self) -> pl.DataFrame:
        """
        Compute credit-to-GDP ratios for all credit categories.

        Returns:
            DataFrame with credit/GDP ratios
        """
        decomp = self.compute_credit_decomposition()

        if "gdp" not in decomp.columns:
            raise ValueError("GDP data not available for normalization")

        # Compute ratios (multiply by 100 for percentage)
        credit_cols = [c for c in decomp.columns if c not in ["date", "gdp"]]

        for col in credit_cols:
            decomp = decomp.with_columns(
                (pl.col(col) / pl.col("gdp") * 100).alias(f"{col}_to_gdp")
            )

        return decomp

    def compute_credit_growth(self, periods: int = 4) -> pl.DataFrame:
        """
        Compute year-over-year credit growth rates.

        Args:
            periods: Number of periods for growth calculation (4 for quarterly YoY)

        Returns:
            DataFrame with growth rates
        """
        decomp = self.compute_credit_decomposition()

        credit_cols = [c for c in decomp.columns if c not in ["date", "gdp"]]

        for col in credit_cols:
            decomp = decomp.with_columns(
                ((pl.col(col) / pl.col(col).shift(periods) - 1) * 100)
                .alias(f"{col}_growth_yoy")
            )

        return decomp

    def get_lender_shares(self) -> pl.DataFrame:
        """
        Compute the share of credit provided by each lender type.

        This is a key metric in Boyarchenko & Elias (2024) - the composition
        of lenders financing a credit expansion determines subsequent outcomes.

        Returns:
            DataFrame with bank_share and nonbank_share columns
        """
        decomp = self.compute_credit_decomposition()

        if "bank_credit" not in decomp.columns or "nonbank_credit" not in decomp.columns:
            raise ValueError("Bank/nonbank decomposition not available")

        total = pl.col("bank_credit") + pl.col("nonbank_credit")

        return decomp.with_columns([
            (pl.col("bank_credit") / total * 100).alias("bank_share"),
            (pl.col("nonbank_credit") / total * 100).alias("nonbank_share"),
        ])


if __name__ == "__main__":
    # Quick test
    data = PrivateCreditData(start_date="1990-01-01")
    df = data.fetch_all()
    print(df)
