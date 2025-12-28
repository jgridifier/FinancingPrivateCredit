"""
FFIEC Call Report Data Fetcher for Funding Stability Metrics

Fetches data from multiple Call Report schedules:
- Schedule RC-O: Other Data for Deposit Insurance (uninsured deposits)
- Schedule RC-M: Memoranda (FHLB advances)
- Schedule RC-E: Deposit Liabilities (brokered deposits)
- Schedule RC-B: Securities (for AOCI calculation)
- Schedule RC-R: Regulatory Capital (for equity denominators)

Data Source: FFIEC Central Data Repository (CDR)
https://cdr.ffiec.gov/public/

MDRM Codes Reference:
- RCON5597: Estimated uninsured deposits (RC-O, Memo Item 2)
- RCON2200: Total deposits (domestic offices)
- RCFD2950: FHLB advances (RC-M, Item 5.a)
- RCON2365: Total liabilities
- RCONHK04: Total brokered deposits (RC-E, Memo 1.b)
- RCFD8434: HTM securities amortized cost
- RCFD8435: HTM securities fair value
- RCFD1773: AFS securities fair value
- RCFD8439: AFS securities amortized cost
- RCFDA222: Tangible common equity
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen

import polars as pl


@dataclass
class FundingMetrics:
    """Funding stability metrics from Call Report data."""

    rssd_id: str
    report_date: datetime

    # Schedule RC-O: Deposit Insurance Data
    uninsured_deposits: float = 0.0  # RCON5597
    total_domestic_deposits: float = 0.0  # RCON2200

    # Schedule RC-M: FHLB Advances
    fhlb_advances: float = 0.0  # RCFD2950
    total_liabilities: float = 0.0  # RCON2365

    # Schedule RC-E: Brokered Deposits
    brokered_deposits: float = 0.0  # RCONHK04

    # Schedule RC-B: Securities (for AOCI)
    htm_amortized_cost: float = 0.0  # RCFD8434
    htm_fair_value: float = 0.0  # RCFD8435
    afs_amortized_cost: float = 0.0  # RCFD8439
    afs_fair_value: float = 0.0  # RCFD1773

    # Schedule RC-R: Regulatory Capital
    tangible_common_equity: float = 0.0  # RCFDA222
    tier1_capital: float = 0.0  # RCFD8274

    # Wholesale funding components
    fed_funds_purchased: float = 0.0  # RCON2800
    repo_liabilities: float = 0.0  # RCFD2800 + others
    large_time_deposits: float = 0.0  # RCON2604 (>$250k)
    foreign_deposits: float = 0.0  # RCFN2200

    # Computed ratios
    @property
    def uninsured_deposit_ratio(self) -> float:
        """Uninsured Deposits / Total Domestic Deposits."""
        if self.total_domestic_deposits <= 0:
            return 0.0
        return self.uninsured_deposits / self.total_domestic_deposits

    @property
    def fhlb_advance_ratio(self) -> float:
        """FHLB Advances / Total Liabilities."""
        if self.total_liabilities <= 0:
            return 0.0
        return self.fhlb_advances / self.total_liabilities

    @property
    def brokered_deposit_ratio(self) -> float:
        """Brokered Deposits / Total Deposits."""
        if self.total_domestic_deposits <= 0:
            return 0.0
        return self.brokered_deposits / self.total_domestic_deposits

    @property
    def aoci_impact_ratio(self) -> float:
        """(HTM + AFS Unrealized Losses) / Tangible Common Equity."""
        htm_loss = max(0, self.htm_amortized_cost - self.htm_fair_value)
        afs_loss = max(0, self.afs_amortized_cost - self.afs_fair_value)
        total_loss = htm_loss + afs_loss

        if self.tangible_common_equity <= 0:
            return 0.0
        return total_loss / self.tangible_common_equity

    @property
    def wholesale_funding_ratio(self) -> float:
        """Non-core funding / Total Liabilities."""
        non_core = (
            self.fed_funds_purchased +
            self.repo_liabilities +
            self.large_time_deposits +
            self.foreign_deposits +
            self.brokered_deposits +
            self.fhlb_advances
        )
        if self.total_liabilities <= 0:
            return 0.0
        return non_core / self.total_liabilities

    @property
    def deposit_funding_ratio(self) -> float:
        """Core Deposits / Total Liabilities."""
        # Core deposits = Total deposits - Brokered - Large time - Foreign
        core_deposits = max(
            0,
            self.total_domestic_deposits -
            self.brokered_deposits -
            self.large_time_deposits
        )
        if self.total_liabilities <= 0:
            return 0.0
        return core_deposits / self.total_liabilities

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with all ratios."""
        return {
            "rssd_id": self.rssd_id,
            "report_date": self.report_date,
            # Raw values
            "uninsured_deposits": self.uninsured_deposits,
            "total_domestic_deposits": self.total_domestic_deposits,
            "fhlb_advances": self.fhlb_advances,
            "total_liabilities": self.total_liabilities,
            "brokered_deposits": self.brokered_deposits,
            "htm_amortized_cost": self.htm_amortized_cost,
            "htm_fair_value": self.htm_fair_value,
            "afs_amortized_cost": self.afs_amortized_cost,
            "afs_fair_value": self.afs_fair_value,
            "tangible_common_equity": self.tangible_common_equity,
            # Computed ratios
            "uninsured_deposit_ratio": self.uninsured_deposit_ratio,
            "fhlb_advance_ratio": self.fhlb_advance_ratio,
            "brokered_deposit_ratio": self.brokered_deposit_ratio,
            "aoci_impact_ratio": self.aoci_impact_ratio,
            "wholesale_funding_ratio": self.wholesale_funding_ratio,
            "deposit_funding_ratio": self.deposit_funding_ratio,
        }


# MDRM Code Mappings for Call Report Schedules
MDRM_CODES = {
    # Schedule RC-O: Deposit Insurance Data
    "uninsured_deposits": ["RCON5597", "RCFD5597"],  # Memo Item 2
    "total_domestic_deposits": ["RCON2200"],

    # Schedule RC-M: Memoranda
    "fhlb_advances": ["RCFD2950", "RCON2950"],  # Item 5.a

    # Schedule RC: Balance Sheet
    "total_liabilities": ["RCFD2948", "RCON2948"],
    "total_assets": ["RCFD2170", "RCON2170"],

    # Schedule RC-E: Deposit Liabilities
    "brokered_deposits": ["RCONHK04", "RCFDHK04"],  # Memo 1.b
    "large_time_deposits": ["RCON2604", "RCFD2604"],  # >$250k

    # Schedule RC-B: Securities
    "htm_amortized_cost": ["RCFD8434", "RCON8434"],
    "htm_fair_value": ["RCFD8435", "RCON8435"],
    "afs_fair_value": ["RCFD1773", "RCON1773"],
    "afs_amortized_cost": ["RCFD8439", "RCON8439"],

    # Schedule RC-R: Regulatory Capital
    "tangible_common_equity": ["RCFDA222", "RCONA222"],
    "tier1_capital": ["RCFD8274", "RCON8274"],
    "cet1_capital": ["RCFAP859", "RCONP859"],

    # Wholesale funding components
    "fed_funds_purchased": ["RCONB993", "RCFDB993"],
    "repo_liabilities": ["RCFD2800", "RCON2800"],
    "foreign_deposits": ["RCFN2200"],

    # Additional useful metrics
    "net_income": ["RIAD4340"],
    "interest_income": ["RIAD4107"],
    "interest_expense": ["RIAD4073"],
    "provision_loan_losses": ["RIAD4230"],
}


class FFIECBulkDataFetcher:
    """
    Fetcher for FFIEC bulk Call Report data.

    The FFIEC provides quarterly bulk downloads of Call Report data
    in a specific format. This class handles parsing that data.
    """

    # Base URL for FFIEC bulk data
    BASE_URL = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"

    # CDR schedule URLs
    SCHEDULE_URLS = {
        "RC-O": "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx?reportType=Call&format=TXT&item=RC-O",
        "RC-M": "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx?reportType=Call&format=TXT&item=RC-M",
        "RC-E": "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx?reportType=Call&format=TXT&item=RC-E",
        "RC-B": "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx?reportType=Call&format=TXT&item=RC-B",
        "RC-R": "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx?reportType=Call&format=TXT&item=RC-R",
        "RC": "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx?reportType=Call&format=TXT&item=RC",
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the fetcher.

        Args:
            cache_dir: Directory to cache downloaded files
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "ffiec_data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data_cache: dict[str, pl.DataFrame] = {}

    def fetch_schedule(
        self,
        schedule: str,
        report_date: str,
        rssd_ids: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """
        Fetch a specific Call Report schedule.

        Args:
            schedule: Schedule name (e.g., "RC-O", "RC-M")
            report_date: Report date in YYYYMMDD format
            rssd_ids: Optional list of RSSD IDs to filter

        Returns:
            DataFrame with schedule data
        """
        cache_key = f"{schedule}_{report_date}"

        if cache_key in self._data_cache:
            df = self._data_cache[cache_key]
        else:
            # In production, this would download from FFIEC
            # For now, return empty DataFrame as placeholder
            print(f"  Note: FFIEC bulk data fetch for {schedule} not implemented")
            print(f"  Would fetch from: {self.SCHEDULE_URLS.get(schedule, 'unknown')}")
            df = pl.DataFrame()
            self._data_cache[cache_key] = df

        if rssd_ids and df.height > 0:
            df = df.filter(pl.col("rssd_id").is_in(rssd_ids))

        return df

    def fetch_funding_metrics(
        self,
        rssd_id: str,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch all funding-related metrics for a bank.

        Aggregates data from multiple schedules (RC-O, RC-M, RC-E, RC-B, RC-R).

        Args:
            rssd_id: Bank's RSSD ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (optional)

        Returns:
            DataFrame with time series of funding metrics
        """
        # This would aggregate data from multiple schedules
        # For now, return placeholder
        return pl.DataFrame()


class UBPRDataFetcher:
    """
    Fetcher for Uniform Bank Performance Report (UBPR) data.

    The UBPR provides pre-calculated ratios that regulators use,
    including the "Net Non-Core Funding Dependence" metric.
    """

    # Key UBPR concepts
    UBPR_CONCEPTS = {
        "net_noncore_funding_dependence": "UBPRE003",
        "volatile_liability_dependence": "UBPRE006",
        "net_loans_to_core_deposits": "UBPRE001",
        "core_deposits_to_total_assets": "UBPRE002",
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "ubpr_data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_ubpr_ratio(
        self,
        rssd_id: str,
        concept: str,
        start_date: str = "2015-01-01",
    ) -> pl.DataFrame:
        """
        Fetch a specific UBPR ratio time series.

        Args:
            rssd_id: Bank's RSSD ID
            concept: UBPR concept code (e.g., "UBPRE003")
            start_date: Start date

        Returns:
            DataFrame with date and ratio values
        """
        # Placeholder - would fetch from FFIEC UBPR system
        return pl.DataFrame()

    def fetch_noncore_funding_dependence(
        self,
        rssd_id: str,
        start_date: str = "2015-01-01",
    ) -> pl.DataFrame:
        """
        Fetch the Net Non-Core Funding Dependence ratio.

        This is the regulatory standard metric for funding stability.

        Args:
            rssd_id: Bank's RSSD ID
            start_date: Start date

        Returns:
            DataFrame with noncore funding dependence time series
        """
        return self.fetch_ubpr_ratio(
            rssd_id,
            self.UBPR_CONCEPTS["net_noncore_funding_dependence"],
            start_date,
        )


class SECFundingDataExtractor:
    """
    Extract funding metrics from SEC EDGAR filings.

    Fallback data source when Call Report data is unavailable.
    Uses XBRL concepts from 10-K/10-Q filings.
    """

    # XBRL concepts for funding data
    XBRL_CONCEPTS = {
        "deposits": [
            ("us-gaap", "Deposits"),
            ("us-gaap", "DepositsDomestic"),
        ],
        "fhlb_advances": [
            ("us-gaap", "FederalHomeLoanBankAdvances"),
            ("us-gaap", "FederalHomeLoanBankAdvancesLongTerm"),
        ],
        "fed_funds_purchased": [
            ("us-gaap", "FederalFundsPurchased"),
        ],
        "securities_sold_repo": [
            ("us-gaap", "SecuritiesSoldUnderAgreementsToRepurchase"),
        ],
        "total_liabilities": [
            ("us-gaap", "Liabilities"),
        ],
        "stockholders_equity": [
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ],
    }

    def __init__(self):
        from ...bank_data import SECEdgarFetcher
        self.sec_fetcher = SECEdgarFetcher()

    def extract_funding_data(self, cik: str, ticker: str) -> pl.DataFrame:
        """
        Extract funding-related data from SEC filings.

        Args:
            cik: SEC Central Index Key
            ticker: Bank ticker symbol

        Returns:
            DataFrame with funding metrics
        """
        facts = self.sec_fetcher.get_company_facts(cik)

        if not facts:
            return pl.DataFrame()

        dfs = {}

        for metric_name, concepts in self.XBRL_CONCEPTS.items():
            best_df = None
            best_date = None

            for taxonomy, concept in concepts:
                df = self.sec_fetcher.extract_metric(facts, taxonomy, concept)
                if df.height > 0:
                    latest_date = df.select(pl.col("date").max()).item()
                    if best_date is None or latest_date > best_date:
                        best_df = df.rename({"value": metric_name})
                        best_date = latest_date

            if best_df is not None:
                dfs[metric_name] = best_df

        if not dfs:
            return pl.DataFrame()

        # Merge all metrics
        result = None
        for name, df in dfs.items():
            if result is None:
                result = df
            else:
                result = result.join(df, on="date", how="outer_coalesce")

        if result is not None:
            result = result.with_columns(pl.lit(ticker).alias("ticker"))

        return result if result is not None else pl.DataFrame()


def calculate_funding_metrics_from_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate funding stability ratios from a DataFrame with raw values.

    Args:
        df: DataFrame with raw funding values

    Returns:
        DataFrame with computed ratios added
    """
    result = df.clone()

    # Uninsured deposit ratio
    if "uninsured_deposits" in result.columns and "total_domestic_deposits" in result.columns:
        result = result.with_columns(
            (pl.col("uninsured_deposits") / pl.col("total_domestic_deposits").clip(lower_bound=1))
            .alias("uninsured_deposit_ratio")
        )

    # FHLB advance ratio
    if "fhlb_advances" in result.columns and "total_liabilities" in result.columns:
        result = result.with_columns(
            (pl.col("fhlb_advances") / pl.col("total_liabilities").clip(lower_bound=1))
            .alias("fhlb_advance_ratio")
        )

    # Brokered deposit ratio
    if "brokered_deposits" in result.columns and "total_domestic_deposits" in result.columns:
        result = result.with_columns(
            (pl.col("brokered_deposits") / pl.col("total_domestic_deposits").clip(lower_bound=1))
            .alias("brokered_deposit_ratio")
        )

    # AOCI impact ratio
    htm_loss_cols = ["htm_amortized_cost", "htm_fair_value"]
    afs_loss_cols = ["afs_amortized_cost", "afs_fair_value"]
    if all(c in result.columns for c in htm_loss_cols + afs_loss_cols + ["tangible_common_equity"]):
        result = result.with_columns([
            (
                (pl.col("htm_amortized_cost") - pl.col("htm_fair_value")).clip(lower_bound=0) +
                (pl.col("afs_amortized_cost") - pl.col("afs_fair_value")).clip(lower_bound=0)
            ).alias("unrealized_securities_loss"),
        ])
        result = result.with_columns(
            (pl.col("unrealized_securities_loss") / pl.col("tangible_common_equity").clip(lower_bound=1))
            .alias("aoci_impact_ratio")
        )

    return result
