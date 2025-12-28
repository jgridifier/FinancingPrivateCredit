"""
Bank-level data module for credit boom leading indicator model.

Collects bank-specific data from:
1. SEC EDGAR 10-K/10-Q filings (via SEC API)
2. FRED bank-specific series where available
3. Call Report data (FFIEC)

Target banks: Major US bank holding companies with consistent data since 2000.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl
import requests

from .data import FREDDataFetcher


@dataclass
class BankInfo:
    """Bank holding company information."""

    ticker: str
    name: str
    cik: str  # SEC Central Index Key
    rssd_id: str  # Federal Reserve RSSD ID for Call Reports
    tier: int  # 1=G-SIB, 2=Large, 3=Regional


# Target banks for analysis (major US bank holding companies)
TARGET_BANKS = {
    "JPM": BankInfo(
        ticker="JPM",
        name="JPMorgan Chase & Co.",
        cik="0000019617",
        rssd_id="1039502",
        tier=1,
    ),
    "BAC": BankInfo(
        ticker="BAC",
        name="Bank of America Corporation",
        cik="0000070858",
        rssd_id="1073757",
        tier=1,
    ),
    "C": BankInfo(
        ticker="C",
        name="Citigroup Inc.",
        cik="0000831001",
        rssd_id="1951350",
        tier=1,
    ),
    "WFC": BankInfo(
        ticker="WFC",
        name="Wells Fargo & Company",
        cik="0000072971",
        rssd_id="1120754",
        tier=1,
    ),
    "GS": BankInfo(
        ticker="GS",
        name="The Goldman Sachs Group, Inc.",
        cik="0000886982",
        rssd_id="2380443",
        tier=2,
    ),
    "MS": BankInfo(
        ticker="MS",
        name="Morgan Stanley",
        cik="0000895421",
        rssd_id="2162966",
        tier=2,
    ),
    "USB": BankInfo(
        ticker="USB",
        name="U.S. Bancorp",
        cik="0000036104",
        rssd_id="1119794",
        tier=3,
    ),
    "PNC": BankInfo(
        ticker="PNC",
        name="The PNC Financial Services Group, Inc.",
        cik="0000713676",
        rssd_id="1069778",
        tier=3,
    ),
    "TFC": BankInfo(
        ticker="TFC",
        name="Truist Financial Corporation",
        cik="0000092230",
        rssd_id="3242838",
        tier=3,
    ),
    "COF": BankInfo(
        ticker="COF",
        name="Capital One Financial Corporation",
        cik="0000927628",
        rssd_id="2277860",
        tier=3,
    ),
    # --- Expanded Bank List for Macro Sensitivity Analysis ---
    "SCHW": BankInfo(
        ticker="SCHW",
        name="The Charles Schwab Corporation",
        cik="0000316709",
        rssd_id="4846951",  # Charles Schwab Bank, SSB
        tier=3,
    ),
    "BK": BankInfo(
        ticker="BK",
        name="The Bank of New York Mellon Corporation",
        cik="0001390777",
        rssd_id="3587146",
        tier=2,
    ),
    "STT": BankInfo(
        ticker="STT",
        name="State Street Corporation",
        cik="0000093751",
        rssd_id="1111435",
        tier=2,
    ),
    "NTRS": BankInfo(
        ticker="NTRS",
        name="Northern Trust Corporation",
        cik="0000073124",
        rssd_id="1199611",
        tier=3,
    ),
    "RJF": BankInfo(
        ticker="RJF",
        name="Raymond James Financial, Inc.",
        cik="0000720005",
        rssd_id="3793882",  # Raymond James Bank
        tier=3,
    ),
    # Note: TD Bank (TD) - Canadian parent; US subsidiary TD Bank, N.A. doesn't file
    # separate SEC reports. Data availability may be limited for macro sensitivity.
    # Note: Barclays (BCS) - UK parent; US subsidiary files limited SEC reports.
    # ADR filings may not contain full NIM/loan data needed for analysis.
}


@dataclass
class BankQuarterlyMetrics:
    """Quarterly metrics for a bank."""

    ticker: str
    date: datetime
    # Loan Portfolio
    total_loans: Optional[float] = None
    ci_loans: Optional[float] = None  # Commercial & Industrial
    cre_loans: Optional[float] = None  # Commercial Real Estate
    consumer_loans: Optional[float] = None
    residential_loans: Optional[float] = None
    # Credit Quality
    allowance: Optional[float] = None  # Allowance for Credit Losses
    provisions: Optional[float] = None  # Provision for Credit Losses
    net_charge_offs: Optional[float] = None
    npl: Optional[float] = None  # Non-performing loans
    # Balance Sheet
    total_assets: Optional[float] = None
    total_deposits: Optional[float] = None
    tier1_capital: Optional[float] = None
    # Profitability
    net_income: Optional[float] = None
    net_interest_income: Optional[float] = None


class SECEdgarFetcher:
    """
    Fetch financial data from SEC EDGAR.

    Uses the SEC's EDGAR API to retrieve company filings and extract
    financial statement data from 10-K and 10-Q filings.
    """

    BASE_URL = "https://data.sec.gov"
    HEADERS = {
        "User-Agent": "FinancingPrivateCredit Research (contact@example.com)",
        "Accept-Encoding": "gzip, deflate",
    }

    def __init__(self):
        self._cache: dict[str, dict] = {}

    def get_company_facts(self, cik: str) -> dict:
        """
        Fetch company facts from SEC EDGAR.

        Args:
            cik: SEC Central Index Key (with leading zeros)

        Returns:
            Dictionary with company facts including all reported values
        """
        if cik in self._cache:
            return self._cache[cik]

        # Remove leading zeros for API call, then pad back
        cik_num = cik.lstrip("0")
        cik_padded = cik_num.zfill(10)

        url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik_padded}.json"

        try:
            response = requests.get(url, headers=self.HEADERS, timeout=30)
            response.raise_for_status()
            data = response.json()
            self._cache[cik] = data
            return data
        except Exception as e:
            print(f"Warning: Failed to fetch SEC data for CIK {cik}: {e}")
            return {}

    def extract_metric(
        self,
        facts: dict,
        taxonomy: str,
        concept: str,
        unit: str = "USD",
    ) -> pl.DataFrame:
        """
        Extract a specific metric from company facts.

        Args:
            facts: Company facts dictionary from SEC API
            taxonomy: XBRL taxonomy (e.g., 'us-gaap')
            concept: XBRL concept name (e.g., 'Loans')
            unit: Unit of measurement

        Returns:
            DataFrame with date and value columns
        """
        try:
            values = facts["facts"][taxonomy][concept]["units"][unit]

            records = []
            for v in values:
                # Only use quarterly (10-Q) and annual (10-K) filings
                if v.get("form") in ["10-Q", "10-K"]:
                    records.append({
                        "date": v.get("end"),
                        "value": v.get("val"),
                        "form": v.get("form"),
                        "filed": v.get("filed"),
                    })

            if not records:
                return pl.DataFrame({"date": [], "value": []})

            df = pl.DataFrame(records)
            df = df.with_columns(
                pl.col("date").str.to_date("%Y-%m-%d")
            )

            # Deduplicate by taking the most recent filing for each period
            df = df.sort(["date", "filed"], descending=[False, True])
            df = df.unique(subset=["date"], keep="first")

            return df.select(["date", "value"]).sort("date")

        except (KeyError, TypeError):
            return pl.DataFrame({"date": [], "value": []})


class BankDataCollector:
    """
    Collect bank-level data for the leading indicator model.

    Combines data from:
    - SEC EDGAR (financial statements)
    - FRED (where available)
    - Manual data input capability
    """

    # XBRL concept mappings for key metrics
    # IMPORTANT: Order matters - most common/preferred concepts listed first
    #
    # Bank-specific notes:
    # - JPM, BAC, C, MS, USB, PNC, TFC: Use FinancingReceivable* concepts
    # - COF: Uses FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss
    # - GS: Uses NotesReceivableGross/Net for loans
    # - WFC: SEC EDGAR data incomplete after Q2 2022 (provisions available, loans/allowance stop)
    #
    XBRL_CONCEPTS = {
        "total_loans": [
            # CECL-era (2020+) - most banks use these now
            ("us-gaap", "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss"),
            ("us-gaap", "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss"),
            # Investment banks (GS, some MS) use Notes Receivable
            ("us-gaap", "NotesReceivableGross"),
            ("us-gaap", "NotesReceivableNet"),
            # Gross carrying amount (alternative)
            ("us-gaap", "LoansAndLeasesReceivableGrossCarryingAmount"),
            # Pre-CECL fallbacks
            ("us-gaap", "LoansAndLeasesReceivableNetReportedAmount"),
            ("us-gaap", "LoansAndLeasesReceivableNetOfDeferredIncome"),
            ("us-gaap", "LoansReceivableNet"),
        ],
        "allowance": [
            # CECL-era - excluding accrued interest (most common for large banks)
            ("us-gaap", "FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest"),
            # CECL-era - standard
            ("us-gaap", "FinancingReceivableAllowanceForCreditLosses"),
            # Collectively evaluated (some banks report this way)
            ("us-gaap", "FinancingReceivableAllowanceForCreditLossesCollectivelyEvaluatedForImpairment"),
            # Pre-CECL
            ("us-gaap", "LoansAndLeasesReceivableAllowance"),
            ("us-gaap", "AllowanceForLoanAndLeaseLosses"),
        ],
        "provisions": [
            # CECL-era - primary provision concept
            ("us-gaap", "ProvisionForCreditLosses"),
            # Used by C, COF - check this before ProvisionForLoanLeaseAndOtherLosses
            # because COF has older data in the latter concept
            ("us-gaap", "ProvisionForLoanLossesExpensed"),
            # Alternative provision concepts (used by JPM, GS, USB, TFC, WFC)
            ("us-gaap", "ProvisionForLoanLeaseAndOtherLosses"),
            # Net provision (includes recoveries)
            ("us-gaap", "AllowanceForLoanAndLeaseLossesProvisionForLossNet"),
            # Credit loss expense concept
            ("us-gaap", "CreditLossExpenseReversal"),
        ],
        "npl": [
            # CECL-era nonaccrual
            ("us-gaap", "FinancingReceivableRecordedInvestmentNonaccrualStatus"),
            ("us-gaap", "FinancingReceivableNonaccrualStatus"),
            # Past due concepts
            ("us-gaap", "FinancingReceivable90DaysOrMorePastDue"),
            ("us-gaap", "FinancingReceivable30To89DaysPastDue"),
            ("us-gaap", "FinancingReceivable30To59DaysPastDue"),
            # Pre-CECL
            ("us-gaap", "LoansAndLeasesReceivableNonperforming"),
        ],
        "net_charge_offs": [
            # CECL-era
            ("us-gaap", "FinancingReceivableAllowanceForCreditLossesWriteOffs"),
            ("us-gaap", "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoff"),
            # Net of recoveries
            ("us-gaap", "AllowanceForLoanAndLeaseLossesWriteOffsNet"),
            # Pre-CECL
            ("us-gaap", "LoansAndLeasesReceivableAllowanceWriteOffsNet"),
        ],
        "total_assets": [
            ("us-gaap", "Assets"),
        ],
        "total_deposits": [
            ("us-gaap", "Deposits"),
            ("us-gaap", "InterestBearingDepositLiabilities"),
        ],
        "net_income": [
            ("us-gaap", "NetIncomeLoss"),
            ("us-gaap", "ProfitLoss"),
        ],
        "net_interest_income": [
            ("us-gaap", "InterestIncomeExpenseNet"),
            ("us-gaap", "InterestIncomeExpenseAfterProvisionForLoanLoss"),
        ],
        "tier1_capital": [
            ("us-gaap", "Tier1Capital"),
            ("us-gaap", "Tier1CapitalToRiskWeightedAssets"),
        ],
    }

    def __init__(self, start_date: str = "2000-01-01"):
        self.start_date = start_date
        self.sec_fetcher = SECEdgarFetcher()
        self.fred_fetcher = FREDDataFetcher()
        self._bank_data: dict[str, pl.DataFrame] = {}

    def fetch_bank_data(self, ticker: str) -> pl.DataFrame:
        """
        Fetch all available data for a single bank.

        Args:
            ticker: Bank ticker symbol

        Returns:
            DataFrame with quarterly bank metrics
        """
        if ticker not in TARGET_BANKS:
            raise ValueError(f"Unknown bank ticker: {ticker}")

        bank = TARGET_BANKS[ticker]
        print(f"Fetching data for {bank.name}...")

        # Get company facts from SEC
        facts = self.sec_fetcher.get_company_facts(bank.cik)

        if not facts:
            print(f"  No SEC data available for {ticker}")
            return pl.DataFrame({"date": [], "ticker": []})

        # Extract each metric
        dfs = {"date": None}

        for metric_name, concepts in self.XBRL_CONCEPTS.items():
            # For each metric, find the concept with the most recent data
            best_df = None
            best_date = None

            for taxonomy, concept in concepts:
                df = self.sec_fetcher.extract_metric(facts, taxonomy, concept)
                if df.height > 0:
                    latest_date = df.select(pl.col("date").max()).item()
                    # Prefer concept with more recent data
                    if best_date is None or latest_date > best_date:
                        best_df = df.rename({"value": metric_name})
                        best_date = latest_date

            if best_df is not None:
                if dfs["date"] is None:
                    dfs["date"] = best_df
                else:
                    dfs["date"] = dfs["date"].join(
                        best_df, on="date", how="outer_coalesce"
                    )

        if dfs["date"] is None:
            return pl.DataFrame({"date": [], "ticker": []})

        result = dfs["date"]

        # Add ticker column
        result = result.with_columns(pl.lit(ticker).alias("ticker"))

        # Filter to start date
        result = result.filter(pl.col("date") >= pl.lit(self.start_date).str.to_date())

        self._bank_data[ticker] = result
        return result

    def fetch_all_banks(self) -> pl.DataFrame:
        """
        Fetch data for all target banks.

        Returns:
            Panel DataFrame with all banks
        """
        all_dfs = []

        for ticker in TARGET_BANKS:
            try:
                df = self.fetch_bank_data(ticker)
                if df.height > 0:
                    all_dfs.append(df)
            except Exception as e:
                print(f"  Error fetching {ticker}: {e}")

        if not all_dfs:
            return pl.DataFrame({"date": [], "ticker": []})

        # Combine all banks
        result = pl.concat(all_dfs, how="diagonal")
        return result.sort(["ticker", "date"])

    def get_data_quality_summary(self) -> pl.DataFrame:
        """
        Generate a summary of data quality/availability for each bank.

        Returns:
            DataFrame with data quality metrics for each bank
        """
        summary_records = []

        for ticker, bank in TARGET_BANKS.items():
            try:
                facts = self.sec_fetcher.get_company_facts(bank.cik)
                if not facts:
                    summary_records.append({
                        "ticker": ticker,
                        "name": bank.name,
                        "tier": bank.tier,
                        "has_loans": False,
                        "loans_latest_date": None,
                        "has_allowance": False,
                        "allowance_latest_date": None,
                        "has_provisions": False,
                        "provisions_latest_date": None,
                        "data_status": "NO_SEC_DATA",
                    })
                    continue

                record = {
                    "ticker": ticker,
                    "name": bank.name,
                    "tier": bank.tier,
                }

                # Check each key metric (prefer concept with most recent data)
                for metric_name, concepts in [
                    ("loans", self.XBRL_CONCEPTS["total_loans"]),
                    ("allowance", self.XBRL_CONCEPTS["allowance"]),
                    ("provisions", self.XBRL_CONCEPTS["provisions"]),
                ]:
                    best_date = None

                    for taxonomy, concept in concepts:
                        df = self.sec_fetcher.extract_metric(facts, taxonomy, concept)
                        if df.height > 0:
                            latest_date = df.select(pl.col("date").max()).item()
                            if best_date is None or latest_date > best_date:
                                best_date = latest_date

                    record[f"has_{metric_name}"] = best_date is not None
                    record[f"{metric_name}_latest_date"] = best_date

                # Determine overall data status based on recency
                from datetime import date as dt_date
                cutoff = dt_date(2024, 1, 1)

                loans_recent = (
                    record.get("loans_latest_date") and
                    record["loans_latest_date"] >= cutoff
                )
                allowance_recent = (
                    record.get("allowance_latest_date") and
                    record["allowance_latest_date"] >= cutoff
                )
                provisions_recent = (
                    record.get("provisions_latest_date") and
                    record["provisions_latest_date"] >= cutoff
                )

                # Add flags for recent data availability
                record["loans_recent"] = loans_recent
                record["allowance_recent"] = allowance_recent
                record["provisions_recent"] = provisions_recent

                if loans_recent and allowance_recent and provisions_recent:
                    record["data_status"] = "COMPLETE"
                elif loans_recent and allowance_recent:
                    record["data_status"] = "COMPLETE_NO_PROV"
                elif record["has_loans"] and not loans_recent:
                    record["data_status"] = "STALE_DATA"
                elif record["has_loans"] and not record["has_allowance"]:
                    record["data_status"] = "PARTIAL"
                else:
                    record["data_status"] = "LIMITED"

                summary_records.append(record)

            except Exception as e:
                summary_records.append({
                    "ticker": ticker,
                    "name": bank.name,
                    "tier": bank.tier,
                    "has_loans": False,
                    "has_allowance": False,
                    "has_provisions": False,
                    "data_status": f"ERROR: {str(e)[:50]}",
                })

        return pl.DataFrame(summary_records)

    def compute_derived_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute derived metrics for analysis.

        Args:
            df: Bank data DataFrame

        Returns:
            DataFrame with additional computed metrics
        """
        result = df.clone()

        # Average loans for denominators
        result = result.with_columns(
            ((pl.col("total_loans") + pl.col("total_loans").shift(1)) / 2)
            .over("ticker")
            .alias("avg_loans")
        )

        # Derive provisions if missing but allowance and net_charge_offs available
        # Formula: Provision = (Ending Allowance - Beginning Allowance) + Net Charge-offs
        if "provisions" in result.columns and "allowance" in result.columns:
            # Calculate derived provisions
            derived_prov = (
                (pl.col("allowance") - pl.col("allowance").shift(1).over("ticker"))
                + pl.col("net_charge_offs").fill_null(0)
            )
            # Fill null provisions with derived values
            result = result.with_columns(
                pl.when(pl.col("provisions").is_null())
                .then(derived_prov)
                .otherwise(pl.col("provisions"))
                .alias("provisions")
            )
            # Add flag indicating whether provision was derived
            result = result.with_columns(
                pl.col("provisions").is_not_null().alias("has_provisions")
            )

        # Provision rate: Provisions / Average Loans
        if "provisions" in result.columns and "avg_loans" in result.columns:
            result = result.with_columns(
                (pl.col("provisions") / pl.col("avg_loans") * 100)
                .alias("provision_rate")
            )

        # NPL ratio: NPL / Total Loans
        if "npl" in result.columns and "total_loans" in result.columns:
            result = result.with_columns(
                (pl.col("npl") / pl.col("total_loans") * 100)
                .alias("npl_ratio")
            )

        # Coverage ratio: Allowance / NPL
        if "allowance" in result.columns and "npl" in result.columns:
            result = result.with_columns(
                (pl.col("allowance") / pl.col("npl") * 100)
                .alias("coverage_ratio")
            )

        # Charge-off rate: Net Charge-offs / Average Loans
        if "net_charge_offs" in result.columns and "avg_loans" in result.columns:
            result = result.with_columns(
                (pl.col("net_charge_offs") / pl.col("avg_loans") * 100)
                .alias("chargeoff_rate")
            )

        # Loan growth (YoY, 4 quarters)
        if "total_loans" in result.columns:
            result = result.with_columns(
                ((pl.col("total_loans") / pl.col("total_loans").shift(4) - 1) * 100)
                .over("ticker")
                .alias("loan_growth_yoy")
            )

        # Asset growth (YoY)
        if "total_assets" in result.columns:
            result = result.with_columns(
                ((pl.col("total_assets") / pl.col("total_assets").shift(4) - 1) * 100)
                .over("ticker")
                .alias("asset_growth_yoy")
            )

        # Loan-to-asset ratio
        if "total_loans" in result.columns and "total_assets" in result.columns:
            result = result.with_columns(
                (pl.col("total_loans") / pl.col("total_assets") * 100)
                .alias("loan_to_asset")
            )

        # ROA (annualized)
        if "net_income" in result.columns and "total_assets" in result.columns:
            result = result.with_columns(
                (pl.col("net_income") / pl.col("total_assets") * 400)  # Annualize quarterly
                .alias("roa")
            )

        # NIM proxy: Net Interest Income / Average Assets
        if "net_interest_income" in result.columns and "total_assets" in result.columns:
            result = result.with_columns(
                (pl.col("net_interest_income") /
                 ((pl.col("total_assets") + pl.col("total_assets").shift(1)) / 2) * 400)
                .over("ticker")
                .alias("nim")
            )

        return result


class SyntheticBankData:
    """
    Generate synthetic bank data for testing when SEC data is unavailable.

    Creates realistic patterns based on historical bank behavior:
    - Procyclical loan growth
    - Lagged provisions responding to credit quality
    - Cross-sectional variation in risk appetite
    """

    def __init__(self, start_date: str = "2000-01-01", end_date: Optional[str] = None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")

    def generate_panel(self, n_banks: int = 10, seed: int = 42) -> pl.DataFrame:
        """
        Generate synthetic panel data for testing.

        Args:
            n_banks: Number of banks to simulate
            seed: Random seed for reproducibility

        Returns:
            Panel DataFrame with synthetic bank data
        """
        np.random.seed(seed)

        # Generate quarterly dates
        dates = pl.date_range(
            pl.date(2000, 1, 1),
            pl.date(2024, 10, 1),
            "3mo",
            eager=True,
        )
        n_quarters = len(dates)

        # Bank characteristics (heterogeneity)
        bank_tickers = [f"BANK{i:02d}" for i in range(n_banks)]
        risk_appetite = np.random.uniform(0.8, 1.2, n_banks)  # Relative aggressiveness
        size_factor = np.random.uniform(0.5, 2.0, n_banks)  # Relative size

        # Economic cycle (shared across banks)
        cycle = np.sin(2 * np.pi * np.arange(n_quarters) / 40)  # ~10-year cycle
        cycle += 0.3 * np.sin(2 * np.pi * np.arange(n_quarters) / 16)  # ~4-year component
        cycle_noise = np.random.normal(0, 0.1, n_quarters)
        econ_cycle = cycle + cycle_noise

        # Generate data for each bank
        records = []

        for i, ticker in enumerate(bank_tickers):
            # Base loan level
            base_loans = 100_000 * size_factor[i]

            # Loan growth responds to economic cycle with bank-specific sensitivity
            loan_growth = (
                0.02 +  # Trend growth
                0.03 * risk_appetite[i] * econ_cycle +  # Cyclical component
                np.random.normal(0, 0.01, n_quarters)  # Idiosyncratic
            )

            # Cumulative loan levels
            loans = base_loans * np.cumprod(1 + loan_growth)

            # Provisions lag the cycle (3-4 years behind)
            # Higher during downturns, especially for aggressive lenders
            lag = 12  # 3 years
            lagged_cycle = np.roll(econ_cycle, lag)
            lagged_cycle[:lag] = 0

            provision_rate = (
                0.005 +  # Base provision rate
                0.008 * risk_appetite[i] * (-lagged_cycle) +  # Counter-cyclical
                np.random.normal(0, 0.001, n_quarters)
            )
            provision_rate = np.clip(provision_rate, 0.001, 0.05)

            provisions = loans * provision_rate

            # NPL ratio also lags the cycle
            npl_ratio = (
                0.01 +
                0.02 * risk_appetite[i] * (-lagged_cycle) +
                np.random.normal(0, 0.002, n_quarters)
            )
            npl_ratio = np.clip(npl_ratio, 0.002, 0.10)
            npl = loans * npl_ratio

            # Allowance is related to provisions and NPLs
            allowance = npl * (1.2 + 0.3 * np.random.randn(n_quarters))
            allowance = np.clip(allowance, npl * 0.8, npl * 2.0)

            # Total assets
            loan_to_asset = 0.6 + 0.1 * np.random.randn(n_quarters)
            loan_to_asset = np.clip(loan_to_asset, 0.4, 0.8)
            total_assets = loans / loan_to_asset

            for j, date in enumerate(dates):
                records.append({
                    "date": date,
                    "ticker": ticker,
                    "total_loans": loans[j],
                    "provisions": provisions[j],
                    "allowance": allowance[j],
                    "npl": npl[j],
                    "total_assets": total_assets[j],
                    "econ_cycle": econ_cycle[j],  # For validation
                    "risk_appetite": risk_appetite[i],  # For validation
                })

        df = pl.DataFrame(records)

        # Compute derived metrics
        df = df.with_columns([
            ((pl.col("total_loans") / pl.col("total_loans").shift(4) - 1) * 100)
            .over("ticker")
            .alias("loan_growth_yoy"),
            (pl.col("provisions") / pl.col("total_loans") * 100)
            .alias("provision_rate"),
            (pl.col("npl") / pl.col("total_loans") * 100)
            .alias("npl_ratio"),
            (pl.col("allowance") / pl.col("npl") * 100)
            .alias("coverage_ratio"),
        ])

        return df.sort(["ticker", "date"])


if __name__ == "__main__":
    # Test synthetic data generation
    print("Generating synthetic bank panel data...")
    synth = SyntheticBankData()
    panel = synth.generate_panel(n_banks=5)

    print(f"Panel shape: {panel.shape}")
    print(f"Banks: {panel['ticker'].unique().to_list()}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")

    print("\nSample data (last 5 quarters for BANK00):")
    sample = panel.filter(pl.col("ticker") == "BANK00").tail(5)
    print(sample.select([
        "date", "ticker", "loan_growth_yoy", "provision_rate", "npl_ratio"
    ]))

    # Test SEC fetcher (may fail due to rate limits)
    print("\nTesting SEC EDGAR fetcher...")
    collector = BankDataCollector()
    try:
        jpm_data = collector.fetch_bank_data("JPM")
        print(f"JPM data: {jpm_data.shape}")
        if jpm_data.height > 0:
            print(jpm_data.tail(5))
    except Exception as e:
        print(f"SEC fetch failed (expected if rate-limited): {e}")
