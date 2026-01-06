#!/usr/bin/env python3
"""
Credit Boom Indicator (LIS) Example

The Lending Intensity Score (LIS) measures relative lending intensity across
banks using cross-sectional standardization and real SEC EDGAR data.

Key concepts:
- Cross-sectional z-score of loan growth vs peer banks
- Cumulative LIS tracks sustained aggressive lending
- Nowcast uses weekly H.8 data for intra-quarter estimates
- Real bank data from SEC EDGAR filings

Risk Thresholds:
- LIS > 2.0: High risk (aggressive lending)
- LIS 1.0-2.0: Elevated risk
- LIS < 1.0: Normal

Reference: Boyarchenko & Elias (2024): Private Credit and the Business Cycle
"""

import polars as pl

from financing_private_credit.indicators import get_indicator
from financing_private_credit.data import FREDDataFetcher
from financing_private_credit.bank_data import BankDataCollector, TARGET_BANKS


def demonstrate_lis_with_real_data():
    """
    Demonstrate LIS calculation using real SEC EDGAR data.
    """
    print("=" * 60)
    print("CREDIT BOOM INDICATOR - REAL SEC DATA")
    print("=" * 60)

    # Get the indicator
    indicator = get_indicator("credit_boom")
    metadata = indicator.get_metadata()

    print(f"\nIndicator: {metadata.name}")
    print(f"Version: {metadata.version}")
    print(f"Paper: {metadata.paper_reference}")

    # Fetch real bank data from SEC EDGAR
    print("\n[1] Fetching real bank data from SEC EDGAR...")
    print(f"    Target banks: {list(TARGET_BANKS.keys())[:6]}...")

    collector = BankDataCollector(start_date="2020-01-01")

    # Fetch data for a subset of banks (to avoid rate limiting)
    bank_dfs = []
    banks_to_fetch = ["JPM", "BAC", "WFC", "C"]  # G-SIBs

    for ticker in banks_to_fetch:
        try:
            df = collector.fetch_bank_data(ticker)
            if df.height > 0:
                bank_dfs.append(df)
                print(f"    {ticker}: {df.height} quarters of data")
        except Exception as e:
            print(f"    {ticker}: Error - {str(e)[:50]}")

    if bank_dfs:
        bank_panel = pl.concat(bank_dfs, how="diagonal")
        print(f"\n   Total panel: {bank_panel.height} bank-quarters")

        # Compute derived metrics
        bank_panel = collector.compute_derived_metrics(bank_panel)

        # Calculate LIS
        print("\n[2] Calculating Lending Intensity Scores...")
        result = indicator.calculate({"bank_panel": bank_panel})

        if result.data.height > 0:
            print(f"   Calculated LIS for {result.data.height} bank-quarters")

            # Show latest LIS by bank
            print("\n   Latest LIS Scores:")
            latest = result.data.group_by("ticker").agg(pl.col("date").max()).join(
                result.data, on=["ticker", "date"]
            )

            for row in latest.sort("lis", descending=True).iter_rows(named=True):
                lis = row.get("lis", 0)
                if lis > 2.0:
                    risk = "HIGH"
                elif lis > 1.0:
                    risk = "ELEVATED"
                else:
                    risk = "NORMAL"
                print(f"     {row['ticker']}: LIS = {lis:.2f} ({risk})")

            return result

    return None


def demonstrate_h8_nowcast():
    """
    Demonstrate nowcasting with real weekly H.8 data.
    """
    print("\n" + "=" * 60)
    print("CREDIT BOOM NOWCAST (Weekly H.8 Data)")
    print("=" * 60)

    # Fetch real H.8 data from FRED
    print("\n[1] Fetching weekly H.8 banking data from FRED...")
    fetcher = FREDDataFetcher()

    h8_series = ["TOTLL", "BUSLOANS", "CONSUMER", "REALLN"]
    h8_data = fetcher.fetch_multiple_series(h8_series, "2023-01-01")

    if h8_data.height > 0:
        print(f"   Fetched {h8_data.height} weeks of H.8 data")

        # Get latest values
        latest = h8_data.tail(1)
        print(f"\n   Latest H.8 Data ({latest['date'][0]}):")

        series_names = {
            "TOTLL": "Total Loans & Leases",
            "BUSLOANS": "C&I Loans",
            "CONSUMER": "Consumer Loans",
            "REALLN": "Real Estate Loans"
        }

        for col in h8_series:
            if col in latest.columns:
                val = latest[col][0]
                if val is not None:
                    print(f"     {series_names[col]}: ${val:,.0f}B")

        # Calculate week-over-week growth
        print("\n   Weekly Growth Rates:")
        for col in h8_series:
            if col in h8_data.columns:
                df = h8_data.filter(pl.col(col).is_not_null())
                if df.height >= 2:
                    current = df[col][-1]
                    previous = df[col][-2]
                    if current is not None and previous is not None and previous > 0:
                        growth = (current / previous - 1) * 100
                        print(f"     {series_names[col]}: {growth:+.3f}%")

        # Calculate year-over-year growth
        print("\n   Year-over-Year Growth:")
        for col in h8_series:
            if col in h8_data.columns:
                df = h8_data.filter(pl.col(col).is_not_null())
                if df.height >= 52:
                    current = df[col][-1]
                    year_ago = df[col][-52]
                    if current is not None and year_ago is not None and year_ago > 0:
                        yoy_growth = (current / year_ago - 1) * 100
                        print(f"     {series_names[col]}: {yoy_growth:+.1f}%")

        # Nowcast signal
        print("\n   Nowcast Credit Signal:")
        if "TOTLL" in h8_data.columns:
            df = h8_data.filter(pl.col("TOTLL").is_not_null())
            if df.height >= 13:  # 3 months
                current = df["TOTLL"][-1]
                three_mo_ago = df["TOTLL"][-13]
                if current and three_mo_ago and three_mo_ago > 0:
                    momentum = (current / three_mo_ago - 1) * 4 * 100  # Annualized
                    print(f"     3-Month Annualized Growth: {momentum:+.1f}%")

                    if momentum > 10:
                        print("     Status: CREDIT ACCELERATION")
                    elif momentum > 5:
                        print("     Status: HEALTHY GROWTH")
                    elif momentum > 0:
                        print("     Status: SLOWING GROWTH")
                    else:
                        print("     Status: CREDIT CONTRACTION")


def demonstrate_loan_composition():
    """
    Show loan composition analysis using real H.8 data.
    """
    print("\n" + "=" * 60)
    print("LOAN COMPOSITION ANALYSIS")
    print("=" * 60)

    fetcher = FREDDataFetcher()

    # Fetch multiple H.8 loan categories
    print("\n[1] Fetching loan composition data...")
    h8_series = ["TOTLL", "BUSLOANS", "CONSUMER", "REALLN", "OTHLN"]
    h8_data = fetcher.fetch_multiple_series(h8_series, "2020-01-01")

    if h8_data.height > 0:
        # Get latest composition
        latest = h8_data.filter(pl.col("TOTLL").is_not_null()).tail(1)

        if latest.height > 0 and latest["TOTLL"][0] is not None:
            total = latest["TOTLL"][0]

            print(f"\n   Loan Composition ({latest['date'][0]}):")
            print(f"   Total Loans & Leases: ${total:,.0f}B")
            print()

            for col, name in [("BUSLOANS", "C&I Loans"),
                              ("CONSUMER", "Consumer Loans"),
                              ("REALLN", "Real Estate Loans")]:
                if col in latest.columns and latest[col][0] is not None:
                    val = latest[col][0]
                    pct = val / total * 100
                    print(f"     {name}: ${val:,.0f}B ({pct:.1f}%)")

        # Calculate composition changes over time
        print("\n   Composition Shift (1-Year Change):")
        if h8_data.height >= 52:
            current = h8_data.tail(1)
            year_ago = h8_data.head(h8_data.height - 52).tail(1)

            for col, name in [("BUSLOANS", "C&I Share"),
                              ("CONSUMER", "Consumer Share"),
                              ("REALLN", "Real Estate Share")]:
                if (col in current.columns and "TOTLL" in current.columns and
                    current[col][0] is not None and current["TOTLL"][0] is not None and
                    year_ago[col][0] is not None and year_ago["TOTLL"][0] is not None):

                    current_share = current[col][0] / current["TOTLL"][0] * 100
                    year_ago_share = year_ago[col][0] / year_ago["TOTLL"][0] * 100
                    change = current_share - year_ago_share

                    print(f"     {name}: {change:+.1f}pp")


def demonstrate_bank_data_quality():
    """
    Show data quality and availability from SEC EDGAR.
    """
    print("\n" + "=" * 60)
    print("BANK DATA QUALITY (SEC EDGAR)")
    print("=" * 60)

    print("\n[1] Checking data availability for target banks...")
    collector = BankDataCollector()
    summary = collector.get_data_quality_summary()

    if summary.height > 0:
        print(f"\n   {'Ticker':<6} {'Tier':<5} {'Status':<18} {'Latest Data':<12}")
        print("   " + "-" * 45)

        for row in summary.iter_rows(named=True):
            status = row.get("data_status", "UNKNOWN")
            loans_date = row.get("loans_latest_date", "N/A")
            if loans_date and loans_date != "N/A":
                loans_date = str(loans_date)[:10]
            else:
                loans_date = "N/A"

            print(f"   {row['ticker']:<6} {row['tier']:<5} {status:<18} {loans_date:<12}")


def main():
    """Main example runner."""
    print("=" * 60)
    print("CREDIT BOOM INDICATOR EXAMPLE")
    print("Lending Intensity Score (LIS) with Real Data")
    print("=" * 60)

    # Run demonstrations with real data
    demonstrate_h8_nowcast()
    demonstrate_loan_composition()
    demonstrate_bank_data_quality()
    demonstrate_lis_with_real_data()

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Weekly H.8 data provides real-time credit signals")
    print("  2. SEC EDGAR provides bank-level loan and provision data")
    print("  3. LIS measures relative lending intensity vs peers")
    print("  4. High LIS (>2.0) signals aggressive lending risk")
    print("  5. Loan composition shifts reveal credit cycle dynamics")


if __name__ == "__main__":
    main()
