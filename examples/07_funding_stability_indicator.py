#!/usr/bin/env python3
"""
Funding Stability Indicator Example

Measures bank funding vulnerabilities through deposit stability metrics
and reliance on wholesale funding sources. Uses real FRED data.

Key concepts:
- Uninsured deposit ratio (run risk)
- FHLB borrowing reliance
- Deposit concentration
- Wholesale funding dependency

Reference: Bank funding stability analysis for systemic risk
"""

import polars as pl

from financing_private_credit.indicators import get_indicator
from financing_private_credit.data import FREDDataFetcher
from financing_private_credit.bank_data import BankDataCollector


def demonstrate_deposit_environment():
    """
    Show aggregate deposit trends from real FRED data.
    """
    print("=" * 60)
    print("FUNDING STABILITY - DEPOSIT ENVIRONMENT")
    print("=" * 60)

    fetcher = FREDDataFetcher()

    # Fetch deposit-related series
    print("\n[1] Fetching deposit data from FRED...")
    deposit_series = ["DPSACBW027SBOG", "DEMDEPSL", "SVSTCBSL", "STDCBSL"]
    deposits = fetcher.fetch_multiple_series(deposit_series, "2020-01-01")

    # Also fetch total bank assets for context
    assets_series = ["TLAACBW027SBOG"]
    assets = fetcher.fetch_multiple_series(assets_series, "2020-01-01")

    if deposits.height > 0:
        print(f"   Fetched {deposits.height} observations")

        # Latest deposit levels
        latest = deposits.tail(1)
        print(f"\n   Aggregate Deposits ({latest['date'][0]}):")

        if "DPSACBW027SBOG" in latest.columns and latest["DPSACBW027SBOG"][0]:
            total_dep = latest["DPSACBW027SBOG"][0]
            print(f"     Total Deposits: ${total_dep:,.0f}B")

        # Deposit growth analysis
        print("\n[2] Deposit Flow Analysis...")
        if deposits.height >= 52:
            year_ago = deposits.head(deposits.height - 52).tail(1)
            current = deposits.tail(1)

            if ("DPSACBW027SBOG" in current.columns and
                current["DPSACBW027SBOG"][0] and year_ago["DPSACBW027SBOG"][0]):
                yoy_growth = (current["DPSACBW027SBOG"][0] / year_ago["DPSACBW027SBOG"][0] - 1) * 100
                print(f"     Deposit Growth (YoY): {yoy_growth:+.1f}%")

                if yoy_growth < -5:
                    print("     Status: DEPOSIT OUTFLOWS (stress signal)")
                elif yoy_growth < 0:
                    print("     Status: MODEST OUTFLOWS")
                elif yoy_growth < 5:
                    print("     Status: STABLE")
                else:
                    print("     Status: STRONG INFLOWS")


def demonstrate_wholesale_funding():
    """
    Analyze wholesale funding sources including FHLB.
    """
    print("\n" + "=" * 60)
    print("WHOLESALE FUNDING ANALYSIS")
    print("=" * 60)

    fetcher = FREDDataFetcher()

    # Fetch FHLB and other funding data
    print("\n[1] Fetching wholesale funding data...")
    # Note: FHLB advances data may have limited availability
    funding_series = ["BOGZ1FL764190005Q", "WFFBAW027NBOG"]
    funding = fetcher.fetch_multiple_series(funding_series, "2020-01-01")

    print("""
    Wholesale Funding Sources:

    ┌──────────────────────────────────────────────────────────────┐
    │  Source                │  Stability  │  Cost Sensitivity    │
    ├────────────────────────┼─────────────┼──────────────────────┤
    │  Core Deposits         │  HIGH       │  LOW                 │
    │  (checking, savings)   │             │                      │
    ├────────────────────────┼─────────────┼──────────────────────┤
    │  Time Deposits (CDs)   │  MEDIUM     │  MEDIUM              │
    │                        │             │                      │
    ├────────────────────────┼─────────────┼──────────────────────┤
    │  Brokered Deposits     │  LOW        │  HIGH                │
    │                        │             │                      │
    ├────────────────────────┼─────────────┼──────────────────────┤
    │  FHLB Advances         │  MEDIUM     │  HIGH                │
    │  (lender of next-      │             │                      │
    │   to-last resort)      │             │                      │
    ├────────────────────────┼─────────────┼──────────────────────┤
    │  Fed Funds/Repo        │  LOW        │  VERY HIGH           │
    │                        │             │                      │
    └────────────────────────┴─────────────┴──────────────────────┘
    """)

    print("""
    FHLB Advance Surge = Warning Signal:

    ┌──────────────────────────────────────────────────────────────┐
    │  When banks lose deposits, they often tap FHLB first:       │
    │                                                              │
    │  • FHLB provides secured lending against mortgages          │
    │  • Surge in FHLB borrowing signals deposit stress           │
    │  • SVB increased FHLB advances 4x before failure            │
    │  • First Republic similarly relied on FHLB in final weeks  │
    └──────────────────────────────────────────────────────────────┘
    """)


def demonstrate_rate_impact_on_deposits():
    """
    Show how rate environment affects deposit stability.
    """
    print("\n" + "=" * 60)
    print("RATE IMPACT ON DEPOSIT STABILITY")
    print("=" * 60)

    fetcher = FREDDataFetcher()

    # Fetch rates and money market fund data
    print("\n[1] Fetching rate and money market data...")
    series = ["FEDFUNDS", "WRMFSL", "MMMFFAQ027S"]
    data = fetcher.fetch_multiple_series(series, "2020-01-01")

    if data.height > 0:
        print(f"   Fetched {data.height} observations")

        # Recent comparison
        if data.height >= 52:
            current = data.tail(1)
            year_ago = data.head(data.height - 52).tail(1)

            print(f"\n   Rate Environment vs Money Market Flows:")
            print(f"   {'Metric':<25} {'1Y Ago':>12} {'Current':>12} {'Change':>12}")
            print("   " + "-" * 55)

            for col, name in [("FEDFUNDS", "Fed Funds Rate (%)"),
                              ("WRMFSL", "Retail MMF ($B)"),
                              ("MMMFFAQ027S", "Total MMF ($B)")]:
                if col in current.columns and col in year_ago.columns:
                    curr_val = current[col][0]
                    prev_val = year_ago[col][0]
                    if curr_val is not None and prev_val is not None:
                        if col == "FEDFUNDS":
                            change = f"{curr_val - prev_val:+.2f}%"
                            print(f"   {name:<25} {prev_val:>11.2f}% {curr_val:>11.2f}% {change:>12}")
                        else:
                            pct_change = (curr_val / prev_val - 1) * 100
                            print(f"   {name:<25} {prev_val:>12,.0f} {curr_val:>12,.0f} {pct_change:>+11.1f}%")

            print("""
    Key Insight: When rates rise, deposits flow to higher-yielding alternatives

    ┌──────────────────────────────────────────────────────────────┐
    │  Rate Hiking Cycle Effects:                                  │
    │                                                              │
    │  1. Money market funds offer competitive yields             │
    │  2. Bank deposit rates lag Fed Funds (deposit beta < 1)    │
    │  3. Sophisticated depositors (uninsured) move first        │
    │  4. Banks face funding cost pressure OR deposit outflows   │
    │                                                              │
    │  Deposit Beta Analysis:                                      │
    │    • High beta: Bank raises deposit rates (protects base)  │
    │    • Low beta: Bank keeps rates low (loses deposits)       │
    │    • Sweet spot: Gradually raise rates, retain core        │
    └──────────────────────────────────────────────────────────────┘
            """)


def demonstrate_bank_funding_metrics():
    """
    Analyze bank-level funding metrics from SEC data.
    """
    print("\n" + "=" * 60)
    print("BANK FUNDING METRICS (SEC EDGAR)")
    print("=" * 60)

    collector = BankDataCollector(start_date="2023-01-01")

    print("\n[1] Fetching bank funding data...")
    banks = ["JPM", "BAC", "WFC", "C"]

    bank_dfs = []
    for ticker in banks:
        try:
            df = collector.fetch_bank_data(ticker)
            if df.height > 0:
                bank_dfs.append(df)
        except Exception:
            pass

    if bank_dfs:
        panel = pl.concat(bank_dfs, how="diagonal")

        print("\n   Bank Funding Overview (Latest Quarter):")
        print(f"   {'Bank':<6} {'Assets ($B)':<14} {'Deposits ($B)':<14} {'Dep/Assets':<10}")
        print("   " + "-" * 48)

        for ticker in banks:
            bank_data = panel.filter(pl.col("ticker") == ticker).tail(1)
            if bank_data.height > 0:
                assets = bank_data["total_assets"][0]
                deposits = bank_data["total_deposits"][0]

                if assets and deposits:
                    ratio = deposits / assets * 100
                    print(f"   {ticker:<6} {assets/1000:>12,.0f}  {deposits/1000:>12,.0f}  {ratio:>8.1f}%")


def demonstrate_funding_risk_framework():
    """
    Present the funding stability risk framework.
    """
    print("\n" + "=" * 60)
    print("FUNDING STABILITY RISK FRAMEWORK")
    print("=" * 60)

    print("""
    Funding Stability Score Components:

    ┌──────────────────────────────────────────────────────────────┐
    │  1. DEPOSIT QUALITY (40% weight)                            │
    │     • Core deposit ratio (checking + savings)               │
    │     • Deposit growth trend (3-year CAGR)                   │
    │     • Deposit concentration (top 10 depositors)            │
    │                                                              │
    │  2. FUNDING DIVERSIFICATION (25% weight)                    │
    │     • Wholesale funding ratio                               │
    │     • FHLB utilization (advances / capacity)               │
    │     • Debt maturity profile                                 │
    │                                                              │
    │  3. LIQUIDITY POSITION (20% weight)                         │
    │     • LCR (Liquidity Coverage Ratio)                       │
    │     • HQLA as % of assets                                   │
    │     • Available Fed borrowing capacity                     │
    │                                                              │
    │  4. RUN RISK (15% weight)                                   │
    │     • Uninsured deposit ratio                               │
    │     • Social media sentiment (deposit run proxy)           │
    │     • Stock price volatility                                │
    └──────────────────────────────────────────────────────────────┘

    Risk Thresholds:
    ┌──────────────────────────────────────────────────────────────┐
    │  Score 80-100: LOW RISK - Strong funding position          │
    │  Score 60-80:  MODERATE - Monitor key metrics              │
    │  Score 40-60:  ELEVATED - Active management needed         │
    │  Score 0-40:   HIGH RISK - Potential funding stress        │
    └──────────────────────────────────────────────────────────────┘
    """)


def main():
    """Main example runner."""
    print("=" * 60)
    print("FUNDING STABILITY INDICATOR EXAMPLE")
    print("Deposit and Wholesale Funding Analysis")
    print("=" * 60)

    # Run demonstrations with real data
    demonstrate_deposit_environment()
    demonstrate_wholesale_funding()
    demonstrate_rate_impact_on_deposits()
    demonstrate_bank_funding_metrics()
    demonstrate_funding_risk_framework()

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Deposit stability is critical for bank funding")
    print("  2. Rising rates drive deposits to money market funds")
    print("  3. FHLB advance surge signals deposit stress")
    print("  4. Uninsured deposits are first to flee in crisis")
    print("  5. Funding stability score integrates multiple risks")


if __name__ == "__main__":
    main()
