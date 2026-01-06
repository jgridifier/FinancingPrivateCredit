#!/usr/bin/env python3
"""
Duration Mismatch Indicator Example

Measures interest rate risk from asset-liability duration mismatch,
similar to the vulnerabilities that caused SVB's failure.

Key concepts:
- Asset duration vs liability duration gap
- Unrealized losses from rate changes
- HTM portfolio exposure
- Deposit stability and uninsured deposit ratio

Uses real FRED data for interest rates and bank data from SEC EDGAR.

Reference: SVB-style rate risk analysis for bank equity screening
"""

import polars as pl

from financing_private_credit.indicators import get_indicator
from financing_private_credit.data import FREDDataFetcher
from financing_private_credit.bank_data import BankDataCollector


def demonstrate_rate_environment():
    """
    Show current interest rate environment and historical context.
    """
    print("=" * 60)
    print("DURATION MISMATCH - RATE ENVIRONMENT")
    print("=" * 60)

    fetcher = FREDDataFetcher()

    # Fetch key rate series
    print("\n[1] Fetching interest rate data from FRED...")
    rate_series = ["FEDFUNDS", "GS2", "GS5", "GS10", "GS30", "MORTGAGE30US"]
    rates = fetcher.fetch_multiple_series(rate_series, "2020-01-01")

    if rates.height > 0:
        print(f"   Fetched {rates.height} observations")

        # Latest yield curve
        latest = rates.filter(pl.col("GS10").is_not_null()).tail(1)
        if latest.height > 0:
            print(f"\n   Current Yield Curve ({latest['date'][0]}):")

            for col, name in [("FEDFUNDS", "Fed Funds"),
                              ("GS2", "2-Year Treasury"),
                              ("GS5", "5-Year Treasury"),
                              ("GS10", "10-Year Treasury"),
                              ("GS30", "30-Year Treasury"),
                              ("MORTGAGE30US", "30-Year Mortgage")]:
                if col in latest.columns and latest[col][0] is not None:
                    print(f"     {name}: {latest[col][0]:.2f}%")

            # Calculate spreads
            if ("GS10" in latest.columns and "GS2" in latest.columns and
                latest["GS10"][0] is not None and latest["GS2"][0] is not None):
                spread_10_2 = latest["GS10"][0] - latest["GS2"][0]
                print(f"\n     10Y-2Y Spread: {spread_10_2:.2f}%")

                if spread_10_2 < 0:
                    print("     → INVERTED: Historically signals recession")
                elif spread_10_2 < 0.5:
                    print("     → FLAT: Banks face NIM pressure")
                else:
                    print("     → NORMAL: Favorable for bank earnings")

        # Rate change analysis
        print("\n[2] Rate Changes (Duration Impact):")
        if rates.height >= 252:  # ~1 year of daily data
            year_ago = rates.head(rates.height - 252).tail(1)
            current = rates.tail(1)

            for col, name in [("GS2", "2-Year"),
                              ("GS5", "5-Year"),
                              ("GS10", "10-Year")]:
                if (col in current.columns and col in year_ago.columns and
                    current[col][0] is not None and year_ago[col][0] is not None):
                    change = current[col][0] - year_ago[col][0]
                    print(f"     {name} Change (1Y): {change:+.2f}%")

            # Impact analysis
            if ("GS5" in current.columns and current["GS5"][0] is not None and
                "GS5" in year_ago.columns and year_ago["GS5"][0] is not None):
                rate_change = current["GS5"][0] - year_ago["GS5"][0]

                # Approximate impact on 5-year duration portfolio
                duration = 5.0
                price_impact = -duration * rate_change

                print(f"\n   Duration Impact Analysis:")
                print(f"     Assuming 5-year duration portfolio")
                print(f"     Estimated Price Impact: {price_impact:+.1f}%")

                if price_impact < -10:
                    print("     → SEVERE: Major unrealized losses likely")
                elif price_impact < -5:
                    print("     → SIGNIFICANT: Material impact on equity")
                elif price_impact < 0:
                    print("     → MODERATE: Manageable losses")
                else:
                    print("     → GAINS: Rising bond prices benefit HTM")


def demonstrate_bank_rate_sensitivity():
    """
    Analyze bank-level rate sensitivity using SEC data.
    """
    print("\n" + "=" * 60)
    print("BANK RATE SENSITIVITY ANALYSIS")
    print("=" * 60)

    collector = BankDataCollector(start_date="2022-01-01")

    # Fetch data for select banks
    print("\n[1] Fetching bank data from SEC EDGAR...")
    banks = ["JPM", "BAC", "WFC", "C"]

    bank_dfs = []
    for ticker in banks:
        try:
            df = collector.fetch_bank_data(ticker)
            if df.height > 0:
                bank_dfs.append(df)
                print(f"    {ticker}: {df.height} quarters")
        except Exception as e:
            print(f"    {ticker}: {str(e)[:40]}")

    if bank_dfs:
        panel = pl.concat(bank_dfs, how="diagonal")
        panel = collector.compute_derived_metrics(panel)

        # Analyze by bank
        print("\n   Bank Asset/Deposit Analysis:")
        print(f"   {'Bank':<6} {'Assets ($B)':<14} {'Deposits ($B)':<14} {'L/D Ratio':<10}")
        print("   " + "-" * 48)

        for ticker in banks:
            bank_data = panel.filter(pl.col("ticker") == ticker).tail(1)
            if bank_data.height > 0:
                assets = bank_data["total_assets"][0]
                deposits = bank_data["total_deposits"][0]

                if assets and deposits:
                    ld_ratio = assets / deposits
                    print(f"   {ticker:<6} {assets/1000:>12,.0f}  {deposits/1000:>12,.0f}  {ld_ratio:>8.2f}")


def demonstrate_duration_concepts():
    """
    Explain duration mismatch concepts with examples.
    """
    print("\n" + "=" * 60)
    print("DURATION MISMATCH CONCEPTS")
    print("=" * 60)

    print("""
    Duration measures interest rate sensitivity:

    ┌──────────────────────────────────────────────────────────────┐
    │  Asset Duration vs Liability Duration                       │
    │                                                              │
    │  ASSETS (what banks own):                                   │
    │    • Securities: Duration 3-10 years                        │
    │    • Mortgages: Duration 5-7 years                          │
    │    • C&I Loans: Duration 1-3 years (often floating)        │
    │                                                              │
    │  LIABILITIES (what banks owe):                              │
    │    • Demand deposits: Duration ~0 (can leave anytime)      │
    │    • Time deposits: Duration 0.5-2 years                   │
    │    • Borrowings: Duration 0-5 years                        │
    └──────────────────────────────────────────────────────────────┘

    Duration GAP = Asset Duration - Liability Duration

    ┌──────────────────────────────────────────────────────────────┐
    │  If GAP > 0 (Asset-Sensitive):                              │
    │    • Rising rates → Assets fall MORE than liabilities      │
    │    • UNREALIZED LOSSES accumulate                           │
    │    • Example: SVB had large positive duration gap          │
    │                                                              │
    │  If GAP < 0 (Liability-Sensitive):                          │
    │    • Rising rates → Liabilities fall more than assets      │
    │    • Bank equity INCREASES                                  │
    │    • Rare for banks (natural asset-sensitive)              │
    └──────────────────────────────────────────────────────────────┘
    """)


def demonstrate_svb_case_study():
    """
    SVB-style duration mismatch case study with real rate data.
    """
    print("\n" + "=" * 60)
    print("CASE STUDY: SVB-STYLE DURATION RISK")
    print("=" * 60)

    fetcher = FREDDataFetcher()

    # Fetch rate data around SVB failure (2022-2023)
    print("\n[1] Rate Environment During SVB Crisis...")
    rates = fetcher.fetch_multiple_series(["GS5", "GS10"], "2021-01-01")

    if rates.height > 0:
        # Find key dates
        # March 2022: Fed starts hiking
        # March 2023: SVB failure

        print("\n   Key Rate Levels:")

        # Early 2022 (before hiking)
        early_2022 = rates.filter(
            (pl.col("date") >= pl.lit("2022-01-01").str.to_date()) &
            (pl.col("date") <= pl.lit("2022-02-01").str.to_date())
        ).head(1)

        if early_2022.height > 0 and "GS5" in early_2022.columns:
            gs5_early = early_2022["GS5"][0]
            if gs5_early:
                print(f"     Jan 2022 (Pre-Hikes): 5Y = {gs5_early:.2f}%")

        # March 2023 (SVB failure)
        march_2023 = rates.filter(
            (pl.col("date") >= pl.lit("2023-03-01").str.to_date()) &
            (pl.col("date") <= pl.lit("2023-03-15").str.to_date())
        ).head(1)

        if march_2023.height > 0 and "GS5" in march_2023.columns:
            gs5_march = march_2023["GS5"][0]
            if gs5_march:
                print(f"     Mar 2023 (SVB Failure): 5Y = {gs5_march:.2f}%")

                if early_2022.height > 0 and early_2022["GS5"][0]:
                    rate_change = gs5_march - early_2022["GS5"][0]
                    print(f"     Rate Change: {rate_change:+.2f}%")

                    # Calculate hypothetical loss
                    duration = 5.0
                    loss_pct = -duration * rate_change
                    print(f"\n   Hypothetical 5-Year Duration Portfolio:")
                    print(f"     Price Impact: {loss_pct:+.1f}%")

                    # SVB context
                    print("""
    SVB Specific Factors:
    ┌──────────────────────────────────────────────────────────────┐
    │  • HTM Securities: $91B (at cost, hiding $15B+ losses)     │
    │  • Duration Gap: ~4-5 years (highly asset-sensitive)        │
    │  • Uninsured Deposits: ~90% (very high run risk)           │
    │  • Deposit Concentration: Tech/VC sector                   │
    │                                                              │
    │  When rates rose 300bps+:                                   │
    │    1. Unrealized losses exceeded equity cushion            │
    │    2. Depositors fled (uninsured, concentrated)            │
    │    3. Forced to sell securities at loss                    │
    │    4. Bank failure in < 48 hours                           │
    └──────────────────────────────────────────────────────────────┘
                    """)


def demonstrate_screening_signals():
    """
    Show duration mismatch screening signals.
    """
    print("\n" + "=" * 60)
    print("DURATION MISMATCH SCREENING SIGNALS")
    print("=" * 60)

    print("""
    Red Flags for Duration Risk:

    ┌──────────────────────────────────────────────────────────────┐
    │  HIGH RISK Indicators:                                      │
    │    • HTM Securities > 25% of assets                        │
    │    • Unrealized losses > 50% of tangible equity            │
    │    • Uninsured deposits > 50% of total deposits            │
    │    • Deposit concentration in volatile sectors             │
    │    • Low liquidity ratios (LCR < 100%)                     │
    │                                                              │
    │  MODERATE RISK Indicators:                                  │
    │    • HTM Securities 10-25% of assets                       │
    │    • Unrealized losses 20-50% of tangible equity           │
    │    • Uninsured deposits 30-50%                             │
    │                                                              │
    │  LOW RISK Indicators:                                       │
    │    • HTM Securities < 10% of assets                        │
    │    • Minimal unrealized losses                              │
    │    • Uninsured deposits < 30%                              │
    │    • Diversified deposit base                               │
    └──────────────────────────────────────────────────────────────┘
    """)


def main():
    """Main example runner."""
    print("=" * 60)
    print("DURATION MISMATCH INDICATOR EXAMPLE")
    print("Interest Rate Risk Analysis with Real Data")
    print("=" * 60)

    # Run demonstrations with real data
    demonstrate_rate_environment()
    demonstrate_bank_rate_sensitivity()
    demonstrate_duration_concepts()
    demonstrate_svb_case_study()
    demonstrate_screening_signals()

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Duration gap measures asset-liability rate sensitivity")
    print("  2. Rising rates create unrealized losses on long-duration assets")
    print("  3. HTM accounting can hide economic losses")
    print("  4. Uninsured deposit concentration increases run risk")
    print("  5. SVB-style failure from duration + deposit concentration")


if __name__ == "__main__":
    main()
