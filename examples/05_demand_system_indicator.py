#!/usr/bin/env python3
"""
Demand System Indicator Example

Replicates the demand system approach from Boyarchenko & Elias (2024) to
decompose private credit by lender type (banks vs nonbanks) and estimate
supply elasticities using real FRED data.

Key concepts:
- Bank vs Nonbank credit shares over time
- Supply elasticity differences (banks more elastic to rates)
- Credit cycle identification
- Crisis probability estimation

Reference: Boyarchenko & Elias (2024): Private Credit and the Business Cycle
"""

import polars as pl

from financing_private_credit.indicators import get_indicator
from financing_private_credit.data import FREDDataFetcher, PrivateCreditData


def demonstrate_credit_decomposition():
    """
    Demonstrate the credit decomposition by lender type using real FRED data.
    """
    print("=" * 60)
    print("DEMAND SYSTEM - CREDIT DECOMPOSITION")
    print("=" * 60)

    indicator = get_indicator("demand_system")
    metadata = indicator.get_metadata()

    print(f"\nIndicator: {metadata.name}")
    print(f"Description: {metadata.description[:80]}...")
    print(f"Paper: {metadata.paper_reference}")

    # Fetch real credit data from FRED
    print("\n[1] Fetching credit data from FRED...")
    credit_data = PrivateCreditData(start_date="2000-01-01")
    raw_data = credit_data.fetch_all()

    if raw_data.height > 0:
        print(f"   Fetched {raw_data.height} observations")
        print(f"   Date range: {raw_data['date'].min()} to {raw_data['date'].max()}")

    # Get decomposition with derived metrics
    print("\n[2] Computing credit decomposition...")
    decomp = credit_data.compute_credit_decomposition()

    if decomp.height > 0:
        # Get latest quarterly data
        quarterly = decomp.filter(pl.col("date").dt.month().is_in([1, 4, 7, 10])).tail(1)

        if quarterly.height > 0:
            latest_date = quarterly["date"][0]
            print(f"\n   Latest Credit Data (as of {latest_date}):")

            # Bank credit (total loans and leases)
            if "bank_credit" in quarterly.columns:
                bank = quarterly["bank_credit"][0]
                if bank is not None:
                    print(f"     Bank Credit (Loans & Leases): ${bank:,.0f}B")

            # Shadow bank and insurance/pension credit
            if "shadow_bank_credit" in quarterly.columns:
                shadow = quarterly["shadow_bank_credit"][0]
                if shadow is not None:
                    print(f"     Shadow Bank Credit: ${shadow:,.0f}B")

            if "insurance_pension_credit" in quarterly.columns:
                ins = quarterly["insurance_pension_credit"][0]
                if ins is not None:
                    print(f"     Insurance/Pension Credit: ${ins:,.0f}B")

            # Total credit from BIS
            if "total_private_credit_bis" in quarterly.columns:
                total = quarterly["total_private_credit_bis"][0]
                if total is not None:
                    print(f"     Total Private Credit (BIS): ${total:,.0f}B")

    # Calculate credit-to-GDP ratio
    print("\n[3] Computing credit-to-GDP ratio...")
    credit_gdp = credit_data.compute_credit_to_gdp()

    if "bank_credit_to_gdp" in credit_gdp.columns:
        latest = credit_gdp.filter(pl.col("bank_credit_to_gdp").is_not_null()).tail(1)
        if latest.height > 0:
            ratio = latest["bank_credit_to_gdp"][0]
            print(f"     Bank Credit / GDP: {ratio:.1f}%")

    # Get lender shares
    print("\n[4] Computing lender shares over time...")
    shares = credit_data.get_lender_shares()

    if "bank_share" in shares.columns:
        recent = shares.filter(pl.col("bank_share").is_not_null()).tail(20)
        if recent.height > 0:
            avg_share = recent["bank_share"].mean()
            latest_share = recent["bank_share"][-1]
            print(f"     Current Bank Share: {latest_share:.1f}%")
            print(f"     5-Year Average: {avg_share:.1f}%")

            if latest_share > avg_share + 2:
                print("     → Bank share ABOVE trend (procyclical expansion)")
            elif latest_share < avg_share - 2:
                print("     → Bank share BELOW trend (banks contracting)")
            else:
                print("     → Bank share near trend (stable conditions)")

    # Run indicator calculation
    print("\n[5] Running demand system estimation...")
    # Pass raw data (indicator computes its own decomposition)
    result = indicator.calculate({"credit_data": raw_data})

    if result.data.height > 0:
        print(f"   Estimation complete: {result.data.height} periods")

        if "crisis_probability" in result.metadata:
            prob = result.metadata["crisis_probability"]
            print(f"   Crisis Probability: {prob * 100:.1f}%")

    return result


def demonstrate_supply_elasticities():
    """
    Demonstrate supply elasticity estimation using real interest rate data.
    """
    print("\n" + "=" * 60)
    print("SUPPLY ELASTICITY ESTIMATION")
    print("=" * 60)

    print("""
    The demand system estimates how banks and nonbanks respond
    differently to interest rate changes:

    ┌──────────────────────────────────────────────────────────────┐
    │  Supply Elasticity to Interest Rates:                       │
    │                                                              │
    │  Banks:     High elasticity (≈ 1.5-2.0)                     │
    │             → Contract lending when rates rise              │
    │             → Subject to deposit funding costs              │
    │             → More sensitive to Fed policy                  │
    │                                                              │
    │  Nonbanks:  Low elasticity (≈ 0.3-0.5)                      │
    │             → Less sensitive to rate changes                │
    │             → Funded by institutional investors             │
    │             → Fill gaps when banks contract                 │
    └──────────────────────────────────────────────────────────────┘
    """)

    # Fetch real rate data from FRED
    print("\n[1] Fetching interest rate data from FRED...")
    fetcher = FREDDataFetcher()

    rate_series = ["FEDFUNDS", "GS10", "BAA10Y", "DFF"]
    rates = fetcher.fetch_multiple_series(rate_series, "2020-01-01")

    if rates.height > 0:
        print(f"   Fetched {rates.height} observations")

        # Get latest values
        latest = rates.filter(
            pl.col("FEDFUNDS").is_not_null() | pl.col("GS10").is_not_null()
        ).tail(1)

        if latest.height > 0:
            print(f"\n   Current Rate Environment (as of {latest['date'][0]}):")

            if "FEDFUNDS" in latest.columns and latest["FEDFUNDS"][0] is not None:
                ff = latest["FEDFUNDS"][0]
                print(f"     Fed Funds Rate: {ff:.2f}%")

            if "GS10" in latest.columns and latest["GS10"][0] is not None:
                gs10 = latest["GS10"][0]
                print(f"     10-Year Treasury: {gs10:.2f}%")

            if "BAA10Y" in latest.columns and latest["BAA10Y"][0] is not None:
                spread = latest["BAA10Y"][0]
                print(f"     Credit Spread (BAA-10Y): {spread:.2f}%")

            # Calculate term spread
            if ("FEDFUNDS" in latest.columns and "GS10" in latest.columns and
                latest["FEDFUNDS"][0] is not None and latest["GS10"][0] is not None):
                term_spread = latest["GS10"][0] - latest["FEDFUNDS"][0]
                print(f"     Term Spread (10Y-FF): {term_spread:.2f}%")

                if term_spread < 0:
                    print("\n   → INVERTED YIELD CURVE")
                    print("   → Historically signals recession risk")
                    print("   → Banks may tighten lending standards")

            # Rate regime interpretation
            ff_val = latest["FEDFUNDS"][0] if "FEDFUNDS" in latest.columns else None
            if ff_val is not None:
                print("\n   Rate Regime Implications:")
                if ff_val > 4.5:
                    print("   → HIGH RATES: Banks contracting, nonbanks gaining share")
                elif ff_val > 2.5:
                    print("   → MODERATE RATES: Balanced credit provision")
                else:
                    print("   → LOW RATES: Banks expanding aggressively")


def demonstrate_credit_growth():
    """
    Demonstrate credit growth analysis with real H.8 data.
    """
    print("\n" + "=" * 60)
    print("CREDIT GROWTH ANALYSIS (H.8 Data)")
    print("=" * 60)

    fetcher = FREDDataFetcher()

    # Fetch H.8 bank credit data (weekly)
    print("\n[1] Fetching H.8 bank credit data...")
    h8_series = ["TOTLL", "BUSLOANS", "CONSUMER", "REALLN"]
    h8_data = fetcher.fetch_multiple_series(h8_series, "2022-01-01")

    if h8_data.height > 0:
        print(f"   Fetched {h8_data.height} weeks of data")

        # Calculate year-over-year growth
        print("\n   Credit Growth Rates (YoY):")
        for series in h8_series:
            if series in h8_data.columns:
                # Get data with sufficient history
                df = h8_data.filter(pl.col(series).is_not_null())
                if df.height >= 52:  # Need at least 1 year of data
                    current = df[series][-1]
                    year_ago = df[series][-52]
                    if current and year_ago and year_ago > 0:
                        growth = (current / year_ago - 1) * 100
                        series_names = {
                            "TOTLL": "Total Loans & Leases",
                            "BUSLOANS": "C&I Loans",
                            "CONSUMER": "Consumer Loans",
                            "REALLN": "Real Estate Loans"
                        }
                        print(f"     {series_names.get(series, series)}: {growth:+.1f}%")

        # Weekly momentum
        print("\n   Recent Weekly Momentum:")
        if "TOTLL" in h8_data.columns:
            recent = h8_data.filter(pl.col("TOTLL").is_not_null()).tail(4)
            if recent.height >= 2:
                first = recent["TOTLL"][0]
                last = recent["TOTLL"][-1]
                if first and last:
                    monthly_change = (last / first - 1) * 100
                    print(f"     4-Week Change in Total Loans: {monthly_change:+.2f}%")


def demonstrate_credit_to_gdp():
    """
    Demonstrate credit-to-GDP analysis for cycle identification.
    """
    print("\n" + "=" * 60)
    print("CREDIT-TO-GDP CYCLE ANALYSIS")
    print("=" * 60)

    credit_data = PrivateCreditData(start_date="1990-01-01")

    print("\n[1] Fetching historical credit-to-GDP data...")
    credit_gdp = credit_data.compute_credit_to_gdp()

    if "CRDQUSAPABIS_to_gdp" in credit_gdp.columns:
        df = credit_gdp.filter(pl.col("CRDQUSAPABIS_to_gdp").is_not_null())

        if df.height > 0:
            print(f"   Data range: {df['date'].min()} to {df['date'].max()}")

            # Calculate statistics
            latest = df["CRDQUSAPABIS_to_gdp"][-1]
            historical_mean = df["CRDQUSAPABIS_to_gdp"].mean()
            historical_std = df["CRDQUSAPABIS_to_gdp"].std()
            historical_max = df["CRDQUSAPABIS_to_gdp"].max()

            print(f"\n   Credit-to-GDP Statistics:")
            print(f"     Current: {latest:.1f}%")
            print(f"     Historical Mean: {historical_mean:.1f}%")
            print(f"     Historical Std Dev: {historical_std:.1f}%")
            print(f"     Historical Max: {historical_max:.1f}%")

            # Gap from trend (simple deviation from mean)
            gap = latest - historical_mean
            gap_zscore = gap / historical_std if historical_std > 0 else 0

            print(f"\n   Cycle Position:")
            print(f"     Gap from Mean: {gap:+.1f}%")
            print(f"     Z-Score: {gap_zscore:+.2f}")

            if gap_zscore > 1.5:
                print("     Status: 🔴 CREDIT BOOM (elevated risk)")
            elif gap_zscore > 0.5:
                print("     Status: 🟠 ABOVE TREND (monitoring)")
            elif gap_zscore > -0.5:
                print("     Status: 🟢 NORMAL")
            else:
                print("     Status: 🔵 BELOW TREND (credit contraction)")


def main():
    """Main example runner."""
    print("=" * 60)
    print("DEMAND SYSTEM INDICATOR EXAMPLE")
    print("Credit Decomposition by Lender Type")
    print("=" * 60)

    # Run demonstrations with real data
    demonstrate_credit_decomposition()
    demonstrate_supply_elasticities()
    demonstrate_credit_growth()
    demonstrate_credit_to_gdp()

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Real-time FRED data shows current credit conditions")
    print("  2. H.8 weekly data provides high-frequency bank credit signals")
    print("  3. Credit-to-GDP gap identifies cycle position")
    print("  4. Bank vs nonbank shares reveal credit regime")
    print("  5. Rate environment affects bank lending elasticity")


if __name__ == "__main__":
    main()
