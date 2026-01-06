#!/usr/bin/env python3
"""
Bank Macro Sensitivity Indicator Example

Measures bank-specific elasticities to macro variables (rates, output gap,
inflation) using real FRED data and SEC EDGAR bank financials.

Key concepts:
- Bank-specific rate sensitivity (NIM response to rate changes)
- Output gap sensitivity (lending in expansions vs recessions)
- Regime advantages (rising rates, falling rates, expansion, recession)
- Trading signals based on macro regime identification

Reference: Bank equity trading based on macro regime identification
"""

import polars as pl

from financing_private_credit.indicators import get_indicator
from financing_private_credit.data import FREDDataFetcher
from financing_private_credit.bank_data import BankDataCollector


def demonstrate_macro_environment():
    """
    Show current macro environment for sensitivity analysis.
    """
    print("=" * 60)
    print("BANK MACRO SENSITIVITY - CURRENT ENVIRONMENT")
    print("=" * 60)

    fetcher = FREDDataFetcher()

    # Fetch key macro variables
    print("\n[1] Fetching macro data from FRED...")
    macro_series = [
        "FEDFUNDS",     # Fed Funds Rate
        "GDPC1",        # Real GDP
        "GDPPOT",       # Potential GDP
        "CPIAUCSL",     # CPI
        "GS10",         # 10-Year Treasury
        "T10Y2Y",       # 10Y-2Y Spread (term structure)
    ]

    macro_data = fetcher.fetch_multiple_series(macro_series, "2020-01-01")

    if macro_data.height > 0:
        print(f"   Fetched {macro_data.height} observations")

        # Latest values
        latest = macro_data.filter(pl.col("FEDFUNDS").is_not_null()).tail(1)

        if latest.height > 0:
            print(f"\n   Current Macro Conditions ({latest['date'][0]}):")

            # Fed Funds
            if "FEDFUNDS" in latest.columns and latest["FEDFUNDS"][0]:
                ff = latest["FEDFUNDS"][0]
                print(f"     Fed Funds Rate: {ff:.2f}%")

                if ff > 4.5:
                    print("       → HIGH RATE REGIME")
                elif ff > 2.5:
                    print("       → MODERATE RATE REGIME")
                else:
                    print("       → LOW RATE REGIME")

            # Term spread
            if "T10Y2Y" in latest.columns and latest["T10Y2Y"][0] is not None:
                spread = latest["T10Y2Y"][0]
                print(f"     Term Spread (10Y-2Y): {spread:.2f}%")

                if spread < 0:
                    print("       → INVERTED CURVE (recession signal)")
                elif spread < 0.5:
                    print("       → FLAT CURVE (caution)")
                else:
                    print("       → NORMAL CURVE (expansion supportive)")

            # 10-Year
            if "GS10" in latest.columns and latest["GS10"][0]:
                gs10 = latest["GS10"][0]
                print(f"     10-Year Treasury: {gs10:.2f}%")

    return macro_data


def calculate_output_gap():
    """
    Calculate output gap from real FRED data.
    """
    print("\n[2] Calculating Output Gap...")

    fetcher = FREDDataFetcher()

    # Fetch GDP data
    gdp_data = fetcher.fetch_multiple_series(["GDPC1", "GDPPOT"], "2015-01-01")

    if gdp_data.height > 0:
        # Calculate output gap = (Actual GDP - Potential GDP) / Potential GDP
        df = gdp_data.filter(
            pl.col("GDPC1").is_not_null() & pl.col("GDPPOT").is_not_null()
        )

        if df.height > 0:
            latest = df.tail(1)
            actual = latest["GDPC1"][0]
            potential = latest["GDPPOT"][0]

            if actual and potential:
                output_gap = (actual - potential) / potential * 100
                print(f"     Actual GDP: ${actual:,.0f}B")
                print(f"     Potential GDP: ${potential:,.0f}B")
                print(f"     Output Gap: {output_gap:+.2f}%")

                if output_gap > 1:
                    print("       → EXPANSION (above potential)")
                elif output_gap > -1:
                    print("       → NEUTRAL (near potential)")
                else:
                    print("       → CONTRACTION (below potential)")

                return output_gap

    return None


def demonstrate_bank_sensitivities():
    """
    Show bank-specific macro sensitivities.
    """
    print("\n" + "=" * 60)
    print("BANK-SPECIFIC MACRO SENSITIVITIES")
    print("=" * 60)

    # Fetch bank data from SEC EDGAR
    print("\n[1] Fetching bank NIM data from SEC EDGAR...")
    collector = BankDataCollector(start_date="2020-01-01")

    banks = ["JPM", "BAC", "WFC", "C", "USB", "PNC"]
    bank_dfs = []

    for ticker in banks:
        try:
            df = collector.fetch_bank_data(ticker)
            if df.height > 0:
                bank_dfs.append(df)
                print(f"    {ticker}: {df.height} quarters")
        except Exception:
            pass

    if bank_dfs:
        panel = pl.concat(bank_dfs, how="diagonal")
        panel = collector.compute_derived_metrics(panel)

        # Show NIM if available
        if "nim" in panel.columns:
            print("\n   Bank NIM Comparison (Latest):")
            print(f"   {'Bank':<6} {'NIM (%)':>10}")
            print("   " + "-" * 18)

            for ticker in banks:
                bank_data = panel.filter(
                    (pl.col("ticker") == ticker) & pl.col("nim").is_not_null()
                ).tail(1)
                if bank_data.height > 0:
                    nim = bank_data["nim"][0]
                    print(f"   {ticker:<6} {nim:>9.2f}%")

    # Show conceptual sensitivity profiles
    print("""
    Bank Macro Sensitivity Profiles:

    ┌──────────────────────────────────────────────────────────────┐
    │  Bank   │  Rate     │  Output   │  Inflation  │ Best Regime │
    │         │  Sens.    │  Sens.    │  Sens.      │             │
    ├─────────┼───────────┼───────────┼─────────────┼─────────────┤
    │  USB    │  HIGH+    │  MEDIUM   │  LOW        │ Rising Rates│
    │  PNC    │  HIGH+    │  MEDIUM   │  LOW        │ Rising Rates│
    │  WFC    │  HIGH     │  LOW      │  NEUTRAL    │ Rising Rates│
    │  BAC    │  MEDIUM   │  MEDIUM   │  LOW-       │ Mixed       │
    │  JPM    │  MEDIUM   │  HIGH     │  LOW-       │ Expansion   │
    │  C      │  LOW      │  HIGH+    │  LOW-       │ Expansion   │
    └─────────┴───────────┴───────────┴─────────────┴─────────────┘

    Interpretation:
    • HIGH+ Rate Sensitivity: NIM expands significantly in rising rates
    • HIGH Output Sensitivity: Revenues tied to capital markets activity
    • Trading banks (C, JPM) less rate-sensitive due to trading income
    • Regional banks (USB, PNC) more rate-sensitive due to NIM focus
    """)


def demonstrate_regime_trading():
    """
    Show trading strategy based on macro regime.
    """
    print("\n" + "=" * 60)
    print("REGIME-BASED TRADING STRATEGY")
    print("=" * 60)

    # Get current macro conditions
    fetcher = FREDDataFetcher()
    rates = fetcher.fetch_multiple_series(["FEDFUNDS", "T10Y2Y"], "2024-01-01")

    current_regime = "NEUTRAL"
    if rates.height > 0:
        latest = rates.filter(pl.col("FEDFUNDS").is_not_null()).tail(1)
        if latest.height > 0:
            ff = latest["FEDFUNDS"][0]
            spread = latest["T10Y2Y"][0] if "T10Y2Y" in latest.columns else None

            print(f"\n   Current Regime Analysis:")
            print(f"     Fed Funds: {ff:.2f}%" if ff else "     Fed Funds: N/A")
            print(f"     Term Spread: {spread:.2f}%" if spread else "     Term Spread: N/A")

            # Determine regime
            if ff and ff > 4.5:
                current_regime = "HIGH_RATES"
            elif ff and ff < 2.0:
                current_regime = "LOW_RATES"

            if spread and spread < 0:
                current_regime = "RECESSION_RISK"

            print(f"     Identified Regime: {current_regime}")

    print(f"""
    Trading Recommendations for {current_regime} Regime:

    ┌──────────────────────────────────────────────────────────────┐
    │  RISING/HIGH RATES Regime:                                  │
    │    OVERWEIGHT: USB, PNC, WFC (high rate sensitivity)       │
    │    UNDERWEIGHT: C (low rate sensitivity)                   │
    │    Rationale: Asset-sensitive banks benefit from NIM       │
    │                                                              │
    │  FALLING/LOW RATES Regime:                                  │
    │    OVERWEIGHT: C, JPM (trading income offsets NIM)         │
    │    UNDERWEIGHT: USB, PNC (NIM compression)                 │
    │    Rationale: Diversified banks weather NIM pressure       │
    │                                                              │
    │  EXPANSION Regime (Output Gap > 0):                         │
    │    OVERWEIGHT: C, JPM (M&A, capital markets activity)      │
    │    UNDERWEIGHT: Defensive banks                             │
    │    Rationale: Investment banking revenues surge            │
    │                                                              │
    │  RECESSION RISK (Inverted Curve):                           │
    │    OVERWEIGHT: WFC, USB (consumer focus, defensive)        │
    │    UNDERWEIGHT: C, GS (cyclical trading exposure)          │
    │    Rationale: Credit quality focus, stable NIM             │
    └──────────────────────────────────────────────────────────────┘
    """)


def demonstrate_historical_performance():
    """
    Show historical regime performance using real data.
    """
    print("\n" + "=" * 60)
    print("HISTORICAL REGIME PERFORMANCE")
    print("=" * 60)

    fetcher = FREDDataFetcher()

    # Fetch historical Fed Funds
    print("\n[1] Analyzing historical rate regimes...")
    rates = fetcher.fetch_multiple_series(["FEDFUNDS"], "2015-01-01")

    if rates.height > 0:
        # Identify regime periods
        print("\n   Key Rate Regime Periods:")

        # 2015-2018: Hiking cycle
        hiking = rates.filter(
            (pl.col("date") >= pl.lit("2015-12-01").str.to_date()) &
            (pl.col("date") <= pl.lit("2018-12-01").str.to_date())
        )
        if hiking.height > 0:
            start_rate = hiking["FEDFUNDS"][0]
            end_rate = hiking["FEDFUNDS"][-1]
            if start_rate and end_rate:
                print(f"     2015-2018 Hiking: {start_rate:.2f}% → {end_rate:.2f}%")
                print("       Winners: USB, PNC, WFC (rate-sensitive)")

        # 2019-2020: Cutting + COVID
        cutting = rates.filter(
            (pl.col("date") >= pl.lit("2019-07-01").str.to_date()) &
            (pl.col("date") <= pl.lit("2020-06-01").str.to_date())
        )
        if cutting.height > 0:
            start_rate = cutting.filter(pl.col("FEDFUNDS").is_not_null())["FEDFUNDS"][0]
            end_rate = cutting.filter(pl.col("FEDFUNDS").is_not_null())["FEDFUNDS"][-1]
            if start_rate and end_rate:
                print(f"     2019-2020 Cutting: {start_rate:.2f}% → {end_rate:.2f}%")
                print("       Winners: C, JPM (diversified)")

        # 2022-2023: Aggressive hiking
        recent = rates.filter(
            (pl.col("date") >= pl.lit("2022-03-01").str.to_date()) &
            (pl.col("date") <= pl.lit("2023-07-01").str.to_date())
        )
        if recent.height > 0:
            start_rate = recent.filter(pl.col("FEDFUNDS").is_not_null())["FEDFUNDS"][0]
            end_rate = recent.filter(pl.col("FEDFUNDS").is_not_null())["FEDFUNDS"][-1]
            if start_rate and end_rate:
                print(f"     2022-2023 Hiking: {start_rate:.2f}% → {end_rate:.2f}%")
                print("       Winners: USB, PNC (initially)")
                print("       Losers: SVB, regional banks (duration mismatch)")


def main():
    """Main example runner."""
    print("=" * 60)
    print("BANK MACRO SENSITIVITY INDICATOR EXAMPLE")
    print("Rate and Output Gap Sensitivity Analysis")
    print("=" * 60)

    # Run demonstrations with real data
    demonstrate_macro_environment()
    calculate_output_gap()
    demonstrate_bank_sensitivities()
    demonstrate_regime_trading()
    demonstrate_historical_performance()

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Banks have different sensitivities to rates, GDP, inflation")
    print("  2. Rate-sensitive banks (USB, PNC, WFC) benefit from rising rates")
    print("  3. Diversified banks (C, JPM) weather rate cuts better")
    print("  4. Use macro regime identification for equity rotation")
    print("  5. Term spread inversion signals defensive positioning")


if __name__ == "__main__":
    main()
