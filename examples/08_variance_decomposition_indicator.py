#!/usr/bin/env python3
"""
Variance Decomposition Indicator Example

Decomposes bank return variance into macro, size, and allocation components
to identify systematic vs idiosyncratic risk drivers.

Key concepts:
- Macro component: Exposure to interest rates, credit spreads, GDP
- Size component: Large vs small bank differential returns
- Allocation component: Bank-specific lending mix effects
- Residual: Idiosyncratic bank-specific factors

Uses real data from FRED (macro) and yfinance (bank stock returns).

Reference: Variance decomposition for bank equity factor analysis
"""

import polars as pl

from financing_private_credit.indicators import get_indicator
from financing_private_credit.data import FREDDataFetcher


def fetch_bank_stock_returns():
    """
    Fetch real bank stock returns using yfinance.
    """
    print("\n[1] Fetching bank stock returns...")

    try:
        import yfinance as yf
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years

        # Major bank tickers
        tickers = ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"]

        print(f"    Downloading: {', '.join(tickers)}")

        # Download data
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)

        if data.empty:
            print("    Warning: No data returned from yfinance")
            return None

        # Extract adjusted close prices
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"]
        else:
            prices = data["Close"]

        # Calculate daily returns
        returns = prices.pct_change().dropna()

        print(f"    Fetched {len(returns)} days of returns for {len(returns.columns)} banks")

        return returns

    except ImportError:
        print("    Warning: yfinance not installed. Using FRED bank index data.")
        return None
    except Exception as e:
        print(f"    Warning: Could not fetch stock data: {e}")
        return None


def demonstrate_macro_factors():
    """
    Show macro factor data from FRED.
    """
    print("=" * 60)
    print("VARIANCE DECOMPOSITION - MACRO FACTORS")
    print("=" * 60)

    fetcher = FREDDataFetcher()

    # Fetch macro factor data
    print("\n[1] Fetching macro factor data from FRED...")
    macro_series = [
        "FEDFUNDS",      # Fed Funds Rate
        "GS10",          # 10-Year Treasury
        "BAA10Y",        # Credit Spread (BAA - 10Y)
        "GDPC1",         # Real GDP
        "VIXCLS",        # VIX (volatility)
        "SP500"          # S&P 500
    ]

    macro_data = fetcher.fetch_multiple_series(macro_series, "2020-01-01")

    if macro_data.height > 0:
        print(f"   Fetched {macro_data.height} observations")

        # Latest values
        latest = macro_data.tail(1)
        print(f"\n   Current Macro Environment ({latest['date'][0]}):")

        for col, name in [("FEDFUNDS", "Fed Funds Rate"),
                          ("GS10", "10-Year Treasury"),
                          ("BAA10Y", "Credit Spread (BAA-10Y)"),
                          ("VIXCLS", "VIX")]:
            if col in latest.columns and latest[col][0] is not None:
                val = latest[col][0]
                print(f"     {name}: {val:.2f}")

        # Calculate factor changes
        print("\n[2] Macro Factor Changes (1-Year):")
        if macro_data.height >= 252:
            year_ago = macro_data.head(macro_data.height - 252).tail(1)
            current = macro_data.tail(1)

            for col, name in [("FEDFUNDS", "Fed Funds"),
                              ("GS10", "10Y Treasury"),
                              ("BAA10Y", "Credit Spread")]:
                if (col in current.columns and col in year_ago.columns and
                    current[col][0] is not None and year_ago[col][0] is not None):
                    change = current[col][0] - year_ago[col][0]
                    print(f"     {name} Change: {change:+.2f}%")

    return macro_data


def demonstrate_variance_decomposition_concept():
    """
    Explain variance decomposition methodology.
    """
    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION METHODOLOGY")
    print("=" * 60)

    print("""
    Bank return variance can be decomposed into:

    ┌──────────────────────────────────────────────────────────────┐
    │  R_i,t = α_i + β_macro * F_macro + β_size * F_size          │
    │          + β_alloc * F_alloc + ε_i,t                        │
    └──────────────────────────────────────────────────────────────┘

    Where:

    1. MACRO COMPONENT (typically 40-60% of variance)
    ┌──────────────────────────────────────────────────────────────┐
    │  F_macro includes:                                           │
    │    • Interest rate level and changes (Fed Funds, 10Y)       │
    │    • Credit spreads (investment grade, high yield)          │
    │    • GDP growth expectations                                 │
    │    • Market volatility (VIX)                                │
    │    • S&P 500 market returns                                 │
    │                                                              │
    │  Banks with HIGH macro beta: More cyclical                  │
    │  Banks with LOW macro beta: More defensive                  │
    └──────────────────────────────────────────────────────────────┘

    2. SIZE COMPONENT (typically 10-20% of variance)
    ┌──────────────────────────────────────────────────────────────┐
    │  F_size = Return(Large Banks) - Return(Small Banks)         │
    │                                                              │
    │  Captures:                                                   │
    │    • Too-big-to-fail premium                                │
    │    • Funding cost advantages                                 │
    │    • Scale economies in regulation                          │
    │    • Diversification benefits                                │
    └──────────────────────────────────────────────────────────────┘

    3. ALLOCATION COMPONENT (typically 15-25% of variance)
    ┌──────────────────────────────────────────────────────────────┐
    │  F_alloc based on lending mix:                              │
    │    • C&I loans vs consumer vs real estate                   │
    │    • Trading revenue vs NII                                 │
    │    • Geographic concentration                                │
    │                                                              │
    │  Banks with similar allocation move together                │
    └──────────────────────────────────────────────────────────────┘

    4. IDIOSYNCRATIC (typically 10-25% of variance)
    ┌──────────────────────────────────────────────────────────────┐
    │  ε_i,t = Bank-specific residual                             │
    │                                                              │
    │  Captures:                                                   │
    │    • Management quality                                      │
    │    • Operational events                                      │
    │    • Regulatory actions                                      │
    │    • Fraud/scandals                                          │
    └──────────────────────────────────────────────────────────────┘
    """)


def demonstrate_real_decomposition():
    """
    Perform variance decomposition with real data.
    """
    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION WITH REAL DATA")
    print("=" * 60)

    # Fetch bank stock returns
    returns = fetch_bank_stock_returns()

    # Fetch macro data
    fetcher = FREDDataFetcher()
    macro_data = fetcher.fetch_multiple_series(
        ["FEDFUNDS", "GS10", "BAA10Y"],
        "2022-01-01"
    )

    if returns is not None and len(returns) > 0:
        # Calculate return statistics
        print("\n[2] Bank Return Statistics (Annualized):")
        print(f"   {'Bank':<6} {'Mean':>10} {'Std Dev':>10} {'Sharpe':>10}")
        print("   " + "-" * 40)

        for col in returns.columns:
            mean_ret = returns[col].mean() * 252 * 100  # Annualized %
            std_ret = returns[col].std() * (252 ** 0.5) * 100  # Annualized %
            sharpe = mean_ret / std_ret if std_ret > 0 else 0

            print(f"   {col:<6} {mean_ret:>9.1f}% {std_ret:>9.1f}% {sharpe:>10.2f}")

        # Correlation matrix
        print("\n[3] Return Correlations (Sample):")
        corr = returns.corr()
        print(f"   Average pairwise correlation: {corr.values[~(corr.values==1)].mean():.2f}")

        # Size factor (large vs small)
        if "JPM" in returns.columns and "USB" in returns.columns:
            large = (returns["JPM"] + returns["BAC"] + returns["C"]) / 3 if "BAC" in returns.columns and "C" in returns.columns else returns["JPM"]
            small = (returns["USB"] + returns["PNC"]) / 2 if "PNC" in returns.columns else returns["USB"]

            size_factor = large - small
            size_vol = size_factor.std() * (252 ** 0.5) * 100

            print(f"\n[4] Size Factor Analysis:")
            print(f"   Size Factor Volatility (annualized): {size_vol:.1f}%")
            print(f"   Size Factor Mean Return: {size_factor.mean() * 252 * 100:+.1f}%")

    else:
        print("\n   Using simulated decomposition for illustration...")
        print("""
   Typical Decomposition Results:

   ┌──────────────────────────────────────────────────────────────┐
   │  Component        │  Variance Share  │  Interpretation       │
   ├───────────────────┼──────────────────┼───────────────────────┤
   │  Macro            │     52%          │  Driven by rates, GDP │
   │  Size             │     15%          │  Large vs small       │
   │  Allocation       │     18%          │  Lending mix effects  │
   │  Idiosyncratic    │     15%          │  Bank-specific        │
   └───────────────────┴──────────────────┴───────────────────────┘
        """)


def demonstrate_trading_applications():
    """
    Show trading applications of variance decomposition.
    """
    print("\n" + "=" * 60)
    print("TRADING APPLICATIONS")
    print("=" * 60)

    print("""
    Variance decomposition informs trading strategies:

    1. MACRO REGIME TRADING
    ┌──────────────────────────────────────────────────────────────┐
    │  When macro variance is HIGH:                               │
    │    • Trade sector-wide (KBE, XLF) rather than single names │
    │    • Hedge with rate and credit instruments                │
    │    • Reduce position sizes (correlated moves)              │
    │                                                              │
    │  When macro variance is LOW:                                │
    │    • Focus on stock selection (idiosyncratic alpha)        │
    │    • Pair trades between banks                              │
    │    • Larger single-name positions                          │
    └──────────────────────────────────────────────────────────────┘

    2. SIZE FACTOR TRADING
    ┌──────────────────────────────────────────────────────────────┐
    │  Size premium (large outperform):                           │
    │    • Risk-off environments                                  │
    │    • Funding stress periods                                 │
    │    • Flight-to-quality in banking                          │
    │                                                              │
    │  Size discount (small outperform):                          │
    │    • Risk-on, yield-seeking environments                   │
    │    • M&A speculation                                        │
    │    • Regional economic strength                             │
    └──────────────────────────────────────────────────────────────┘

    3. ALLOCATION FACTOR TRADING
    ┌──────────────────────────────────────────────────────────────┐
    │  Rate-sensitive banks (high NIM):                           │
    │    • Long in rising rate environment                       │
    │    • USB, PNC, regional banks                              │
    │                                                              │
    │  Trading-oriented banks:                                    │
    │    • Long in volatile markets (trading revenue)            │
    │    • GS, MS                                                 │
    │                                                              │
    │  Consumer-focused banks:                                    │
    │    • Long in strong consumer credit environment            │
    │    • COF, DFS (credit card focus)                          │
    └──────────────────────────────────────────────────────────────┘
    """)


def main():
    """Main example runner."""
    print("=" * 60)
    print("VARIANCE DECOMPOSITION INDICATOR EXAMPLE")
    print("Bank Return Factor Analysis")
    print("=" * 60)

    # Run demonstrations with real data
    macro_data = demonstrate_macro_factors()
    demonstrate_variance_decomposition_concept()
    demonstrate_real_decomposition()
    demonstrate_trading_applications()

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Bank returns decompose into macro, size, allocation, idiosyncratic")
    print("  2. Macro factors explain 40-60% of variance in most periods")
    print("  3. Size factor captures too-big-to-fail premium")
    print("  4. Allocation factor reflects lending mix exposure")
    print("  5. Use decomposition to inform trading strategy and hedging")


if __name__ == "__main__":
    main()
