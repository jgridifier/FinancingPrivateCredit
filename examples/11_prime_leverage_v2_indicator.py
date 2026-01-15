#!/usr/bin/env python3
"""
Prime Brokerage Leverage Lead Indicator V2 (PB-LLI) Example

This example demonstrates the PB-LLI V2 indicator, a forecasting-centric system
designed to predict prime brokerage performance 1-2 quarters ahead.

Architecture:
1. Quarterly Anchor Index: Fed Z.1 hedge fund leverage & dealer supply data
2. Weekly Nowcast Factor: CFTC COT and NY Fed Primary Dealer data
3. Composite Forecast Signal: Combined output with run-rate regime classification

Key outputs demonstrated:
- PB_Intensity: Hedge fund prime broker borrowing / HF equity
- Dealer_SupplyGrowth: Broker-dealer customer receivables growth
- PB_Lead: Composite leading signal for 1-2 quarter forecasting
- Run-Rate Regimes: Accelerating / Stable / Decelerating
- Attribution: Contribution decomposition by source

References:
- Adrian, T., & Shin, H. S. (2010). "Liquidity and Leverage"
- Fed Z.1 Financial Accounts (hedge fund sector)
- CFTC Traders in Financial Futures (TFF)
- NY Fed Primary Dealer Statistics
"""

from datetime import datetime
import polars as pl

from financing_private_credit.indicators import get_indicator
from financing_private_credit.indicators.prime_leverage_v2 import (
    PrimeLeverageV2Indicator,
    PBLLISpec,
)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def demonstrate_indicator_registration():
    """Verify indicator is properly registered in the framework."""
    print_section("1. INDICATOR REGISTRATION")

    indicator = get_indicator("prime_leverage_v2")
    metadata = indicator.get_metadata()

    print(f"\n  Name: {metadata.name}")
    print(f"  Short Name: {metadata.short_name}")
    print(f"  Version: {metadata.version}")
    print(f"  Update Frequency: {metadata.update_frequency}")
    print(f"  Lookback Periods: {metadata.lookback_periods}")
    print(f"\n  Description: {metadata.description[:100]}...")
    print(f"\n  Data Sources:")
    for source in metadata.data_sources:
        print(f"    - {source[:70]}...")
    print(f"\n  Reference: {metadata.paper_reference}")

    return indicator


def fetch_and_calculate(indicator: PrimeLeverageV2Indicator):
    """Fetch data and calculate the indicator."""
    print_section("2. FETCHING DATA")

    print("\n  Quarterly Fed Z.1 Data:")
    print("    - BOGZ1FL624123035Q: HF Prime Broker Margin Loans")
    print("    - BOGZ1FL624090005Q: HF Total Financial Assets")
    print("    - BOGZ1FL624190005Q: HF Total Financial Liabilities")
    print("    - BOGZ1FL622151005Q: HF Repo Liabilities")
    print("    - BOGZ1FL663067003Q: B-D Customer Receivables")
    print("\n  Weekly Data:")
    print("    - CFTC COT TFF: Leveraged Funds positioning")
    print("    - NY Fed PD: Repo volumes and settlement fails")

    try:
        data = indicator.fetch_data("2005-01-01")

        quarterly_data = data.get("quarterly_data", pl.DataFrame())
        weekly_cot = data.get("weekly_cot", pl.DataFrame())
        weekly_pd = data.get("weekly_pd", pl.DataFrame())

        print(f"\n  Quarterly observations: {quarterly_data.height}")
        print(f"  Weekly COT observations: {weekly_cot.height}")
        print(f"  Weekly PD observations: {weekly_pd.height}")

    except Exception as e:
        print(f"\n  Warning: Error fetching data: {e}")
        print("  Using synthetic data for demonstration...")
        data = create_synthetic_data()

    # Calculate indicator
    print_section("3. CALCULATING PB-LLI")
    result = indicator.calculate(data)

    if result.data.height == 0:
        print("\n  Calculation failed, using synthetic data...")
        data = create_synthetic_data()
        result = indicator.calculate(data)

    return result, data


def create_synthetic_data() -> dict[str, pl.DataFrame]:
    """Create synthetic data for demonstration."""
    import numpy as np

    # Generate quarterly dates from 2005 to present
    dates = pl.date_range(
        datetime(2005, 3, 31),
        datetime.now(),
        interval="1q",
        eager=True,
    )

    n = len(dates)
    np.random.seed(42)

    # Generate realistic hedge fund and broker-dealer data (in millions USD)
    # HF margin loans: ~$200-400B range
    hf_pb_borrowing = 250000 + np.cumsum(np.random.randn(n) * 10000)
    hf_pb_borrowing = np.maximum(hf_pb_borrowing, 150000)

    # HF total assets: ~$4-6T range
    hf_assets = 5000000 + np.cumsum(np.random.randn(n) * 50000)
    hf_assets = np.maximum(hf_assets, 3000000)

    # HF total liabilities: ~$3-4T range (assets - equity)
    hf_liabilities = hf_assets * 0.65 + np.random.randn(n) * 100000

    # HF repo liabilities: ~$500B-1T range
    hf_repo_liab = 750000 + np.cumsum(np.random.randn(n) * 20000)
    hf_repo_liab = np.maximum(hf_repo_liab, 400000)

    # B-D customer receivables: ~$150-300B range
    bd_cust_recv = 200000 + np.cumsum(np.random.randn(n) * 8000)
    bd_cust_recv = np.maximum(bd_cust_recv, 100000)

    # Add crisis effects (2008-2009, 2020)
    crisis_2008 = np.exp(-((np.arange(n) - 14) ** 2) / 8)
    crisis_2020 = np.exp(-((np.arange(n) - 60) ** 2) / 4)

    hf_pb_borrowing = hf_pb_borrowing * (1 - 0.3 * crisis_2008 - 0.15 * crisis_2020)
    bd_cust_recv = bd_cust_recv * (1 - 0.25 * crisis_2008 - 0.10 * crisis_2020)

    quarterly_data = pl.DataFrame({
        "date": dates,
        "BOGZ1FL624123035Q": hf_pb_borrowing,
        "BOGZ1FL624090005Q": hf_assets,
        "BOGZ1FL624190005Q": hf_liabilities,
        "BOGZ1FL622151005Q": hf_repo_liab,
        "BOGZ1FL663067003Q": bd_cust_recv,
    })

    # Generate weekly COT data (synthetic)
    weekly_dates = pl.date_range(
        datetime(2005, 1, 4),
        datetime.now(),
        interval="1w",
        eager=True,
    )
    n_weeks = len(weekly_dates)

    cot_data = []
    for contract in ["ES", "NQ", "TU", "FV", "TY", "US", "EC", "JY"]:
        np.random.seed(hash(contract) % (2**32))
        base = 100000 if contract in ["ES", "TY"] else 50000
        net_positions = np.cumsum(np.random.randn(n_weeks) * base * 0.02)
        z_scores = (net_positions - net_positions.mean()) / net_positions.std()

        for i, d in enumerate(weekly_dates):
            cot_data.append({
                "report_date": d.date(),
                "contract": contract,
                "leveraged_net": int(net_positions[i]),
                "net_pct_oi_z_5y": z_scores[i],
            })

    weekly_cot = pl.DataFrame(cot_data).with_columns(
        pl.col("report_date").cast(pl.Date)
    )

    # Generate weekly NY Fed PD data (synthetic)
    pd_data = []
    for series in ["PD_RP_T_TOT", "PD_RRP_T_TOT", "PD_AFtD_AG", "PD_AFtR_AG"]:
        np.random.seed(hash(series) % (2**32))
        base = 2500 if "RP" in series else 50
        values = base + np.cumsum(np.random.randn(n_weeks) * base * 0.01)
        z_scores = (values - values.mean()) / values.std()

        for i, d in enumerate(weekly_dates):
            pd_data.append({
                "week_ending": d.date(),
                "series": series,
                "value": values[i],
                "value_z_3y": z_scores[i],
            })

    weekly_pd = pl.DataFrame(pd_data).with_columns(
        pl.col("week_ending").cast(pl.Date)
    )

    return {
        "quarterly_data": quarterly_data,
        "weekly_cot": weekly_cot,
        "weekly_pd": weekly_pd,
    }


def display_quarterly_anchor(result):
    """Display the quarterly anchor index values."""
    print_section("4. QUARTERLY ANCHOR INDEX")

    if result.data.height == 0:
        print("\n  No data available")
        return

    df = result.data

    print("\n  Key Quarterly Series (last 12 quarters):")
    print("-" * 100)

    display_df = df.tail(12).select([
        pl.col("date").dt.strftime("%Y-Q%q").alias("Quarter"),
        (pl.col("pb_intensity") * 100).round(2).alias("PB_Intensity_%"),
        (pl.col("pb_intensity_g") * 100).round(2).alias("PB_Int_Growth_%"),
        (pl.col("dealer_supply_g") * 100).round(2).alias("Dealer_Supply_G_%"),
        pl.col("pb_intensity_g_z").round(2).alias("PB_Int_Z"),
        pl.col("dealer_supply_g_z").round(2).alias("Dealer_Z"),
    ])

    print(display_df)


def display_composite_forecast(result):
    """Display the composite PB_Lead forecast signal."""
    print_section("5. COMPOSITE FORECAST SIGNAL")

    if result.data.height == 0:
        print("\n  No data available")
        return

    df = result.data

    print("\n  PB_Lead Composite Signal (last 12 quarters):")
    print("-" * 100)

    display_df = df.tail(12).select([
        pl.col("date").dt.strftime("%Y-Q%q").alias("Quarter"),
        (pl.col("pb_lead") * 100).round(2).alias("PB_Lead_%"),
        pl.col("pb_lead_z").round(2).alias("PB_Lead_Z"),
        pl.col("pb_lead_percentile").round(1).alias("Percentile"),
        pl.col("run_rate_regime").alias("Regime"),
        pl.col("stress_flag").alias("Stress"),
    ])

    print(display_df)

    # Regime distribution
    print("\n  Run-Rate Regime Distribution:")
    print("-" * 50)
    summary = result.metadata.get("summary", {})
    if summary:
        print(f"    Accelerating: {summary.get('pct_accelerating', 0) * 100:.1f}%")
        print(f"    Stable: {summary.get('pct_stable', 0) * 100:.1f}%")
        print(f"    Decelerating: {summary.get('pct_decelerating', 0) * 100:.1f}%")


def display_current_reading(result, indicator: PrimeLeverageV2Indicator):
    """Display current reading with interpretation."""
    print_section("6. CURRENT READING & INTERPRETATION")

    if result.data.height == 0:
        print("\n  No data available")
        return

    current_signal = result.metadata.get("current_signal", {})
    current_regime = result.metadata.get("current_regime")

    print("\n  Current Signal:")
    print("-" * 50)

    signal_emoji = {1: "GREEN", 0: "YELLOW", -1: "RED"}.get(
        current_signal.get("signal", 0), "UNKNOWN"
    )
    print(f"    Signal: [{signal_emoji}] {current_signal.get('reason', 'Unknown')}")
    print(f"    Regime: {current_signal.get('regime', 'Unknown')}")
    print(f"    PB_Lead Z-Score: {current_signal.get('pb_lead_z', 'N/A')}")
    print(f"    Stress Flag: {current_signal.get('stress_flag', False)}")

    # Get interpretation
    if current_regime:
        interpretation = indicator.get_regime_interpretation(current_regime)
        print("\n  Regime Interpretation:")
        print("-" * 50)
        for key, value in interpretation.items():
            print(f"    {key.replace('_', ' ').title()}: {value}")


def display_attribution(result):
    """Display attribution decomposition."""
    print_section("7. ATTRIBUTION DECOMPOSITION")

    attribution = result.metadata.get("attribution", {})

    if not attribution:
        print("\n  No attribution data available")
        return

    print("\n  Current Period Attribution:")
    print("-" * 60)

    print(f"\n  PB_Lead Total: {attribution.get('pb_lead_total', 0) * 100:.2f}%")
    print("\n  Contribution Breakdown:")
    print(f"    HF Demand (Nowcast):     {attribution.get('hf_demand_contribution', 0) * 100:+.2f}%  "
          f"({attribution.get('hf_demand_pct', 0):.1f}%)")
    print(f"    Lagged Intensity:        {attribution.get('lagged_intensity_contribution', 0) * 100:+.2f}%  "
          f"({attribution.get('lagged_intensity_pct', 0):.1f}%)")
    print(f"    Dealer Supply:           {attribution.get('dealer_supply_contribution', 0) * 100:+.2f}%  "
          f"({attribution.get('dealer_supply_pct', 0):.1f}%)")

    # Visual bar chart
    print("\n  Attribution Chart:")
    print("-" * 60)

    for name, pct_key in [
        ("HF Demand", "hf_demand_pct"),
        ("Lagged", "lagged_intensity_pct"),
        ("Dealer", "dealer_supply_pct"),
    ]:
        pct = attribution.get(pct_key, 0)
        bar_len = int(pct / 2)  # Scale to 50 chars max
        bar = "#" * bar_len
        print(f"    {name:12} |{bar:<50}| {pct:.1f}%")


def display_forecasts(result):
    """Display t+1 and t+2 forecasts."""
    print_section("8. FORECASTS")

    forecasts = result.metadata.get("forecasts", {})

    if not forecasts:
        print("\n  No forecast data available")
        return

    print("\n  Quarterly Forecasts (PB_Lead):")
    print("-" * 60)

    for horizon, label in [("t_plus_1", "t+1 (Next Quarter)"), ("t_plus_2", "t+2 (Two Quarters)")]:
        fc = forecasts.get(horizon, {})
        if fc:
            forecast = fc.get("forecast", 0) * 100
            lower = fc.get("lower_bound", 0) * 100
            upper = fc.get("upper_bound", 0) * 100
            print(f"\n  {label}:")
            print(f"    Point Forecast: {forecast:+.2f}%")
            print(f"    95% Confidence: [{lower:+.2f}%, {upper:+.2f}%]")


def display_weekly_nowcast_status(result):
    """Display weekly nowcast availability and status."""
    print_section("9. WEEKLY NOWCAST STATUS")

    weekly_available = result.metadata.get("weekly_nowcast_available", False)

    print(f"\n  Weekly Nowcast Available: {'Yes' if weekly_available else 'No'}")

    if weekly_available:
        print("\n  Weekly Data Sources:")
        print("    - CFTC COT TFF: Leveraged Funds positioning in futures")
        print("    - NY Fed PD: Primary dealer repo/reverse repo and fails")
        print("\n  Nowcast Update Frequency:")
        print("    - CFTC COT: Friday 3:30pm ET (Tuesday data)")
        print("    - NY Fed PD: Thursday ~4:15pm ET (prior week)")
    else:
        print("\n  Note: Weekly nowcast enhances quarterly anchor with")
        print("  higher-frequency signals. Enable by ensuring weekly")
        print("  data sources are properly configured.")


def demonstrate_spec_customization():
    """Show how to customize the indicator specification."""
    print_section("10. SPECIFICATION CUSTOMIZATION")

    print("\n  Default Specification:")
    default_spec = PBLLISpec()
    print(f"    Nowcast Weight: {default_spec.nowcast_weight}")
    print(f"    Lagged Intensity Weight: {default_spec.lagged_intensity_weight}")
    print(f"    Dealer Supply Weight: {default_spec.dealer_supply_weight}")
    print(f"    Accelerating Percentile: {default_spec.accelerating_percentile}")
    print(f"    Decelerating Percentile: {default_spec.decelerating_percentile}")

    print("\n  Custom Specification (more weight on nowcast):")
    custom_spec = PBLLISpec(
        name="nowcast_heavy",
        description="Higher weight on weekly nowcast signal",
        nowcast_weight=0.50,
        lagged_intensity_weight=0.30,
        dealer_supply_weight=0.20,
    )
    print(f"    Nowcast Weight: {custom_spec.nowcast_weight}")
    print(f"    Lagged Intensity Weight: {custom_spec.lagged_intensity_weight}")
    print(f"    Dealer Supply Weight: {custom_spec.dealer_supply_weight}")

    print("\n  Note: Weights should sum to 1.0 and be calibrated via")
    print("  historical backtest on out-of-sample data.")


def main():
    """Main example runner."""
    print("=" * 80)
    print("PRIME BROKERAGE LEVERAGE LEAD INDICATOR V2 (PB-LLI) - MONITORING EXAMPLE")
    print("Forecasting Prime Brokerage Performance 1-2 Quarters Ahead")
    print("=" * 80)

    # 1. Verify registration
    indicator = demonstrate_indicator_registration()

    # 2. Fetch data and calculate
    result, data = fetch_and_calculate(indicator)

    if result is None or result.data.height == 0:
        print("\n  Could not calculate indicator. Check data availability.")
        return

    # 3. Display quarterly anchor
    display_quarterly_anchor(result)

    # 4. Display composite forecast
    display_composite_forecast(result)

    # 5. Current reading and interpretation
    display_current_reading(result, indicator)

    # 6. Attribution decomposition
    display_attribution(result)

    # 7. Forecasts
    display_forecasts(result)

    # 8. Weekly nowcast status
    display_weekly_nowcast_status(result)

    # 9. Specification customization
    demonstrate_spec_customization()

    # Summary
    print_section("SUMMARY")
    print("""
    The Prime Brokerage Leverage Lead Indicator V2 (PB-LLI) provides:

    1. QUARTERLY ANCHOR INDEX (Structural Truth)
       - PB_Intensity: HF prime broker borrowing / HF equity
       - Dealer_SupplyGrowth: B-D customer receivables growth rate
       - Fed Z.1 data with ~6-8 week publication lag

    2. WEEKLY NOWCAST FACTOR (Timeliness Edge)
       - CFTC COT: Leveraged fund positioning (risk appetite proxy)
       - NY Fed PD: Dealer balance sheet intermediation pulse
       - Updates weekly inside the quarter

    3. COMPOSITE FORECAST SIGNAL (PB_Lead)
       - Weighted combination of demand + supply factors
       - 1-2 quarter ahead projections with confidence bands
       - Attribution decomposition for PM insight

    Run-Rate Regimes (Normal-Times Focus):
       ACCELERATING: Top tercile - improving balances/revenue momentum
       STABLE:       Middle tercile - neutral run-rate
       DECELERATING: Bottom tercile - slowing balances/revenue momentum

    Stress Overlay (Secondary Diagnostic):
       Triggered on sharp drops in PB_Intensity or Dealer_Supply
       Widens forecast intervals and adds downside skew note

    Integration Points:
       - Combine with Leverage Cycle Index V1 for comprehensive view
       - Use with Rehypothecation Liquidity Index for funding stress
       - Map to named primes via EDGAR extraction (Phase 2)
    """)

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
