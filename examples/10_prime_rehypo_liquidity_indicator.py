#!/usr/bin/env python3
"""
Prime Rehypothecation Liquidity Index (RLI) Example

This example demonstrates the RLI indicator using real broker-dealer data from FRED
to measure liquidity creation through collateral rehypothecation.

RLI = (Margin_Receivables + Repo_Liabilities) × Haircut_Spread

Key signals:
- RLI_Velocity < -20%: Severe liquidity contraction
- Haircut_Spread > 6%: Extreme stress regime
- RLI_Normalized < 1%: Below crisis levels (compare to 2008: 0.8%)

References:
- Eren, E. (2014). "Intermediary Funding Liquidity and Rehypothecation"
- Singh, M., & Aitken, J. (2010). "The (Sizable) Role of Rehypothecation"
- Infante, S. (2015). "Liquidity Windfalls: The Consequences of Repo Rehypothecation"
"""

from datetime import datetime
import polars as pl

from financing_private_credit.indicators import get_indicator
from financing_private_credit.indicators.prime_rehypo_liquidity import (
    PrimeRehypoLiquidityIndicator,
    RehypoLiquiditySpec,
)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def demonstrate_indicator_registration():
    """Verify indicator is properly registered in the framework."""
    print_section("1. INDICATOR REGISTRATION")

    indicator = get_indicator("prime_rehypo_liquidity")
    metadata = indicator.get_metadata()

    print(f"\n  Name: {metadata.name}")
    print(f"  Short Name: {metadata.short_name}")
    print(f"  Version: {metadata.version}")
    print(f"  Update Frequency: {metadata.update_frequency}")
    print(f"  Lookback Periods: {metadata.lookback_periods}")
    print(f"\n  Description: {metadata.description[:100]}...")
    print(f"\n  Data Sources: {', '.join(metadata.data_sources)}")
    print(f"  Reference: {metadata.paper_reference}")

    return indicator


def fetch_and_calculate(indicator: PrimeRehypoLiquidityIndicator):
    """Fetch real data and calculate the indicator."""
    print_section("2. FETCHING REAL DATA FROM FRED")

    print("\n  Fetching broker-dealer and VIX data...")
    print("  Series:")
    print("    - BOGZ1FL663067003Q: B-D Margin Receivables (quarterly)")
    print("    - BOGZ1FL662151003Q: B-D Repo Liabilities (quarterly)")
    print("    - BOGZ1FL664090005Q: B-D Total Assets (quarterly)")
    print("    - VIXCLS: CBOE Volatility Index (daily)")

    try:
        data = indicator.fetch_data("2000-01-01")
        macro_data = data.get("macro_data", pl.DataFrame())

        if macro_data.height > 0:
            print(f"\n  ✓ Fetched {macro_data.height} observations")
            print(f"  Columns: {macro_data.columns}")
        else:
            print("\n  ⚠ No data fetched from FRED")
            return None, None

    except Exception as e:
        print(f"\n  ⚠ Error fetching data: {e}")
        print("  Using synthetic data for demonstration...")
        data, macro_data = create_synthetic_data()

    # Calculate indicator
    print_section("3. CALCULATING RLI")
    result = indicator.calculate(data)

    if result.data.height == 0:
        print("\n  ⚠ Calculation failed, using synthetic data...")
        data, macro_data = create_synthetic_data()
        result = indicator.calculate(data)

    return result, data


def create_synthetic_data() -> tuple[dict, pl.DataFrame]:
    """Create synthetic data for demonstration when FRED is unavailable."""
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

    # Generate realistic broker-dealer data (in millions USD)
    # Base trends with cyclical components
    base_margin = 150000 + np.cumsum(np.random.randn(n) * 5000)
    base_repo = 200000 + np.cumsum(np.random.randn(n) * 8000)
    base_assets = 500000 + np.cumsum(np.random.randn(n) * 10000)

    # Add crisis dips (2008-2009, 2020)
    crisis_2008 = np.exp(-((np.arange(n) - 14) ** 2) / 8) * 80000
    crisis_2020 = np.exp(-((np.arange(n) - 60) ** 2) / 4) * 50000

    margin_recv = np.maximum(base_margin - crisis_2008 - crisis_2020, 50000)
    repo_liab = np.maximum(base_repo - crisis_2008 * 1.2 - crisis_2020, 80000)
    total_assets = np.maximum(base_assets - crisis_2008 * 0.5 - crisis_2020 * 0.3, 300000)

    # Generate VIX with spikes during crises
    base_vix = 18 + np.random.randn(n) * 3
    vix_spike_2008 = np.exp(-((np.arange(n) - 14) ** 2) / 4) * 50
    vix_spike_2020 = np.exp(-((np.arange(n) - 60) ** 2) / 2) * 40
    vix = np.clip(base_vix + vix_spike_2008 + vix_spike_2020, 10, 80)

    macro_data = pl.DataFrame({
        "date": dates,
        "BOGZ1FL663067003Q": margin_recv,
        "BOGZ1FL662151003Q": repo_liab,
        "BOGZ1FL664090005Q": total_assets,
        "VIXCLS": vix,
    })

    return {"macro_data": macro_data}, macro_data


def display_historical_readings(result):
    """Display historical RLI readings in a polars table."""
    print_section("4. HISTORICAL READINGS")

    if result.data.height == 0:
        print("\n  No data available")
        return

    df = result.data

    # Show full historical data (last 20 quarters)
    print("\n  Last 20 Quarters of RLI Data:")
    print("-" * 100)

    display_df = df.tail(20).select([
        pl.col("date").dt.strftime("%Y-Q%q").alias("Quarter"),
        (pl.col("margin_receivables") / 1000).round(1).alias("Margin_Recv_$B"),
        (pl.col("repo_liabilities") / 1000).round(1).alias("Repo_Liab_$B"),
        (pl.col("collateral_base") / 1000).round(1).alias("Coll_Base_$B"),
        pl.col("vix_avg").round(1).alias("VIX_Avg"),
        pl.col("vol_regime").alias("Vol_Regime"),
        (pl.col("haircut_spread") * 100).round(2).alias("Haircut_%"),
        (pl.col("rli") / 1000).round(2).alias("RLI_$B"),
        (pl.col("rli_normalized") * 100).round(2).alias("RLI_Norm_%"),
        pl.col("rli_velocity").round(1).alias("Velocity_%"),
    ])

    print(display_df)

    # Summary statistics
    print("\n  Summary Statistics:")
    print("-" * 50)
    summary = result.metadata.get("summary", {})
    if summary:
        print(f"    RLI Mean: ${summary.get('rli_mean', 0) / 1000:.1f}B")
        print(f"    RLI Std Dev: ${summary.get('rli_std', 0) / 1000:.1f}B")
        print(f"    RLI Range: ${summary.get('rli_min', 0) / 1000:.1f}B - ${summary.get('rli_max', 0) / 1000:.1f}B")
        print(f"    RLI 10th Percentile: ${summary.get('rli_p10', 0) / 1000:.1f}B")
        print(f"    RLI 90th Percentile: ${summary.get('rli_p90', 0) / 1000:.1f}B")
        print(f"    High Vol Regime %: {summary.get('pct_high_vol_regime', 0) * 100:.1f}%")


def benchmark_against_crisis_events(result, indicator: PrimeRehypoLiquidityIndicator):
    """Compare current and historical readings against crisis benchmarks."""
    print_section("5. BENCHMARK TESTING: CRISIS EVENTS")

    if result.data.height == 0:
        print("\n  No data available")
        return

    df = result.data
    benchmarks = indicator.get_historical_benchmarks()

    print("\n  Historical Crisis Benchmarks:")
    print("-" * 70)

    benchmark_data = []
    for event_name, benchmark in benchmarks.items():
        if event_name == "normal_conditions":
            continue
        benchmark_data.append({
            "Event": benchmark.get("description", event_name),
            "Expected_Velocity_%": benchmark.get("expected_rli_velocity", "N/A"),
            "Expected_Haircut_%": f"{benchmark.get('expected_haircut_spread', 0) * 100:.1f}" if benchmark.get('expected_haircut_spread') else "N/A",
            "Notes": benchmark.get("notes", ""),
        })

    benchmark_df = pl.DataFrame(benchmark_data)
    print(benchmark_df)

    # Find periods that match crisis patterns
    print("\n  Historical Periods Matching Crisis Patterns:")
    print("-" * 70)

    crisis_periods = df.filter(
        (pl.col("rli_velocity") < -15) | (pl.col("haircut_spread") > 0.05)
    ).select([
        pl.col("date").dt.strftime("%Y-Q%q").alias("Quarter"),
        pl.col("rli_velocity").round(1).alias("Velocity_%"),
        (pl.col("haircut_spread") * 100).round(2).alias("Haircut_%"),
        pl.col("vol_regime").alias("Regime"),
    ])

    if crisis_periods.height > 0:
        print(crisis_periods)
    else:
        print("  No crisis-level periods detected in data.")


def display_current_reading(result, indicator: PrimeRehypoLiquidityIndicator):
    """Display current reading with interpretation."""
    print_section("6. CURRENT READING & INTERPRETATION")

    if result.data.height == 0:
        print("\n  No data available")
        return

    df = result.data
    latest = df.tail(1)

    print("\n  Current Quarter Values:")
    print("-" * 50)

    # Extract current values
    current_date = latest["date"][0]
    margin_recv = latest["margin_receivables"][0]
    repo_liab = latest["repo_liabilities"][0]
    collateral_base = latest["collateral_base"][0]
    vix_avg = latest["vix_avg"][0]
    vol_regime = latest["vol_regime"][0]
    haircut_spread = latest["haircut_spread"][0]
    rli = latest["rli"][0]
    rli_normalized = latest["rli_normalized"][0]
    rli_velocity = latest["rli_velocity"][0]

    print(f"    Date: {current_date}")
    print(f"    Margin Receivables: ${margin_recv / 1000:.1f}B")
    print(f"    Repo Liabilities: ${repo_liab / 1000:.1f}B")
    print(f"    Collateral Base: ${collateral_base / 1000:.1f}B")
    print(f"    VIX Average: {vix_avg:.1f}")
    print(f"    Volatility Regime: {vol_regime}")
    print(f"    Haircut Spread: {haircut_spread * 100:.2f}%")
    print(f"    RLI: ${rli / 1000:.2f}B")
    print(f"    RLI Normalized: {rli_normalized * 100:.2f}% of B-D assets")
    print(f"    RLI Velocity: {rli_velocity:.1f}% QoQ")

    # Get current regime and signal from metadata
    current_regime = result.metadata.get("current_regime", {})
    current_signal = result.metadata.get("current_signal", {})

    print("\n  Current Regime:")
    print("-" * 50)
    print(f"    Volatility Regime: {current_regime.get('vol_regime', 'Unknown')}")
    print(f"    VIX Level: {current_regime.get('vix_avg', 'N/A')}")
    print(f"    Haircut Spread: {current_regime.get('haircut_spread_pct', 'N/A')}")

    print("\n  Signal Assessment:")
    print("-" * 50)
    signal_value = current_signal.get("signal", 0)
    signal_emoji = "🟢" if signal_value >= 0 else "🔴"
    print(f"    Signal: {signal_emoji} {current_signal.get('reason', 'Unknown')}")

    if current_signal.get("all_signals"):
        print("\n    All Active Signals:")
        for sig in current_signal["all_signals"]:
            severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(
                sig.get("severity"), "⚪"
            )
            print(f"      {severity_emoji} {sig.get('type')}: {sig.get('message')}")

    # Get interpretation
    if rli_velocity is not None and haircut_spread is not None:
        interpretation = indicator.get_stress_interpretation(rli_velocity, haircut_spread)
        print("\n  Interpretation:")
        print("-" * 50)
        print(f"    Velocity: {interpretation['velocity_interpretation']}")
        print(f"    Spread: {interpretation['spread_interpretation']}")
        print(f"    Overall Assessment: {interpretation['overall']}")


def demonstrate_spec_customization():
    """Show how to customize the indicator specification."""
    print_section("7. SPECIFICATION CUSTOMIZATION")

    print("\n  Default Specification:")
    default_spec = RehypoLiquiditySpec()
    print(f"    Low Vol Base Spread (α_low): {default_spec.alpha_low * 100:.1f}%")
    print(f"    Low Vol Beta (β_low): {default_spec.beta_low * 100:.2f}% per VIX point")
    print(f"    High Vol Base Spread (α_high): {default_spec.alpha_high * 100:.1f}%")
    print(f"    High Vol Beta (β_high): {default_spec.beta_high * 100:.2f}% per VIX point")
    print(f"    VIX Regime Threshold: {default_spec.vix_regime_threshold}")

    print("\n  Conservative Specification (wider spreads):")
    conservative_spec = RehypoLiquiditySpec(
        name="conservative",
        description="Conservative haircut spread assumptions",
        alpha_low=0.025,  # 2.5% base in low vol
        beta_low=0.001,  # 1 bp per VIX point
        alpha_high=0.04,  # 4% base in high vol
        beta_high=0.003,  # 3 bps per VIX point
    )
    print(f"    Low Vol Base Spread: {conservative_spec.alpha_low * 100:.1f}%")
    print(f"    High Vol Base Spread: {conservative_spec.alpha_high * 100:.1f}%")

    print("\n  Aggressive Specification (narrower spreads):")
    aggressive_spec = RehypoLiquiditySpec(
        name="aggressive",
        description="Aggressive haircut spread assumptions",
        alpha_low=0.015,  # 1.5% base in low vol
        beta_low=0.0003,  # 0.3 bps per VIX point
        alpha_high=0.025,  # 2.5% base in high vol
        beta_high=0.0015,  # 1.5 bps per VIX point
    )
    print(f"    Low Vol Base Spread: {aggressive_spec.alpha_low * 100:.1f}%")
    print(f"    High Vol Base Spread: {aggressive_spec.alpha_high * 100:.1f}%")

    print("\n  Note: Different specifications allow sensitivity analysis")
    print("  comparing RLI under various haircut assumptions.")


def run_sensitivity_analysis(indicator, data):
    """Run sensitivity analysis with different spec assumptions."""
    print_section("8. SENSITIVITY ANALYSIS")

    if data.get("macro_data", pl.DataFrame()).height == 0:
        print("\n  No data available for sensitivity analysis")
        return

    specs = [
        RehypoLiquiditySpec(name="aggressive", alpha_low=0.015, alpha_high=0.025),
        RehypoLiquiditySpec(name="default", alpha_low=0.02, alpha_high=0.03),
        RehypoLiquiditySpec(name="conservative", alpha_low=0.025, alpha_high=0.04),
    ]

    print("\n  RLI Sensitivity to Haircut Spread Assumptions:")
    print("-" * 70)

    results_data = []
    for spec in specs:
        result = indicator.calculate(data, spec=spec)
        if result.data.height > 0:
            latest = result.data.tail(1)
            results_data.append({
                "Specification": spec.name.title(),
                "Low_Vol_α": f"{spec.alpha_low * 100:.1f}%",
                "High_Vol_α": f"{spec.alpha_high * 100:.1f}%",
                "Current_RLI_$B": f"{latest['rli'][0] / 1000:.2f}",
                "Haircut_%": f"{latest['haircut_spread'][0] * 100:.2f}",
            })

    if results_data:
        sensitivity_df = pl.DataFrame(results_data)
        print(sensitivity_df)

        print("\n  Interpretation:")
        print("    - RLI varies significantly with haircut assumptions")
        print("    - Conservative estimates provide larger liquidity buffer estimates")
        print("    - Use multiple specs for robust risk assessment")


def main():
    """Main example runner."""
    print("=" * 70)
    print("PRIME REHYPOTHECATION LIQUIDITY INDEX (RLI) - INDICATOR EXAMPLE")
    print("Measuring Liquidity Creation Through Collateral Reuse")
    print("=" * 70)

    # 1. Verify registration
    indicator = demonstrate_indicator_registration()

    # 2. Fetch data and calculate
    result, data = fetch_and_calculate(indicator)

    if result is None or result.data.height == 0:
        print("\n⚠ Could not calculate indicator. Check data availability.")
        return

    # 3. Display historical readings
    display_historical_readings(result)

    # 4. Benchmark against crisis events
    benchmark_against_crisis_events(result, indicator)

    # 5. Current reading and interpretation
    display_current_reading(result, indicator)

    # 6. Specification customization
    demonstrate_spec_customization()

    # 7. Sensitivity analysis
    run_sensitivity_analysis(indicator, data)

    # Summary
    print_section("SUMMARY")
    print("""
    The Rehypothecation Liquidity Index (RLI) measures:

    1. COLLATERAL BASE: Margin receivables + repo liabilities
       → Total volume eligible for rehypothecation

    2. HAIRCUT SPREAD: VIX-regime-dependent differential
       → Liquidity generated per dollar of collateral

    3. RLI: Collateral Base × Haircut Spread
       → Dollar amount of funding liquidity created

    Key Risk Signals:
    🔴 RLI Velocity < -20%: Severe contraction, expect revenue decline
    🟠 Haircut Spread > 6%: Extreme stress, historical -25% revenue impact
    🟡 RLI Normalized < 1%: Below 2008 crisis levels

    Integration with Revenue Models:
    - RLI provides leading indicator of prime brokerage capacity
    - Velocity changes signal upcoming revenue pressure
    - Combine with Leverage Cycle Index for comprehensive risk view
    """)

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
