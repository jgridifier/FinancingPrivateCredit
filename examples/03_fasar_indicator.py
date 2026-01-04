#!/usr/bin/env python3
"""
FASAR (Flex-Adjusted Syndicate Absorption Ratio) Example

This example demonstrates the FASAR indicator using real market data from FRED
to calculate CLO velocity (market absorption capacity).

FASAR = Σ(Volume × Rigidity) / (Base Capacity × CLO Velocity)

Key signals:
- FASAR > 2.0: "Trapped" - Bank must fund but cannot syndicate (Hung Loan Risk)
- FASAR 1.5-2.0: "Elevated" - Syndication stress emerging
- FASAR 1.0-1.5: "Watch" - Monitor closely
- FASAR < 1.0: "Normal" - Market absorbing debt

Reference: Ivashina & Scharfstein (2010): Loan Syndication and Credit Cycles
"""

from datetime import datetime, timedelta
import polars as pl

from financing_private_credit.indicators.FASAR import (
    FASARIndicator,
    FASARSpec,
    CommitmentDeal,
    RigidityScorer,
    CLOVelocityCalculator,
    FASARNowcaster,
    ScarTissueForecaster,
    extract_rigidity_evidence,
    compute_preliminary_rigidity,
)
from financing_private_credit.data import FREDDataFetcher


def fetch_real_spread_data(lookback_days: int = 90) -> pl.DataFrame:
    """
    Fetch real credit spread data from FRED.

    Uses:
    - BAMLC0A0CM: ICE BofA US Corporate Index Option-Adjusted Spread
    - BAMLH0A0HYM2: ICE BofA US High Yield Index Option-Adjusted Spread
    """
    fetcher = FREDDataFetcher()
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    print(f"  Fetching credit spread data from FRED (last {lookback_days} days)...")

    # Fetch High Yield OAS (Option-Adjusted Spread)
    hy_spread = fetcher.fetch_series("BAMLH0A0HYM2", start_date)

    if hy_spread.height > 0 and "value" in hy_spread.columns:
        # Rename value column to spread and convert units
        # FRED reports in percentage points (e.g., 4.5 = 450bps)
        spread_data = hy_spread.select([
            "date",
            (pl.col("value") * 100).alias("spread"),  # Convert to basis points
        ]).drop_nulls()

        print(f"  ✓ Fetched {spread_data.height} observations")

        if spread_data.height > 0:
            latest = spread_data.tail(1)
            print(f"  Latest HY spread: {latest['spread'][0]:.0f} bps on {latest['date'][0]}")

        return spread_data

    # Fallback to Investment Grade if HY not available
    print("  HY spread not available, trying Investment Grade...")
    ig_spread = fetcher.fetch_series("BAMLC0A0CM", start_date)

    if ig_spread.height > 0 and "value" in ig_spread.columns:
        spread_data = ig_spread.select([
            "date",
            (pl.col("value") * 100).alias("spread"),
        ]).drop_nulls()

        print(f"  ✓ Fetched {spread_data.height} IG spread observations")
        return spread_data

    print("  ⚠ Could not fetch spread data from FRED")
    return pl.DataFrame({"date": [], "spread": []})


def demonstrate_rigidity_scoring():
    """
    Demonstrate the rigidity scoring system using real 8-K text patterns.

    In production, this would parse actual SEC EDGAR 8-K filings.
    """
    print("\n" + "=" * 60)
    print("RIGIDITY SCORING FROM 8-K TEXT")
    print("=" * 60)

    # Real-world 8-K commitment letter text pattern (SunGard-style)
    sample_8k_text = """
    BRIDGE COMMITMENT LETTER

    Subject to the terms and conditions set forth herein, the Commitment Parties
    hereby commit to provide the full amount of the Bridge Facility on a "certain
    funds" basis, subject only to the Limited Conditionality Provisions set forth
    in Exhibit A.

    The Commitment Parties' obligations are not subject to any conditions relating
    to the accuracy of representations regarding the Target's business, financial
    condition, or prospects, or to any material adverse change in the Target or
    market conditions.

    Market flex provisions are limited to: (i) OID adjustments up to 50 basis points,
    and (ii) coupon increases up to 25 basis points.

    The Funding shall occur on the Closing Date regardless of the syndication status
    of the Facility, provided that Successful Syndication is not a condition to
    funding.
    """

    # Extract evidence using regex patterns
    evidence = extract_rigidity_evidence(sample_8k_text)
    score = compute_preliminary_rigidity(evidence)

    print("\nSample 8-K Analysis:")
    print(f"  Rigidity Score: {score:.2f}")
    print("\n  Detected Provisions:")
    print(f"    - SunGard Clause: {evidence.sungard_clause}")
    print(f"    - Limited Conditionality: {evidence.limited_conditionality}")
    print(f"    - Certain Funds: {evidence.certain_funds}")
    print(f"    - Successful Syndication Condition: {evidence.successful_syndication_condition}")
    print(f"    - Market Flex Mentioned: {evidence.market_flex_mentioned}")
    print(f"    - Flex Cap: {evidence.flex_is_capped} ({evidence.flex_cap_amount or 'N/A'})")
    print(f"    - MAE Excludes Market: {evidence.mae_excludes_market}")

    # Score interpretation
    if score >= 0.8:
        classification = "TRAPPED - Bank must fund regardless of market conditions"
    elif score >= 0.5:
        classification = "CONSTRAINED - Limited flexibility, syndication-dependent"
    else:
        classification = "FLEXIBLE - Has market outs, can exit if needed"

    print(f"\n  Classification: {classification}")

    return score


def calculate_clo_velocity(spread_data: pl.DataFrame) -> float:
    """
    Calculate CLO velocity from real spread data.
    """
    print("\n" + "=" * 60)
    print("CLO VELOCITY FROM REAL SPREAD DATA")
    print("=" * 60)

    if spread_data.height == 0:
        print("  No spread data available, using default velocity = 1.0")
        return 1.0

    calculator = CLOVelocityCalculator()
    result = calculator.calculate(spread_data=spread_data)

    print(f"\n  CLO Velocity: {result.velocity:.3f}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Data Sources: {', '.join(result.data_sources_used) or 'none'}")

    if result.spread_signal is not None:
        print(f"  Spread Signal: {result.spread_signal:.3f}")

    print("\n  Interpretation:")
    if result.velocity > 1.2:
        print("    → STRONG absorption capacity - CLO market healthy")
    elif result.velocity > 0.8:
        print("    → NORMAL absorption capacity - market functioning")
    elif result.velocity > 0.5:
        print("    → WEAK absorption capacity - syndication challenging")
    else:
        print("    → IMPAIRED market - elevated hung loan risk")

    if result.choke_signal:
        print("\n  ⚠ CHOKE SIGNAL DETECTED: JBBB/HYG divergence indicates CLO stress")

    return result.velocity


def create_current_deals() -> list[CommitmentDeal]:
    """
    Create deals representing current market conditions.

    In production, these would be extracted from SEC EDGAR 8-K filings.
    For demonstration, we create realistic current-period deals.
    """
    today = datetime.now()

    # Create deals with varying rigidity levels
    deals = [
        # Large LBO with SunGard-style commitment (high rigidity)
        CommitmentDeal(
            deal_id="deal_001",
            bank_ticker="JPM",
            announcement_date=today - timedelta(days=3),
            commitment_amount=2500.0,  # $2.5B
            target_company="Target Corp A",
            acquirer_company="PE Firm Alpha",
            rigidity_score=0.95,
            has_sungard_clause=True,
            has_limited_conditionality=True,
            has_market_flex=True,
            flex_is_capped=True,
        ),

        # Mid-size deal with moderate flex
        CommitmentDeal(
            deal_id="deal_002",
            bank_ticker="GS",
            announcement_date=today - timedelta(days=2),
            commitment_amount=1800.0,  # $1.8B
            target_company="Tech Target",
            acquirer_company="PE Firm Beta",
            rigidity_score=0.75,
            has_market_flex=True,
            flex_is_capped=True,
        ),

        # Bank of America - flexible terms (low rigidity)
        CommitmentDeal(
            deal_id="deal_003",
            bank_ticker="BAC",
            announcement_date=today - timedelta(days=1),
            commitment_amount=1200.0,  # $1.2B
            target_company="Consumer Target",
            acquirer_company="Strategic Corp",
            rigidity_score=0.3,
            has_market_flex=True,
            flex_is_capped=False,
        ),

        # Morgan Stanley - high rigidity deal
        CommitmentDeal(
            deal_id="deal_004",
            bank_ticker="MS",
            announcement_date=today - timedelta(days=1),
            commitment_amount=2200.0,  # $2.2B
            target_company="Healthcare Target",
            acquirer_company="PE Firm Delta",
            rigidity_score=0.85,
            has_sungard_clause=True,
        ),

        # Citi - already syndicated (excluded from FASAR)
        CommitmentDeal(
            deal_id="deal_005",
            bank_ticker="C",
            announcement_date=today - timedelta(days=10),
            commitment_amount=2000.0,
            target_company="Industrial Target",
            acquirer_company="PE Firm Gamma",
            rigidity_score=0.8,
            is_syndicated=True,
            syndication_date=today - timedelta(days=3),
        ),
    ]

    return deals


def calculate_fasar(deals: list[CommitmentDeal], clo_velocity: float):
    """
    Calculate FASAR scores for all banks using real velocity data.
    """
    print("\n" + "=" * 60)
    print("FASAR CALCULATION WITH REAL MARKET DATA")
    print("=" * 60)

    active_deals = [d for d in deals if not d.is_syndicated]
    print(f"\n  Active deals: {len(active_deals)} (excluding {len(deals) - len(active_deals)} syndicated)")

    # Create spec with standard parameters
    spec = FASARSpec(
        name="current_market",
        description="Current market FASAR calculation",
        base_syndication_capacity=5000.0,  # $5B weekly capacity
        high_risk_threshold=2.0,
        elevated_threshold=1.5,
        normal_threshold=1.0,
    )

    indicator = FASARIndicator()

    result = indicator.calculate_from_deals(
        deals=deals,
        clo_velocity=clo_velocity,
    )

    if result.height == 0:
        print("\n  No active deals to calculate FASAR.")
        return result

    print(f"\n  {'Bank':<8} {'FASAR':>8} {'Risk Level':<12} {'Trapped Vol':>12} {'# Deals':>8}")
    print("  " + "-" * 52)

    for row in result.iter_rows(named=True):
        emoji = {"HIGH_RISK": "🔴", "ELEVATED": "🟠", "WATCH": "🟡", "NORMAL": "🟢"}.get(
            row['risk_level'], "⚪"
        )
        print(
            f"  {row['ticker']:<8} "
            f"{row['effective_fasar']:>8.2f} "
            f"{emoji} {row['risk_level']:<10} "
            f"${row['total_trapped_volume']:>10,.0f}M "
            f"{row['n_active_deals']:>8}"
        )

    # Summary
    print("\n  " + "-" * 52)
    avg_fasar = result["effective_fasar"].mean()
    max_fasar = result["effective_fasar"].max()
    max_bank = result.filter(pl.col("effective_fasar") == max_fasar)["ticker"][0]
    high_risk_count = result.filter(pl.col("risk_level") == "HIGH_RISK").height

    print(f"\n  Summary Statistics:")
    print(f"    Average FASAR: {avg_fasar:.2f}")
    print(f"    Maximum FASAR: {max_fasar:.2f} ({max_bank})")
    print(f"    High Risk Banks: {high_risk_count}")
    print(f"    CLO Velocity Used: {clo_velocity:.3f}")

    return result


def demonstrate_nowcast(fasar_result: pl.DataFrame, spread_data: pl.DataFrame):
    """
    Show nowcast capabilities for real-time monitoring.
    """
    print("\n" + "=" * 60)
    print("FASAR NOWCAST (Real-Time Adjustments)")
    print("=" * 60)

    nowcaster = FASARNowcaster()

    nowcast_result = nowcaster.nowcast(
        base_fasar=fasar_result,
        spread_data=spread_data,
        etf_data=pl.DataFrame(),
    )

    print("\n  Nowcast Signals:")
    print(f"    AAA Spread Widening: {nowcast_result.metadata.get('aaa_widening_signal', False)}")
    print(f"    Velocity Adjustment: {nowcast_result.metadata.get('velocity_adjustment', 1.0):.2f}x")

    if nowcast_result.data.height > 0 and "nowcast_fasar" in nowcast_result.data.columns:
        print("\n  Nowcast-Adjusted FASAR by Bank:")
        for row in nowcast_result.data.iter_rows(named=True):
            base = row.get('effective_fasar', 0)
            nowcast = row.get('nowcast_fasar', base)
            change = ((nowcast / base) - 1) * 100 if base > 0 else 0
            print(f"    {row['ticker']}: {nowcast:.2f} ({change:+.1f}% from base)")


def run_stress_scenario(scenario_name: str, clo_velocity: float, deals: list[CommitmentDeal]):
    """
    Run a stress scenario with given parameters.
    """
    indicator = FASARIndicator()
    result = indicator.calculate_from_deals(deals, clo_velocity)

    if result.height > 0:
        max_fasar = result["effective_fasar"].max()
        max_bank = result.filter(pl.col("effective_fasar") == max_fasar)["ticker"][0]
        max_risk = result.filter(pl.col("effective_fasar") == max_fasar)["risk_level"][0]

        emoji, status = indicator.get_warning_level(max_fasar)
        print(f"\n  {scenario_name}:")
        print(f"    CLO Velocity: {clo_velocity}")
        print(f"    Peak FASAR: {max_fasar:.2f} ({max_bank})")
        print(f"    Status: {emoji} {status}")


def main():
    """Main example runner."""
    print("=" * 60)
    print("FASAR INDICATOR - REAL DATA EXAMPLE")
    print("Flex-Adjusted Syndicate Absorption Ratio")
    print("=" * 60)

    # Step 1: Fetch real market data
    print("\n[1] FETCHING REAL MARKET DATA")
    print("-" * 40)
    spread_data = fetch_real_spread_data(lookback_days=90)

    # Step 2: Demonstrate rigidity scoring
    demonstrate_rigidity_scoring()

    # Step 3: Calculate CLO velocity from real data
    clo_velocity = calculate_clo_velocity(spread_data)

    # Step 4: Create current deals and calculate FASAR
    print("\n[4] CREATING CURRENT PERIOD DEALS")
    print("-" * 40)
    deals = create_current_deals()
    active = sum(1 for d in deals if not d.is_syndicated)
    print(f"  Created {len(deals)} deals ({active} active, from last 3 days)")

    fasar_result = calculate_fasar(deals, clo_velocity)

    # Step 5: Demonstrate nowcast
    if fasar_result.height > 0:
        demonstrate_nowcast(fasar_result, spread_data)

    # Step 6: Stress scenario comparison
    print("\n" + "=" * 60)
    print("STRESS SCENARIO COMPARISON")
    print("=" * 60)
    print("\n  How current deals would perform under historical stress:")

    # Use current deals with different velocity scenarios
    scenarios = [
        ("Current Market", clo_velocity),
        ("2015 Energy Stress (velocity=0.35)", 0.35),
        ("2020 COVID Shock (velocity=0.40)", 0.40),
        ("2008 Crisis (velocity=0.30)", 0.30),
    ]

    for name, vel in scenarios:
        run_stress_scenario(name, vel, deals)

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nThe FASAR indicator combines:")
    print("  1. Real credit spread data from FRED")
    print("  2. Deal rigidity from 8-K NLP analysis")
    print("  3. Market absorption capacity (CLO velocity)")
    print("\nTo produce early warning signals for hung loan risk.")


if __name__ == "__main__":
    main()
