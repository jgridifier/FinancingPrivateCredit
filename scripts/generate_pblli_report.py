#!/usr/bin/env python3
"""
Generate Prime Brokerage Leverage Lead Indicator V2 (PB-LLI) Reports

This script:
1. Fetches real data from FRED, CFTC, and NY Fed
2. Calculates the PB-LLI indicator
3. Saves detailed CSV reports
4. Outputs summary statistics for market report generation
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import polars as pl

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from financing_private_credit.indicators import get_indicator
from financing_private_credit.indicators.prime_leverage_v2 import (
    PrimeLeverageV2Indicator,
    PBLLISpec,
)


def create_output_dir() -> Path:
    """Create output directory for reports."""
    output_dir = project_root / "reports" / "pblli" / datetime.now().strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_indicator() -> tuple:
    """Run the PB-LLI indicator and return results."""
    print("=" * 80)
    print("PRIME BROKERAGE LEVERAGE LEAD INDICATOR V2 (PB-LLI)")
    print("Report Generation")
    print("=" * 80)

    # Get indicator instance
    indicator = get_indicator("prime_leverage_v2")
    metadata = indicator.get_metadata()

    print(f"\nIndicator: {metadata.name}")
    print(f"Version: {metadata.version}")
    print(f"Data Sources: {metadata.data_sources}")

    # Fetch data from 2005 for sufficient history
    print("\n" + "-" * 40)
    print("FETCHING DATA...")
    print("-" * 40)

    data = indicator.fetch_data("2005-01-01")

    quarterly_data = data.get("quarterly_data", pl.DataFrame())
    weekly_cot = data.get("weekly_cot", pl.DataFrame())
    weekly_pd = data.get("weekly_pd", pl.DataFrame())

    print(f"\nQuarterly observations: {quarterly_data.height}")
    print(f"Weekly COT observations: {weekly_cot.height}")
    print(f"Weekly PD observations: {weekly_pd.height}")

    if quarterly_data.height > 0:
        print(f"\nQuarterly data columns: {quarterly_data.columns}")
        print(f"Date range: {quarterly_data['date'].min()} to {quarterly_data['date'].max()}")

    # Calculate indicator
    print("\n" + "-" * 40)
    print("CALCULATING PB-LLI...")
    print("-" * 40)

    result = indicator.calculate(data)

    return indicator, result, data


def save_quarterly_anchor_report(result, output_dir: Path) -> pl.DataFrame:
    """Save the quarterly anchor index report."""
    print("\nSaving Quarterly Anchor Index...")

    if result.data.height == 0:
        print("  No data available")
        return pl.DataFrame()

    df = result.data

    # Select key quarterly anchor columns
    cols_to_export = ["date"]

    # Add available columns
    optional_cols = [
        "pb_intensity", "pb_intensity_g", "pb_intensity_g_z",
        "dealer_supply_g", "dealer_supply_g_z",
        "hf_equity", "repo_intensity", "repo_intensity_g",
        "year", "quarter"
    ]

    for col in optional_cols:
        if col in df.columns:
            cols_to_export.append(col)

    anchor_df = df.select(cols_to_export)

    # Save to CSV
    csv_path = output_dir / "quarterly_anchor_index.csv"
    anchor_df.write_csv(csv_path)
    print(f"  Saved: {csv_path}")
    print(f"  Rows: {anchor_df.height}")

    return anchor_df


def save_composite_forecast_report(result, output_dir: Path) -> pl.DataFrame:
    """Save the composite forecast signal report."""
    print("\nSaving Composite Forecast Signal...")

    if result.data.height == 0:
        print("  No data available")
        return pl.DataFrame()

    df = result.data

    # Select composite forecast columns
    cols_to_export = ["date"]

    optional_cols = [
        "pb_lead", "pb_lead_z", "pb_lead_percentile",
        "run_rate_regime", "stress_flag",
        "pb_intensity_g_nowcast", "pb_intensity_g_lag1"
    ]

    for col in optional_cols:
        if col in df.columns:
            cols_to_export.append(col)

    forecast_df = df.select(cols_to_export)

    # Save to CSV
    csv_path = output_dir / "composite_forecast_signal.csv"
    forecast_df.write_csv(csv_path)
    print(f"  Saved: {csv_path}")
    print(f"  Rows: {forecast_df.height}")

    return forecast_df


def save_full_dataset(result, output_dir: Path) -> None:
    """Save the complete dataset with all columns."""
    print("\nSaving Full Dataset...")

    if result.data.height == 0:
        print("  No data available")
        return

    csv_path = output_dir / "pblli_full_dataset.csv"
    result.data.write_csv(csv_path)
    print(f"  Saved: {csv_path}")
    print(f"  Columns: {result.data.columns}")
    print(f"  Rows: {result.data.height}")


def save_metadata_report(result, output_dir: Path) -> dict:
    """Save metadata and summary statistics."""
    print("\nSaving Metadata Report...")

    import json

    metadata = result.metadata

    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif hasattr(obj, '__float__'):
            return float(obj)
        elif hasattr(obj, '__int__'):
            return int(obj)
        else:
            return str(obj)

    serializable_metadata = make_serializable(metadata)

    json_path = output_dir / "pblli_metadata.json"
    with open(json_path, "w") as f:
        json.dump(serializable_metadata, f, indent=2)
    print(f"  Saved: {json_path}")

    return metadata


def save_regime_history(result, output_dir: Path) -> pl.DataFrame:
    """Save regime classification history."""
    print("\nSaving Regime History...")

    if result.data.height == 0:
        print("  No data available")
        return pl.DataFrame()

    df = result.data

    if "run_rate_regime" not in df.columns:
        print("  No regime data available")
        return pl.DataFrame()

    # Create regime summary by year
    regime_df = df.select([
        "date", "run_rate_regime", "pb_lead", "pb_lead_z", "stress_flag"
    ]).filter(pl.col("run_rate_regime").is_not_null())

    csv_path = output_dir / "regime_history.csv"
    regime_df.write_csv(csv_path)
    print(f"  Saved: {csv_path}")

    # Regime distribution
    if regime_df.height > 0:
        dist = regime_df.group_by("run_rate_regime").agg(
            pl.count().alias("count")
        ).with_columns(
            (pl.col("count") / pl.col("count").sum() * 100).alias("pct")
        )
        print("\n  Regime Distribution:")
        for row in dist.iter_rows(named=True):
            print(f"    {row['run_rate_regime']}: {row['count']} ({row['pct']:.1f}%)")

    return regime_df


def print_current_reading(result, indicator) -> dict:
    """Print and return current reading analysis."""
    print("\n" + "=" * 80)
    print("CURRENT READING & FORECAST")
    print("=" * 80)

    current_signal = result.metadata.get("current_signal", {})
    current_regime = result.metadata.get("current_regime")
    attribution = result.metadata.get("attribution", {})
    forecasts = result.metadata.get("forecasts", {})

    print("\n--- Current Signal ---")
    signal_label = {1: "POSITIVE", 0: "NEUTRAL", -1: "NEGATIVE"}.get(
        current_signal.get("signal", 0), "UNKNOWN"
    )
    print(f"Signal: {signal_label}")
    print(f"Regime: {current_signal.get('regime', 'Unknown')}")
    print(f"PB_Lead Z-Score: {current_signal.get('pb_lead_z', 'N/A')}")
    print(f"Stress Flag: {current_signal.get('stress_flag', False)}")
    print(f"Reason: {current_signal.get('reason', 'Unknown')}")

    if current_regime:
        interpretation = indicator.get_regime_interpretation(current_regime)
        print("\n--- Regime Interpretation ---")
        for key, value in interpretation.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

    print("\n--- Attribution Decomposition ---")
    if attribution:
        total = attribution.get('pb_lead_total', 0)
        print(f"PB_Lead Total: {total * 100:.2f}%")
        print(f"  HF Demand (Nowcast):  {attribution.get('hf_demand_contribution', 0) * 100:+.2f}%  ({attribution.get('hf_demand_pct', 0):.1f}%)")
        print(f"  Lagged Intensity:     {attribution.get('lagged_intensity_contribution', 0) * 100:+.2f}%  ({attribution.get('lagged_intensity_pct', 0):.1f}%)")
        print(f"  Dealer Supply:        {attribution.get('dealer_supply_contribution', 0) * 100:+.2f}%  ({attribution.get('dealer_supply_pct', 0):.1f}%)")

    print("\n--- Forecasts ---")
    if forecasts:
        for horizon, label in [("t_plus_1", "t+1 (Next Quarter)"), ("t_plus_2", "t+2 (Two Quarters)")]:
            fc = forecasts.get(horizon, {})
            if fc:
                print(f"\n{label}:")
                print(f"  Point Forecast: {fc.get('forecast', 0) * 100:+.2f}%")
                print(f"  95% Confidence: [{fc.get('lower_bound', 0) * 100:+.2f}%, {fc.get('upper_bound', 0) * 100:+.2f}%]")

    return {
        "current_signal": current_signal,
        "current_regime": current_regime,
        "attribution": attribution,
        "forecasts": forecasts,
    }


def print_historical_analysis(result) -> dict:
    """Print historical statistics and key periods."""
    print("\n" + "=" * 80)
    print("HISTORICAL ANALYSIS")
    print("=" * 80)

    summary = result.metadata.get("summary", {})

    print("\n--- Summary Statistics ---")
    if summary:
        print(f"PB_Lead Mean: {summary.get('pb_lead_mean', 0) * 100:.2f}%")
        print(f"PB_Lead Std: {summary.get('pb_lead_std', 0) * 100:.2f}%")
        print(f"PB Intensity Growth Mean: {summary.get('pb_intensity_g_mean', 0) * 100:.2f}%")
        print(f"Dealer Supply Growth Mean: {summary.get('dealer_supply_g_mean', 0) * 100:.2f}%")

    df = result.data
    if df.height > 0:
        # Key historical periods
        print("\n--- Notable Historical Periods ---")

        # Find extreme values
        if "pb_lead_z" in df.columns:
            max_idx = df["pb_lead_z"].arg_max()
            min_idx = df["pb_lead_z"].arg_min()

            if max_idx is not None and min_idx is not None:
                max_row = df[max_idx]
                min_row = df[min_idx]

                print(f"\nHighest PB_Lead Z-Score:")
                print(f"  Date: {max_row['date'][0]}")
                print(f"  Z-Score: {max_row['pb_lead_z'][0]:.2f}")
                if "run_rate_regime" in max_row.columns:
                    print(f"  Regime: {max_row['run_rate_regime'][0]}")

                print(f"\nLowest PB_Lead Z-Score:")
                print(f"  Date: {min_row['date'][0]}")
                print(f"  Z-Score: {min_row['pb_lead_z'][0]:.2f}")
                if "run_rate_regime" in min_row.columns:
                    print(f"  Regime: {min_row['run_rate_regime'][0]}")

        # Recent trend
        print("\n--- Recent Trend (Last 8 Quarters) ---")
        recent = df.tail(8)
        if "pb_lead" in recent.columns and "run_rate_regime" in recent.columns:
            for row in recent.iter_rows(named=True):
                date_str = str(row["date"])[:10] if row.get("date") else "N/A"
                pb_lead = row.get("pb_lead", 0) or 0
                regime = row.get("run_rate_regime", "N/A")
                print(f"  {date_str}: PB_Lead={pb_lead*100:+.2f}%, Regime={regime}")

    return summary


def main():
    """Main entry point."""
    # Create output directory
    output_dir = create_output_dir()
    print(f"\nOutput directory: {output_dir}")

    try:
        # Run indicator
        indicator, result, data = run_indicator()

        if result.data.height == 0:
            print("\nERROR: No data calculated. Check data sources.")
            return 1

        # Save reports
        print("\n" + "=" * 80)
        print("SAVING REPORTS")
        print("=" * 80)

        save_quarterly_anchor_report(result, output_dir)
        save_composite_forecast_report(result, output_dir)
        save_full_dataset(result, output_dir)
        save_metadata_report(result, output_dir)
        save_regime_history(result, output_dir)

        # Print analysis
        print_current_reading(result, indicator)
        print_historical_analysis(result)

        print("\n" + "=" * 80)
        print("REPORT GENERATION COMPLETE")
        print(f"Reports saved to: {output_dir}")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
