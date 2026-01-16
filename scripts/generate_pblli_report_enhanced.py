#!/usr/bin/env python3
"""
Generate Enhanced PB-LLI Report with Shadow Nowcast Extension

This script:
1. Fetches real data from FRED using API key
2. Calculates the PB-LLI indicator through official data
3. Extends with shadow/nowcast estimates for unreported quarters
4. Generates 2-quarter forward outlook
5. Creates comprehensive CSV and Markdown reports
"""

import os
import sys
import json
from datetime import datetime, date
from pathlib import Path

# Set API keys from environment or directly
FRED_API_KEY = os.environ.get("FRED_API_KEY", "b91d6a27ef2aa703e7ccba3a5d52e457")
NASDAQ_API_KEY = os.environ.get("NASDAQ_API_KEY", "nA1riYWQGB6kf-1bofh2")

# Set environment variables for the modules to use
os.environ["FRED_API_KEY"] = FRED_API_KEY
os.environ["NASDAQ_API_KEY"] = NASDAQ_API_KEY

import polars as pl
import numpy as np

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


def run_indicator_with_shadow() -> tuple:
    """Run the PB-LLI indicator with shadow nowcast extension."""
    print("=" * 80)
    print("PRIME BROKERAGE LEVERAGE LEAD INDICATOR V2 (PB-LLI)")
    print("Enhanced Report with Shadow Nowcast Extension")
    print("=" * 80)
    print(f"\nReport Date: {datetime.now().strftime('%B %d, %Y')}")
    print(f"FRED API Key: {FRED_API_KEY[:8]}...")

    # Get indicator instance
    indicator = get_indicator("prime_leverage_v2")
    metadata = indicator.get_metadata()

    print(f"\nIndicator: {metadata.name}")
    print(f"Version: {metadata.version}")

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

    if result.data.height == 0:
        print("ERROR: No data calculated")
        return indicator, result, data, {}

    print(f"Calculated {result.data.height} quarterly observations")
    print(f"Last official quarter: {result.data['date'].max()}")

    # Calculate shadow extension
    print("\n" + "-" * 40)
    print("CALCULATING SHADOW NOWCAST EXTENSION...")
    print("-" * 40)

    shadow_extension = indicator.calculate_shadow_extension(
        data, result, as_of_date=datetime.now()
    )

    if shadow_extension.get("success"):
        print(f"Shadow model R²: {shadow_extension['model_fit'].get('r_squared', 0):.3f}")
        print(f"Shadow estimates generated: {len(shadow_extension.get('shadow_estimates', []))}")
        print(f"Forward forecasts: {len(shadow_extension.get('forecasts', []))}")
    else:
        print(f"Shadow extension failed: {shadow_extension.get('reason', 'Unknown')}")

    return indicator, result, data, shadow_extension


def save_enhanced_reports(result, shadow_extension, output_dir: Path) -> dict:
    """Save all enhanced reports."""
    print("\n" + "=" * 80)
    print("SAVING ENHANCED REPORTS")
    print("=" * 80)

    saved_files = {}

    # 1. Save quarterly anchor index
    print("\n1. Saving Quarterly Anchor Index...")
    anchor_df = save_quarterly_anchor(result, output_dir)
    saved_files["quarterly_anchor"] = str(output_dir / "quarterly_anchor_index.csv")

    # 2. Save composite forecast signal
    print("\n2. Saving Composite Forecast Signal...")
    forecast_df = save_composite_forecast(result, output_dir)
    saved_files["composite_forecast"] = str(output_dir / "composite_forecast_signal.csv")

    # 3. Save shadow nowcast estimates
    if shadow_extension.get("success"):
        print("\n3. Saving Shadow Nowcast Estimates...")
        shadow_df = save_shadow_estimates(shadow_extension, output_dir)
        saved_files["shadow_estimates"] = str(output_dir / "shadow_nowcast_estimates.csv")

    # 4. Save full dataset
    print("\n4. Saving Full Dataset...")
    result.data.write_csv(output_dir / "pblli_full_dataset.csv")
    saved_files["full_dataset"] = str(output_dir / "pblli_full_dataset.csv")

    # 5. Save enhanced metadata
    print("\n5. Saving Enhanced Metadata...")
    metadata = save_enhanced_metadata(result, shadow_extension, output_dir)
    saved_files["metadata"] = str(output_dir / "pblli_metadata_enhanced.json")

    # 6. Generate enhanced markdown report
    print("\n6. Generating Enhanced Markdown Report...")
    report_path = generate_enhanced_markdown_report(result, shadow_extension, output_dir)
    saved_files["report"] = str(report_path)

    return saved_files


def save_quarterly_anchor(result, output_dir: Path) -> pl.DataFrame:
    """Save quarterly anchor index."""
    df = result.data
    cols = ["date", "pb_intensity", "pb_intensity_g", "pb_intensity_g_z",
            "dealer_supply_g", "dealer_supply_g_z", "hf_equity", "year", "quarter"]
    available_cols = [c for c in cols if c in df.columns]
    anchor_df = df.select(available_cols)
    anchor_df.write_csv(output_dir / "quarterly_anchor_index.csv")
    print(f"   Saved: {output_dir / 'quarterly_anchor_index.csv'} ({anchor_df.height} rows)")
    return anchor_df


def save_composite_forecast(result, output_dir: Path) -> pl.DataFrame:
    """Save composite forecast signal."""
    df = result.data
    cols = ["date", "pb_lead", "pb_lead_z", "pb_lead_percentile",
            "run_rate_regime", "stress_flag", "pb_intensity_g_nowcast"]
    available_cols = [c for c in cols if c in df.columns]
    forecast_df = df.select(available_cols)
    forecast_df.write_csv(output_dir / "composite_forecast_signal.csv")
    print(f"   Saved: {output_dir / 'composite_forecast_signal.csv'} ({forecast_df.height} rows)")
    return forecast_df


def save_shadow_estimates(shadow_extension: dict, output_dir: Path) -> pl.DataFrame:
    """Save shadow nowcast estimates."""
    estimates = shadow_extension.get("shadow_estimates", [])
    forecasts = shadow_extension.get("forecasts", [])

    rows = []

    # Shadow/nowcast estimates
    for est in estimates:
        rows.append({
            "quarter": est["quarter"],
            "quarter_label": est["quarter_label"],
            "type": est["estimate_type"],
            "pb_intensity_g_pct": est["pb_intensity_g_pct"],
            "pb_intensity_level": est["pb_intensity_level"],
            "confidence_lower_pct": est["confidence_lower_pct"],
            "confidence_upper_pct": est["confidence_upper_pct"],
            "completeness_weekly": est["completeness_weekly"],
            "as_of_date": est["as_of_date"],
        })

    # Forward forecasts
    for fc in forecasts:
        rows.append({
            "quarter": fc["quarter"],
            "quarter_label": fc["quarter_label"],
            "type": "forecast",
            "pb_intensity_g_pct": None,
            "pb_intensity_level": None,
            "pb_lead_forecast_pct": fc["pb_lead_forecast"] * 100,
            "confidence_lower_pct": fc["confidence_lower"] * 100,
            "confidence_upper_pct": fc["confidence_upper"] * 100,
            "completeness_weekly": None,
            "as_of_date": None,
        })

    shadow_df = pl.DataFrame(rows)
    shadow_df.write_csv(output_dir / "shadow_nowcast_estimates.csv")
    print(f"   Saved: {output_dir / 'shadow_nowcast_estimates.csv'} ({shadow_df.height} rows)")
    return shadow_df


def save_enhanced_metadata(result, shadow_extension: dict, output_dir: Path) -> dict:
    """Save enhanced metadata."""
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif hasattr(obj, '__float__'):
            return float(obj)
        elif hasattr(obj, '__int__'):
            return int(obj)
        else:
            return str(obj)

    metadata = {
        "report_date": datetime.now().isoformat(),
        "indicator_version": "2.0.0",
        "official_data": make_serializable(result.metadata),
        "shadow_extension": make_serializable(shadow_extension),
    }

    with open(output_dir / "pblli_metadata_enhanced.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   Saved: {output_dir / 'pblli_metadata_enhanced.json'}")
    return metadata


def generate_enhanced_markdown_report(result, shadow_extension: dict, output_dir: Path) -> Path:
    """Generate enhanced markdown report with shadow/nowcast extension."""

    # Get key values
    current_signal = result.metadata.get("current_signal", {})
    attribution = result.metadata.get("attribution", {})
    summary = result.metadata.get("summary", {})

    # Last official quarter
    last_date = result.data["date"].max()
    last_year = last_date.year
    last_quarter = (last_date.month - 1) // 3 + 1
    last_q_label = f"Q{last_quarter} {last_year}"

    # Current values
    pb_lead = result.data["pb_lead"][-1]
    pb_lead_z = result.data["pb_lead_z"][-1]
    pb_intensity = result.data["pb_intensity"][-1]
    pb_intensity_g = result.data["pb_intensity_g"][-1]
    regime = result.data["run_rate_regime"][-1]

    # Shadow estimates
    shadow_estimates = shadow_extension.get("shadow_estimates", [])
    forecasts = shadow_extension.get("forecasts", [])

    # Build report
    report = f"""# Prime Brokerage Leverage Lead Indicator (PB-LLI) Market Report
## {last_q_label} Assessment with Shadow Nowcast Extension | Report Date: {datetime.now().strftime('%B %d, %Y')}

---

## Executive Summary

**Current Signal: {regime.upper()} ({'Positive' if regime == 'Accelerating' else 'Neutral' if regime == 'Stable' else 'Negative'})**

The PB-LLI composite signal stands at **{pb_lead*100:+.2f}%** with a z-score of **{pb_lead_z:+.2f}**, indicating {current_signal.get('reason', 'N/A')}.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| PB_Lead | {pb_lead*100:+.2f}% | {'Above-average' if pb_lead > summary.get('pb_lead_mean', 0) else 'Below-average'} growth signal |
| Z-Score | {pb_lead_z:+.2f} | {'Positive' if pb_lead_z > 0 else 'Negative'} |
| Regime | {regime} | {'Upper' if regime == 'Accelerating' else 'Middle' if regime == 'Stable' else 'Lower'} tercile |
| Stress Flag | {'ON' if current_signal.get('stress_flag') else 'OFF'} | {'Stress detected' if current_signal.get('stress_flag') else 'No acute stress'} |

---

## 1. Official Data Analysis ({last_q_label})

### 1.1 Quarterly Anchor Index

The Fed Z.1 data through {last_q_label} shows:

- **PB Intensity (HF Borrowing/HF Equity)**: **{pb_intensity*100:.1f}%**
- **PB Intensity Growth**: **{pb_intensity_g*100:+.2f}% QoQ**
- **Dealer Supply Growth**: **{result.data['dealer_supply_g'][-1]*100:+.2f}% QoQ**

### 1.2 Attribution Decomposition

| Component | Contribution | Share |
|-----------|-------------|-------|
| HF Demand (Nowcast) | {attribution.get('hf_demand_contribution', 0)*100:+.2f}% | {attribution.get('hf_demand_pct', 0):.1f}% |
| Dealer Supply | {attribution.get('dealer_supply_contribution', 0)*100:+.2f}% | {attribution.get('dealer_supply_pct', 0):.1f}% |
| Lagged Intensity | {attribution.get('lagged_intensity_contribution', 0)*100:+.2f}% | {attribution.get('lagged_intensity_pct', 0):.1f}% |

---

## 2. Shadow Nowcast Extension

"""

    if shadow_extension.get("success"):
        model_fit = shadow_extension.get("model_fit", {})
        report += f"""### 2.1 Shadow Model Diagnostics

The shadow nowcast model extends official Z.1 data using weekly leverage appetite proxies.

| Metric | Value |
|--------|-------|
| Model R² | {model_fit.get('r_squared', 0):.3f} |
| Residual Std | {model_fit.get('residual_std', 0)*100:.2f}% |
| Training Observations | {model_fit.get('n_observations', 0)} |
| Ridge Penalty (λ) | 5.0 |

### 2.2 Shadow/Nowcast Estimates

These estimates extend the indicator beyond the official Z.1 release:

| Quarter | Type | PB Intensity Growth | 95% CI | Completeness |
|---------|------|---------------------|--------|--------------|
"""
        for est in shadow_estimates:
            report += f"| {est['quarter_label']} | {est['estimate_type'].title()} | {est['pb_intensity_g_pct']:+.2f}% | [{est['confidence_lower_pct']:+.2f}%, {est['confidence_upper_pct']:+.2f}%] | {est['completeness_weekly']*100:.0f}% weekly |\n"

        report += """
**Interpretation**: Shadow estimates bridge the gap between the last official Z.1 release and the current quarter. These estimates use weekly CFTC positioning and NY Fed dealer statistics as leading indicators. Uncertainty bands are wider for quarters with less data coverage.

"""
    else:
        report += f"""### 2.1 Shadow Extension Status

Shadow nowcast extension not available: {shadow_extension.get('reason', 'Unknown reason')}

"""

    # Forward forecasts
    report += """---

## 3. Forward Outlook (2 Quarters)

### 3.1 PB_Lead Forecasts

"""
    if forecasts:
        report += """| Quarter | PB_Lead Forecast | 95% CI Lower | 95% CI Upper |
|---------|------------------|--------------|--------------|
"""
        for fc in forecasts:
            report += f"| {fc['quarter_label']} | {fc['pb_lead_forecast']*100:+.2f}% | {fc['confidence_lower']*100:+.2f}% | {fc['confidence_upper']*100:+.2f}% |\n"

        report += """
### 3.2 Outlook Interpretation

"""
        if forecasts[0]['pb_lead_forecast'] > summary.get('pb_lead_mean', 0.01):
            report += """The model projects **continued positive momentum** over the next two quarters, though with natural mean-reversion from current levels. Key implications:

1. **Prime brokerage balances** expected to grow faster than trend
2. **Revenue momentum** for equity prime businesses should remain positive
3. **Risk factors**: Equity market corrections, funding stress, or regulatory shocks could shift outlook
"""
        else:
            report += """The model projects **moderating momentum** over the next two quarters as the indicator mean-reverts. Key implications:

1. **Prime brokerage growth** expected to normalize toward historical averages
2. **Revenue momentum** may face headwinds relative to recent quarters
3. **Watch for**: Signs of re-acceleration in weekly data or bank earnings
"""

    # Historical context
    report += f"""
---

## 4. Historical Context

### 4.1 Sample Statistics ({result.data['date'].min().strftime('%Y')} - {last_year})

| Statistic | PB_Lead | PB Intensity Growth | Dealer Supply Growth |
|-----------|---------|---------------------|---------------------|
| Mean | {summary.get('pb_lead_mean', 0)*100:+.2f}% | {summary.get('pb_intensity_g_mean', 0)*100:+.2f}% | {summary.get('dealer_supply_g_mean', 0)*100:+.2f}% |
| Std Dev | {summary.get('pb_lead_std', 0)*100:.2f}% | {summary.get('pb_intensity_g_std', 0)*100:.2f}% | {summary.get('dealer_supply_g_std', 0)*100:.2f}% |

### 4.2 Regime Distribution

- **Accelerating**: {summary.get('pct_accelerating', 0)*100:.1f}% of quarters
- **Stable**: {summary.get('pct_stable', 0)*100:.1f}% of quarters
- **Decelerating**: {summary.get('pct_decelerating', 0)*100:.1f}% of quarters

### 4.3 Recent Trend (Last 8 Quarters)

| Quarter | PB_Lead | Regime |
|---------|---------|--------|
"""

    recent = result.data.tail(8)
    for i in range(recent.height):
        row_date = recent["date"][i]
        q_label = f"Q{(row_date.month-1)//3+1}'{str(row_date.year)[2:]}"
        pb_lead_val = recent["pb_lead"][i]
        regime_val = recent["run_rate_regime"][i]
        report += f"| {q_label} | {pb_lead_val*100:+.2f}% | {regime_val} |\n"

    # Methodology note
    report += f"""
---

## 5. Methodology & Data Quality

### 5.1 Three-Layer Publication System

1. **Official Layer** (through {last_q_label}): Fed Z.1 HF sector data
2. **Shadow Backfill Layer**: Ridge regression estimates for completed unreported quarters
3. **Progressive Nowcast Layer**: Weekly-updated estimates for current quarter

### 5.2 Data Sources

| Source | Series | Frequency | Role |
|--------|--------|-----------|------|
| FRED (Fed Z.1) | HF balance sheet | Quarterly | Official anchor |
| CFTC COT (TFF) | Leveraged Funds | Weekly | Nowcast input |
| NY Fed PD Stats | Dealer repo/fails | Weekly | Nowcast input |

### 5.3 Shadow Model Specification

```
y_t = α + β₁·x̄_t + β₂·D_t + ε_t

Where:
- y_t = Δln(PB_Intensity)
- x̄_t = Quarterly mean of weekly leverage appetite factor
- D_t = Dealer supply growth
- Ridge penalty λ = 5.0
```

### 5.4 Limitations

- Shadow estimates are model-dependent and should be interpreted with wider confidence bands
- Weekly data uses synthetic representative values pending full API integration
- Estimates will be replaced with official Z.1 values when released

---

*Report generated by PB-LLI V2.0.0 with Shadow Nowcast Extension*
*Data through {last_q_label} (official) | Shadow estimates as of {datetime.now().strftime('%Y-%m-%d')}*
"""

    # Write report
    report_path = output_dir / f"PBLLI_Enhanced_Report_{last_q_label.replace(' ', '_')}.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"   Saved: {report_path}")
    return report_path


def print_summary(result, shadow_extension: dict):
    """Print summary to console."""
    print("\n" + "=" * 80)
    print("REPORT SUMMARY")
    print("=" * 80)

    current_signal = result.metadata.get("current_signal", {})
    print(f"\nCurrent Regime: {current_signal.get('regime', 'Unknown')}")
    print(f"PB_Lead: {result.data['pb_lead'][-1]*100:+.2f}%")
    print(f"Z-Score: {result.data['pb_lead_z'][-1]:+.2f}")

    if shadow_extension.get("success"):
        print("\nShadow Extension:")
        for est in shadow_extension.get("shadow_estimates", []):
            print(f"  {est['quarter_label']} ({est['estimate_type']}): {est['pb_intensity_g_pct']:+.2f}% "
                  f"[{est['confidence_lower_pct']:+.2f}%, {est['confidence_upper_pct']:+.2f}%]")

        print("\nForward Forecasts:")
        for fc in shadow_extension.get("forecasts", []):
            print(f"  {fc['quarter_label']}: PB_Lead = {fc['pb_lead_forecast']*100:+.2f}% "
                  f"[{fc['confidence_lower']*100:+.2f}%, {fc['confidence_upper']*100:+.2f}%]")


def main():
    """Main entry point."""
    output_dir = create_output_dir()
    print(f"\nOutput directory: {output_dir}")

    try:
        # Run indicator with shadow extension
        indicator, result, data, shadow_extension = run_indicator_with_shadow()

        if result.data.height == 0:
            print("\nERROR: No data calculated. Check data sources.")
            return 1

        # Save enhanced reports
        saved_files = save_enhanced_reports(result, shadow_extension, output_dir)

        # Print summary
        print_summary(result, shadow_extension)

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
