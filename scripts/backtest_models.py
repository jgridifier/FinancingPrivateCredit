#!/usr/bin/env python3
"""
Model Backtesting Script

Compare multiple model specifications using rolling-window backtesting.
Outputs performance metrics and comparison charts.

Usage:
    python scripts/backtest_models.py --specs config/model_specs/*.json --output results/
    python scripts/backtest_models.py --specs config/model_specs/baseline_aplr.json config/model_specs/lis_enhanced.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from financing_private_credit.bank_data import BankDataCollector
from financing_private_credit.forecast_models import (
    BacktestResult,
    ModelSpecification,
    get_forecaster,
)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """
    Compute forecast evaluation metrics.

    Args:
        actual: Array of actual values
        predicted: Array of predicted values

    Returns:
        Dictionary with MAE, RMSE, MAPE, directional accuracy
    """
    # Remove any NaN pairs
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "directional_accuracy": np.nan}

    errors = actual - predicted
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs(errors / actual)) * 100)
        mape = np.nan if np.isinf(mape) else mape

    # Directional accuracy
    if len(actual) > 1:
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        directional_accuracy = float(np.mean(actual_direction == predicted_direction) * 100)
    else:
        directional_accuracy = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
    }


def rolling_backtest(
    panel: pl.DataFrame,
    spec: ModelSpecification,
    initial_window: int = 20,
    step: int = 1,
    horizons: list[int] = None,
) -> BacktestResult:
    """
    Perform rolling-window backtest of a model specification.

    Args:
        panel: Panel data with all banks
        spec: Model specification to test
        initial_window: Initial training window size (quarters)
        step: Step size for rolling window
        horizons: Forecast horizons to evaluate

    Returns:
        BacktestResult with metrics
    """
    if horizons is None:
        horizons = [1, 2, 4]

    all_forecasts = []
    all_actuals = []
    forecast_banks = []
    forecast_horizons = []

    banks = panel["ticker"].unique().to_list()

    print(f"\nBacktesting {spec.name}...")
    print(f"  Banks: {len(banks)}")

    for bank in banks:
        bank_df = panel.filter(pl.col("ticker") == bank).sort("date")

        if bank_df.height < initial_window + max(horizons):
            print(f"  Skipping {bank}: insufficient data ({bank_df.height} rows)")
            continue

        # Get sorted unique dates
        dates = bank_df["date"].to_list()
        n_periods = len(dates)

        # Rolling forecast loop
        for t in range(initial_window, n_periods - max(horizons), step):
            train_end_date = dates[t - 1]

            # Create training data
            train_data = panel.filter(pl.col("date") <= train_end_date)

            try:
                # Fit model
                forecaster = get_forecaster(spec)
                forecaster.fit(train_data)

                # Generate forecasts
                forecasts = forecaster.predict(train_data, bank, horizon=max(horizons))

                # Evaluate at each horizon
                for h in horizons:
                    forecast_result = next((f for f in forecasts if f.horizon == h), None)
                    if forecast_result is None:
                        continue

                    # Get actual value at horizon h
                    actual_idx = t + h - 1
                    if actual_idx < n_periods:
                        actual_row = bank_df[actual_idx]
                        actual_value = actual_row[spec.target][0] if spec.target in actual_row.columns else None

                        if actual_value is not None and not np.isnan(actual_value):
                            all_forecasts.append(forecast_result.point_forecast)
                            all_actuals.append(actual_value)
                            forecast_banks.append(bank)
                            forecast_horizons.append(h)

            except Exception as e:
                # Skip this window on error
                continue

    # Compute aggregate metrics
    all_forecasts = np.array(all_forecasts)
    all_actuals = np.array(all_actuals)
    overall_metrics = compute_metrics(all_actuals, all_forecasts)

    # Compute metrics by bank
    by_bank = {}
    for bank in banks:
        mask = np.array(forecast_banks) == bank
        if np.sum(mask) > 0:
            by_bank[bank] = compute_metrics(all_actuals[mask], all_forecasts[mask])

    # Compute metrics by horizon
    by_horizon = {}
    for h in horizons:
        mask = np.array(forecast_horizons) == h
        if np.sum(mask) > 0:
            by_horizon[h] = compute_metrics(all_actuals[mask], all_forecasts[mask])

    return BacktestResult(
        spec_name=spec.name,
        mae=overall_metrics["mae"],
        rmse=overall_metrics["rmse"],
        mape=overall_metrics["mape"],
        directional_accuracy=overall_metrics["directional_accuracy"],
        n_forecasts=len(all_forecasts),
        by_bank=by_bank,
        by_horizon=by_horizon,
    )


def compare_specifications(
    panel: pl.DataFrame,
    spec_paths: list[str | Path],
    output_dir: Optional[str | Path] = None,
) -> list[BacktestResult]:
    """
    Compare multiple model specifications.

    Args:
        panel: Panel data with all banks
        spec_paths: Paths to specification JSON files
        output_dir: Directory to save results

    Returns:
        List of BacktestResult objects
    """
    results = []

    for path in spec_paths:
        try:
            spec = ModelSpecification.from_json(path)
            result = rolling_backtest(panel, spec)
            results.append(result)
            print(f"\n{spec.name}:")
            print(f"  MAE:  {result.mae:.4f}")
            print(f"  RMSE: {result.rmse:.4f}")
            print(f"  MAPE: {result.mape:.2f}%")
            print(f"  Directional: {result.directional_accuracy:.1f}%")
            print(f"  N forecasts: {result.n_forecasts}")
        except Exception as e:
            print(f"Error processing {path}: {e}")

    # Save comparison summary
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary table
        summary = []
        for r in results:
            summary.append({
                "specification": r.spec_name,
                "mae": r.mae,
                "rmse": r.rmse,
                "mape": r.mape,
                "directional_accuracy": r.directional_accuracy,
                "n_forecasts": r.n_forecasts,
            })

        summary_df = pl.DataFrame(summary)
        summary_df.write_csv(output_dir / "comparison_summary.csv")
        print(f"\nSaved summary to {output_dir / 'comparison_summary.csv'}")

        # Detailed results JSON
        detailed = {
            "run_date": datetime.now().isoformat(),
            "results": [asdict(r) for r in results],
        }
        with open(output_dir / "detailed_results.json", "w") as f:
            json.dump(detailed, f, indent=2, default=str)
        print(f"Saved detailed results to {output_dir / 'detailed_results.json'}")

    return results


def generate_comparison_report(results: list[BacktestResult]) -> str:
    """
    Generate a markdown comparison report.

    Args:
        results: List of backtest results

    Returns:
        Markdown formatted report
    """
    report = []
    report.append("# Model Comparison Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Overall summary table
    report.append("## Overall Performance\n")
    report.append("| Specification | MAE | RMSE | MAPE (%) | Direction (%) | N |")
    report.append("|--------------|-----|------|----------|---------------|---|")

    # Sort by RMSE
    sorted_results = sorted(results, key=lambda x: x.rmse if not np.isnan(x.rmse) else float("inf"))

    for r in sorted_results:
        report.append(
            f"| {r.spec_name} | {r.mae:.4f} | {r.rmse:.4f} | "
            f"{r.mape:.1f} | {r.directional_accuracy:.1f} | {r.n_forecasts} |"
        )

    # Best model callout
    if sorted_results:
        best = sorted_results[0]
        report.append(f"\n**Best model (by RMSE):** {best.spec_name}\n")

    # By-horizon breakdown
    report.append("\n## Performance by Forecast Horizon\n")

    horizons = set()
    for r in results:
        horizons.update(r.by_horizon.keys())

    for h in sorted(horizons):
        report.append(f"\n### Horizon {h}\n")
        report.append("| Specification | MAE | RMSE |")
        report.append("|--------------|-----|------|")

        for r in sorted_results:
            if h in r.by_horizon:
                metrics = r.by_horizon[h]
                report.append(f"| {r.spec_name} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} |")

    # By-bank breakdown (top 5 banks by data)
    report.append("\n## Performance by Bank (Selected)\n")

    # Get common banks
    common_banks = set()
    for r in results:
        common_banks.update(r.by_bank.keys())

    for bank in sorted(list(common_banks))[:5]:
        report.append(f"\n### {bank}\n")
        report.append("| Specification | MAE | RMSE |")
        report.append("|--------------|-----|------|")

        for r in sorted_results:
            if bank in r.by_bank:
                metrics = r.by_bank[bank]
                report.append(f"| {r.spec_name} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} |")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Backtest model specifications")
    parser.add_argument(
        "--specs",
        nargs="+",
        required=True,
        help="Paths to model specification JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/backtest",
        help="Output directory for results",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Start year for data",
    )
    parser.add_argument(
        "--initial-window",
        type=int,
        default=16,
        help="Initial training window (quarters)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Model Backtesting")
    print("=" * 60)

    # Load data
    print("\nLoading bank data...")
    collector = BankDataCollector(start_date=f"{args.start_year}-01-01")
    panel = collector.fetch_all_banks()
    panel = collector.compute_derived_metrics(panel)

    print(f"Loaded {panel.height} rows for {panel['ticker'].n_unique()} banks")

    # Add LIS scores if needed (for models that use it)
    from financing_private_credit.dashboard import compute_lis_scores
    panel = compute_lis_scores(panel)

    # Run comparison
    results = compare_specifications(
        panel,
        args.specs,
        output_dir=args.output,
    )

    # Generate markdown report
    if results:
        report = generate_comparison_report(results)
        output_dir = Path(args.output)
        with open(output_dir / "comparison_report.md", "w") as f:
            f.write(report)
        print(f"\nSaved report to {output_dir / 'comparison_report.md'}")

    print("\n" + "=" * 60)
    print("Backtesting complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
