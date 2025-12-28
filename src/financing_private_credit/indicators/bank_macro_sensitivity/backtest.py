"""
Backtesting Framework for Bank Macro Sensitivity Models

Implements:
1. Rolling window backtests
2. Expanding window backtests
3. Cross-validation for time series
4. Performance metrics (RMSE, MAE, directional accuracy)
5. Regime-specific performance analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import polars as pl

from .indicator import MacroSensitivitySpec, get_sensitivity_model


@dataclass
class BacktestMetrics:
    """Performance metrics for a backtest."""

    # Overall metrics
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    r_squared: float

    # Directional metrics
    directional_accuracy: float  # % of correct direction predictions
    up_accuracy: float  # Accuracy when actual went up
    down_accuracy: float  # Accuracy when actual went down

    # Risk metrics
    max_error: float
    avg_bias: float  # Positive = overpredict, Negative = underpredict

    n_predictions: int


@dataclass
class BankBacktestResult:
    """Backtest results for a single bank."""

    ticker: str
    metrics: BacktestMetrics

    # Time series of predictions vs actuals
    predictions: pl.DataFrame  # [date, actual, predicted, error]

    # Regime-specific performance
    regime_metrics: dict[str, BacktestMetrics] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete backtest results."""

    spec_name: str
    backtest_start: datetime
    backtest_end: datetime

    # Overall performance
    aggregate_metrics: BacktestMetrics

    # Bank-level performance
    bank_results: dict[str, BankBacktestResult]

    # Rankings
    bank_ranking: pl.DataFrame  # Banks ranked by performance

    # Regime analysis
    regime_performance: dict[str, BacktestMetrics]

    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate text summary of backtest results."""
        lines = [
            f"Backtest Results: {self.spec_name}",
            f"Period: {self.backtest_start.date()} to {self.backtest_end.date()}",
            "-" * 50,
            "Aggregate Metrics:",
            f"  MAE:  {self.aggregate_metrics.mae:.4f}",
            f"  RMSE: {self.aggregate_metrics.rmse:.4f}",
            f"  R²:   {self.aggregate_metrics.r_squared:.4f}",
            f"  Directional Accuracy: {self.aggregate_metrics.directional_accuracy:.1%}",
            "-" * 50,
            f"N Banks: {len(self.bank_results)}",
            f"N Predictions: {self.aggregate_metrics.n_predictions}",
        ]

        if self.regime_performance:
            lines.append("\nRegime Performance:")
            for regime, metrics in self.regime_performance.items():
                lines.append(f"  {regime}: MAE={metrics.mae:.4f}, Dir.Acc={metrics.directional_accuracy:.1%}")

        return "\n".join(lines)


class MacroSensitivityBacktester:
    """
    Backtesting engine for macro sensitivity models.

    Supports:
    - Rolling window: Fixed training window moves forward
    - Expanding window: Training window grows over time
    - Blocked time series CV: Non-overlapping train/test splits
    """

    def __init__(
        self,
        spec: MacroSensitivitySpec,
        method: str = "expanding",  # "rolling", "expanding", "blocked_cv"
        min_train_periods: int = 20,
        test_periods: int = 4,
        step_size: int = 1,
    ):
        """
        Initialize backtester.

        Args:
            spec: Model specification
            method: Backtest method
            min_train_periods: Minimum training periods
            test_periods: Periods to test at each step
            step_size: How many periods to step forward
        """
        self.spec = spec
        self.method = method
        self.min_train_periods = min_train_periods
        self.test_periods = test_periods
        self.step_size = step_size

    def _compute_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> BacktestMetrics:
        """Compute performance metrics."""
        errors = actual - predicted
        abs_errors = np.abs(errors)

        # Basic metrics
        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        # MAPE (handle zeros)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_errors = np.abs(errors / actual) * 100
            pct_errors = pct_errors[np.isfinite(pct_errors)]
            mape = float(np.mean(pct_errors)) if len(pct_errors) > 0 else np.nan

        # R-squared
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0

        # Directional accuracy
        if len(actual) > 1:
            actual_direction = np.sign(np.diff(actual))
            predicted_direction = np.sign(np.diff(predicted))
            direction_correct = actual_direction == predicted_direction
            directional_accuracy = float(np.mean(direction_correct))

            # Up/down specific
            up_mask = actual_direction > 0
            down_mask = actual_direction < 0

            up_accuracy = float(np.mean(direction_correct[up_mask])) if up_mask.sum() > 0 else np.nan
            down_accuracy = float(np.mean(direction_correct[down_mask])) if down_mask.sum() > 0 else np.nan
        else:
            directional_accuracy = np.nan
            up_accuracy = np.nan
            down_accuracy = np.nan

        return BacktestMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            r_squared=r_squared,
            directional_accuracy=directional_accuracy,
            up_accuracy=up_accuracy,
            down_accuracy=down_accuracy,
            max_error=float(np.max(abs_errors)),
            avg_bias=float(np.mean(errors)),
            n_predictions=len(actual),
        )

    def _classify_regime(
        self,
        macro_data: pl.DataFrame,
        date: datetime,
    ) -> str:
        """Classify macro regime at a given date."""
        # Get macro values at date
        row = macro_data.filter(pl.col("date") <= date).tail(1)

        if row.height == 0:
            return "unknown"

        # Classify based on output gap and rate environment
        output_gap = row["output_gap"][0] if "output_gap" in row.columns else 0
        rate_spread = row.get("rate_spread", [0])[0] if "rate_spread" in row.columns else 0

        if output_gap is None:
            output_gap = 0
        if rate_spread is None:
            rate_spread = 0

        if output_gap > 1.0:
            if rate_spread > 2.0:
                return "expansion_high_rates"
            else:
                return "expansion_low_rates"
        elif output_gap < -1.0:
            if rate_spread > 1.0:
                return "recession_high_rates"
            else:
                return "recession_low_rates"
        else:
            return "neutral"

    def backtest(
        self,
        bank_panel: pl.DataFrame,
        macro_data: pl.DataFrame,
    ) -> BacktestResult:
        """
        Run full backtest.

        Args:
            bank_panel: Bank data with NIM
            macro_data: Macro variables

        Returns:
            BacktestResult with all metrics
        """
        # Get unique sorted dates
        all_dates = (
            bank_panel.select("date")
            .unique()
            .sort("date")
            ["date"]
            .to_list()
        )

        if len(all_dates) < self.min_train_periods + self.test_periods:
            raise ValueError(
                f"Insufficient data: {len(all_dates)} periods, "
                f"need at least {self.min_train_periods + self.test_periods}"
            )

        # Initialize storage for predictions
        bank_predictions: dict[str, list[dict]] = {}

        # Run backtest based on method
        if self.method == "expanding":
            results = self._expanding_window_backtest(
                bank_panel, macro_data, all_dates
            )
        elif self.method == "rolling":
            results = self._rolling_window_backtest(
                bank_panel, macro_data, all_dates
            )
        else:
            results = self._expanding_window_backtest(
                bank_panel, macro_data, all_dates
            )

        return self._compile_results(results, all_dates, macro_data)

    def _expanding_window_backtest(
        self,
        bank_panel: pl.DataFrame,
        macro_data: pl.DataFrame,
        all_dates: list,
    ) -> dict[str, list[dict]]:
        """Run expanding window backtest."""
        predictions = {}

        # Start after minimum training period
        start_idx = self.min_train_periods

        for i in range(start_idx, len(all_dates) - self.test_periods + 1, self.step_size):
            train_end_date = all_dates[i - 1]
            test_dates = all_dates[i:i + self.test_periods]

            # Training data
            train_panel = bank_panel.filter(pl.col("date") <= train_end_date)
            train_macro = macro_data.filter(pl.col("date") <= train_end_date)

            # Fit model on training data
            model = get_sensitivity_model(self.spec)

            try:
                model.fit(train_panel, train_macro)
            except Exception as e:
                print(f"Warning: Model fitting failed at {train_end_date}: {e}")
                continue

            # Generate predictions for test period
            for test_date in test_dates:
                # Get macro values at test date
                test_macro_row = macro_data.filter(pl.col("date") == test_date)

                if test_macro_row.height == 0:
                    continue

                test_macro_dict = {
                    col: float(test_macro_row[col][0])
                    for col in self.spec.macro_features
                    if col in test_macro_row.columns and test_macro_row[col][0] is not None
                }

                # Get actual NIM at test date
                test_bank_data = bank_panel.filter(pl.col("date") == test_date)

                for bank in test_bank_data["ticker"].unique().to_list():
                    if bank not in predictions:
                        predictions[bank] = []

                    bank_row = test_bank_data.filter(pl.col("ticker") == bank)

                    if bank_row.height == 0 or self.spec.target not in bank_row.columns:
                        continue

                    actual = bank_row[self.spec.target][0]
                    if actual is None:
                        continue

                    try:
                        predicted = model.predict(bank, test_macro_dict)

                        predictions[bank].append({
                            "date": test_date,
                            "actual": float(actual),
                            "predicted": float(predicted),
                            "error": float(actual - predicted),
                            "train_end": train_end_date,
                        })
                    except Exception:
                        continue

        return predictions

    def _rolling_window_backtest(
        self,
        bank_panel: pl.DataFrame,
        macro_data: pl.DataFrame,
        all_dates: list,
    ) -> dict[str, list[dict]]:
        """Run rolling window backtest with fixed window size."""
        predictions = {}

        window_size = self.min_train_periods

        for i in range(window_size, len(all_dates) - self.test_periods + 1, self.step_size):
            train_start_date = all_dates[i - window_size]
            train_end_date = all_dates[i - 1]
            test_dates = all_dates[i:i + self.test_periods]

            # Training data (fixed window)
            train_panel = bank_panel.filter(
                (pl.col("date") >= train_start_date) &
                (pl.col("date") <= train_end_date)
            )
            train_macro = macro_data.filter(
                (pl.col("date") >= train_start_date) &
                (pl.col("date") <= train_end_date)
            )

            # Same prediction logic as expanding window
            model = get_sensitivity_model(self.spec)

            try:
                model.fit(train_panel, train_macro)
            except Exception:
                continue

            for test_date in test_dates:
                test_macro_row = macro_data.filter(pl.col("date") == test_date)

                if test_macro_row.height == 0:
                    continue

                test_macro_dict = {
                    col: float(test_macro_row[col][0])
                    for col in self.spec.macro_features
                    if col in test_macro_row.columns and test_macro_row[col][0] is not None
                }

                test_bank_data = bank_panel.filter(pl.col("date") == test_date)

                for bank in test_bank_data["ticker"].unique().to_list():
                    if bank not in predictions:
                        predictions[bank] = []

                    bank_row = test_bank_data.filter(pl.col("ticker") == bank)

                    if bank_row.height == 0 or self.spec.target not in bank_row.columns:
                        continue

                    actual = bank_row[self.spec.target][0]
                    if actual is None:
                        continue

                    try:
                        predicted = model.predict(bank, test_macro_dict)

                        predictions[bank].append({
                            "date": test_date,
                            "actual": float(actual),
                            "predicted": float(predicted),
                            "error": float(actual - predicted),
                            "train_end": train_end_date,
                        })
                    except Exception:
                        continue

        return predictions

    def _compile_results(
        self,
        predictions: dict[str, list[dict]],
        all_dates: list,
        macro_data: pl.DataFrame,
    ) -> BacktestResult:
        """Compile predictions into final results."""
        bank_results = {}
        all_actuals = []
        all_predicted = []

        regime_predictions: dict[str, tuple[list, list]] = {}

        for bank, preds in predictions.items():
            if not preds:
                continue

            pred_df = pl.DataFrame(preds)

            actuals = np.array([p["actual"] for p in preds])
            predicted = np.array([p["predicted"] for p in preds])

            all_actuals.extend(actuals)
            all_predicted.extend(predicted)

            # Bank-level metrics
            metrics = self._compute_metrics(actuals, predicted)

            # Regime-specific metrics
            regime_metrics = {}
            for pred in preds:
                regime = self._classify_regime(macro_data, pred["date"])

                if regime not in regime_predictions:
                    regime_predictions[regime] = ([], [])

                regime_predictions[regime][0].append(pred["actual"])
                regime_predictions[regime][1].append(pred["predicted"])

            bank_results[bank] = BankBacktestResult(
                ticker=bank,
                metrics=metrics,
                predictions=pred_df,
                regime_metrics=regime_metrics,
            )

        # Aggregate metrics
        if all_actuals:
            aggregate_metrics = self._compute_metrics(
                np.array(all_actuals),
                np.array(all_predicted)
            )
        else:
            aggregate_metrics = BacktestMetrics(
                mae=np.nan, rmse=np.nan, mape=np.nan, r_squared=np.nan,
                directional_accuracy=np.nan, up_accuracy=np.nan,
                down_accuracy=np.nan, max_error=np.nan, avg_bias=np.nan,
                n_predictions=0
            )

        # Regime-level metrics
        regime_performance = {}
        for regime, (actuals, predicted) in regime_predictions.items():
            if actuals:
                regime_performance[regime] = self._compute_metrics(
                    np.array(actuals),
                    np.array(predicted)
                )

        # Bank ranking by RMSE
        ranking_data = [
            {"ticker": bank, "rmse": res.metrics.rmse, "mae": res.metrics.mae,
             "r_squared": res.metrics.r_squared, "dir_accuracy": res.metrics.directional_accuracy}
            for bank, res in bank_results.items()
        ]
        bank_ranking = pl.DataFrame(ranking_data).sort("rmse")

        # Determine date range
        all_pred_dates = []
        for preds in predictions.values():
            all_pred_dates.extend([p["date"] for p in preds])

        if all_pred_dates:
            backtest_start = min(all_pred_dates)
            backtest_end = max(all_pred_dates)
        else:
            backtest_start = all_dates[0] if all_dates else datetime.now()
            backtest_end = all_dates[-1] if all_dates else datetime.now()

        return BacktestResult(
            spec_name=self.spec.name,
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            aggregate_metrics=aggregate_metrics,
            bank_results=bank_results,
            bank_ranking=bank_ranking,
            regime_performance=regime_performance,
            metadata={
                "method": self.method,
                "min_train_periods": self.min_train_periods,
                "test_periods": self.test_periods,
                "step_size": self.step_size,
            },
        )

    def compare_specifications(
        self,
        specs: list[MacroSensitivitySpec],
        bank_panel: pl.DataFrame,
        macro_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Compare multiple model specifications.

        Args:
            specs: List of specifications to compare
            bank_panel: Bank data
            macro_data: Macro data

        Returns:
            DataFrame comparing specifications
        """
        results = []

        for spec in specs:
            backtester = MacroSensitivityBacktester(
                spec=spec,
                method=self.method,
                min_train_periods=self.min_train_periods,
                test_periods=self.test_periods,
            )

            try:
                result = backtester.backtest(bank_panel, macro_data)

                results.append({
                    "spec_name": spec.name,
                    "mae": result.aggregate_metrics.mae,
                    "rmse": result.aggregate_metrics.rmse,
                    "r_squared": result.aggregate_metrics.r_squared,
                    "directional_accuracy": result.aggregate_metrics.directional_accuracy,
                    "n_predictions": result.aggregate_metrics.n_predictions,
                })

            except Exception as e:
                results.append({
                    "spec_name": spec.name,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r_squared": np.nan,
                    "directional_accuracy": np.nan,
                    "n_predictions": 0,
                    "error": str(e),
                })

        return pl.DataFrame(results).sort("rmse")
