"""
Backtesting Framework for Duration Mismatch Models

Tests the predictive relationship:
    duration_exposure → earnings/stock volatility

Evaluation metrics:
1. Directional accuracy: Does high duration predict high volatility?
2. Magnitude: Does the predicted impact match actual?
3. Timing: How many quarters ahead is the signal useful?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import polars as pl
from scipy import stats

from .forecast import ARDLModel, ARDLSpec


@dataclass
class BacktestMetrics:
    """Performance metrics for backtest."""

    # Prediction accuracy
    mae: float
    rmse: float
    mape: float
    r_squared: float

    # Directional accuracy
    directional_accuracy: float
    hit_rate_high_vol: float  # Correctly predicted high volatility
    hit_rate_low_vol: float   # Correctly predicted low volatility

    # Cross-sectional metrics
    rank_correlation: float  # Spearman correlation of rankings
    top_quartile_accuracy: float  # Accuracy for most vulnerable banks

    n_predictions: int


@dataclass
class BankBacktestResult:
    """Backtest results for a single bank."""

    ticker: str
    metrics: BacktestMetrics
    predictions: pl.DataFrame
    coefficient_stability: dict[str, float]  # Stability of coefficients over time


@dataclass
class BacktestResult:
    """Complete backtest results."""

    model_name: str
    backtest_start: datetime
    backtest_end: datetime

    # Aggregate metrics
    aggregate_metrics: BacktestMetrics

    # Bank-level results
    bank_results: dict[str, BankBacktestResult]

    # Cross-sectional performance
    ranking_accuracy: pl.DataFrame  # How well did we rank banks?

    # Time-varying performance
    rolling_metrics: pl.DataFrame

    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"Duration Mismatch Backtest: {self.model_name}",
            f"Period: {self.backtest_start.date()} to {self.backtest_end.date()}",
            "-" * 50,
            "Aggregate Metrics:",
            f"  MAE:  {self.aggregate_metrics.mae:.4f}",
            f"  RMSE: {self.aggregate_metrics.rmse:.4f}",
            f"  R²:   {self.aggregate_metrics.r_squared:.4f}",
            f"  Directional Accuracy: {self.aggregate_metrics.directional_accuracy:.1%}",
            f"  Rank Correlation: {self.aggregate_metrics.rank_correlation:.3f}",
            "-" * 50,
            f"N Banks: {len(self.bank_results)}",
            f"N Predictions: {self.aggregate_metrics.n_predictions}",
        ]
        return "\n".join(lines)


class DurationMismatchBacktester:
    """
    Backtest duration → volatility prediction models.

    Key tests:
    1. Does predicted impact correlate with actual volatility?
    2. Can we rank banks by vulnerability correctly?
    3. Is the signal stable over time?
    """

    def __init__(
        self,
        target: str = "earnings_volatility",
        method: str = "expanding",
        min_train_periods: int = 12,
        test_periods: int = 4,
        step_size: int = 1,
    ):
        """
        Initialize backtester.

        Args:
            target: Variable to predict
            method: "expanding" or "rolling"
            min_train_periods: Minimum training quarters
            test_periods: Periods to test at each step
            step_size: How many periods to step forward
        """
        self.target = target
        self.method = method
        self.min_train_periods = min_train_periods
        self.test_periods = test_periods
        self.step_size = step_size

    def backtest(
        self,
        duration_data: pl.DataFrame,
    ) -> BacktestResult:
        """
        Run full backtest.

        Args:
            duration_data: DataFrame with duration exposure and volatility

        Returns:
            BacktestResult with all metrics
        """
        # Get unique dates
        all_dates = (
            duration_data
            .select("date")
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

        # Run backtest
        predictions = self._run_backtest(duration_data, all_dates)

        # Compile results
        return self._compile_results(predictions, all_dates, duration_data)

    def _run_backtest(
        self,
        data: pl.DataFrame,
        all_dates: list,
    ) -> dict[str, list[dict]]:
        """Run the backtest loop."""
        predictions: dict[str, list[dict]] = {}

        start_idx = self.min_train_periods

        for i in range(start_idx, len(all_dates) - self.test_periods + 1, self.step_size):
            if self.method == "expanding":
                train_data = data.filter(pl.col("date") <= all_dates[i - 1])
            else:
                # Rolling
                window_start = all_dates[max(0, i - self.min_train_periods)]
                train_data = data.filter(
                    (pl.col("date") >= window_start) &
                    (pl.col("date") <= all_dates[i - 1])
                )

            test_dates = all_dates[i:i + self.test_periods]

            # Fit model for each bank
            for ticker in data["ticker"].unique().to_list():
                bank_train = train_data.filter(pl.col("ticker") == ticker)
                bank_test = data.filter(
                    (pl.col("ticker") == ticker) &
                    (pl.col("date").is_in(test_dates))
                )

                if bank_train.height < 8:  # Minimum for ARDL
                    continue

                if self.target not in bank_train.columns:
                    continue

                # Check for target data
                train_target = bank_train.filter(pl.col(self.target).is_not_null())
                if train_target.height < 8:
                    continue

                try:
                    # Fit ARDL model
                    spec = ARDLSpec(name="backtest", target=self.target)
                    model = ARDLModel(spec)
                    model.fit(bank_train)

                    # Generate predictions
                    for test_date in test_dates:
                        test_row = bank_test.filter(pl.col("date") == test_date)

                        if test_row.height == 0:
                            continue

                        actual = test_row[self.target][0]
                        if actual is None:
                            continue

                        # Predict
                        forecast = model.predict(bank_train, horizon=1)

                        if not forecast:
                            continue

                        if ticker not in predictions:
                            predictions[ticker] = []

                        predictions[ticker].append({
                            "date": test_date,
                            "actual": float(actual),
                            "predicted": forecast[0].point_forecast,
                            "lower": forecast[0].lower_bound,
                            "upper": forecast[0].upper_bound,
                            "train_end": all_dates[i - 1],
                        })

                except Exception:
                    continue

        return predictions

    def _compile_results(
        self,
        predictions: dict[str, list[dict]],
        all_dates: list,
        data: pl.DataFrame,
    ) -> BacktestResult:
        """Compile predictions into final results."""
        bank_results = {}
        all_actuals = []
        all_predicted = []
        all_rankings = []

        for ticker, preds in predictions.items():
            if not preds:
                continue

            pred_df = pl.DataFrame(preds)
            actuals = np.array([p["actual"] for p in preds])
            predicted = np.array([p["predicted"] for p in preds])

            all_actuals.extend(actuals)
            all_predicted.extend(predicted)

            # Bank-level metrics
            metrics = self._compute_metrics(actuals, predicted)

            # Coefficient stability (std of predictions relative to actuals)
            coef_stability = {
                "prediction_std": float(np.std(predicted)),
                "actual_std": float(np.std(actuals)),
                "correlation": float(np.corrcoef(actuals, predicted)[0, 1]) if len(actuals) > 1 else 0,
            }

            bank_results[ticker] = BankBacktestResult(
                ticker=ticker,
                metrics=metrics,
                predictions=pred_df,
                coefficient_stability=coef_stability,
            )

        # Cross-sectional ranking accuracy
        ranking_accuracy = self._compute_ranking_accuracy(predictions, data)

        # Aggregate metrics
        if all_actuals:
            aggregate_metrics = self._compute_metrics(
                np.array(all_actuals),
                np.array(all_predicted),
            )

            # Add rank correlation
            if ranking_accuracy.height > 0:
                aggregate_metrics.rank_correlation = self._compute_rank_correlation(ranking_accuracy)
            else:
                aggregate_metrics.rank_correlation = 0.0
        else:
            aggregate_metrics = self._empty_metrics()

        # Rolling metrics
        rolling_metrics = self._compute_rolling_metrics(predictions)

        # Date range
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
            model_name=f"ARDL_{self.target}",
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            aggregate_metrics=aggregate_metrics,
            bank_results=bank_results,
            ranking_accuracy=ranking_accuracy,
            rolling_metrics=rolling_metrics,
            metadata={
                "method": self.method,
                "min_train_periods": self.min_train_periods,
                "test_periods": self.test_periods,
            },
        )

    def _compute_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> BacktestMetrics:
        """Compute prediction metrics."""
        if len(actual) == 0:
            return self._empty_metrics()

        errors = actual - predicted
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        # MAPE
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
            actual_dir = np.sign(np.diff(actual))
            pred_dir = np.sign(np.diff(predicted))
            dir_correct = actual_dir == pred_dir
            directional_accuracy = float(np.mean(dir_correct))

            # High/low volatility accuracy
            high_vol_mask = actual[1:] > np.median(actual)
            low_vol_mask = ~high_vol_mask

            hit_high = float(np.mean(dir_correct[high_vol_mask])) if high_vol_mask.sum() > 0 else 0.5
            hit_low = float(np.mean(dir_correct[low_vol_mask])) if low_vol_mask.sum() > 0 else 0.5
        else:
            directional_accuracy = 0.5
            hit_high = 0.5
            hit_low = 0.5

        return BacktestMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            r_squared=r_squared,
            directional_accuracy=directional_accuracy,
            hit_rate_high_vol=hit_high,
            hit_rate_low_vol=hit_low,
            rank_correlation=0.0,  # Computed separately
            top_quartile_accuracy=0.0,  # Computed separately
            n_predictions=len(actual),
        )

    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics."""
        return BacktestMetrics(
            mae=np.nan, rmse=np.nan, mape=np.nan, r_squared=np.nan,
            directional_accuracy=np.nan, hit_rate_high_vol=np.nan,
            hit_rate_low_vol=np.nan, rank_correlation=np.nan,
            top_quartile_accuracy=np.nan, n_predictions=0,
        )

    def _compute_ranking_accuracy(
        self,
        predictions: dict[str, list[dict]],
        data: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute how well we rank banks by vulnerability."""
        records = []

        # Group predictions by date
        date_preds: dict[Any, list[dict]] = {}
        for ticker, preds in predictions.items():
            for p in preds:
                date = p["date"]
                if date not in date_preds:
                    date_preds[date] = []
                date_preds[date].append({
                    "ticker": ticker,
                    "predicted": p["predicted"],
                    "actual": p["actual"],
                })

        for date, preds in date_preds.items():
            if len(preds) < 3:
                continue

            df = pl.DataFrame(preds)

            # Rank by predicted and actual
            df = df.with_columns([
                pl.col("predicted").rank(descending=True).alias("predicted_rank"),
                pl.col("actual").rank(descending=True).alias("actual_rank"),
            ])

            # Spearman correlation
            try:
                rho, _ = stats.spearmanr(df["predicted_rank"], df["actual_rank"])
            except Exception:
                rho = 0.0

            # Top quartile accuracy
            n_top = max(1, len(preds) // 4)
            top_predicted = set(df.sort("predicted_rank").head(n_top)["ticker"].to_list())
            top_actual = set(df.sort("actual_rank").head(n_top)["ticker"].to_list())
            top_overlap = len(top_predicted & top_actual) / n_top

            records.append({
                "date": date,
                "n_banks": len(preds),
                "rank_correlation": rho,
                "top_quartile_overlap": top_overlap,
            })

        return pl.DataFrame(records)

    def _compute_rank_correlation(self, ranking_df: pl.DataFrame) -> float:
        """Compute average rank correlation across dates."""
        if ranking_df.height == 0:
            return 0.0

        return float(ranking_df["rank_correlation"].mean())

    def _compute_rolling_metrics(
        self,
        predictions: dict[str, list[dict]],
        window: int = 8,
    ) -> pl.DataFrame:
        """Compute rolling metrics over time."""
        # Combine all predictions with dates
        all_preds = []
        for ticker, preds in predictions.items():
            for p in preds:
                all_preds.append({
                    "date": p["date"],
                    "ticker": ticker,
                    "actual": p["actual"],
                    "predicted": p["predicted"],
                    "error": p["actual"] - p["predicted"],
                    "abs_error": abs(p["actual"] - p["predicted"]),
                })

        if not all_preds:
            return pl.DataFrame()

        df = pl.DataFrame(all_preds).sort("date")

        # Rolling metrics by date
        rolling = (
            df.group_by("date")
            .agg([
                pl.col("abs_error").mean().alias("mae"),
                (pl.col("error") ** 2).mean().sqrt().alias("rmse"),
                pl.count().alias("n_predictions"),
            ])
            .sort("date")
        )

        # Add rolling average
        rolling = rolling.with_columns([
            pl.col("mae").rolling_mean(window_size=window).alias("rolling_mae"),
            pl.col("rmse").rolling_mean(window_size=window).alias("rolling_rmse"),
        ])

        return rolling

    def test_predictive_power(
        self,
        data: pl.DataFrame,
        horizons: list[int] = [1, 2, 4, 8],
    ) -> pl.DataFrame:
        """
        Test predictive power at different horizons.

        Args:
            data: Duration data
            horizons: List of horizons to test

        Returns:
            DataFrame with performance by horizon
        """
        results = []

        for h in horizons:
            # Modify test_periods and run
            original_test = self.test_periods
            self.test_periods = h

            try:
                result = self.backtest(data)

                results.append({
                    "horizon": h,
                    "mae": result.aggregate_metrics.mae,
                    "rmse": result.aggregate_metrics.rmse,
                    "r_squared": result.aggregate_metrics.r_squared,
                    "directional_accuracy": result.aggregate_metrics.directional_accuracy,
                    "rank_correlation": result.aggregate_metrics.rank_correlation,
                    "n_predictions": result.aggregate_metrics.n_predictions,
                })

            except Exception as e:
                results.append({
                    "horizon": h,
                    "mae": np.nan,
                    "error": str(e),
                })

            self.test_periods = original_test

        return pl.DataFrame(results)
