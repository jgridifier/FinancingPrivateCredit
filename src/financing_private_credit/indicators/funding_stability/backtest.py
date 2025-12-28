"""
Backtesting Framework for Funding Stability Score

Tests the predictive power of the Funding Resilience Score for:
1. Procyclical lending behavior (credit growth in downturns)
2. Loan loss provisions
3. Stock performance during stress periods
4. FHLB advance drawdowns

Key hypothesis: Banks with lower scores should:
- Cut lending more in downturns
- Have higher provisions during stress
- Underperform in credit boom periods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Target variables to predict
    targets: list[str] = field(default_factory=lambda: [
        "lending_growth",
        "provision_ratio",
        "stock_return",
    ])

    # Backtesting method
    method: str = "expanding"  # "expanding", "rolling", "walk_forward"
    min_train_periods: int = 12  # Minimum quarters for training
    test_periods: int = 4  # Periods to test at each step

    # Evaluation horizons
    horizons: list[int] = field(default_factory=lambda: [1, 2, 4])

    # Classification thresholds
    score_percentile_low: float = 0.33  # Bottom third = "vulnerable"
    score_percentile_high: float = 0.67  # Top third = "resilient"


@dataclass
class BacktestResult:
    """Results from backtesting."""

    config: BacktestConfig
    test_periods: list[datetime]

    # Prediction accuracy
    directional_accuracy: dict[str, float]  # % correct direction
    ranking_accuracy: dict[str, float]  # Rank correlation

    # Quantile analysis
    low_score_outcomes: dict[str, float]  # Avg outcome for low score banks
    high_score_outcomes: dict[str, float]  # Avg outcome for high score banks
    spread: dict[str, float]  # High - Low difference

    # Statistical tests
    t_stats: dict[str, float]
    p_values: dict[str, float]

    # Time series of predictions
    predictions_df: Optional[pl.DataFrame] = None

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            "Funding Stability Score Backtest Results",
            "=" * 50,
            f"Test periods: {len(self.test_periods)}",
            "",
            "Predictive Power by Target:",
        ]

        for target in self.config.targets:
            lines.extend([
                f"\n  {target}:",
                f"    Directional Accuracy: {self.directional_accuracy.get(target, 0):.1%}",
                f"    Rank Correlation: {self.ranking_accuracy.get(target, 0):.3f}",
                f"    Low Score Outcome: {self.low_score_outcomes.get(target, 0):.3f}",
                f"    High Score Outcome: {self.high_score_outcomes.get(target, 0):.3f}",
                f"    Spread: {self.spread.get(target, 0):.3f}",
                f"    t-stat: {self.t_stats.get(target, 0):.2f}",
            ])

        return "\n".join(lines)


class FundingStabilityBacktester:
    """
    Backtester for Funding Stability Score.

    Tests whether the score predicts procyclical behavior.
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
    ):
        """
        Initialize backtester.

        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self._results: Optional[BacktestResult] = None

    def backtest(
        self,
        funding_data: pl.DataFrame,
        outcome_data: pl.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            funding_data: Historical funding stability scores
            outcome_data: Outcome variables (lending growth, provisions, etc.)

        Returns:
            BacktestResult with predictive power metrics
        """
        # Merge data
        merged = funding_data.join(
            outcome_data,
            on=["date", "ticker"],
            how="inner"
        ).sort(["ticker", "date"])

        if merged.height == 0:
            return self._empty_result()

        # Get unique dates
        dates = merged["date"].unique().sort().to_list()

        if len(dates) < self.config.min_train_periods + self.config.test_periods:
            return self._empty_result()

        # Run expanding window backtest
        predictions = []
        test_dates = []

        for i in range(self.config.min_train_periods, len(dates) - max(self.config.horizons)):
            train_end = dates[i]
            test_dates.append(train_end)

            # Get train data (up to train_end)
            train = merged.filter(pl.col("date") <= train_end)

            # For each horizon, get outcome
            for h in self.config.horizons:
                if i + h >= len(dates):
                    continue

                outcome_date = dates[i + h]
                outcome = merged.filter(pl.col("date") == outcome_date)

                # Score at train_end
                scores = train.group_by("ticker").agg(
                    pl.col("funding_resilience_score").last().alias("score")
                )

                # Merge with outcomes
                pred = scores.join(
                    outcome.select(["ticker"] + self.config.targets),
                    on="ticker",
                    how="inner"
                )

                pred = pred.with_columns([
                    pl.lit(train_end).alias("score_date"),
                    pl.lit(outcome_date).alias("outcome_date"),
                    pl.lit(h).alias("horizon"),
                ])

                predictions.append(pred)

        if not predictions:
            return self._empty_result()

        predictions_df = pl.concat(predictions)

        # Calculate metrics
        result = self._calculate_metrics(predictions_df, test_dates)
        result.predictions_df = predictions_df

        self._results = result
        return result

    def _calculate_metrics(
        self,
        predictions_df: pl.DataFrame,
        test_dates: list[datetime],
    ) -> BacktestResult:
        """Calculate backtest metrics."""
        directional_accuracy = {}
        ranking_accuracy = {}
        low_score_outcomes = {}
        high_score_outcomes = {}
        spread = {}
        t_stats = {}
        p_values = {}

        for target in self.config.targets:
            if target not in predictions_df.columns:
                continue

            df = predictions_df.filter(pl.col(target).is_not_null())
            if df.height == 0:
                continue

            # Calculate percentile thresholds
            low_thresh = df["score"].quantile(self.config.score_percentile_low)
            high_thresh = df["score"].quantile(self.config.score_percentile_high)

            # Low vs high score outcomes
            low_df = df.filter(pl.col("score") <= low_thresh)
            high_df = df.filter(pl.col("score") >= high_thresh)

            low_mean = low_df[target].mean() if low_df.height > 0 else 0
            high_mean = high_df[target].mean() if high_df.height > 0 else 0

            low_score_outcomes[target] = low_mean
            high_score_outcomes[target] = high_mean
            spread[target] = high_mean - low_mean

            # Directional accuracy
            # For lending_growth and stock_return: high score should predict better outcome
            # For provision_ratio: high score should predict lower provisions
            expected_sign = -1 if target == "provision_ratio" else 1

            median_score = df["score"].median()
            median_outcome = df[target].median()

            correct = df.with_columns(
                (
                    ((pl.col("score") >= median_score) & (pl.col(target) * expected_sign >= median_outcome * expected_sign)) |
                    ((pl.col("score") < median_score) & (pl.col(target) * expected_sign < median_outcome * expected_sign))
                ).alias("correct")
            )
            directional_accuracy[target] = correct["correct"].mean() if correct.height > 0 else 0.5

            # Rank correlation
            scores = df["score"].to_numpy()
            outcomes = df[target].to_numpy()
            try:
                from scipy import stats
                corr, pval = stats.spearmanr(scores, outcomes * expected_sign)
                ranking_accuracy[target] = corr
                p_values[target] = pval
            except ImportError:
                # Simple correlation
                corr = np.corrcoef(scores, outcomes)[0, 1] * expected_sign
                ranking_accuracy[target] = corr
                p_values[target] = 0.05

            # t-test for spread
            if low_df.height > 1 and high_df.height > 1:
                low_vals = low_df[target].to_numpy()
                high_vals = high_df[target].to_numpy()
                try:
                    from scipy import stats
                    t, pval = stats.ttest_ind(high_vals * expected_sign, low_vals * expected_sign)
                    t_stats[target] = t
                    p_values[target] = pval
                except ImportError:
                    t_stats[target] = spread[target] / (np.std(outcomes) / np.sqrt(df.height) + 1e-10)

        return BacktestResult(
            config=self.config,
            test_periods=test_dates,
            directional_accuracy=directional_accuracy,
            ranking_accuracy=ranking_accuracy,
            low_score_outcomes=low_score_outcomes,
            high_score_outcomes=high_score_outcomes,
            spread=spread,
            t_stats=t_stats,
            p_values=p_values,
        )

    def _empty_result(self) -> BacktestResult:
        """Return empty result when data is insufficient."""
        return BacktestResult(
            config=self.config,
            test_periods=[],
            directional_accuracy={},
            ranking_accuracy={},
            low_score_outcomes={},
            high_score_outcomes={},
            spread={},
            t_stats={},
            p_values={},
        )

    def test_stress_periods(
        self,
        funding_data: pl.DataFrame,
        outcome_data: pl.DataFrame,
        stress_periods: list[tuple[datetime, datetime]],
    ) -> pl.DataFrame:
        """
        Test predictive power specifically during stress periods.

        Args:
            funding_data: Historical funding stability scores
            outcome_data: Outcome variables
            stress_periods: List of (start, end) date tuples for stress periods

        Returns:
            DataFrame with stress period analysis
        """
        results = []

        for start, end in stress_periods:
            # Get scores at start of stress period
            pre_stress = funding_data.filter(
                pl.col("date") < start
            ).group_by("ticker").agg(
                pl.col("funding_resilience_score").last().alias("pre_stress_score")
            )

            # Get outcomes during stress period
            stress_outcomes = outcome_data.filter(
                (pl.col("date") >= start) & (pl.col("date") <= end)
            ).group_by("ticker").agg([
                pl.col(t).mean().alias(f"stress_{t}")
                for t in self.config.targets
                if t in outcome_data.columns
            ])

            # Merge
            merged = pre_stress.join(stress_outcomes, on="ticker", how="inner")

            # Classify by score
            median = merged["pre_stress_score"].median()
            merged = merged.with_columns([
                (pl.col("pre_stress_score") >= median).alias("high_score"),
                pl.lit(start).alias("stress_start"),
                pl.lit(end).alias("stress_end"),
            ])

            results.append(merged)

        if not results:
            return pl.DataFrame()

        return pl.concat(results)

    def get_feature_importance(self) -> pl.DataFrame:
        """
        Analyze which score components are most predictive.

        Returns:
            DataFrame with component importance rankings
        """
        if self._results is None or self._results.predictions_df is None:
            return pl.DataFrame()

        # Would correlate individual components with outcomes
        # Placeholder for now
        return pl.DataFrame()


# Notable stress periods for backtesting
STRESS_PERIODS = [
    (datetime(2008, 9, 1), datetime(2009, 3, 31)),   # GFC
    (datetime(2020, 2, 1), datetime(2020, 6, 30)),   # COVID
    (datetime(2023, 3, 1), datetime(2023, 5, 31)),   # SVB/regional bank crisis
]
