"""
Common Utilities

Shared utility functions for:
- Date/time operations
- Statistical calculations
- Formatting
- Data transformations
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Union

import numpy as np
import polars as pl


# =============================================================================
# Date Utilities
# =============================================================================

def to_quarterly(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    """
    Aggregate data to quarterly frequency.

    Uses end-of-quarter values for alignment with financial reporting.
    """
    if date_col not in df.columns:
        return df

    df = df.with_columns(
        pl.col(date_col).dt.truncate("1q").alias("quarter")
    )

    value_cols = [c for c in df.columns if c not in [date_col, "quarter"]]

    quarterly = (
        df
        .group_by("quarter")
        .agg([pl.col(c).drop_nulls().last().alias(c) for c in value_cols])
        .sort("quarter")
        .rename({"quarter": date_col})
    )

    return quarterly


def get_quarter(date: datetime) -> int:
    """Get quarter number (1-4) for a date."""
    return (date.month - 1) // 3 + 1


def get_quarter_start(date: datetime) -> datetime:
    """Get the start date of the quarter containing the given date."""
    quarter = get_quarter(date)
    month = (quarter - 1) * 3 + 1
    return datetime(date.year, month, 1)


def get_quarter_end(date: datetime) -> datetime:
    """Get the end date of the quarter containing the given date."""
    quarter = get_quarter(date)
    if quarter == 4:
        return datetime(date.year, 12, 31)
    else:
        next_quarter_start = datetime(date.year, quarter * 3 + 1, 1)
        return next_quarter_start - timedelta(days=1)


def quarters_between(start: datetime, end: datetime) -> int:
    """Calculate number of quarters between two dates."""
    start_q = get_quarter(start) + (start.year * 4)
    end_q = get_quarter(end) + (end.year * 4)
    return end_q - start_q


# =============================================================================
# Statistical Utilities
# =============================================================================

def compute_yoy_growth(
    df: pl.DataFrame,
    value_col: str,
    periods: int = 4,
    suffix: str = "_growth_yoy",
) -> pl.DataFrame:
    """
    Compute year-over-year growth rate.

    Args:
        df: DataFrame with value column
        value_col: Column to compute growth for
        periods: Number of periods for lag (4 for quarterly, 12 for monthly)
        suffix: Suffix for new column name

    Returns:
        DataFrame with growth column added
    """
    growth_col = f"{value_col}{suffix}"
    return df.with_columns(
        ((pl.col(value_col) / pl.col(value_col).shift(periods) - 1) * 100)
        .alias(growth_col)
    )


def compute_rolling_stats(
    df: pl.DataFrame,
    value_col: str,
    window: int = 20,
    stats: list[str] = None,
    group_by: Optional[str] = None,
) -> pl.DataFrame:
    """
    Compute rolling statistics.

    Args:
        df: DataFrame
        value_col: Column for statistics
        window: Rolling window size
        stats: Statistics to compute ["mean", "std", "min", "max", "median"]
        group_by: Optional column to group by before rolling

    Returns:
        DataFrame with rolling stat columns added
    """
    if stats is None:
        stats = ["mean", "std"]

    result = df.clone()

    for stat in stats:
        col_name = f"{value_col}_rolling_{stat}_{window}"

        if stat == "mean":
            expr = pl.col(value_col).rolling_mean(window_size=window, min_periods=1)
        elif stat == "std":
            expr = pl.col(value_col).rolling_std(window_size=window, min_periods=2)
        elif stat == "min":
            expr = pl.col(value_col).rolling_min(window_size=window, min_periods=1)
        elif stat == "max":
            expr = pl.col(value_col).rolling_max(window_size=window, min_periods=1)
        elif stat == "median":
            expr = pl.col(value_col).rolling_median(window_size=window, min_periods=1)
        else:
            continue

        if group_by:
            expr = expr.over(group_by)

        result = result.with_columns(expr.alias(col_name))

    return result


def compute_zscore(
    df: pl.DataFrame,
    value_col: str,
    group_by: Optional[str] = None,
) -> pl.DataFrame:
    """
    Compute cross-sectional z-score.

    Args:
        df: DataFrame
        value_col: Column to standardize
        group_by: Column defining cross-section groups (e.g., "date")

    Returns:
        DataFrame with z-score column added
    """
    zscore_col = f"{value_col}_zscore"

    if group_by:
        # Cross-sectional z-score (standardize within each group)
        result = df.with_columns([
            pl.col(value_col).mean().over(group_by).alias("_mean"),
            pl.col(value_col).std().over(group_by).alias("_std"),
        ])

        result = result.with_columns(
            pl.when(pl.col("_std") > 0.001)
            .then((pl.col(value_col) - pl.col("_mean")) / pl.col("_std"))
            .otherwise(0.0)
            .alias(zscore_col)
        ).drop(["_mean", "_std"])

    else:
        # Time-series z-score
        mean_val = df.select(pl.col(value_col).mean()).item()
        std_val = df.select(pl.col(value_col).std()).item()

        if std_val and std_val > 0.001:
            result = df.with_columns(
                ((pl.col(value_col) - mean_val) / std_val).alias(zscore_col)
            )
        else:
            result = df.with_columns(pl.lit(0.0).alias(zscore_col))

    return result


def winsorize(
    values: np.ndarray,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> np.ndarray:
    """
    Winsorize values at specified percentiles.

    Args:
        values: Array of values
        lower_pct: Lower percentile (default 1%)
        upper_pct: Upper percentile (default 99%)

    Returns:
        Winsorized array
    """
    lower = np.nanpercentile(values, lower_pct * 100)
    upper = np.nanpercentile(values, upper_pct * 100)
    return np.clip(values, lower, upper)


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_pct(value: float, decimals: int = 1) -> str:
    """Format a value as percentage string."""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_currency(
    value: float,
    unit: str = "B",
    decimals: int = 1,
) -> str:
    """
    Format a value as currency string.

    Args:
        value: Value in base units
        unit: Display unit ("B" for billions, "M" for millions, "K" for thousands)
        decimals: Number of decimal places

    Returns:
        Formatted string like "$1.5B"
    """
    if value is None or np.isnan(value):
        return "N/A"

    divisors = {"B": 1e9, "M": 1e6, "K": 1e3, "": 1}
    divisor = divisors.get(unit, 1)

    scaled = value / divisor
    return f"${scaled:,.{decimals}f}{unit}"


def format_delta(value: float, decimals: int = 2) -> str:
    """Format a change value with +/- sign."""
    if value is None or np.isnan(value):
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}"


# =============================================================================
# Data Validation Utilities
# =============================================================================

def check_data_coverage(
    df: pl.DataFrame,
    required_cols: list[str],
    min_rows: int = 10,
) -> dict[str, Union[bool, str, int]]:
    """
    Check data coverage and completeness.

    Returns:
        Dictionary with coverage stats and issues
    """
    result = {
        "valid": True,
        "issues": [],
        "row_count": df.height,
        "column_count": len(df.columns),
    }

    # Check row count
    if df.height < min_rows:
        result["valid"] = False
        result["issues"].append(f"Insufficient rows: {df.height} < {min_rows}")

    # Check required columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        result["valid"] = False
        result["issues"].append(f"Missing columns: {missing_cols}")

    # Check null rates
    for col in required_cols:
        if col in df.columns:
            null_rate = df.select(pl.col(col).is_null().mean()).item()
            if null_rate and null_rate > 0.5:
                result["issues"].append(f"High null rate for {col}: {null_rate:.1%}")

    return result
