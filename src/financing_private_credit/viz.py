"""
Visualization module using Vega-Altair.

Reproduces key charts from Boyarchenko & Elias (2024) and extends
with real-time monitoring dashboards.
"""

from __future__ import annotations

from typing import Optional

import altair as alt
import polars as pl


# Configure Altair for better defaults
alt.data_transformers.disable_max_rows()


def chart_credit_to_gdp(
    data: pl.DataFrame,
    date_col: str = "date",
    title: str = "Private Credit to GDP Ratio",
) -> alt.Chart:
    """
    Create a time series chart of credit-to-GDP ratios.

    Args:
        data: DataFrame with date and credit/GDP ratio columns
        date_col: Name of date column
        title: Chart title

    Returns:
        Altair Chart object
    """
    # Find credit-to-GDP columns
    ratio_cols = [c for c in data.columns if c.endswith("_to_gdp")]

    if not ratio_cols:
        raise ValueError("No credit-to-GDP columns found")

    # Melt to long format for Altair
    df_long = data.select([date_col] + ratio_cols).unpivot(
        index=date_col,
        variable_name="series",
        value_name="ratio",
    )

    # Clean series names
    df_long = df_long.with_columns(
        pl.col("series").str.replace("_to_gdp", "").str.replace("_", " ").str.to_titlecase()
    )

    chart = alt.Chart(df_long.to_pandas()).mark_line(strokeWidth=2).encode(
        x=alt.X(f"{date_col}:T", title="Date"),
        y=alt.Y("ratio:Q", title="Credit / GDP (%)"),
        color=alt.Color("series:N", title="Credit Type", legend=alt.Legend(orient="bottom")),
        tooltip=[
            alt.Tooltip(f"{date_col}:T", title="Date", format="%Y-%m"),
            alt.Tooltip("series:N", title="Type"),
            alt.Tooltip("ratio:Q", title="Ratio", format=".1f"),
        ],
    ).properties(
        width=700,
        height=400,
        title=title,
    )

    return chart


def chart_lender_composition(
    data: pl.DataFrame,
    date_col: str = "date",
    bank_col: str = "bank_share",
    nonbank_col: str = "nonbank_share",
    title: str = "Lender Composition: Bank vs Nonbank Share",
) -> alt.Chart:
    """
    Create stacked area chart showing bank vs nonbank share of credit.

    This is a key visualization from the paper showing the secular
    shift from bank to nonbank financing.

    Args:
        data: DataFrame with date and share columns
        date_col: Name of date column
        bank_col: Name of bank share column
        nonbank_col: Name of nonbank share column
        title: Chart title

    Returns:
        Altair Chart object
    """
    df = data.select([date_col, bank_col, nonbank_col])

    # Melt to long format
    df_long = df.unpivot(
        index=date_col,
        variable_name="lender_type",
        value_name="share",
    )

    # Clean names
    df_long = df_long.with_columns(
        pl.when(pl.col("lender_type") == bank_col)
        .then(pl.lit("Banks"))
        .otherwise(pl.lit("Nonbanks"))
        .alias("lender_type")
    )

    # Define custom color scale
    colors = ["#1f77b4", "#ff7f0e"]  # Blue for banks, orange for nonbanks

    chart = alt.Chart(df_long.to_pandas()).mark_area().encode(
        x=alt.X(f"{date_col}:T", title="Date"),
        y=alt.Y(
            "share:Q",
            title="Share of Credit (%)",
            stack="normalize",
            scale=alt.Scale(domain=[0, 100]),
        ),
        color=alt.Color(
            "lender_type:N",
            title="Lender Type",
            scale=alt.Scale(range=colors),
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip(f"{date_col}:T", title="Date", format="%Y-%m"),
            alt.Tooltip("lender_type:N", title="Lender"),
            alt.Tooltip("share:Q", title="Share", format=".1f"),
        ],
    ).properties(
        width=700,
        height=400,
        title=title,
    )

    return chart


def chart_credit_growth_decomposition(
    data: pl.DataFrame,
    date_col: str = "date",
    title: str = "Credit Growth Decomposition",
) -> alt.Chart:
    """
    Create chart showing contributions to credit growth by lender type.

    Args:
        data: DataFrame with growth contribution columns
        date_col: Name of date column
        title: Chart title

    Returns:
        Altair Chart object
    """
    # Find contribution columns
    contrib_cols = [c for c in data.columns if "contribution" in c.lower()]

    if not contrib_cols:
        # Fall back to growth columns
        contrib_cols = [c for c in data.columns if "growth" in c.lower() and c != "total_credit_growth"]

    if not contrib_cols:
        raise ValueError("No growth/contribution columns found")

    df_long = data.select([date_col] + contrib_cols).unpivot(
        index=date_col,
        variable_name="source",
        value_name="contribution",
    )

    # Clean source names
    df_long = df_long.with_columns(
        pl.col("source")
        .str.replace("_contribution_to_growth", "")
        .str.replace("_growth", "")
        .str.replace("_", " ")
        .str.to_titlecase()
        .alias("source")
    )

    chart = alt.Chart(df_long.to_pandas()).mark_bar().encode(
        x=alt.X(f"{date_col}:T", title="Date"),
        y=alt.Y("contribution:Q", title="Contribution to Growth (pp)"),
        color=alt.Color("source:N", title="Source", legend=alt.Legend(orient="bottom")),
        tooltip=[
            alt.Tooltip(f"{date_col}:T", title="Date", format="%Y-%m"),
            alt.Tooltip("source:N", title="Source"),
            alt.Tooltip("contribution:Q", title="Contribution", format=".2f"),
        ],
    ).properties(
        width=700,
        height=400,
        title=title,
    )

    return chart


def chart_cyclical_comparison(
    data: pl.DataFrame,
    date_col: str = "date",
    bank_cycle_col: str = "bank_credit_cycle",
    nonbank_cycle_col: str = "nonbank_credit_cycle",
    gdp_cycle_col: str = "gdp_cycle",
    title: str = "Cyclical Properties of Bank vs Nonbank Credit",
) -> alt.Chart:
    """
    Create chart comparing cyclical behavior of bank vs nonbank credit.

    Shows the key finding: bank credit is more procyclical than nonbank.

    Args:
        data: DataFrame with cycle columns
        date_col: Name of date column
        bank_cycle_col: Name of bank credit cycle column
        nonbank_cycle_col: Name of nonbank credit cycle column
        gdp_cycle_col: Name of GDP cycle column
        title: Chart title

    Returns:
        Altair Chart object
    """
    cols = [date_col]
    if bank_cycle_col in data.columns:
        cols.append(bank_cycle_col)
    if nonbank_cycle_col in data.columns:
        cols.append(nonbank_cycle_col)
    if gdp_cycle_col in data.columns:
        cols.append(gdp_cycle_col)

    df = data.select(cols)

    # Melt to long format
    df_long = df.unpivot(
        index=date_col,
        variable_name="variable",
        value_name="cycle",
    )

    # Clean names
    df_long = df_long.with_columns(
        pl.col("variable")
        .str.replace("_cycle", "")
        .str.replace("_", " ")
        .str.to_titlecase()
        .alias("variable")
    )

    base = alt.Chart(df_long.to_pandas())

    lines = base.mark_line(strokeWidth=2).encode(
        x=alt.X(f"{date_col}:T", title="Date"),
        y=alt.Y("cycle:Q", title="Deviation from Trend (%)"),
        color=alt.Color("variable:N", title="Variable", legend=alt.Legend(orient="bottom")),
        tooltip=[
            alt.Tooltip(f"{date_col}:T", title="Date", format="%Y-%m"),
            alt.Tooltip("variable:N", title="Variable"),
            alt.Tooltip("cycle:Q", title="Deviation", format=".2f"),
        ],
    )

    # Add zero line
    zero_line = alt.Chart(pl.DataFrame({"y": [0]}).to_pandas()).mark_rule(
        strokeDash=[4, 4],
        strokeWidth=1,
        color="gray",
    ).encode(y="y:Q")

    chart = (lines + zero_line).properties(
        width=700,
        height=400,
        title=title,
    )

    return chart


def chart_crisis_probability_indicator(
    data: pl.DataFrame,
    date_col: str = "date",
    risk_col: str = "elevated_crisis_risk",
    credit_growth_col: str = "total_credit_growth",
    bank_share_col: str = "bank_share",
    title: str = "Crisis Risk Indicator",
) -> alt.Chart:
    """
    Create visualization of crisis probability indicator.

    Based on paper's finding: high credit growth + high bank share = elevated risk.

    Args:
        data: DataFrame with risk indicator data
        date_col: Name of date column
        risk_col: Name of crisis risk indicator column
        credit_growth_col: Name of credit growth column
        bank_share_col: Name of bank share column
        title: Chart title

    Returns:
        Altair Chart object
    """
    # Main chart: credit growth with risk shading
    base = alt.Chart(data.to_pandas())

    # Credit growth line
    growth_line = base.mark_line(color="#1f77b4", strokeWidth=2).encode(
        x=alt.X(f"{date_col}:T", title="Date"),
        y=alt.Y(f"{credit_growth_col}:Q", title="Credit Growth (% YoY)"),
    )

    # Risk indicator as background
    if risk_col in data.columns:
        risk_areas = base.mark_rect(opacity=0.3, color="red").encode(
            x=alt.X(f"{date_col}:T"),
            x2=alt.X2(f"{date_col}:T"),
        ).transform_filter(
            alt.datum[risk_col] == 1
        )

        chart = (risk_areas + growth_line)
    else:
        chart = growth_line

    chart = chart.properties(
        width=700,
        height=400,
        title=title,
    )

    return chart


def chart_elasticity_comparison(
    bank_elasticity: float,
    nonbank_elasticity: float,
    bank_se: float,
    nonbank_se: float,
    title: str = "Credit Supply Elasticities",
) -> alt.Chart:
    """
    Create bar chart comparing bank vs nonbank elasticities.

    Args:
        bank_elasticity: Bank credit elasticity estimate
        nonbank_elasticity: Nonbank credit elasticity estimate
        bank_se: Standard error for bank elasticity
        nonbank_se: Standard error for nonbank elasticity
        title: Chart title

    Returns:
        Altair Chart object
    """
    df = pl.DataFrame({
        "lender_type": ["Banks", "Nonbanks"],
        "elasticity": [bank_elasticity, nonbank_elasticity],
        "se": [bank_se, nonbank_se],
        "ci_lower": [bank_elasticity - 1.96 * bank_se, nonbank_elasticity - 1.96 * nonbank_se],
        "ci_upper": [bank_elasticity + 1.96 * bank_se, nonbank_elasticity + 1.96 * nonbank_se],
    })

    base = alt.Chart(df.to_pandas())

    # Bars
    bars = base.mark_bar(width=50).encode(
        x=alt.X("lender_type:N", title="Lender Type", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("elasticity:Q", title="Output Elasticity"),
        color=alt.Color(
            "lender_type:N",
            scale=alt.Scale(range=["#1f77b4", "#ff7f0e"]),
            legend=None,
        ),
    )

    # Error bars
    error_bars = base.mark_errorbar().encode(
        x=alt.X("lender_type:N"),
        y=alt.Y("ci_lower:Q", title=""),
        y2=alt.Y2("ci_upper:Q"),
    )

    chart = (bars + error_bars).properties(
        width=300,
        height=400,
        title=title,
    )

    return chart


def create_dashboard(
    credit_data: pl.DataFrame,
    title: str = "Private Credit Monitor",
) -> alt.VConcatChart:
    """
    Create comprehensive dashboard combining multiple visualizations.

    Args:
        credit_data: DataFrame with credit and macro data
        title: Dashboard title

    Returns:
        Altair VConcatChart combining multiple views
    """
    charts = []

    # 1. Credit to GDP if available
    ratio_cols = [c for c in credit_data.columns if c.endswith("_to_gdp")]
    if ratio_cols:
        charts.append(chart_credit_to_gdp(credit_data))

    # 2. Lender composition if available
    if "bank_share" in credit_data.columns and "nonbank_share" in credit_data.columns:
        charts.append(chart_lender_composition(credit_data))

    # 3. Credit growth decomposition if available
    growth_cols = [c for c in credit_data.columns if "growth" in c.lower()]
    if growth_cols:
        charts.append(chart_credit_growth_decomposition(credit_data))

    if not charts:
        # Create a simple line chart of available numeric columns
        numeric_cols = [c for c in credit_data.columns if c != "date"]
        if numeric_cols:
            df_long = credit_data.unpivot(
                index="date",
                variable_name="series",
                value_name="value",
            )
            charts.append(
                alt.Chart(df_long.to_pandas()).mark_line().encode(
                    x="date:T",
                    y="value:Q",
                    color="series:N",
                ).properties(width=700, height=400, title="Credit Data")
            )

    if len(charts) == 1:
        return charts[0]

    return alt.vconcat(*charts).properties(title=title)


# =============================================================================
# Credit Boom Leading Indicator Visualizations
# =============================================================================


def chart_lis_heatmap(
    data: pl.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    lis_col: str = "lis",
    title: str = "Lending Intensity Score Heatmap",
) -> alt.Chart:
    """
    Create heatmap of LIS across banks and time.

    Red = aggressive lending (LIS > 1)
    Blue = conservative lending (LIS < -1)

    Args:
        data: DataFrame with date, ticker, lis columns
        date_col: Name of date column
        ticker_col: Name of ticker column
        lis_col: Name of LIS column
        title: Chart title

    Returns:
        Altair Chart object
    """
    chart = alt.Chart(data.to_pandas()).mark_rect().encode(
        x=alt.X(f"{date_col}:T", title="Date"),
        y=alt.Y(f"{ticker_col}:N", title="Bank"),
        color=alt.Color(
            f"{lis_col}:Q",
            title="LIS (SDs)",
            scale=alt.Scale(
                domain=[-2, 0, 2],
                range=["#2166ac", "#f7f7f7", "#b2182b"],
            ),
        ),
        tooltip=[
            alt.Tooltip(f"{date_col}:T", title="Date"),
            alt.Tooltip(f"{ticker_col}:N", title="Bank"),
            alt.Tooltip(f"{lis_col}:Q", title="LIS", format=".2f"),
        ],
    ).properties(
        width=700,
        height=300,
        title=title,
    )

    return chart


def chart_lis_timeseries(
    data: pl.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    lis_col: str = "lis",
    title: str = "Lending Intensity Score Over Time",
) -> alt.Chart:
    """
    Create line chart of LIS for each bank over time.

    Args:
        data: DataFrame with date, ticker, lis columns
        date_col: Name of date column
        ticker_col: Name of ticker column
        lis_col: Name of LIS column
        title: Chart title

    Returns:
        Altair Chart object
    """
    base = alt.Chart(data.to_pandas())

    lines = base.mark_line(strokeWidth=2).encode(
        x=alt.X(f"{date_col}:T", title="Date"),
        y=alt.Y(f"{lis_col}:Q", title="LIS (Standard Deviations)"),
        color=alt.Color(f"{ticker_col}:N", title="Bank"),
        tooltip=[
            alt.Tooltip(f"{date_col}:T", title="Date"),
            alt.Tooltip(f"{ticker_col}:N", title="Bank"),
            alt.Tooltip(f"{lis_col}:Q", title="LIS", format=".2f"),
        ],
    )

    # Add threshold lines at +/- 1 SD
    thresholds = alt.Chart(
        pl.DataFrame({"y": [-1, 0, 1, 2]}).to_pandas()
    ).mark_rule(strokeDash=[4, 4], opacity=0.5).encode(
        y="y:Q",
        color=alt.value("gray"),
    )

    # Add annotations for threshold levels
    threshold_labels = alt.Chart(
        pl.DataFrame({
            "y": [1, 2],
            "label": ["Warning (+1 SD)", "Alert (+2 SD)"]
        }).to_pandas()
    ).mark_text(align="left", dx=5, dy=-5).encode(
        y="y:Q",
        text="label:N",
        color=alt.value("gray"),
    )

    chart = (lines + thresholds).properties(
        width=700,
        height=400,
        title=title,
    )

    return chart


def chart_provision_forecast(
    data: pl.DataFrame,
    date_col: str = "forecast_date",
    ticker_col: str = "ticker",
    forecast_col: str = "point_forecast",
    lower_col: str = "ci_lower",
    upper_col: str = "ci_upper",
    title: str = "Provision Rate Forecasts (3-4 Year Horizon)",
) -> alt.Chart:
    """
    Create chart showing provision forecasts with confidence intervals.

    Args:
        data: DataFrame with forecast data
        date_col: Name of forecast date column
        ticker_col: Name of ticker column
        forecast_col: Name of point forecast column
        lower_col: Name of CI lower bound column
        upper_col: Name of CI upper bound column
        title: Chart title

    Returns:
        Altair Chart object
    """
    base = alt.Chart(data.to_pandas())

    # Confidence interval band
    band = base.mark_area(opacity=0.3).encode(
        x=alt.X(f"{date_col}:T", title="Date"),
        y=alt.Y(f"{lower_col}:Q", title=""),
        y2=alt.Y2(f"{upper_col}:Q"),
        color=alt.Color(f"{ticker_col}:N", legend=None),
    )

    # Point forecast line
    line = base.mark_line(strokeWidth=2).encode(
        x=alt.X(f"{date_col}:T"),
        y=alt.Y(f"{forecast_col}:Q", title="Provision Rate (%)"),
        color=alt.Color(f"{ticker_col}:N", title="Bank"),
        tooltip=[
            alt.Tooltip(f"{date_col}:T", title="Date"),
            alt.Tooltip(f"{ticker_col}:N", title="Bank"),
            alt.Tooltip(f"{forecast_col}:Q", title="Forecast", format=".2f"),
        ],
    )

    chart = (band + line).properties(
        width=700,
        height=400,
        title=title,
    )

    return chart


def chart_early_warning_dashboard(
    signals: pl.DataFrame,
    title: str = "Credit Boom Early Warning Dashboard",
) -> alt.VConcatChart:
    """
    Create comprehensive early warning dashboard.

    Args:
        signals: DataFrame with LIS and risk signals for each bank
        title: Dashboard title

    Returns:
        Altair compound chart
    """
    charts = []

    # 1. Current LIS bar chart (ranked by risk)
    if "lis" in signals.columns and "ticker" in signals.columns:
        bar_data = signals.select(["ticker", "lis", "risk_classification"]).drop_nulls()

        lis_bars = alt.Chart(bar_data.to_pandas()).mark_bar().encode(
            y=alt.Y("ticker:N", sort="-x", title="Bank"),
            x=alt.X("lis:Q", title="Lending Intensity Score (SDs)"),
            color=alt.Color(
                "risk_classification:N",
                title="Risk Level",
                scale=alt.Scale(
                    domain=["LOW", "MEDIUM", "HIGH"],
                    range=["#2ca02c", "#ff7f0e", "#d62728"],
                ),
            ),
            tooltip=[
                alt.Tooltip("ticker:N", title="Bank"),
                alt.Tooltip("lis:Q", title="LIS", format=".2f"),
                alt.Tooltip("risk_classification:N", title="Risk"),
            ],
        ).properties(
            width=400,
            height=250,
            title="Current Lending Intensity by Bank",
        )

        charts.append(lis_bars)

    # 2. Cumulative LIS (sustained exposure)
    if "lis_cumulative_12q" in signals.columns:
        cum_data = signals.select(["ticker", "lis_cumulative_12q", "risk_classification"]).drop_nulls()

        cum_bars = alt.Chart(cum_data.to_pandas()).mark_bar().encode(
            y=alt.Y("ticker:N", sort="-x", title="Bank"),
            x=alt.X("lis_cumulative_12q:Q", title="Cumulative LIS (12 Quarters)"),
            color=alt.Color(
                "risk_classification:N",
                legend=None,
                scale=alt.Scale(
                    domain=["LOW", "MEDIUM", "HIGH"],
                    range=["#2ca02c", "#ff7f0e", "#d62728"],
                ),
            ),
        ).properties(
            width=400,
            height=250,
            title="Sustained Lending Intensity (3 Years)",
        )

        charts.append(cum_bars)

    # 3. Expected provision impact
    if "expected_provision_impact_bp" in signals.columns:
        impact_data = signals.select(
            ["ticker", "expected_provision_impact_bp", "risk_classification"]
        ).drop_nulls()

        impact_bars = alt.Chart(impact_data.to_pandas()).mark_bar().encode(
            y=alt.Y("ticker:N", sort="-x", title="Bank"),
            x=alt.X(
                "expected_provision_impact_bp:Q",
                title="Expected Provision Impact (bp)",
            ),
            color=alt.Color(
                "risk_classification:N",
                legend=None,
                scale=alt.Scale(
                    domain=["LOW", "MEDIUM", "HIGH"],
                    range=["#2ca02c", "#ff7f0e", "#d62728"],
                ),
            ),
        ).properties(
            width=400,
            height=250,
            title="Projected Provision Increase (3-4 Years)",
        )

        charts.append(impact_bars)

    if not charts:
        return alt.Chart().mark_text().encode(text=alt.value("No data available"))

    # Combine charts
    if len(charts) >= 2:
        row1 = alt.hconcat(charts[0], charts[1]) if len(charts) >= 2 else charts[0]
        if len(charts) >= 3:
            dashboard = alt.vconcat(row1, charts[2])
        else:
            dashboard = row1
    else:
        dashboard = charts[0]

    return dashboard.properties(title=title)


def chart_historical_validation(
    actual: pl.DataFrame,
    predicted: pl.DataFrame,
    date_col: str = "date",
    actual_col: str = "provision_rate",
    predicted_col: str = "predicted_provision",
    title: str = "Model Validation: Actual vs. Predicted Provisions",
) -> alt.Chart:
    """
    Create chart comparing actual vs predicted provisions.

    Args:
        actual: DataFrame with actual provision rates
        predicted: DataFrame with predicted rates
        date_col: Name of date column
        actual_col: Name of actual column
        predicted_col: Name of predicted column
        title: Chart title

    Returns:
        Altair Chart object
    """
    # Merge actual and predicted
    if "ticker" in actual.columns and "ticker" in predicted.columns:
        # Panel data
        merged = actual.join(predicted, on=["date", "ticker"], how="inner")
    else:
        merged = actual.join(predicted, on="date", how="inner")

    df_long = merged.unpivot(
        index=[date_col],
        on=[actual_col, predicted_col],
        variable_name="series",
        value_name="rate",
    )

    chart = alt.Chart(df_long.to_pandas()).mark_line().encode(
        x=alt.X(f"{date_col}:T", title="Date"),
        y=alt.Y("rate:Q", title="Provision Rate (%)"),
        color=alt.Color(
            "series:N",
            title="",
            scale=alt.Scale(
                domain=[actual_col, predicted_col],
                range=["#1f77b4", "#ff7f0e"],
            ),
        ),
        strokeDash=alt.condition(
            alt.datum.series == predicted_col,
            alt.value([4, 4]),
            alt.value([0]),
        ),
    ).properties(
        width=700,
        height=400,
        title=title,
    )

    return chart


if __name__ == "__main__":
    import numpy as np

    # Create sample data for testing
    n = 100
    dates = pl.date_range(pl.date(1990, 1, 1), pl.date(2015, 1, 1), eager=True)[:n]

    np.random.seed(42)

    df = pl.DataFrame({
        "date": dates,
        "bank_share": 60 - np.linspace(0, 20, n) + np.random.normal(0, 2, n),
        "nonbank_share": 40 + np.linspace(0, 20, n) + np.random.normal(0, 2, n),
        "bank_credit_to_gdp": 40 + np.linspace(0, 10, n) + np.random.normal(0, 1, n),
        "nonbank_credit_to_gdp": 30 + np.linspace(0, 15, n) + np.random.normal(0, 1, n),
        "bank_credit_growth": np.random.normal(5, 3, n),
        "nonbank_credit_growth": np.random.normal(7, 2, n),
    })

    # Test charts
    print("Creating charts...")

    chart1 = chart_credit_to_gdp(df)
    chart1.save("/tmp/credit_to_gdp.html")

    chart2 = chart_lender_composition(df)
    chart2.save("/tmp/lender_composition.html")

    print("Charts saved to /tmp/")
