"""
Credit Boom Leading Indicator Dashboard

Interactive Streamlit dashboard for monitoring bank credit risk metrics
and early warning signals based on the NY Fed private credit methodology.

Run with: streamlit run src/financing_private_credit/dashboard.py
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import altair as alt
import numpy as np
import polars as pl
import streamlit as st

from .bank_data import BankDataCollector, TARGET_BANKS
from .leading_indicator import LendingIntensityScore, ARDLModel
from .nowcast import CreditNowcaster, FinancialConditionsMonitor


# Page configuration
st.set_page_config(
    page_title="Credit Boom Leading Indicator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_bank_data(start_date: str = "2015-01-01") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and cache bank data from SEC EDGAR."""
    collector = BankDataCollector(start_date=start_date)

    # Get data quality summary
    quality_summary = collector.get_data_quality_summary()

    # Fetch all bank data
    panel = collector.fetch_all_banks()

    # Compute derived metrics
    panel_with_metrics = collector.compute_derived_metrics(panel)

    return panel_with_metrics, quality_summary


@st.cache_data(ttl=3600)
def compute_lis_scores(panel: pl.DataFrame) -> pl.DataFrame:
    """Compute Lending Intensity Scores for all banks."""
    if panel.height == 0 or "loan_growth_yoy" not in panel.columns:
        return pl.DataFrame()

    # Filter to rows with valid loan growth
    panel_valid = panel.filter(pl.col("loan_growth_yoy").is_not_null())

    if panel_valid.height == 0:
        return panel

    # Compute system-wide statistics at each date (cross-sectional)
    system_stats = (
        panel_valid
        .group_by("date")
        .agg([
            pl.col("loan_growth_yoy").mean().alias("system_growth"),
            pl.col("loan_growth_yoy").std().alias("system_std"),
            pl.col("loan_growth_yoy").count().alias("n_banks"),
        ])
        .sort("date")
    )

    # Join system stats back to panel
    result = panel.join(system_stats, on="date", how="left")

    # Compute LIS = (bank_growth - system_growth) / system_std
    # Handle case where std is 0 or very small
    result = result.with_columns(
        pl.when(pl.col("system_std") > 0.01)
        .then((pl.col("loan_growth_yoy") - pl.col("system_growth")) / pl.col("system_std"))
        .otherwise(0.0)
        .alias("lis")
    )

    # Compute cumulative LIS (rolling 12-quarter sum)
    result = result.sort(["ticker", "date"]).with_columns(
        pl.col("lis")
        .rolling_sum(window_size=12, min_periods=1)
        .over("ticker")
        .alias("lis_cumulative_12q")
    )

    # Compute percentile rank among peers at each date
    result = result.with_columns(
        pl.col("loan_growth_yoy").rank().over("date").alias("growth_rank")
    )

    # Compute growth percentile
    result = result.with_columns(
        pl.when(pl.col("growth_rank").max().over("date") > 0)
        .then(pl.col("growth_rank") / pl.col("growth_rank").max().over("date") * 100)
        .otherwise(50.0)
        .alias("growth_percentile")
    )

    return result


@st.cache_data(ttl=3600)
def fit_ardl_model(panel: pl.DataFrame, forecast_horizon: int = 12) -> tuple[dict, pl.DataFrame]:
    """Fit ARDL model and generate forecasts."""
    if panel.height == 0 or "provision_rate" not in panel.columns:
        return {}, pl.DataFrame()

    # First compute LIS if not present
    if "lis" not in panel.columns:
        panel = compute_lis_scores(panel)

    if panel.height == 0 or "lis" not in panel.columns:
        return {}, pl.DataFrame()

    try:
        ardl = ARDLModel(
            data=panel,
            dep_var="provision_rate",
            lis_var="lis",
            ar_lags=4,
            lis_lags=[4, 8, 12]  # Shorter lags for limited data
        )
        ardl.fit()

        metrics = {}
        if hasattr(ardl, "_result") and ardl._result is not None:
            metrics["r_squared"] = ardl._result.r_squared if hasattr(ardl._result, "r_squared") else None

        # Generate simple forecasts (extend last values)
        forecasts = generate_simple_forecast(panel, forecast_horizon)

        return metrics, forecasts
    except Exception as e:
        # Fallback to simple forecast
        forecasts = generate_simple_forecast(panel, forecast_horizon)
        return {"error": str(e)}, forecasts


@st.cache_data(ttl=900)  # Cache for 15 minutes (more frequent for nowcasting)
def load_nowcast_data() -> tuple[pl.DataFrame, dict]:
    """Load high-frequency nowcast data from FRED."""
    try:
        nowcaster = CreditNowcaster(lookback_years=5)

        # Fetch weekly proxy data
        proxy_data = nowcaster.fetch_proxy_data()

        # Compute credit growth nowcast
        growth_nowcast = nowcaster.compute_credit_growth_nowcast()

        # Get financial conditions
        monitor = FinancialConditionsMonitor()
        conditions = monitor.assess_credit_environment()

        return growth_nowcast, conditions
    except Exception as e:
        return pl.DataFrame(), {"status": "error", "message": str(e)}


@st.cache_data(ttl=900)  # 15-minute cache
def load_weekly_h8_data() -> pl.DataFrame:
    """Load weekly H.8 bank credit data for monitoring."""
    try:
        from .data import FREDDataFetcher
        fetcher = FREDDataFetcher()

        # Weekly H.8 series
        h8_series = ["TOTLL", "BUSLOANS", "CONSUMER", "REALLN"]

        # Fetch last 2 years of weekly data
        from datetime import datetime, timedelta
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        data = fetcher.fetch_multiple_series(h8_series, start_date=start_date)

        if data.height > 0:
            # Compute YoY growth for each series
            for col in h8_series:
                if col in data.columns:
                    data = data.with_columns(
                        ((pl.col(col) / pl.col(col).shift(52) - 1) * 100)
                        .alias(f"{col}_growth_yoy")
                    )

            # Compute total bank credit growth
            if "TOTLL" in data.columns:
                data = data.with_columns(
                    ((pl.col("TOTLL") / pl.col("TOTLL").shift(52) - 1) * 100)
                    .alias("total_bank_credit_growth")
                )

        return data
    except Exception as e:
        return pl.DataFrame()


def create_weekly_credit_chart(weekly_data: pl.DataFrame) -> alt.Chart:
    """Create weekly bank credit growth chart."""
    if weekly_data.height == 0 or "total_bank_credit_growth" not in weekly_data.columns:
        return alt.Chart().mark_text().encode(text=alt.value("No weekly data available"))

    df = weekly_data.select(["date", "total_bank_credit_growth"]).drop_nulls().to_pandas()

    if len(df) == 0:
        return alt.Chart().mark_text().encode(text=alt.value("No weekly data available"))

    chart = alt.Chart(df).mark_line(strokeWidth=2, color="#1f77b4").encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("total_bank_credit_growth:Q", title="YoY Growth (%)"),
        tooltip=["date:T", alt.Tooltip("total_bank_credit_growth:Q", format=".2f", title="Growth (%)")]
    ).properties(
        height=300,
        title="Total Bank Credit Growth (Weekly H.8 Data)"
    ).interactive()

    # Add zero line
    zero_line = alt.Chart(pl.DataFrame({"y": [0]}).to_pandas()).mark_rule(
        strokeDash=[5, 5], color="gray"
    ).encode(y="y:Q")

    return chart + zero_line


def create_credit_components_chart(weekly_data: pl.DataFrame) -> alt.Chart:
    """Create chart showing credit components (C&I, Consumer, Real Estate)."""
    if weekly_data.height == 0:
        return alt.Chart().mark_text().encode(text=alt.value("No data available"))

    growth_cols = ["BUSLOANS_growth_yoy", "CONSUMER_growth_yoy", "REALLN_growth_yoy"]
    available = [c for c in growth_cols if c in weekly_data.columns]

    if not available:
        return alt.Chart().mark_text().encode(text=alt.value("No component data available"))

    # Melt to long format
    df = weekly_data.select(["date"] + available).drop_nulls()

    # Rename for display
    rename_map = {
        "BUSLOANS_growth_yoy": "C&I Loans",
        "CONSUMER_growth_yoy": "Consumer Loans",
        "REALLN_growth_yoy": "Real Estate Loans"
    }

    records = []
    for _, row in df.to_pandas().iterrows():
        for col in available:
            if col in rename_map:
                records.append({
                    "date": row["date"],
                    "category": rename_map[col],
                    "growth": row[col]
                })

    if not records:
        return alt.Chart().mark_text().encode(text=alt.value("No data available"))

    import pandas as pd
    long_df = pd.DataFrame(records)

    chart = alt.Chart(long_df).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("growth:Q", title="YoY Growth (%)"),
        color=alt.Color("category:N", title="Loan Type"),
        tooltip=["date:T", "category:N", alt.Tooltip("growth:Q", format=".2f")]
    ).properties(
        height=300,
        title="Bank Credit Components Growth (Weekly)"
    ).interactive()

    return chart


def generate_simple_forecast(panel: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Generate simple persistence-based forecasts."""
    if panel.height == 0:
        return pl.DataFrame()

    forecasts = []

    for ticker in panel["ticker"].unique().to_list():
        bank_data = panel.filter(pl.col("ticker") == ticker).sort("date")

        if bank_data.height == 0 or "provision_rate" not in bank_data.columns:
            continue

        # Get last known values
        last_row = bank_data.filter(pl.col("provision_rate").is_not_null()).tail(1)

        if last_row.height == 0:
            continue

        last_date = last_row["date"][0]
        last_prov = last_row["provision_rate"][0]

        # Simple mean reversion forecast
        historical_mean = bank_data.select(pl.col("provision_rate").mean()).item()

        for q in range(1, horizon + 1):
            # Next quarter date
            forecast_date = last_date + timedelta(days=91 * q)

            # Mean reversion with decay
            decay = 0.9 ** q
            forecast_prov = decay * last_prov + (1 - decay) * historical_mean

            forecasts.append({
                "date": forecast_date,
                "ticker": ticker,
                "provision_rate": forecast_prov,
                "type": "forecast"
            })

    if not forecasts:
        return pl.DataFrame()

    return pl.DataFrame(forecasts)


def create_loan_growth_chart(panel: pl.DataFrame, selected_banks: list[str]) -> alt.Chart:
    """Create loan growth time series chart."""
    data = panel.filter(pl.col("ticker").is_in(selected_banks))

    if data.height == 0:
        return alt.Chart().mark_text().encode(text=alt.value("No data available"))

    # Convert to pandas for Altair
    df = data.select(["date", "ticker", "loan_growth_yoy"]).drop_nulls().to_pandas()

    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("loan_growth_yoy:Q", title="Loan Growth YoY (%)"),
        color=alt.Color("ticker:N", title="Bank"),
        tooltip=["date:T", "ticker:N", alt.Tooltip("loan_growth_yoy:Q", format=".2f")]
    ).properties(
        height=350,
        title="Loan Growth Year-over-Year by Bank"
    ).interactive()

    # Add zero line
    zero_line = alt.Chart(pl.DataFrame({"y": [0]}).to_pandas()).mark_rule(
        strokeDash=[5, 5], color="gray"
    ).encode(y="y:Q")

    return chart + zero_line


def create_provision_rate_chart(panel: pl.DataFrame, selected_banks: list[str]) -> alt.Chart:
    """Create provision rate time series chart."""
    data = panel.filter(pl.col("ticker").is_in(selected_banks))

    if data.height == 0:
        return alt.Chart().mark_text().encode(text=alt.value("No data available"))

    df = data.select(["date", "ticker", "provision_rate"]).drop_nulls().to_pandas()

    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("provision_rate:Q", title="Provision Rate (%)"),
        color=alt.Color("ticker:N", title="Bank"),
        tooltip=["date:T", "ticker:N", alt.Tooltip("provision_rate:Q", format=".3f")]
    ).properties(
        height=350,
        title="Provision Rate by Bank"
    ).interactive()

    return chart


def create_lis_chart(lis_data: pl.DataFrame, selected_banks: list[str]) -> alt.Chart:
    """Create LIS (Lending Intensity Score) chart."""
    if lis_data.height == 0:
        return alt.Chart().mark_text().encode(text=alt.value("No LIS data available"))

    data = lis_data.filter(pl.col("ticker").is_in(selected_banks))

    if "lis" not in data.columns:
        return alt.Chart().mark_text().encode(text=alt.value("LIS column not found"))

    df = data.select(["date", "ticker", "lis"]).drop_nulls().to_pandas()

    if len(df) == 0:
        return alt.Chart().mark_text().encode(text=alt.value("No LIS data available"))

    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("lis:Q", title="Lending Intensity Score"),
        color=alt.Color("ticker:N", title="Bank"),
        tooltip=["date:T", "ticker:N", alt.Tooltip("lis:Q", format=".2f")]
    ).properties(
        height=350,
        title="Lending Intensity Score (LIS) by Bank"
    ).interactive()

    # Add threshold lines
    thresholds = pl.DataFrame({
        "y": [1.5, -1.5],
        "label": ["Aggressive Lending", "Conservative Lending"]
    }).to_pandas()

    threshold_lines = alt.Chart(thresholds).mark_rule(
        strokeDash=[5, 5]
    ).encode(
        y="y:Q",
        color=alt.value("red")
    )

    zero_line = alt.Chart(pl.DataFrame({"y": [0]}).to_pandas()).mark_rule(
        color="gray"
    ).encode(y="y:Q")

    return chart + threshold_lines + zero_line


def create_lis_heatmap(lis_data: pl.DataFrame) -> alt.Chart:
    """Create LIS heatmap across banks and time."""
    if lis_data.height == 0 or "lis" not in lis_data.columns:
        return alt.Chart().mark_text().encode(text=alt.value("No LIS data available"))

    # Get last 12 quarters
    df = (
        lis_data
        .select(["date", "ticker", "lis"])
        .drop_nulls()
        .sort("date", descending=True)
        .group_by("ticker")
        .head(12)
        .to_pandas()
    )

    if len(df) == 0:
        return alt.Chart().mark_text().encode(text=alt.value("No LIS data available"))

    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X("yearquarter(date):O", title="Quarter"),
        y=alt.Y("ticker:N", title="Bank", sort=list(TARGET_BANKS.keys())),
        color=alt.Color(
            "lis:Q",
            title="LIS",
            scale=alt.Scale(scheme="redblue", domain=[-3, 3], reverse=True)
        ),
        tooltip=["date:T", "ticker:N", alt.Tooltip("lis:Q", format=".2f")]
    ).properties(
        height=300,
        title="Lending Intensity Score Heatmap (Last 12 Quarters)"
    )

    return chart


def create_early_warning_table(lis_data: pl.DataFrame) -> pl.DataFrame:
    """Create early warning signals table."""
    if lis_data.height == 0 or "lis" not in lis_data.columns:
        return pl.DataFrame()

    # Get latest LIS for each bank
    latest = (
        lis_data
        .filter(pl.col("lis").is_not_null())
        .sort("date", descending=True)
        .group_by("ticker")
        .first()
    )

    if latest.height == 0:
        return pl.DataFrame()

    # Calculate cumulative LIS if available
    if "lis_cumulative_12q" in lis_data.columns:
        cum_lis = (
            lis_data
            .filter(pl.col("lis_cumulative_12q").is_not_null())
            .sort("date", descending=True)
            .group_by("ticker")
            .first()
            .select(["ticker", "lis_cumulative_12q"])
        )
        latest = latest.join(cum_lis, on="ticker", how="left")

    # Determine warning level
    def warning_level(lis: float) -> str:
        if lis > 2.0:
            return "🔴 HIGH RISK"
        elif lis > 1.5:
            return "🟠 ELEVATED"
        elif lis > 1.0:
            return "🟡 MODERATE"
        elif lis < -1.5:
            return "🟢 CONSERVATIVE"
        else:
            return "⚪ NORMAL"

    result = latest.select([
        "ticker",
        "date",
        pl.col("lis").round(2).alias("Current LIS"),
        pl.col("loan_growth_yoy").round(2).alias("Loan Growth (%)") if "loan_growth_yoy" in latest.columns else pl.lit(None).alias("Loan Growth (%)"),
    ])

    return result


def create_forecast_chart(
    panel: pl.DataFrame,
    forecasts: pl.DataFrame,
    selected_bank: str
) -> alt.Chart:
    """Create forecast visualization for a single bank."""
    if forecasts.height == 0:
        return alt.Chart().mark_text().encode(text=alt.value("No forecast available"))

    # Historical data
    hist = panel.filter(pl.col("ticker") == selected_bank).select([
        "date", "provision_rate"
    ]).drop_nulls()

    if hist.height == 0:
        return alt.Chart().mark_text().encode(text=alt.value("No historical data"))

    hist_df = hist.with_columns(pl.lit("Historical").alias("type")).to_pandas()

    # Forecast data
    if "ticker" in forecasts.columns:
        fcst = forecasts.filter(pl.col("ticker") == selected_bank)
    else:
        fcst = forecasts

    if fcst.height > 0 and "provision_rate" in fcst.columns:
        fcst_df = fcst.select(["date", "provision_rate"]).with_columns(
            pl.lit("Forecast").alias("type")
        ).to_pandas()

        combined = pl.concat([
            hist.with_columns(pl.lit("Historical").alias("type")),
            fcst.select(["date", "provision_rate"]).with_columns(pl.lit("Forecast").alias("type"))
        ]).to_pandas()
    else:
        combined = hist_df

    chart = alt.Chart(combined).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("provision_rate:Q", title="Provision Rate (%)"),
        color=alt.Color("type:N", title="", scale=alt.Scale(
            domain=["Historical", "Forecast"],
            range=["#1f77b4", "#ff7f0e"]
        )),
        strokeDash=alt.StrokeDash("type:N", scale=alt.Scale(
            domain=["Historical", "Forecast"],
            range=[[0], [5, 5]]
        ))
    ).properties(
        height=300,
        title=f"Provision Rate Forecast: {selected_bank}"
    ).interactive()

    return chart


def create_metrics_comparison(panel: pl.DataFrame) -> alt.Chart:
    """Create scatter plot comparing loan growth vs provision rate."""
    # Get latest data for each bank
    latest = (
        panel
        .filter(
            pl.col("loan_growth_yoy").is_not_null() &
            pl.col("provision_rate").is_not_null()
        )
        .sort("date", descending=True)
        .group_by("ticker")
        .first()
    )

    if latest.height == 0:
        return alt.Chart().mark_text().encode(text=alt.value("No data available"))

    df = latest.select([
        "ticker", "loan_growth_yoy", "provision_rate", "total_loans"
    ]).to_pandas()

    chart = alt.Chart(df).mark_circle(size=200).encode(
        x=alt.X("loan_growth_yoy:Q", title="Loan Growth YoY (%)"),
        y=alt.Y("provision_rate:Q", title="Provision Rate (%)"),
        color=alt.Color("ticker:N", title="Bank"),
        size=alt.Size("total_loans:Q", title="Total Loans", scale=alt.Scale(range=[100, 1000])),
        tooltip=[
            "ticker:N",
            alt.Tooltip("loan_growth_yoy:Q", format=".2f", title="Loan Growth (%)"),
            alt.Tooltip("provision_rate:Q", format=".3f", title="Provision Rate (%)"),
            alt.Tooltip("total_loans:Q", format=",.0f", title="Total Loans ($)")
        ]
    ).properties(
        height=400,
        title="Loan Growth vs Provision Rate (Latest Quarter)"
    ).interactive()

    return chart


def main():
    """Main dashboard application."""
    st.title("📊 Credit Boom Leading Indicator Dashboard")
    st.markdown("""
    *Based on NY Fed Staff Report 1111: Financing Private Credit methodology*

    This dashboard monitors bank credit risk using real SEC EDGAR data and the
    Lending Intensity Score (LIS) framework to identify early warning signals.
    """)

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    start_year = st.sidebar.slider(
        "Start Year",
        min_value=2010,
        max_value=2023,
        value=2018,
        help="Historical data start year"
    )

    # Load data
    with st.spinner("Loading bank data from SEC EDGAR..."):
        try:
            panel, quality_summary = load_bank_data(f"{start_year}-01-01")
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

    if panel.height == 0:
        st.error("No bank data available. Please check your internet connection.")
        st.stop()

    # Bank selection
    available_banks = sorted(panel["ticker"].unique().to_list())

    st.sidebar.header("🏦 Bank Selection")
    select_all = st.sidebar.checkbox("Select All Banks", value=True)

    if select_all:
        selected_banks = available_banks
    else:
        selected_banks = st.sidebar.multiselect(
            "Select Banks",
            options=available_banks,
            default=available_banks[:5] if len(available_banks) >= 5 else available_banks
        )

    if not selected_banks:
        st.warning("Please select at least one bank.")
        st.stop()

    # Compute LIS
    with st.spinner("Computing Lending Intensity Scores..."):
        lis_data = compute_lis_scores(panel)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview",
        "🎯 LIS Analysis",
        "⚠️ Early Warnings",
        "📡 Nowcast (Weekly)",
        "🔮 Forecasts",
        "📋 Data Quality"
    ])

    # Tab 1: Overview
    with tab1:
        st.header("Bank Credit Metrics Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Loan Growth")
            loan_chart = create_loan_growth_chart(panel, selected_banks)
            st.altair_chart(loan_chart, use_container_width=True)

        with col2:
            st.subheader("Provision Rate")
            prov_chart = create_provision_rate_chart(panel, selected_banks)
            st.altair_chart(prov_chart, use_container_width=True)

        st.subheader("Loan Growth vs Provision Rate")
        scatter = create_metrics_comparison(panel)
        st.altair_chart(scatter, use_container_width=True)

        # Latest metrics table
        st.subheader("Latest Quarterly Metrics")
        latest_data = (
            panel
            .filter(pl.col("ticker").is_in(selected_banks))
            .sort("date", descending=True)
            .group_by("ticker")
            .first()
            .select([
                "ticker",
                "date",
                pl.col("total_loans").cast(pl.Float64) / 1e9,
                pl.col("allowance").cast(pl.Float64) / 1e9,
                pl.col("provisions").cast(pl.Float64) / 1e9,
                "loan_growth_yoy",
                "provision_rate",
            ])
            .rename({
                "total_loans": "Loans ($B)",
                "allowance": "Allowance ($B)",
                "provisions": "Provisions ($B)",
                "loan_growth_yoy": "Loan Growth (%)",
                "provision_rate": "Prov. Rate (%)",
            })
            .sort("ticker")
        )
        st.dataframe(
            latest_data.to_pandas().round(2),
            use_container_width=True,
            hide_index=True
        )

    # Tab 2: LIS Analysis
    with tab2:
        st.header("Lending Intensity Score (LIS) Analysis")

        st.markdown("""
        **LIS = (Bank Loan Growth - System Average) / System Std Dev**

        - **LIS > 1.5**: Aggressive lending relative to peers (elevated risk)
        - **LIS < -1.5**: Conservative lending relative to peers
        - **Cumulative LIS (12Q)**: Sustained deviation indicates persistent risk build-up
        """)

        if lis_data.height > 0 and "lis" in lis_data.columns:
            # LIS time series
            st.subheader("LIS Time Series")
            lis_chart = create_lis_chart(lis_data, selected_banks)
            st.altair_chart(lis_chart, use_container_width=True)

            # LIS heatmap
            st.subheader("LIS Heatmap")
            heatmap = create_lis_heatmap(lis_data)
            st.altair_chart(heatmap, use_container_width=True)

            # Cumulative LIS if available
            if "lis_cumulative_12q" in lis_data.columns:
                st.subheader("Cumulative LIS (12 Quarters)")
                cum_data = lis_data.filter(pl.col("ticker").is_in(selected_banks))
                df = cum_data.select(["date", "ticker", "lis_cumulative_12q"]).drop_nulls().to_pandas()

                if len(df) > 0:
                    cum_chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("lis_cumulative_12q:Q", title="Cumulative LIS (12Q)"),
                        color=alt.Color("ticker:N", title="Bank"),
                    ).properties(height=350).interactive()
                    st.altair_chart(cum_chart, use_container_width=True)
        else:
            st.info("LIS data not available. Ensure sufficient historical data is loaded.")

    # Tab 3: Early Warnings
    with tab3:
        st.header("⚠️ Early Warning Signals")

        st.markdown("""
        Banks are flagged based on their Lending Intensity Score:
        - 🔴 **HIGH RISK**: LIS > 2.0 - Significantly aggressive lending
        - 🟠 **ELEVATED**: LIS > 1.5 - Above-average lending intensity
        - 🟡 **MODERATE**: LIS > 1.0 - Slightly elevated
        - ⚪ **NORMAL**: -1.5 < LIS < 1.0
        - 🟢 **CONSERVATIVE**: LIS < -1.5 - Below-average lending
        """)

        if lis_data.height > 0 and "lis" in lis_data.columns:
            # Create warning table
            warning_df = create_early_warning_table(lis_data)

            if warning_df.height > 0:
                # Add warning level column
                latest_lis = (
                    lis_data
                    .filter(pl.col("lis").is_not_null())
                    .sort("date", descending=True)
                    .group_by("ticker")
                    .first()
                    .select(["ticker", "lis"])
                )

                # Display as cards
                cols = st.columns(min(5, len(selected_banks)))

                for idx, bank in enumerate(selected_banks[:10]):
                    bank_lis = latest_lis.filter(pl.col("ticker") == bank)

                    if bank_lis.height > 0:
                        lis_val = bank_lis["lis"][0]

                        if lis_val > 2.0:
                            color = "🔴"
                            status = "HIGH RISK"
                        elif lis_val > 1.5:
                            color = "🟠"
                            status = "ELEVATED"
                        elif lis_val > 1.0:
                            color = "🟡"
                            status = "MODERATE"
                        elif lis_val < -1.5:
                            color = "🟢"
                            status = "CONSERVATIVE"
                        else:
                            color = "⚪"
                            status = "NORMAL"

                        with cols[idx % len(cols)]:
                            st.metric(
                                label=f"{color} {bank}",
                                value=f"{lis_val:.2f}",
                                delta=status,
                                delta_color="off"
                            )

                st.divider()

                # Detailed table
                st.subheader("Detailed Metrics")
                st.dataframe(warning_df.to_pandas(), use_container_width=True, hide_index=True)
        else:
            st.info("Early warning data not available.")

    # Tab 4: Nowcast (Weekly Data)
    with tab4:
        st.header("📡 Weekly Credit Nowcast")

        st.markdown("""
        **High-frequency monitoring using Federal Reserve H.8 weekly bank credit data.**

        This tab provides real-time credit conditions using:
        - Weekly Total Loans & Leases (updated every Friday)
        - Credit component breakdown (C&I, Consumer, Real Estate)
        - Financial conditions indicators (NFCI, credit spreads)

        *Data refreshes every 15 minutes when dashboard is active.*
        """)

        # Load weekly H.8 data
        with st.spinner("Loading weekly H.8 data from FRED..."):
            weekly_data = load_weekly_h8_data()

        if weekly_data.height > 0:
            # Latest reading
            latest_weekly = weekly_data.filter(pl.col("total_bank_credit_growth").is_not_null()).tail(1)

            if latest_weekly.height > 0:
                col1, col2, col3 = st.columns(3)

                with col1:
                    latest_growth = latest_weekly["total_bank_credit_growth"][0]
                    st.metric(
                        "Total Bank Credit Growth (YoY)",
                        f"{latest_growth:.2f}%",
                        delta=None
                    )

                with col2:
                    latest_date = latest_weekly["date"][0]
                    st.metric("Latest Data", latest_date.strftime("%Y-%m-%d"))

                with col3:
                    # Compare to 4 weeks ago
                    if weekly_data.height > 4:
                        four_weeks_ago = weekly_data.filter(
                            pl.col("total_bank_credit_growth").is_not_null()
                        ).tail(5).head(1)
                        if four_weeks_ago.height > 0:
                            prev_growth = four_weeks_ago["total_bank_credit_growth"][0]
                            delta = latest_growth - prev_growth
                            st.metric(
                                "4-Week Change",
                                f"{delta:+.2f}pp",
                                delta="accelerating" if delta > 0 else "decelerating",
                                delta_color="normal"
                            )

            st.divider()

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Total Bank Credit Growth")
                weekly_chart = create_weekly_credit_chart(weekly_data)
                st.altair_chart(weekly_chart, use_container_width=True)

            with col2:
                st.subheader("Credit Components")
                components_chart = create_credit_components_chart(weekly_data)
                st.altair_chart(components_chart, use_container_width=True)

            # Financial conditions
            st.divider()
            st.subheader("Financial Conditions Assessment")

            with st.spinner("Loading financial conditions..."):
                _, conditions = load_nowcast_data()

            if conditions.get("status") != "error":
                cond_cols = st.columns(4)

                overall = conditions.get("overall", "unknown")
                overall_emoji = {"tight": "🔴", "loose": "🟢", "neutral": "🟡"}.get(overall, "⚪")

                with cond_cols[0]:
                    st.metric("Overall Conditions", f"{overall_emoji} {overall.upper()}")

                indicators = conditions.get("indicators", {})

                if "NFCI" in indicators:
                    nfci = indicators["NFCI"]
                    with cond_cols[1]:
                        st.metric(
                            "NFCI",
                            f"{nfci['value']:.2f}" if nfci.get('value') else "N/A",
                            delta=nfci.get('interpretation', '').upper()
                        )

                if "HY_spread" in indicators:
                    hy = indicators["HY_spread"]
                    with cond_cols[2]:
                        st.metric(
                            "HY Spread (bps)",
                            f"{hy['value']:.0f}" if hy.get('value') else "N/A",
                            delta=hy.get('interpretation', '').upper()
                        )

                with cond_cols[3]:
                    cond_date = conditions.get("date")
                    if cond_date:
                        st.metric("As of", cond_date.strftime("%Y-%m-%d") if hasattr(cond_date, 'strftime') else str(cond_date))
            else:
                st.warning("Could not load financial conditions data.")

            # Recent weekly data table
            st.divider()
            st.subheader("Recent Weekly Data")

            display_cols = ["date", "TOTLL", "total_bank_credit_growth"]
            component_growth = ["BUSLOANS_growth_yoy", "CONSUMER_growth_yoy", "REALLN_growth_yoy"]
            display_cols.extend([c for c in component_growth if c in weekly_data.columns])

            recent = weekly_data.select([c for c in display_cols if c in weekly_data.columns]).tail(10)
            st.dataframe(recent.to_pandas().round(2), use_container_width=True, hide_index=True)

        else:
            st.warning("Could not load weekly H.8 data. Check FRED API connection.")

    # Tab 5: Forecasts
    with tab5:
        st.header("🔮 Provision Rate Forecasts")

        forecast_bank = st.selectbox(
            "Select Bank for Forecast",
            options=selected_banks,
            index=0
        )

        forecast_horizon = st.slider(
            "Forecast Horizon (Quarters)",
            min_value=4,
            max_value=16,
            value=8
        )

        with st.spinner("Fitting ARDL model..."):
            model_metrics, forecasts = fit_ardl_model(panel, forecast_horizon)

        if model_metrics:
            col1, col2 = st.columns([2, 1])

            with col1:
                forecast_chart = create_forecast_chart(panel, forecasts, forecast_bank)
                st.altair_chart(forecast_chart, use_container_width=True)

            with col2:
                st.subheader("Model Metrics")
                if model_metrics.get("r_squared"):
                    st.metric("R-squared", f"{model_metrics['r_squared']:.3f}")

                st.markdown("""
                **ARDL Model**

                Autoregressive Distributed Lag model with:
                - 4 lags of provision rate
                - Macro indicators (GDP growth, credit spreads)
                - Bank fixed effects
                """)
        else:
            st.info("Forecast model could not be fitted. Ensure sufficient data is available.")

    # Tab 6: Data Quality
    with tab6:
        st.header("📋 Data Quality Summary")

        st.markdown("""
        Data is sourced from SEC EDGAR XBRL filings. Status indicates:
        - **COMPLETE**: All metrics (loans, allowance, provisions) current
        - **COMPLETE_NO_PROV**: Loans/allowance current, provisions derived
        - **STALE_DATA**: Data not current (e.g., WFC data stops at Q2 2022)
        """)

        if quality_summary.height > 0:
            display_cols = [
                "ticker", "name", "tier",
                "loans_recent", "allowance_recent", "provisions_recent",
                "data_status"
            ]
            available_cols = [c for c in display_cols if c in quality_summary.columns]

            st.dataframe(
                quality_summary.select(available_cols).to_pandas(),
                use_container_width=True,
                hide_index=True
            )

            # Summary stats
            col1, col2, col3 = st.columns(3)

            complete = quality_summary.filter(pl.col("data_status") == "COMPLETE").height
            partial = quality_summary.filter(pl.col("data_status") == "COMPLETE_NO_PROV").height
            stale = quality_summary.filter(pl.col("data_status") == "STALE_DATA").height

            with col1:
                st.metric("Complete Data", complete)
            with col2:
                st.metric("Derived Provisions", partial)
            with col3:
                st.metric("Stale Data", stale)

    # Footer
    st.divider()
    st.markdown("""
    ---
    **Credit Boom Leading Indicator Dashboard** | Based on NY Fed Staff Report 1111

    *Data refreshes hourly. Last updated: {}*
    """.format(date.today().strftime("%Y-%m-%d")))


if __name__ == "__main__":
    main()
