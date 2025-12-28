"""
Visualization Module for Duration Mismatch Indicator

Generates a comprehensive set of charts using Vega-Altair to tell the story:
1. Duration exposure across banks
2. Predicted vs actual volatility
3. Vulnerability rankings
4. Rate shock scenarios
5. Backtest performance

Charts are returned as a dictionary with numbered keys for ordering.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import polars as pl

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False


class DurationMismatchVisualizer:
    """
    Generate Vega-Altair visualizations for duration mismatch analysis.

    Returns charts as a dictionary with keys like:
    - "01_duration_by_bank"
    - "02_vulnerability_ranking"
    - etc.

    This ordering tells the story progressively.
    """

    # Color palette for banks
    BANK_COLORS = {
        "JPM": "#003087",
        "BAC": "#012169",
        "C": "#003C71",
        "WFC": "#D71E28",
        "GS": "#6BB9F0",
        "MS": "#00A4E4",
        "USB": "#0C2340",
        "PNC": "#FF8200",
        "TFC": "#4A0D67",
        "COF": "#D03027",
        "SCHW": "#00A0DF",
        "BK": "#231F20",
        "STT": "#003366",
        "NTRS": "#006341",
        "RJF": "#002855",
    }

    # Default chart dimensions
    WIDTH = 600
    HEIGHT = 400

    def __init__(self, duration_data: pl.DataFrame):
        """
        Initialize visualizer.

        Args:
            duration_data: DataFrame with duration exposure data
        """
        if not ALTAIR_AVAILABLE:
            raise ImportError(
                "altair is required for visualizations. "
                "Install with: pip install altair"
            )

        self.data = duration_data
        alt.data_transformers.enable('default', max_rows=None)

    def generate_all_charts(
        self,
        forecast_data: Optional[pl.DataFrame] = None,
        backtest_data: Optional[pl.DataFrame] = None,
        scenario_data: Optional[pl.DataFrame] = None,
    ) -> dict[str, alt.Chart]:
        """
        Generate all charts for the duration mismatch story.

        Args:
            forecast_data: Optional forecast results
            backtest_data: Optional backtest results
            scenario_data: Optional scenario analysis results

        Returns:
            Dictionary of charts with numbered keys
        """
        charts = {}

        # Part 1: Understanding Duration Exposure
        charts["01_duration_by_bank"] = self.chart_duration_by_bank()
        charts["02_duration_over_time"] = self.chart_duration_time_series()
        charts["03_portfolio_composition"] = self.chart_portfolio_composition()

        # Part 2: Vulnerability Analysis
        charts["04_vulnerability_ranking"] = self.chart_vulnerability_ranking()
        charts["05_duration_vs_volatility_scatter"] = self.chart_duration_volatility_scatter()
        charts["06_dv01_comparison"] = self.chart_dv01_comparison()

        # Part 3: Rate Sensitivity
        charts["07_yield_sensitivity_heatmap"] = self.chart_yield_sensitivity_heatmap()

        if scenario_data is not None and scenario_data.height > 0:
            charts["08_scenario_impact"] = self.chart_scenario_impact(scenario_data)

        # Part 4: Predictive Relationship
        if "predicted_impact_pct" in self.data.columns and "earnings_volatility" in self.data.columns:
            charts["09_predicted_vs_actual"] = self.chart_predicted_vs_actual()

        # Part 5: Forecasts
        if forecast_data is not None and forecast_data.height > 0:
            charts["10_forecast_comparison"] = self.chart_forecast_comparison(forecast_data)
            charts["11_forecast_uncertainty"] = self.chart_forecast_uncertainty(forecast_data)

        # Part 6: Backtest Performance
        if backtest_data is not None and backtest_data.height > 0:
            charts["12_backtest_performance"] = self.chart_backtest_performance(backtest_data)

        # Part 7: Summary
        charts["13_executive_summary"] = self.chart_executive_summary()

        return charts

    def chart_duration_by_bank(self) -> alt.Chart:
        """Bar chart of current duration exposure by bank."""
        # Get latest duration for each bank
        latest = (
            self.data
            .sort("date", descending=True)
            .group_by("ticker")
            .head(1)
        )

        if latest.height == 0:
            return alt.Chart().mark_text().encode(text=alt.value("No data"))

        df = latest.select(["ticker", "estimated_duration"]).to_pandas()

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("ticker:N", sort="-y", title="Bank"),
            y=alt.Y("estimated_duration:Q", title="Estimated Duration (years)"),
            color=alt.Color(
                "ticker:N",
                scale=alt.Scale(domain=list(self.BANK_COLORS.keys()),
                               range=list(self.BANK_COLORS.values())),
                legend=None
            ),
            tooltip=["ticker", alt.Tooltip("estimated_duration:Q", format=".2f")]
        ).properties(
            title="Duration Exposure by Bank",
            width=self.WIDTH,
            height=self.HEIGHT
        )

        # Add average line
        avg_duration = df["estimated_duration"].mean()
        rule = alt.Chart(df).mark_rule(color="red", strokeDash=[5, 5]).encode(
            y=alt.datum(avg_duration)
        )

        return chart + rule

    def chart_duration_time_series(self) -> alt.Chart:
        """Line chart of duration over time by bank."""
        df = self.data.select(["date", "ticker", "estimated_duration"]).to_pandas()

        if df.empty:
            return alt.Chart().mark_text().encode(text=alt.value("No data"))

        chart = alt.Chart(df).mark_line().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("estimated_duration:Q", title="Estimated Duration (years)"),
            color=alt.Color(
                "ticker:N",
                scale=alt.Scale(domain=list(self.BANK_COLORS.keys()),
                               range=list(self.BANK_COLORS.values())),
            ),
            tooltip=["ticker", "date:T", alt.Tooltip("estimated_duration:Q", format=".2f")]
        ).properties(
            title="Duration Exposure Over Time",
            width=self.WIDTH,
            height=self.HEIGHT
        )

        return chart

    def chart_portfolio_composition(self) -> alt.Chart:
        """Stacked bar showing AFS vs HTM composition."""
        if "afs_securities" not in self.data.columns:
            return alt.Chart().mark_text().encode(text=alt.value("No portfolio data"))

        latest = (
            self.data
            .sort("date", descending=True)
            .group_by("ticker")
            .head(1)
        )

        # Reshape for stacked bar
        records = []
        for row in latest.iter_rows(named=True):
            afs = row.get("afs_securities", 0) or 0
            htm = row.get("htm_securities", 0) or 0
            records.append({"ticker": row["ticker"], "type": "AFS", "value": afs})
            records.append({"ticker": row["ticker"], "type": "HTM", "value": htm})

        df = pl.DataFrame(records).to_pandas()

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("ticker:N", sort="-y", title="Bank"),
            y=alt.Y("value:Q", title="Securities ($ millions)"),
            color=alt.Color("type:N", title="Type",
                          scale=alt.Scale(domain=["AFS", "HTM"],
                                        range=["#4C78A8", "#F58518"])),
            tooltip=["ticker", "type", alt.Tooltip("value:Q", format=",.0f")]
        ).properties(
            title="Securities Portfolio Composition",
            width=self.WIDTH,
            height=self.HEIGHT
        )

        return chart

    def chart_vulnerability_ranking(self) -> alt.Chart:
        """Horizontal bar chart ranking banks by vulnerability."""
        if "vulnerability_score" not in self.data.columns:
            return alt.Chart().mark_text().encode(text=alt.value("No vulnerability data"))

        latest = (
            self.data
            .sort("date", descending=True)
            .group_by("ticker")
            .head(1)
            .sort("vulnerability_score", descending=True)
        )

        df = latest.select(["ticker", "vulnerability_score"]).to_pandas()

        chart = alt.Chart(df).mark_bar().encode(
            y=alt.Y("ticker:N", sort="-x", title="Bank"),
            x=alt.X("vulnerability_score:Q", title="Vulnerability Score"),
            color=alt.Color(
                "vulnerability_score:Q",
                scale=alt.Scale(scheme="reds"),
                legend=None
            ),
            tooltip=["ticker", alt.Tooltip("vulnerability_score:Q", format=".3f")]
        ).properties(
            title="Vulnerability Ranking (Higher = More Vulnerable)",
            width=self.WIDTH,
            height=self.HEIGHT
        )

        return chart

    def chart_duration_volatility_scatter(self) -> alt.Chart:
        """Scatter plot of duration vs stock volatility."""
        if "stock_volatility" not in self.data.columns:
            return alt.Chart().mark_text().encode(text=alt.value("No volatility data"))

        latest = (
            self.data
            .filter(pl.col("stock_volatility").is_not_null())
            .sort("date", descending=True)
            .group_by("ticker")
            .head(1)
        )

        df = latest.select(["ticker", "estimated_duration", "stock_volatility"]).to_pandas()

        if df.empty:
            return alt.Chart().mark_text().encode(text=alt.value("No data"))

        # Scatter with trend line
        points = alt.Chart(df).mark_circle(size=100).encode(
            x=alt.X("estimated_duration:Q", title="Estimated Duration (years)"),
            y=alt.Y("stock_volatility:Q", title="Stock Volatility (60-day)"),
            color=alt.Color(
                "ticker:N",
                scale=alt.Scale(domain=list(self.BANK_COLORS.keys()),
                               range=list(self.BANK_COLORS.values())),
            ),
            tooltip=["ticker",
                    alt.Tooltip("estimated_duration:Q", format=".2f"),
                    alt.Tooltip("stock_volatility:Q", format=".4f")]
        )

        # Regression line
        trend = points.transform_regression(
            "estimated_duration", "stock_volatility"
        ).mark_line(color="gray", strokeDash=[5, 5])

        chart = (points + trend).properties(
            title="Duration vs Stock Volatility",
            width=self.WIDTH,
            height=self.HEIGHT
        )

        return chart

    def chart_dv01_comparison(self) -> alt.Chart:
        """Bar chart comparing DV01 across banks."""
        if "dv01" not in self.data.columns:
            return alt.Chart().mark_text().encode(text=alt.value("No DV01 data"))

        latest = (
            self.data
            .sort("date", descending=True)
            .group_by("ticker")
            .head(1)
        )

        df = latest.select(["ticker", "dv01"]).to_pandas()

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("ticker:N", sort="-y", title="Bank"),
            y=alt.Y("dv01:Q", title="DV01 ($ millions per bp)"),
            color=alt.Color(
                "ticker:N",
                scale=alt.Scale(domain=list(self.BANK_COLORS.keys()),
                               range=list(self.BANK_COLORS.values())),
                legend=None
            ),
            tooltip=["ticker", alt.Tooltip("dv01:Q", format=",.2f")]
        ).properties(
            title="Dollar Value of 1bp Rate Move (DV01)",
            width=self.WIDTH,
            height=self.HEIGHT
        )

        return chart

    def chart_yield_sensitivity_heatmap(self) -> alt.Chart:
        """Heatmap showing sensitivity to different yield curve points."""
        # Create synthetic sensitivity data based on duration
        latest = (
            self.data
            .sort("date", descending=True)
            .group_by("ticker")
            .head(1)
        )

        if latest.height == 0:
            return alt.Chart().mark_text().encode(text=alt.value("No data"))

        # Assume sensitivity proportional to duration * maturity weight
        maturity_weights = {"1Y": 0.1, "2Y": 0.2, "5Y": 0.25, "10Y": 0.3, "30Y": 0.15}

        records = []
        for row in latest.iter_rows(named=True):
            dur = row.get("estimated_duration", 5.0) or 5.0
            for mat, weight in maturity_weights.items():
                # Sensitivity in bps NIM impact per 100bp move
                sens = dur * weight * 0.1  # Simplified
                records.append({
                    "ticker": row["ticker"],
                    "maturity": mat,
                    "sensitivity": sens
                })

        df = pl.DataFrame(records).to_pandas()

        chart = alt.Chart(df).mark_rect().encode(
            x=alt.X("maturity:O", title="Yield Curve Point",
                   sort=["1Y", "2Y", "5Y", "10Y", "30Y"]),
            y=alt.Y("ticker:N", title="Bank"),
            color=alt.Color("sensitivity:Q",
                          scale=alt.Scale(scheme="blues"),
                          title="Sensitivity"),
            tooltip=["ticker", "maturity", alt.Tooltip("sensitivity:Q", format=".3f")]
        ).properties(
            title="Yield Curve Sensitivity Heatmap",
            width=self.WIDTH,
            height=self.HEIGHT
        )

        return chart

    def chart_scenario_impact(self, scenario_data: pl.DataFrame) -> alt.Chart:
        """Bar chart showing impact under different rate scenarios."""
        df = scenario_data.to_pandas()

        if df.empty:
            return alt.Chart().mark_text().encode(text=alt.value("No scenario data"))

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("ticker:N", title="Bank"),
            y=alt.Y("vol_change:Q", title="Volatility Change"),
            color=alt.Color("rate_change_bp:Q",
                          scale=alt.Scale(scheme="redblue", domainMid=0),
                          title="Rate Change (bp)"),
            column=alt.Column("rate_change_bp:N", title="Scenario"),
            tooltip=["ticker", "rate_change_bp", alt.Tooltip("vol_change:Q", format=".4f")]
        ).properties(
            title="Volatility Impact by Rate Scenario",
            width=150,
            height=300
        )

        return chart

    def chart_predicted_vs_actual(self) -> alt.Chart:
        """Scatter plot comparing predicted impact to actual volatility."""
        df = self.data.filter(
            pl.col("predicted_impact_pct").is_not_null() &
            pl.col("earnings_volatility").is_not_null()
        ).to_pandas()

        if df.empty:
            return alt.Chart().mark_text().encode(text=alt.value("No prediction data"))

        points = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X("predicted_impact_pct:Q", title="Predicted Impact (% of portfolio)"),
            y=alt.Y("earnings_volatility:Q", title="Actual Earnings Volatility"),
            color=alt.Color("ticker:N", legend=None),
            tooltip=["ticker", "date:T",
                    alt.Tooltip("predicted_impact_pct:Q", format=".3f"),
                    alt.Tooltip("earnings_volatility:Q", format=".3f")]
        )

        # 45-degree reference line
        line = alt.Chart(df).mark_line(color="gray", strokeDash=[5, 5]).encode(
            x="predicted_impact_pct:Q",
            y="predicted_impact_pct:Q"
        )

        chart = (points + line).properties(
            title="Predicted Impact vs Actual Volatility",
            width=self.WIDTH,
            height=self.HEIGHT
        )

        return chart

    def chart_forecast_comparison(self, forecast_data: pl.DataFrame) -> alt.Chart:
        """Line chart comparing forecasts across banks."""
        df = forecast_data.to_pandas()

        if df.empty:
            return alt.Chart().mark_text().encode(text=alt.value("No forecast data"))

        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("horizon:O", title="Forecast Horizon (quarters)"),
            y=alt.Y("forecast_volatility:Q", title="Forecast Volatility"),
            color=alt.Color("ticker:N",
                          scale=alt.Scale(domain=list(self.BANK_COLORS.keys()),
                                        range=list(self.BANK_COLORS.values()))),
            tooltip=["ticker", "horizon",
                    alt.Tooltip("forecast_volatility:Q", format=".4f")]
        ).properties(
            title="Volatility Forecasts by Horizon",
            width=self.WIDTH,
            height=self.HEIGHT
        )

        return chart

    def chart_forecast_uncertainty(self, forecast_data: pl.DataFrame) -> alt.Chart:
        """Error bar chart showing forecast uncertainty."""
        df = forecast_data.to_pandas()

        if df.empty or "lower_bound" not in df.columns:
            return alt.Chart().mark_text().encode(text=alt.value("No forecast data"))

        # Select 4-quarter horizon
        df_h4 = df[df["horizon"] == 4] if "horizon" in df.columns else df

        points = alt.Chart(df_h4).mark_point(size=100).encode(
            x=alt.X("ticker:N", title="Bank"),
            y=alt.Y("forecast_volatility:Q", title="Forecast Volatility"),
            color=alt.Color("ticker:N", legend=None)
        )

        error_bars = alt.Chart(df_h4).mark_errorbar().encode(
            x=alt.X("ticker:N"),
            y=alt.Y("lower_bound:Q", title=""),
            y2=alt.Y2("upper_bound:Q")
        )

        chart = (points + error_bars).properties(
            title="4-Quarter Forecast with 95% Confidence Interval",
            width=self.WIDTH,
            height=self.HEIGHT
        )

        return chart

    def chart_backtest_performance(self, backtest_data: pl.DataFrame) -> alt.Chart:
        """Line chart showing backtest metrics over time."""
        df = backtest_data.to_pandas()

        if df.empty:
            return alt.Chart().mark_text().encode(text=alt.value("No backtest data"))

        # Use rolling MAE if available
        y_col = "rolling_mae" if "rolling_mae" in df.columns else "mae"

        chart = alt.Chart(df).mark_line().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(f"{y_col}:Q", title="Rolling MAE"),
            tooltip=["date:T", alt.Tooltip(f"{y_col}:Q", format=".4f")]
        ).properties(
            title="Backtest Performance Over Time",
            width=self.WIDTH,
            height=self.HEIGHT
        )

        return chart

    def chart_executive_summary(self) -> alt.Chart:
        """Summary dashboard with key metrics."""
        latest = (
            self.data
            .sort("date", descending=True)
            .group_by("ticker")
            .head(1)
        )

        if latest.height == 0:
            return alt.Chart().mark_text().encode(text=alt.value("No data"))

        # Key metrics
        avg_duration = float(latest["estimated_duration"].mean())
        max_duration_bank = latest.sort("estimated_duration", descending=True).head(1)["ticker"][0]

        if "vulnerability_score" in latest.columns:
            most_vulnerable = latest.sort("vulnerability_score", descending=True).head(1)["ticker"][0]
            least_vulnerable = latest.sort("vulnerability_score").head(1)["ticker"][0]
        else:
            most_vulnerable = "N/A"
            least_vulnerable = "N/A"

        # Create text summary as a simple chart
        summary_text = f"""
        DURATION MISMATCH SUMMARY
        ========================

        Average Duration: {avg_duration:.2f} years
        Highest Duration: {max_duration_bank}

        Most Vulnerable: {most_vulnerable}
        Least Vulnerable: {least_vulnerable}

        Banks Analyzed: {latest.height}
        As of: {datetime.now().strftime('%Y-%m-%d')}
        """

        chart = alt.Chart({"values": [{"text": summary_text}]}).mark_text(
            align="left",
            baseline="top",
            fontSize=14,
            font="monospace"
        ).encode(
            text="text:N"
        ).properties(
            width=self.WIDTH,
            height=200,
            title="Executive Summary"
        )

        return chart

    def save_all_charts(
        self,
        output_dir: str,
        format: str = "html",
        **kwargs,
    ) -> dict[str, str]:
        """
        Save all charts to files.

        Args:
            output_dir: Directory to save charts
            format: "html", "png", or "svg"

        Returns:
            Dictionary mapping chart names to file paths
        """
        from pathlib import Path

        charts = self.generate_all_charts(**kwargs)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        for name, chart in charts.items():
            filename = f"{name}.{format}"
            filepath = output_path / filename

            if format == "html":
                chart.save(str(filepath))
            elif format in ["png", "svg"]:
                chart.save(str(filepath))

            saved_files[name] = str(filepath)

        return saved_files
