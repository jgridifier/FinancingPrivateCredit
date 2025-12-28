"""
Visualization Module for Funding Stability Indicator

Uses Vega-Altair for interactive visualizations.
Charts are numbered for consistent storytelling.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import polars as pl

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False


class FundingStabilityVisualizer:
    """
    Visualizer for Funding Stability Score analysis.

    Generates numbered charts for consistent narratives.
    """

    def __init__(self, funding_data: pl.DataFrame):
        """
        Initialize visualizer.

        Args:
            funding_data: DataFrame with funding stability data
        """
        self.funding_data = funding_data
        self._charts: dict[str, Any] = {}

    def _check_altair(self):
        """Check if altair is available."""
        if not HAS_ALTAIR:
            raise ImportError(
                "altair is required for visualization. "
                "Install with: pip install altair"
            )

    def generate_all_charts(
        self,
        forecast_data: Optional[pl.DataFrame] = None,
        scenario_data: Optional[pl.DataFrame] = None,
        backtest_data: Optional[pl.DataFrame] = None,
    ) -> dict[str, Any]:
        """
        Generate all charts.

        Args:
            forecast_data: Forecast results (optional)
            scenario_data: Scenario analysis results (optional)
            backtest_data: Backtest results (optional)

        Returns:
            Dictionary of numbered charts
        """
        self._check_altair()

        charts = {}

        # Core charts
        charts["01_resilience_ranking"] = self.plot_resilience_ranking()
        charts["02_score_distribution"] = self.plot_score_distribution()
        charts["03_score_over_time"] = self.plot_score_time_series()
        charts["04_component_breakdown"] = self.plot_component_breakdown()
        charts["05_risk_tier_composition"] = self.plot_risk_tier_composition()
        charts["06_stress_indicators"] = self.plot_stress_indicators()
        charts["07_uninsured_deposit_heatmap"] = self.plot_uninsured_deposits()
        charts["08_fhlb_dependence"] = self.plot_fhlb_dependence()
        charts["09_aoci_impact"] = self.plot_aoci_impact()

        # Forecast charts
        if forecast_data is not None:
            charts["10_forecast_scenarios"] = self.plot_forecast_scenarios(forecast_data)
            charts["11_monte_carlo_distribution"] = self.plot_monte_carlo_distribution(forecast_data)

        # Scenario charts
        if scenario_data is not None:
            charts["12_scenario_comparison"] = self.plot_scenario_comparison(scenario_data)

        # Backtest charts
        if backtest_data is not None:
            charts["13_backtest_performance"] = self.plot_backtest_performance(backtest_data)

        # Summary
        charts["14_executive_summary"] = self.plot_executive_summary()

        self._charts = charts
        return charts

    def plot_resilience_ranking(self) -> Any:
        """01: Bar chart ranking banks by funding resilience score."""
        self._check_altair()

        # Get latest scores
        latest = self.funding_data.group_by("ticker").agg(
            pl.all().sort_by("date").last()
        ).sort("funding_resilience_score", descending=True)

        df_pd = latest.to_pandas()

        chart = alt.Chart(df_pd).mark_bar().encode(
            x=alt.X("funding_resilience_score:Q", title="Funding Resilience Score"),
            y=alt.Y("ticker:N", sort="-x", title="Bank"),
            color=alt.Color(
                "risk_tier:N",
                scale=alt.Scale(
                    domain=["Low", "Moderate", "High", "Critical"],
                    range=["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
                ),
                title="Risk Tier"
            ),
            tooltip=["ticker", "funding_resilience_score", "risk_tier"]
        ).properties(
            title="01: Funding Resilience Ranking",
            width=500,
            height=400
        )

        return chart

    def plot_score_distribution(self) -> Any:
        """02: Histogram of score distribution across banks."""
        self._check_altair()

        latest = self.funding_data.group_by("ticker").agg(
            pl.col("funding_resilience_score").last()
        )

        df_pd = latest.to_pandas()

        chart = alt.Chart(df_pd).mark_bar().encode(
            x=alt.X("funding_resilience_score:Q", bin=alt.Bin(maxbins=20), title="Score"),
            y=alt.Y("count():Q", title="Number of Banks"),
        ).properties(
            title="02: Score Distribution",
            width=500,
            height=300
        )

        return chart

    def plot_score_time_series(self) -> Any:
        """03: Line chart of scores over time by bank."""
        self._check_altair()

        df_pd = self.funding_data.to_pandas()

        chart = alt.Chart(df_pd).mark_line().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("funding_resilience_score:Q", title="Score"),
            color=alt.Color("ticker:N", title="Bank"),
            tooltip=["ticker", "date", "funding_resilience_score"]
        ).properties(
            title="03: Funding Resilience Score Over Time",
            width=600,
            height=400
        )

        return chart

    def plot_component_breakdown(self) -> Any:
        """04: Stacked bar showing component contributions."""
        self._check_altair()

        # Get latest with contributions
        contrib_cols = [c for c in self.funding_data.columns if c.startswith("contrib_")]

        if not contrib_cols:
            return self._placeholder_chart("04: Component Breakdown (no contribution data)")

        latest = self.funding_data.group_by("ticker").agg(
            pl.all().sort_by("date").last()
        )

        # Melt to long format
        df_pd = latest.select(["ticker"] + contrib_cols).to_pandas()
        df_long = df_pd.melt(id_vars=["ticker"], var_name="component", value_name="contribution")
        df_long["component"] = df_long["component"].str.replace("contrib_", "")

        chart = alt.Chart(df_long).mark_bar().encode(
            x=alt.X("contribution:Q", stack="zero", title="Contribution to Score"),
            y=alt.Y("ticker:N", title="Bank"),
            color=alt.Color("component:N", title="Component"),
            tooltip=["ticker", "component", "contribution"]
        ).properties(
            title="04: Score Component Breakdown",
            width=500,
            height=400
        )

        return chart

    def plot_risk_tier_composition(self) -> Any:
        """05: Pie chart of risk tier distribution."""
        self._check_altair()

        latest = self.funding_data.group_by("ticker").agg(
            pl.col("risk_tier").last()
        )

        tier_counts = latest.group_by("risk_tier").len()
        df_pd = tier_counts.to_pandas()

        chart = alt.Chart(df_pd).mark_arc().encode(
            theta=alt.Theta("len:Q"),
            color=alt.Color(
                "risk_tier:N",
                scale=alt.Scale(
                    domain=["Low", "Moderate", "High", "Critical"],
                    range=["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
                )
            ),
            tooltip=["risk_tier", "len"]
        ).properties(
            title="05: Risk Tier Composition",
            width=300,
            height=300
        )

        return chart

    def plot_stress_indicators(self) -> Any:
        """06: Heatmap of stress indicator flags."""
        self._check_altair()

        stress_cols = ["is_fhlb_dependent", "is_run_vulnerable", "has_aoci_stress"]
        available = [c for c in stress_cols if c in self.funding_data.columns]

        if not available:
            return self._placeholder_chart("06: Stress Indicators (no data)")

        latest = self.funding_data.group_by("ticker").agg(
            pl.all().sort_by("date").last()
        )

        df_pd = latest.select(["ticker"] + available).to_pandas()
        df_long = df_pd.melt(id_vars=["ticker"], var_name="indicator", value_name="flagged")
        df_long["indicator"] = df_long["indicator"].str.replace("is_", "").str.replace("has_", "")

        chart = alt.Chart(df_long).mark_rect().encode(
            x=alt.X("indicator:N", title="Stress Indicator"),
            y=alt.Y("ticker:N", title="Bank"),
            color=alt.Color(
                "flagged:N",
                scale=alt.Scale(domain=[False, True], range=["#ecf0f1", "#e74c3c"])
            ),
            tooltip=["ticker", "indicator", "flagged"]
        ).properties(
            title="06: Stress Indicator Flags",
            width=300,
            height=400
        )

        return chart

    def plot_uninsured_deposits(self) -> Any:
        """07: Bar chart of uninsured deposit ratios."""
        self._check_altair()

        if "uninsured_deposit_ratio" not in self.funding_data.columns:
            return self._placeholder_chart("07: Uninsured Deposits (no data)")

        latest = self.funding_data.group_by("ticker").agg(
            pl.col("uninsured_deposit_ratio").last()
        ).sort("uninsured_deposit_ratio", descending=True)

        df_pd = latest.to_pandas()

        chart = alt.Chart(df_pd).mark_bar().encode(
            x=alt.X("uninsured_deposit_ratio:Q", title="Uninsured Deposit Ratio"),
            y=alt.Y("ticker:N", sort="-x", title="Bank"),
            color=alt.condition(
                alt.datum.uninsured_deposit_ratio > 0.5,
                alt.value("#e74c3c"),
                alt.value("#3498db")
            ),
            tooltip=["ticker", "uninsured_deposit_ratio"]
        ).properties(
            title="07: Uninsured Deposit Exposure (Run Risk)",
            width=500,
            height=400
        )

        return chart

    def plot_fhlb_dependence(self) -> Any:
        """08: Bar chart of FHLB advance utilization."""
        self._check_altair()

        if "fhlb_advance_ratio" not in self.funding_data.columns:
            return self._placeholder_chart("08: FHLB Dependence (no data)")

        latest = self.funding_data.group_by("ticker").agg(
            pl.col("fhlb_advance_ratio").last()
        ).sort("fhlb_advance_ratio", descending=True)

        df_pd = latest.to_pandas()

        chart = alt.Chart(df_pd).mark_bar().encode(
            x=alt.X("fhlb_advance_ratio:Q", title="FHLB Advance Ratio"),
            y=alt.Y("ticker:N", sort="-x", title="Bank"),
            color=alt.condition(
                alt.datum.fhlb_advance_ratio > 0.1,
                alt.value("#e74c3c"),
                alt.value("#2ecc71")
            ),
            tooltip=["ticker", "fhlb_advance_ratio"]
        ).properties(
            title="08: FHLB Advance Utilization (Desperation Signal)",
            width=500,
            height=400
        )

        return chart

    def plot_aoci_impact(self) -> Any:
        """09: Scatter plot of AOCI impact vs score."""
        self._check_altair()

        if "aoci_impact_ratio" not in self.funding_data.columns:
            return self._placeholder_chart("09: AOCI Impact (no data)")

        latest = self.funding_data.group_by("ticker").agg(
            pl.all().sort_by("date").last()
        )

        df_pd = latest.to_pandas()

        chart = alt.Chart(df_pd).mark_circle(size=100).encode(
            x=alt.X("aoci_impact_ratio:Q", title="AOCI Impact (% of TCE)"),
            y=alt.Y("funding_resilience_score:Q", title="Resilience Score"),
            color=alt.Color("risk_tier:N"),
            tooltip=["ticker", "aoci_impact_ratio", "funding_resilience_score"]
        ).properties(
            title="09: AOCI Impact vs Resilience Score",
            width=500,
            height=400
        )

        return chart

    def plot_forecast_scenarios(self, forecast_data: pl.DataFrame) -> Any:
        """10: Line chart comparing forecast scenarios."""
        self._check_altair()

        if "scenario" not in forecast_data.columns:
            return self._placeholder_chart("10: Forecast Scenarios")

        df_pd = forecast_data.to_pandas()

        chart = alt.Chart(df_pd).mark_bar().encode(
            x=alt.X("scenario:N", title="Scenario"),
            y=alt.Y("mean(funding_resilience_score):Q", title="Avg Score"),
            color=alt.Color("scenario:N"),
            column=alt.Column("ticker:N", title="Bank")
        ).properties(
            title="10: Forecast Under Different Scenarios",
            width=100,
            height=200
        )

        return chart

    def plot_monte_carlo_distribution(self, forecast_data: pl.DataFrame) -> Any:
        """11: Histogram of Monte Carlo simulation results."""
        self._check_altair()

        df_pd = forecast_data.to_pandas()

        chart = alt.Chart(df_pd).mark_bar().encode(
            x=alt.X("funding_resilience_score:Q", bin=alt.Bin(maxbins=50), title="Score"),
            y=alt.Y("count():Q", title="Frequency"),
        ).properties(
            title="11: Monte Carlo Score Distribution",
            width=500,
            height=300
        )

        return chart

    def plot_scenario_comparison(self, scenario_data: pl.DataFrame) -> Any:
        """12: Grouped bar chart comparing scenarios."""
        self._check_altair()

        df_pd = scenario_data.to_pandas()

        chart = alt.Chart(df_pd).mark_bar().encode(
            x=alt.X("ticker:N", title="Bank"),
            y=alt.Y("funding_resilience_score:Q", title="Score"),
            color=alt.Color("scenario:N", title="Scenario"),
            xOffset="scenario:N"
        ).properties(
            title="12: Scenario Comparison",
            width=600,
            height=400
        )

        return chart

    def plot_backtest_performance(self, backtest_data: pl.DataFrame) -> Any:
        """13: Line chart of backtest accuracy over time."""
        self._check_altair()

        return self._placeholder_chart("13: Backtest Performance")

    def plot_executive_summary(self) -> Any:
        """14: Summary dashboard with key metrics."""
        self._check_altair()

        # Get summary statistics
        latest = self.funding_data.group_by("ticker").agg(
            pl.all().sort_by("date").last()
        )

        avg_score = latest["funding_resilience_score"].mean()
        n_critical = latest.filter(pl.col("risk_tier") == "Critical").height
        n_high = latest.filter(pl.col("risk_tier") == "High").height

        summary = pl.DataFrame({
            "metric": ["Average Score", "Critical Risk Banks", "High Risk Banks"],
            "value": [avg_score, n_critical, n_high],
        }).to_pandas()

        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X("value:Q", title="Value"),
            y=alt.Y("metric:N", title="Metric"),
        ).properties(
            title="14: Executive Summary",
            width=400,
            height=200
        )

        return chart

    def _placeholder_chart(self, title: str) -> Any:
        """Create placeholder chart when data is missing."""
        self._check_altair()

        df = pl.DataFrame({"x": [0], "y": [0], "label": ["No data available"]}).to_pandas()

        return alt.Chart(df).mark_text(size=16).encode(
            x="x:Q",
            y="y:Q",
            text="label:N"
        ).properties(title=title, width=400, height=200)

    def save_all_charts(
        self,
        output_dir: str | Path,
        format: str = "html",
    ) -> None:
        """
        Save all charts to files.

        Args:
            output_dir: Output directory
            format: Output format ("html", "png", "svg")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, chart in self._charts.items():
            file_path = output_path / f"{name}.{format}"
            if format == "html":
                chart.save(str(file_path))
            elif format in ("png", "svg"):
                chart.save(str(file_path))
            print(f"Saved: {file_path}")
