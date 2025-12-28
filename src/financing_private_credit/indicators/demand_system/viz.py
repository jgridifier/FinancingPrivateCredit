"""
Visualization Module for Demand System Indicator

Reproduces key charts from Boyarchenko & Elias (2024) and extends
with real-time monitoring dashboards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

try:
    import altair as alt
    alt.data_transformers.disable_max_rows()
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False


class DemandSystemVisualizer:
    """
    Visualizer for Demand System analysis.

    Generates numbered charts for consistent storytelling.
    """

    def __init__(self, data: pl.DataFrame):
        """
        Initialize visualizer.

        Args:
            data: DataFrame with demand system metrics
        """
        self.data = data
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
        elasticity_results: Optional[dict] = None,
        irf_data: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Generate all charts.

        Args:
            elasticity_results: Elasticity estimation results
            irf_data: Impulse response function data

        Returns:
            Dictionary of numbered charts
        """
        self._check_altair()

        charts = {}

        charts["01_credit_to_gdp"] = self.chart_credit_to_gdp()
        charts["02_lender_composition"] = self.chart_lender_composition()
        charts["03_credit_growth"] = self.chart_credit_growth()
        charts["04_growth_contributions"] = self.chart_growth_contributions()
        charts["05_crisis_risk_timeline"] = self.chart_crisis_risk_timeline()

        if elasticity_results:
            charts["06_elasticity_comparison"] = self.chart_elasticity_comparison(elasticity_results)

        if irf_data:
            charts["07_impulse_responses"] = self.chart_impulse_responses(irf_data)

        charts["08_bank_share_vs_growth"] = self.chart_bank_share_vs_growth()
        charts["09_executive_summary"] = self.chart_executive_summary()

        self._charts = charts
        return charts

    def chart_credit_to_gdp(
        self,
        date_col: str = "date",
        title: str = "Private Credit to GDP Ratio",
    ) -> Any:
        """01: Time series chart of credit-to-GDP ratios."""
        self._check_altair()

        # Find credit-to-GDP columns
        ratio_cols = [c for c in self.data.columns if c.endswith("_to_gdp") or "credit_to_gdp" in c]

        if not ratio_cols:
            # Try to compute it
            if "total_credit" in self.data.columns and "gdp" in self.data.columns:
                df = self.data.with_columns(
                    (pl.col("total_credit") / pl.col("gdp") * 100).alias("credit_to_gdp")
                )
                ratio_cols = ["credit_to_gdp"]
            else:
                return self._placeholder_chart("01: Credit to GDP (no data)")

        # Melt to long format for Altair
        df_long = self.data.select([date_col] + ratio_cols).unpivot(
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
            color=alt.Color("series:N", title="Credit Type"),
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
        self,
        date_col: str = "date",
        bank_col: str = "bank_share",
        nonbank_col: str = "nonbank_share",
        title: str = "Lender Composition: Bank vs Nonbank Share",
    ) -> Any:
        """02: Stacked area chart showing bank vs nonbank share of credit."""
        self._check_altair()

        if bank_col not in self.data.columns or nonbank_col not in self.data.columns:
            return self._placeholder_chart("02: Lender Composition (no data)")

        df = self.data.select([date_col, bank_col, nonbank_col])

        # Melt to long format
        df_long = df.unpivot(
            index=date_col,
            variable_name="lender_type",
            value_name="share",
        )

        df_long = df_long.with_columns(
            pl.col("lender_type")
            .str.replace("_share", "")
            .str.replace("_", " ")
            .str.to_titlecase()
        )

        chart = alt.Chart(df_long.to_pandas()).mark_area().encode(
            x=alt.X(f"{date_col}:T", title="Date"),
            y=alt.Y("share:Q", stack="normalize", title="Share (%)"),
            color=alt.Color(
                "lender_type:N",
                title="Lender Type",
                scale=alt.Scale(
                    domain=["Bank", "Nonbank"],
                    range=["#3498db", "#e74c3c"]
                )
            ),
            tooltip=[
                alt.Tooltip(f"{date_col}:T", title="Date", format="%Y-%m"),
                alt.Tooltip("lender_type:N", title="Type"),
                alt.Tooltip("share:Q", title="Share", format=".1f"),
            ],
        ).properties(
            width=700,
            height=400,
            title=title,
        )

        return chart

    def chart_credit_growth(
        self,
        date_col: str = "date",
        title: str = "Credit Growth by Lender Type",
    ) -> Any:
        """03: Line chart of credit growth rates."""
        self._check_altair()

        growth_cols = [c for c in self.data.columns if c.endswith("_growth")]

        if not growth_cols:
            return self._placeholder_chart("03: Credit Growth (no data)")

        df_long = self.data.select([date_col] + growth_cols).unpivot(
            index=date_col,
            variable_name="series",
            value_name="growth",
        )

        df_long = df_long.with_columns(
            pl.col("series")
            .str.replace("_growth", "")
            .str.replace("_credit", "")
            .str.replace("_", " ")
            .str.to_titlecase()
        )

        chart = alt.Chart(df_long.to_pandas()).mark_line(strokeWidth=2).encode(
            x=alt.X(f"{date_col}:T", title="Date"),
            y=alt.Y("growth:Q", title="YoY Growth (%)"),
            color=alt.Color("series:N", title="Credit Type"),
            tooltip=[
                alt.Tooltip(f"{date_col}:T", title="Date"),
                alt.Tooltip("series:N", title="Type"),
                alt.Tooltip("growth:Q", title="Growth (%)", format=".1f"),
            ],
        ).properties(
            width=700,
            height=400,
            title=title,
        )

        # Add zero line
        zero_line = alt.Chart(pl.DataFrame({"y": [0]}).to_pandas()).mark_rule(
            color="gray",
            strokeDash=[3, 3]
        ).encode(y="y:Q")

        return chart + zero_line

    def chart_growth_contributions(
        self,
        date_col: str = "date",
        title: str = "Contribution to Total Credit Growth",
    ) -> Any:
        """04: Stacked bar chart of growth contributions."""
        self._check_altair()

        contrib_cols = [c for c in self.data.columns if "contribution_to_growth" in c]

        if not contrib_cols:
            return self._placeholder_chart("04: Growth Contributions (no data)")

        df_long = self.data.select([date_col] + contrib_cols).unpivot(
            index=date_col,
            variable_name="lender",
            value_name="contribution",
        )

        df_long = df_long.with_columns(
            pl.col("lender")
            .str.replace("_contribution_to_growth", "")
            .str.to_titlecase()
        )

        chart = alt.Chart(df_long.to_pandas()).mark_bar().encode(
            x=alt.X(f"{date_col}:T", title="Date"),
            y=alt.Y("contribution:Q", title="Contribution (pp)"),
            color=alt.Color(
                "lender:N",
                title="Lender",
                scale=alt.Scale(
                    domain=["Bank", "Nonbank"],
                    range=["#3498db", "#e74c3c"]
                )
            ),
            tooltip=[
                alt.Tooltip(f"{date_col}:T", title="Date"),
                alt.Tooltip("lender:N", title="Lender"),
                alt.Tooltip("contribution:Q", title="Contribution", format=".2f"),
            ],
        ).properties(
            width=700,
            height=400,
            title=title,
        )

        return chart

    def chart_crisis_risk_timeline(
        self,
        date_col: str = "date",
        title: str = "Elevated Crisis Risk Periods",
    ) -> Any:
        """05: Timeline highlighting high-risk periods."""
        self._check_altair()

        if "elevated_crisis_risk" not in self.data.columns:
            return self._placeholder_chart("05: Crisis Risk Timeline (no data)")

        df = self.data.select([date_col, "elevated_crisis_risk", "total_credit_growth", "bank_share"])

        # Main line: credit growth
        growth_line = alt.Chart(df.to_pandas()).mark_line(
            color="#3498db",
            strokeWidth=2
        ).encode(
            x=alt.X(f"{date_col}:T", title="Date"),
            y=alt.Y("total_credit_growth:Q", title="Credit Growth (%)"),
        )

        # Highlight risk periods
        risk_periods = alt.Chart(df.to_pandas()).mark_rect(
            opacity=0.3,
            color="#e74c3c"
        ).encode(
            x=alt.X(f"{date_col}:T"),
            x2=alt.X2(f"{date_col}:T"),
        ).transform_filter(
            alt.datum.elevated_crisis_risk == 1
        )

        chart = (risk_periods + growth_line).properties(
            width=700,
            height=400,
            title=title,
        )

        return chart

    def chart_elasticity_comparison(
        self,
        elasticity_results: dict,
        title: str = "Supply Elasticity: Bank vs Nonbank",
    ) -> Any:
        """06: Bar chart comparing elasticities."""
        self._check_altair()

        df = pl.DataFrame({
            "lender": ["Bank", "Nonbank"],
            "elasticity": [
                elasticity_results.get("bank_elasticity", 0),
                elasticity_results.get("nonbank_elasticity", 0)
            ],
            "se": [
                elasticity_results.get("bank_se", 0),
                elasticity_results.get("nonbank_se", 0)
            ],
        })

        # Calculate error bars
        df = df.with_columns([
            (pl.col("elasticity") - 1.96 * pl.col("se")).alias("lower"),
            (pl.col("elasticity") + 1.96 * pl.col("se")).alias("upper"),
        ])

        bars = alt.Chart(df.to_pandas()).mark_bar().encode(
            x=alt.X("lender:N", title="Lender Type"),
            y=alt.Y("elasticity:Q", title="Supply Elasticity"),
            color=alt.Color(
                "lender:N",
                scale=alt.Scale(
                    domain=["Bank", "Nonbank"],
                    range=["#3498db", "#e74c3c"]
                ),
                legend=None
            ),
        )

        # Error bars
        errors = alt.Chart(df.to_pandas()).mark_errorbar().encode(
            x="lender:N",
            y=alt.Y("lower:Q", title=""),
            y2="upper:Q",
        )

        chart = (bars + errors).properties(
            width=400,
            height=400,
            title=title,
        )

        return chart

    def chart_impulse_responses(
        self,
        irf_data: dict,
        title: str = "Impulse Response: Credit to Output Shock",
    ) -> Any:
        """07: Line chart of impulse responses."""
        self._check_altair()

        horizons = irf_data.get("horizons", np.arange(20))
        bank_irf = irf_data.get("bank_credit_response", np.zeros_like(horizons))
        nonbank_irf = irf_data.get("nonbank_credit_response", np.zeros_like(horizons))

        df = pl.DataFrame({
            "horizon": list(horizons) * 2,
            "response": list(bank_irf) + list(nonbank_irf),
            "lender": ["Bank"] * len(horizons) + ["Nonbank"] * len(horizons),
        })

        chart = alt.Chart(df.to_pandas()).mark_line(strokeWidth=2).encode(
            x=alt.X("horizon:Q", title="Quarters After Shock"),
            y=alt.Y("response:Q", title="Response to 1% Output Shock"),
            color=alt.Color(
                "lender:N",
                title="Credit Type",
                scale=alt.Scale(
                    domain=["Bank", "Nonbank"],
                    range=["#3498db", "#e74c3c"]
                )
            ),
        ).properties(
            width=600,
            height=400,
            title=title,
        )

        # Add zero line
        zero_line = alt.Chart(pl.DataFrame({"y": [0]}).to_pandas()).mark_rule(
            color="gray",
            strokeDash=[3, 3]
        ).encode(y="y:Q")

        return chart + zero_line

    def chart_bank_share_vs_growth(
        self,
        title: str = "Bank Share vs Credit Growth",
    ) -> Any:
        """08: Scatter plot of bank share vs credit growth."""
        self._check_altair()

        if "bank_share" not in self.data.columns or "total_credit_growth" not in self.data.columns:
            return self._placeholder_chart("08: Bank Share vs Growth (no data)")

        chart = alt.Chart(self.data.to_pandas()).mark_circle(size=60).encode(
            x=alt.X("bank_share:Q", title="Bank Share (%)"),
            y=alt.Y("total_credit_growth:Q", title="Credit Growth (%)"),
            color=alt.condition(
                alt.datum.elevated_crisis_risk == 1,
                alt.value("#e74c3c"),
                alt.value("#3498db")
            ) if "elevated_crisis_risk" in self.data.columns else alt.value("#3498db"),
            tooltip=["date:T", "bank_share:Q", "total_credit_growth:Q"],
        ).properties(
            width=500,
            height=400,
            title=title,
        )

        return chart

    def chart_executive_summary(self) -> Any:
        """09: Summary statistics dashboard."""
        self._check_altair()

        # Calculate summary stats
        stats = []

        if "bank_share" in self.data.columns:
            latest_bank_share = self.data.filter(
                pl.col("bank_share").is_not_null()
            )["bank_share"].tail(1).item() if self.data.height > 0 else 0
            stats.append({"metric": "Current Bank Share", "value": latest_bank_share})

        if "total_credit_growth" in self.data.columns:
            latest_growth = self.data.filter(
                pl.col("total_credit_growth").is_not_null()
            )["total_credit_growth"].tail(1).item() if self.data.height > 0 else 0
            stats.append({"metric": "Current Credit Growth", "value": latest_growth})

        if "elevated_crisis_risk" in self.data.columns:
            n_risk = self.data["elevated_crisis_risk"].sum()
            stats.append({"metric": "High Risk Periods", "value": float(n_risk)})

        if not stats:
            return self._placeholder_chart("09: Executive Summary")

        df = pl.DataFrame(stats)

        chart = alt.Chart(df.to_pandas()).mark_bar().encode(
            x=alt.X("value:Q", title="Value"),
            y=alt.Y("metric:N", title="Metric"),
            color=alt.value("#3498db"),
        ).properties(
            width=400,
            height=200,
            title="09: Executive Summary",
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
            chart.save(str(file_path))
            print(f"Saved: {file_path}")
