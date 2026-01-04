"""
Template Indicator

Copy this file as a starting point for new indicators.
Replace 'Template' with your indicator name throughout.

Required methods:
- get_metadata(): Describe your indicator
- fetch_data(): Gather required data (use DataRegistry for shared data)
- calculate(): Compute the indicator values

Optional methods (have defaults):
- nowcast(): High-frequency updates (set supports_nowcast=True to enable)
- get_dashboard_components(): Streamlit dashboard config
- get_required_data_sources(): Document data dependencies
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import polars as pl

from ..base import (
    BaseIndicator,
    IndicatorMetadata,
    IndicatorResult,
    register_indicator,
)


@dataclass
class TemplateSpec:
    """Configuration for the template indicator."""

    name: str = "default"
    description: str = "Default template configuration"

    # Add your parameters here
    window: int = 20
    threshold: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "TemplateSpec":
        """Create spec from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Uncomment and rename when ready to register
# @register_indicator("template")
class TemplateIndicator(BaseIndicator):
    """
    Template Indicator - Replace with your description.

    This indicator measures [what it measures] to help identify
    [what insights it provides].

    Key metrics:
    - Metric 1: Description
    - Metric 2: Description
    """

    # Set to True if you implement nowcast()
    supports_nowcast: bool = False

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self._spec: Optional[TemplateSpec] = None

    def get_metadata(self) -> IndicatorMetadata:
        """Return indicator metadata."""
        return IndicatorMetadata(
            name="Template Indicator",  # Full name
            short_name="Template",  # Abbreviation
            description=(
                "Replace this with a detailed description of what "
                "your indicator measures and why it matters."
            ),
            version="1.0.0",
            paper_reference="Optional: Citation or reference",
            data_sources=["SEC EDGAR", "FRED"],  # List your data sources
            update_frequency="quarterly",  # quarterly, monthly, weekly, daily
            lookback_periods=20,  # Quarters of history needed
        )

    def get_required_data_sources(self) -> list[str]:
        """
        Document what data sources this indicator needs.

        Used by DataRegistry for smart pre-fetching.
        """
        return ["bank_panel", "macro_data"]

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch all data required for this indicator.

        Uses DataRegistry for shared data (bank panels, FRED macro).
        Add indicator-specific data sources as needed.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date

        Returns:
            Dictionary of DataFrames with required data
        """
        # Use DataRegistry for shared data sources (recommended)
        # This provides caching and avoids redundant API calls
        from ...core import DataRegistry

        registry = DataRegistry.get_instance()

        # 1. Get shared bank panel data (cached automatically)
        bank_panel = registry.get_bank_panel(
            start_date=start_date,
            compute_derived=True,  # Include derived metrics
        )

        # 2. Get FRED macro data (cached automatically)
        macro_series = ["FEDFUNDS", "DGS10", "BAA10Y"]
        macro_data = registry.get_macro_series(macro_series, start_date)

        # 3. Add indicator-specific data sources here
        # Option A: Register a custom source for reuse
        #
        # def fetch_my_custom_data(start_date: str) -> pl.DataFrame:
        #     # Your custom fetching logic
        #     ...
        #
        # registry.register_source("my_custom_data", fetch_my_custom_data)
        # custom_data = registry.get("my_custom_data", start_date=start_date)
        #
        # Option B: Fetch directly if one-time use
        # custom_data = self._fetch_custom_data(start_date)

        return {
            "bank_panel": bank_panel,
            "macro_data": macro_data,
            # "custom_data": custom_data,  # Add your custom data
        }

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        spec: Optional[TemplateSpec] = None,
        **kwargs,
    ) -> IndicatorResult:
        """
        Calculate the indicator values.

        Args:
            data: Dictionary of DataFrames from fetch_data()
            spec: Optional specification override
            **kwargs: Additional parameters

        Returns:
            IndicatorResult with calculated values
        """
        bank_panel = data.get("bank_panel", pl.DataFrame())
        macro_data = data.get("macro_data", pl.DataFrame())

        if bank_panel.height == 0:
            return IndicatorResult(
                indicator_name="template",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No bank data available"},
            )

        # Use provided spec or default
        if spec is not None:
            self._spec = spec
        elif self._spec is None:
            self._spec = TemplateSpec()

        # =====================================================
        # YOUR CALCULATION LOGIC HERE
        # =====================================================
        #
        # Example: Calculate a simple metric for each bank
        #
        # results = []
        # for ticker in bank_panel["ticker"].unique().to_list():
        #     bank_df = bank_panel.filter(pl.col("ticker") == ticker)
        #     metric = self._calculate_metric(bank_df)
        #     results.append({"ticker": ticker, "metric": metric})
        #
        # result_df = pl.DataFrame(results)
        #
        # =====================================================

        # Placeholder result
        result_df = bank_panel.select(["ticker", "date"]).unique()

        return IndicatorResult(
            indicator_name="template",
            calculation_date=datetime.now(),
            data=result_df,
            metadata={
                "spec": self._spec.name,
                "n_banks": result_df["ticker"].n_unique(),
                "date_range": {
                    "start": str(result_df["date"].min()),
                    "end": str(result_df["date"].max()),
                },
            },
        )

    def _calculate_metric(self, bank_df: pl.DataFrame) -> float:
        """
        Calculate your indicator metric for a single bank.

        Args:
            bank_df: DataFrame with bank's time series

        Returns:
            Calculated metric value
        """
        # Implement your calculation logic
        return 0.0

    # =========================================================================
    # OPTIONAL: Nowcasting support
    # =========================================================================
    # Uncomment and implement if your indicator supports nowcasting.
    # Don't forget to set `supports_nowcast = True` above.
    #
    # def nowcast(
    #     self,
    #     data: dict[str, pl.DataFrame],
    #     **kwargs,
    # ) -> IndicatorResult:
    #     """
    #     Generate high-frequency nowcast estimates.
    #
    #     Uses proxy variables (stock prices, CDS spreads, etc.) to update
    #     the indicator between quarterly data releases.
    #
    #     Args:
    #         data: Dictionary with "quarterly_results" and "stock_data"
    #         **kwargs: Additional parameters
    #
    #     Returns:
    #         IndicatorResult with nowcast values
    #     """
    #     from .nowcast import TemplateNowcaster
    #
    #     nowcaster = TemplateNowcaster()
    #     return nowcaster.nowcast(
    #         quarterly_data=data.get("quarterly_results", pl.DataFrame()),
    #         proxy_data={"stock_data": data.get("stock_data", pl.DataFrame())},
    #     )

    # =========================================================================
    # OPTIONAL: Custom dashboard configuration
    # =========================================================================
    # Uncomment to provide custom dashboard configuration.
    # The default implementation returns a minimal config.
    #
    # def get_dashboard_components(self) -> dict[str, Any]:
    #     """Return dashboard configuration."""
    #     return {
    #         "tabs": [
    #             {"name": "Overview", "icon": "chart"},
    #             {"name": "Bank Details", "icon": "bank"},
    #         ],
    #         "primary_metric": "my_main_metric",
    #         "alert_fields": ["risk_flag", "warning_flag"],
    #     }
