"""
Template Indicator

Copy this file as a starting point for new indicators.
Replace 'Template' with your indicator name throughout.
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

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch all data required for this indicator.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date

        Returns:
            Dictionary of DataFrames with required data
        """
        from ...bank_data import BankDataCollector, TARGET_BANKS
        from ...cache import CachedFREDFetcher

        # 1. Fetch bank-level data from SEC EDGAR
        collector = BankDataCollector(start_date=start_date)
        bank_panel = collector.fetch_all_banks()
        bank_panel = collector.compute_derived_metrics(bank_panel)

        # 2. Fetch macro data from FRED (customize series as needed)
        fred = CachedFREDFetcher(max_age_hours=6)
        macro_series = ["FEDFUNDS", "DGS10", "BAA10Y"]
        macro_data = fred.fetch_multiple_series(macro_series, start_date=start_date)

        # 3. Add any additional data sources here

        return {
            "bank_panel": bank_panel,
            "macro_data": macro_data,
            # Add other data as needed
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
