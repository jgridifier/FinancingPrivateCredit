"""
Demand System Indicator - Original Paper Replication

Implements the demand system approach from Boyarchenko & Elias (2024):
"Financing Private Credit: The Role of Lender Type in Credit Booms"

This indicator replicates the core analysis from the paper:
1. Credit decomposition by lender type (bank vs nonbank)
2. Supply elasticity estimation
3. Crisis probability computation
4. Schularick-Taylor credit expansion predictor

Key finding: Credit expansions financed primarily by banks are
associated with higher crisis probability than those financed by nonbanks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
from scipy import stats

from ..base import (
    BaseIndicator,
    IndicatorMetadata,
    IndicatorResult,
    register_indicator,
)


@dataclass
class DemandSystemSpec:
    """Specification for demand system model."""

    name: str
    description: str

    # Decomposition settings
    growth_periods: int = 4  # Quarters for growth calculation
    cycle_window: int = 40  # Quarters for trend calculation (10 years)

    # Crisis probability thresholds
    credit_growth_threshold: float = 10.0  # YoY growth threshold (%)
    bank_share_threshold: float = 60.0  # Bank share threshold (%)

    # Elasticity estimation
    elasticity_method: str = "ols"  # "ols", "iv", "gmm"

    # IRF settings
    irf_horizons: int = 20
    persistence: float = 0.9

    @classmethod
    def from_json(cls, path: str | Path) -> "DemandSystemSpec":
        """Load specification from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        """Save specification to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class ElasticityResults:
    """Results from elasticity estimation."""

    bank_elasticity: float
    nonbank_elasticity: float
    bank_se: float
    nonbank_se: float
    r_squared: float
    n_obs: int

    def bank_more_procyclical(self) -> bool:
        """Check if bank credit is more procyclical than nonbank."""
        return self.bank_elasticity > self.nonbank_elasticity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bank_elasticity": self.bank_elasticity,
            "nonbank_elasticity": self.nonbank_elasticity,
            "bank_se": self.bank_se,
            "nonbank_se": self.nonbank_se,
            "r_squared": self.r_squared,
            "n_obs": self.n_obs,
            "bank_more_procyclical": self.bank_more_procyclical(),
        }


class CreditDecomposition:
    """
    Decompose private credit by lender type following the paper's methodology.

    The paper distinguishes between:
    - Banks: Commercial banks, credit unions, savings institutions
    - Nonbanks: Shadow banks (MMFs, ABS issuers, broker-dealers, finance companies),
                insurance companies, pension funds, and other financial intermediaries
    """

    def __init__(self, data: pl.DataFrame):
        """
        Initialize with credit data.

        Args:
            data: DataFrame with bank_credit, nonbank_credit, and total columns
        """
        self.data = data

    def compute_shares(self) -> pl.DataFrame:
        """
        Compute the share of credit from each lender type.

        Returns:
            DataFrame with bank_share and nonbank_share columns
        """
        df = self.data.clone()

        # Compute total from components if not present
        if "total_credit" not in df.columns:
            if "bank_credit" in df.columns and "nonbank_credit" in df.columns:
                df = df.with_columns(
                    (pl.col("bank_credit") + pl.col("nonbank_credit")).alias("total_credit")
                )

        # Compute shares
        if "total_credit" in df.columns:
            df = df.with_columns([
                (pl.col("bank_credit") / pl.col("total_credit") * 100).alias("bank_share"),
                (pl.col("nonbank_credit") / pl.col("total_credit") * 100).alias("nonbank_share"),
            ])

        return df

    def compute_growth_decomposition(self, periods: int = 4) -> pl.DataFrame:
        """
        Decompose credit growth into bank and nonbank contributions.

        This shows how much of total credit growth came from each lender type.

        Args:
            periods: Number of periods for growth calculation

        Returns:
            DataFrame with growth contributions from each sector
        """
        df = self.compute_shares()

        # Compute changes in credit
        for col in ["bank_credit", "nonbank_credit", "total_credit"]:
            if col in df.columns:
                df = df.with_columns([
                    (pl.col(col) - pl.col(col).shift(periods)).alias(f"{col}_change"),
                    ((pl.col(col) / pl.col(col).shift(periods) - 1) * 100).alias(f"{col}_growth"),
                ])

        # Contribution to total growth
        if "total_credit_change" in df.columns:
            if "bank_credit_change" in df.columns:
                df = df.with_columns(
                    (pl.col("bank_credit_change") / pl.col("total_credit").shift(periods) * 100)
                    .alias("bank_contribution_to_growth")
                )
            if "nonbank_credit_change" in df.columns:
                df = df.with_columns(
                    (pl.col("nonbank_credit_change") / pl.col("total_credit").shift(periods) * 100)
                    .alias("nonbank_contribution_to_growth")
                )

        return df

    def compute_cyclical_properties(self, gdp_col: str = "gdp", window: int = 40) -> pl.DataFrame:
        """
        Compute cyclical properties of bank vs nonbank credit.

        Key finding from the paper: Bank credit is more sensitive to economic
        downturns than nonbank credit.

        Args:
            gdp_col: Name of GDP column for cycle identification
            window: Rolling window for trend calculation

        Returns:
            DataFrame with cyclical metrics
        """
        df = self.data.clone()

        if gdp_col not in df.columns:
            raise ValueError(f"GDP column '{gdp_col}' not found")

        # Compute HP-filtered trend (simplified using rolling mean as proxy)
        for col in ["bank_credit", "nonbank_credit", gdp_col]:
            if col in df.columns:
                # Trend component (rolling mean)
                df = df.with_columns(
                    pl.col(col).rolling_mean(window_size=window, min_periods=1).alias(f"{col}_trend")
                )
                # Cycle component (deviation from trend)
                df = df.with_columns(
                    ((pl.col(col) - pl.col(f"{col}_trend")) / pl.col(f"{col}_trend") * 100)
                    .alias(f"{col}_cycle")
                )

        return df


class DemandSystemModel:
    """
    Implement the demand system approach from Boyarchenko & Elias (2024).

    The model jointly estimates credit demand and supply to compute equilibrium
    elasticities showing how credit quantities respond to economic conditions.

    Key model components:
    1. Credit demand: Depends on GDP, interest rates, economic conditions
    2. Credit supply: Differs by lender type (banks more procyclical)
    3. Equilibrium: Where demand meets supply from all lender types
    """

    def __init__(self, data: pl.DataFrame):
        """
        Initialize the demand system model.

        Args:
            data: DataFrame with credit and macro data
        """
        self.data = data
        self._elasticities: Optional[ElasticityResults] = None

    def estimate_supply_elasticities(
        self,
        output_var: str = "gdp",
        bank_var: str = "bank_credit",
        nonbank_var: str = "nonbank_credit",
    ) -> ElasticityResults:
        """
        Estimate supply elasticities for bank and nonbank credit.

        Uses simple OLS regression of credit growth on output growth.

        Args:
            output_var: Name of output/GDP variable
            bank_var: Name of bank credit variable
            nonbank_var: Name of nonbank credit variable

        Returns:
            ElasticityResults with estimated elasticities
        """
        df = self.data.drop_nulls()

        # Convert to numpy for regression
        data_np = df.to_numpy()
        cols = df.columns

        # Get column indices
        try:
            output_idx = cols.index(output_var)
            bank_idx = cols.index(bank_var)
            nonbank_idx = cols.index(nonbank_var)
        except ValueError as e:
            raise ValueError(f"Required column not found: {e}")

        # Compute log differences (growth rates)
        output_growth = np.diff(np.log(data_np[:, output_idx].astype(float)))
        bank_growth = np.diff(np.log(data_np[:, bank_idx].astype(float)))
        nonbank_growth = np.diff(np.log(data_np[:, nonbank_idx].astype(float)))

        # Remove any NaN/Inf values
        valid_mask = (
            np.isfinite(output_growth) &
            np.isfinite(bank_growth) &
            np.isfinite(nonbank_growth)
        )
        output_growth = output_growth[valid_mask]
        bank_growth = bank_growth[valid_mask]
        nonbank_growth = nonbank_growth[valid_mask]

        # Estimate elasticities via OLS
        bank_result = stats.linregress(output_growth, bank_growth)
        nonbank_result = stats.linregress(output_growth, nonbank_growth)

        self._elasticities = ElasticityResults(
            bank_elasticity=bank_result.slope,
            nonbank_elasticity=nonbank_result.slope,
            bank_se=bank_result.stderr,
            nonbank_se=nonbank_result.stderr,
            r_squared=(bank_result.rvalue**2 + nonbank_result.rvalue**2) / 2,
            n_obs=len(output_growth),
        )

        return self._elasticities

    def compute_crisis_probability(
        self,
        credit_growth_threshold: float = 10.0,
        bank_share_threshold: float = 60.0,
    ) -> pl.DataFrame:
        """
        Compute crisis probability indicator based on credit conditions.

        Key finding from paper: Credit expansions financed primarily by banks
        are associated with higher crisis probability than those financed by nonbanks.

        Args:
            credit_growth_threshold: YoY credit growth threshold (%)
            bank_share_threshold: Bank share of credit threshold (%)

        Returns:
            DataFrame with crisis probability indicators
        """
        df = self.data.clone()

        # Compute credit growth if not present
        if "total_credit_growth" not in df.columns:
            for col in ["bank_credit", "nonbank_credit"]:
                if col in df.columns:
                    df = df.with_columns(
                        ((pl.col(col) / pl.col(col).shift(4) - 1) * 100).alias(f"{col}_growth")
                    )

            if "bank_credit" in df.columns and "nonbank_credit" in df.columns:
                total = pl.col("bank_credit") + pl.col("nonbank_credit")
                df = df.with_columns([
                    total.alias("total_credit"),
                    ((total / total.shift(4) - 1) * 100).alias("total_credit_growth"),
                ])

        # Compute bank share
        if "bank_share" not in df.columns and "bank_credit" in df.columns:
            df = df.with_columns(
                (pl.col("bank_credit") / pl.col("total_credit") * 100).alias("bank_share")
            )

        # Crisis indicator: high credit growth + high bank share
        df = df.with_columns([
            (pl.col("total_credit_growth") > credit_growth_threshold).alias("high_credit_growth"),
            (pl.col("bank_share") > bank_share_threshold).alias("high_bank_share"),
        ])

        # Combined risk indicator
        df = df.with_columns(
            (pl.col("high_credit_growth") & pl.col("high_bank_share"))
            .cast(pl.Int32)
            .alias("elevated_crisis_risk")
        )

        return df

    def compute_impulse_responses(
        self,
        horizons: int = 20,
        persistence: float = 0.9,
    ) -> dict[str, np.ndarray]:
        """
        Compute simplified impulse responses of credit to output shocks.

        Shows differential response of bank vs nonbank credit to economic shocks.

        Args:
            horizons: Number of periods for IRF
            persistence: AR(1) persistence parameter

        Returns:
            Dictionary with IRF arrays for bank and nonbank credit
        """
        if self._elasticities is None:
            self.estimate_supply_elasticities()

        # Compute IRFs
        horizons_arr = np.arange(horizons)
        shock_path = persistence ** horizons_arr

        bank_irf = self._elasticities.bank_elasticity * shock_path
        nonbank_irf = self._elasticities.nonbank_elasticity * shock_path

        return {
            "horizons": horizons_arr,
            "bank_credit_response": bank_irf,
            "nonbank_credit_response": nonbank_irf,
            "shock_path": shock_path,
        }


def compute_schularick_taylor_predictor(
    data: pl.DataFrame,
    credit_col: str = "total_credit",
    gdp_col: str = "gdp",
    window: int = 20,
) -> pl.DataFrame:
    """
    Compute the Schularick-Taylor credit expansion predictor.

    From Schularick & Taylor (2012): Credit expansions predict subsequent
    financial crises. The paper extends this to show lender composition matters.

    Args:
        data: DataFrame with credit and GDP data
        credit_col: Name of credit column
        gdp_col: Name of GDP column
        window: Rolling window for credit expansion measure

    Returns:
        DataFrame with credit expansion predictor
    """
    df = data.clone()

    # Credit to GDP ratio
    df = df.with_columns(
        (pl.col(credit_col) / pl.col(gdp_col)).alias("credit_to_gdp")
    )

    # Change in credit/GDP ratio (the predictor)
    df = df.with_columns(
        (pl.col("credit_to_gdp") - pl.col("credit_to_gdp").shift(window))
        .alias("credit_expansion_predictor")
    )

    return df


@register_indicator("demand_system")
class DemandSystemIndicator(BaseIndicator):
    """
    Demand System Indicator - Paper Replication

    Replicates the core analysis from Boyarchenko & Elias (2024):
    1. Credit decomposition by lender type
    2. Supply elasticity estimation
    3. Crisis probability computation

    Key output: Crisis risk indicator based on credit composition.
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self._spec: Optional[DemandSystemSpec] = None
        self._model: Optional[DemandSystemModel] = None
        self._decomposition: Optional[CreditDecomposition] = None

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="Demand System - Paper Replication",
            short_name="DemandSystem",
            description=(
                "Replicates the demand system approach from Boyarchenko & Elias (2024). "
                "Decomposes credit by lender type (bank vs nonbank) and estimates "
                "supply elasticities. Key insight: bank-financed credit expansions "
                "are associated with higher crisis probability."
            ),
            version="1.0.0",
            paper_reference="Boyarchenko & Elias (2024) - NY Fed Staff Report 1111",
            data_sources=["Flow of Funds (Z.1)", "FRED"],
            update_frequency="quarterly",
            lookback_periods=40,
        )

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch credit and macro data from Flow of Funds and FRED."""
        from ...cache import CachedFREDFetcher

        # Fetch macro data from FRED
        fred = CachedFREDFetcher(max_age_hours=6)

        # Key series for demand system analysis
        credit_series = [
            "BOGZ1FL154104005Q",  # Total credit market instruments, all sectors
            "CRDQUSAPABIS",       # Credit to private non-financial sector
            "BUSLOANS",           # Commercial and Industrial Loans
            "REALLN",             # Real Estate Loans
        ]

        macro_series = [
            "GDP",        # Gross Domestic Product
            "GDPC1",      # Real GDP
            "FEDFUNDS",   # Federal Funds Rate
            "DGS10",      # 10-Year Treasury
            "UNRATE",     # Unemployment Rate
        ]

        credit_data = fred.fetch_multiple_series(credit_series, start_date=start_date)
        macro_data = fred.fetch_multiple_series(macro_series, start_date=start_date)

        # Note: For full paper replication, would need Flow of Funds Z.1 data
        # to properly decompose by bank vs nonbank. This is a simplified version.

        return {
            "credit_data": credit_data,
            "macro_data": macro_data,
        }

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        spec: Optional[DemandSystemSpec] = None,
        **kwargs,
    ) -> IndicatorResult:
        """
        Calculate demand system metrics.

        Steps:
        1. Decompose credit by lender type
        2. Estimate supply elasticities
        3. Compute crisis probability indicators
        """
        credit_data = data.get("credit_data", pl.DataFrame())
        macro_data = data.get("macro_data", pl.DataFrame())

        if spec is None:
            spec = DemandSystemSpec(
                name="default",
                description="Default demand system specification"
            )
        self._spec = spec

        if credit_data.height == 0 and macro_data.height == 0:
            return IndicatorResult(
                indicator_name="demand_system",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No data available"},
            )

        # Merge credit and macro data
        if credit_data.height > 0 and macro_data.height > 0:
            merged = credit_data.join(macro_data, on="date", how="outer_coalesce")
        elif credit_data.height > 0:
            merged = credit_data
        else:
            merged = macro_data

        # For simplified analysis, use available series
        # Create synthetic bank/nonbank split if full Z.1 data not available
        if "bank_credit" not in merged.columns:
            if "BUSLOANS" in merged.columns:
                # Use C&I loans as proxy for bank credit
                merged = merged.with_columns([
                    pl.col("BUSLOANS").alias("bank_credit"),
                ])
            if "CRDQUSAPABIS" in merged.columns and "bank_credit" in merged.columns:
                merged = merged.with_columns(
                    (pl.col("CRDQUSAPABIS") - pl.col("bank_credit")).alias("nonbank_credit")
                )

        # Initialize decomposition
        self._decomposition = CreditDecomposition(merged)

        # Compute shares and growth
        result = self._decomposition.compute_growth_decomposition(spec.growth_periods)

        # Initialize demand system model
        if "bank_credit" in result.columns and "nonbank_credit" in result.columns:
            # Add GDP for elasticity estimation
            if "GDP" in result.columns:
                result = result.rename({"GDP": "gdp"})
            elif "GDPC1" in result.columns:
                result = result.rename({"GDPC1": "gdp"})

            self._model = DemandSystemModel(result)

            # Estimate elasticities
            try:
                elasticities = self._model.estimate_supply_elasticities()
                elasticity_dict = elasticities.to_dict()
            except Exception as e:
                print(f"Warning: Could not estimate elasticities: {e}")
                elasticity_dict = {}

            # Compute crisis probability
            result = self._model.compute_crisis_probability(
                spec.credit_growth_threshold,
                spec.bank_share_threshold,
            )
        else:
            elasticity_dict = {}

        return IndicatorResult(
            indicator_name="demand_system",
            calculation_date=datetime.now(),
            data=result,
            metadata={
                "spec_name": spec.name,
                "elasticities": elasticity_dict,
                "n_obs": result.height,
            },
        )

    def get_elasticity_results(self) -> Optional[ElasticityResults]:
        """Get elasticity estimation results."""
        if self._model:
            return self._model._elasticities
        return None

    def get_impulse_responses(self) -> dict[str, np.ndarray]:
        """Get impulse response functions."""
        if self._model:
            return self._model.compute_impulse_responses(
                horizons=self._spec.irf_horizons if self._spec else 20,
                persistence=self._spec.persistence if self._spec else 0.9,
            )
        return {}

    def get_dashboard_components(self) -> dict[str, Any]:
        """Return dashboard configuration."""
        return {
            "tabs": [
                {"name": "Credit Composition", "icon": "pie_chart"},
                {"name": "Elasticities", "icon": "trending_up"},
                {"name": "Crisis Probability", "icon": "warning"},
                {"name": "Impulse Responses", "icon": "timeline"},
            ],
            "primary_metric": "bank_share",
            "alert_fields": ["elevated_crisis_risk"],
        }
