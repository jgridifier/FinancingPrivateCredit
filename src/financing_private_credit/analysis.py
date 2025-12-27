"""
Analysis module implementing the demand system approach from Boyarchenko & Elias (2024).

The key insight from the paper is that the sectoral composition of lenders
financing a credit expansion is a key determinant for:
1. Subsequent real activity
2. Crisis probability

The demand system approach allows joint modeling of credit demand and supply,
enabling computation of equilibrium elasticities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from scipy import stats


@dataclass
class ElasticityResults:
    """Results from elasticity estimation."""

    bank_elasticity: float
    nonbank_elasticity: float
    bank_se: float
    nonbank_se: float
    r_squared: float
    n_obs: int


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

    def compute_cyclical_properties(self, gdp_col: str = "gdp") -> pl.DataFrame:
        """
        Compute cyclical properties of bank vs nonbank credit.

        Key finding from the paper: Bank credit is more sensitive to economic
        downturns than nonbank credit.

        Args:
            gdp_col: Name of GDP column for cycle identification

        Returns:
            DataFrame with cyclical metrics
        """
        df = self.data.clone()

        if gdp_col not in df.columns:
            raise ValueError(f"GDP column '{gdp_col}' not found")

        # Compute HP-filtered trend (simplified using rolling mean as proxy)
        # For rigorous analysis, use statsmodels HP filter
        window = 40  # 10-year window for quarterly data

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
        For the full demand system approach, see estimate_full_system().

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
        # Bank credit elasticity with respect to output
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

    def estimate_full_system(
        self,
        instruments: Optional[list[str]] = None,
    ) -> dict:
        """
        Estimate the full demand system with supply and demand.

        This implements a simplified version of the Boyarchenko & Elias approach.
        The full model would use:
        1. Demand equation: Q_d = α + β*Y + γ*r + ε_d
        2. Supply equations:
           - Q_s_bank = α_b + β_b*Y + γ_b*r + ε_b
           - Q_s_nonbank = α_n + β_n*Y + γ_n*r + ε_n
        3. Equilibrium: Q = Q_d = Q_s_bank + Q_s_nonbank

        Returns:
            Dictionary with estimation results
        """
        # For full implementation, would need:
        # - Instrument variables (Gabaix-Koijen style)
        # - 2SLS or GMM estimation
        # - Proper identification strategy

        # Simplified: just compute correlations and elasticities
        elasticities = self.estimate_supply_elasticities()

        return {
            "supply_elasticities": elasticities,
            "bank_more_procyclical": elasticities.bank_elasticity > elasticities.nonbank_elasticity,
            "methodology_note": "Simplified elasticity estimation. Full demand system requires IV/GMM.",
        }

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
        shock_var: str = "gdp",
        horizons: int = 20,
    ) -> dict[str, np.ndarray]:
        """
        Compute simplified impulse responses of credit to output shocks.

        Shows differential response of bank vs nonbank credit to economic shocks.

        Args:
            shock_var: Variable to shock
            horizons: Number of periods for IRF

        Returns:
            Dictionary with IRF arrays for bank and nonbank credit
        """
        if self._elasticities is None:
            self.estimate_supply_elasticities()

        # Simple AR(1) persistence assumption
        persistence = 0.9

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
    window: int = 20,  # 5 years of quarterly data
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


if __name__ == "__main__":
    # Quick test with sample data
    import numpy as np

    # Create sample data
    n = 100
    dates = pl.date_range(
        pl.date(1990, 1, 1),
        pl.date(2015, 1, 1),
        eager=True,
    )[:n]

    np.random.seed(42)
    gdp = 10000 * np.cumprod(1 + np.random.normal(0.005, 0.01, n))
    bank_credit = 5000 * np.cumprod(1 + np.random.normal(0.006, 0.02, n))
    nonbank_credit = 3000 * np.cumprod(1 + np.random.normal(0.007, 0.015, n))

    df = pl.DataFrame({
        "date": dates,
        "gdp": gdp,
        "bank_credit": bank_credit,
        "nonbank_credit": nonbank_credit,
    })

    # Test decomposition
    decomp = CreditDecomposition(df)
    shares = decomp.compute_shares()
    print("Credit Shares:")
    print(shares.tail())

    # Test demand system
    model = DemandSystemModel(df)
    results = model.estimate_full_system()
    print("\nDemand System Results:")
    print(f"Bank elasticity: {results['supply_elasticities'].bank_elasticity:.3f}")
    print(f"Nonbank elasticity: {results['supply_elasticities'].nonbank_elasticity:.3f}")
    print(f"Bank more procyclical: {results['bank_more_procyclical']}")
