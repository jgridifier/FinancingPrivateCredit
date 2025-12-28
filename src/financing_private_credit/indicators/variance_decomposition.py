"""
Cross-Bank Variance Decomposition Indicator

Decomposes each bank's quarterly loan growth into contributions from:
1. Macroeconomic conditions (M)
2. Bank size growth (S)
3. Portfolio allocation decisions (A)
4. Idiosyncratic bank-specific factors (ε)

This extends the paper's variance decomposition (Tables 5-6) from country-level
to individual banks, revealing which banks are "macro-driven" versus
"strategically-driven" in their lending behavior.

Key Equation:
    Δ(Loans_{bank,t}) = M_{bank,t} + S_{bank,t} + A_{bank,t} + ε_{bank,t}

Data Sources:
- SEC EDGAR: Bank-level loan portfolio data
- FRED: Macroeconomic variables (GDP, inflation, rates, spreads)
- FRED H.8: System-wide loan aggregates

Paper Reference: NY Fed Staff Report 1111, Tables 5-6, Section 6
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import polars as pl
from scipy import stats

from .base import (
    BaseDecomposition,
    IndicatorMetadata,
    IndicatorResult,
    register_indicator,
)


@dataclass
class DecompositionResult:
    """Results from variance decomposition for a single bank."""

    bank: str
    total_variance: float
    macro_contribution: float
    size_contribution: float
    allocation_contribution: float
    idiosyncratic_contribution: float
    covariance_contribution: float
    macro_pct: float
    size_pct: float
    allocation_pct: float
    idiosyncratic_pct: float
    n_observations: int


@dataclass
class BankArchetype:
    """Classification of bank based on variance decomposition."""

    name: str
    description: str
    risk_profile: str
    investment_implication: str


# Bank archetypes based on dominant variance component
ARCHETYPES = {
    "macro_follower": BankArchetype(
        name="Macro Follower",
        description="Lending closely tracks macroeconomic conditions",
        risk_profile="High beta to economic cycle, procyclical",
        investment_implication="Higher beta, amplifies booms/busts",
    ),
    "strategic_allocator": BankArchetype(
        name="Strategic Allocator",
        description="Actively rebalances portfolio allocation",
        risk_profile="Can shift into/out of risk at inopportune times",
        investment_implication="Harder to predict, monitor management commentary",
    ),
    "steady_grower": BankArchetype(
        name="Steady Grower",
        description="Scales all activities proportionally with size",
        risk_profile="Lower volatility, more predictable",
        investment_implication="Defensive, lower beta",
    ),
    "idiosyncratic_specialist": BankArchetype(
        name="Idiosyncratic Specialist",
        description="Unique strategy driven by bank-specific factors",
        risk_profile="Depends on bank-specific factors, less diversified",
        investment_implication="Requires deep fundamental analysis",
    ),
}


@register_indicator("variance_decomposition")
class VarianceDecompositionIndicator(BaseDecomposition):
    """
    Cross-Bank Variance Decomposition Indicator.

    Decomposes loan growth volatility into macro, size, allocation,
    and idiosyncratic components following the methodology in
    NY Fed Staff Report 1111.
    """

    # Macroeconomic variables for the macro effect estimation
    MACRO_VARIABLES = {
        "GDPC1": "Real GDP",
        "GDPPOT": "Potential GDP",
        "CPIAUCSL": "CPI (All Urban Consumers)",
        "GS10": "10-Year Treasury Yield",
        "FEDFUNDS": "Federal Funds Rate",
        "BAA10Y": "Baa-10Y Spread (Credit Spread)",
    }

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self._macro_models: dict[str, dict] = {}
        self._decompositions: dict[str, pl.DataFrame] = {}

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="Cross-Bank Variance Decomposition",
            short_name="VarDecomp",
            description=(
                "Decomposes bank loan growth into macro, size, allocation, "
                "and idiosyncratic components. Identifies whether banks are "
                "macro-driven or strategically-driven."
            ),
            version="1.0.0",
            paper_reference="NY Fed Staff Report 1111, Tables 5-6",
            data_sources=["SEC EDGAR (10-K/10-Q)", "FRED (Macro)", "FRED H.8"],
            update_frequency="quarterly",
            lookback_periods=40,  # 10 years of quarterly data
        )

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch bank-level and macro data."""
        from ..bank_data import BankDataCollector
        from ..cache import CachedFREDFetcher

        # Fetch bank-level data from SEC EDGAR
        collector = BankDataCollector(start_date=start_date)
        bank_panel = collector.fetch_all_banks()
        bank_panel = collector.compute_derived_metrics(bank_panel)

        # Fetch macro data from FRED
        fetcher = CachedFREDFetcher(max_age_hours=24)  # Daily macro data cached longer

        macro_series = list(self.MACRO_VARIABLES.keys())
        macro_data = fetcher.fetch_multiple_series(macro_series, start_date=start_date)

        # Aggregate macro to quarterly
        if macro_data.height > 0 and "date" in macro_data.columns:
            macro_data = self._aggregate_to_quarterly(macro_data)

        # Fetch H.8 system data for comparison
        h8_series = ["TOTLL", "BUSLOANS", "CONSUMER", "REALLN"]
        system_data = fetcher.fetch_multiple_series(h8_series, start_date=start_date)

        return {
            "bank_panel": bank_panel,
            "macro_data": macro_data,
            "system_h8": system_data,
        }

    def _aggregate_to_quarterly(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate data to quarterly frequency using end-of-quarter values."""
        if "date" not in df.columns:
            return df

        df = df.with_columns(
            pl.col("date").dt.truncate("1q").alias("quarter")
        )

        value_cols = [c for c in df.columns if c not in ["date", "quarter"]]

        quarterly = (
            df
            .group_by("quarter")
            .agg([pl.col(c).drop_nulls().last().alias(c) for c in value_cols])
            .sort("quarter")
            .rename({"quarter": "date"})
        )

        return quarterly

    def _compute_output_gap(self, macro_data: pl.DataFrame) -> pl.DataFrame:
        """Compute output gap from GDP and potential GDP."""
        if "GDPC1" not in macro_data.columns or "GDPPOT" not in macro_data.columns:
            return macro_data

        return macro_data.with_columns(
            ((pl.col("GDPC1") - pl.col("GDPPOT")) / pl.col("GDPPOT") * 100)
            .alias("output_gap")
        )

    def _compute_inflation(self, macro_data: pl.DataFrame) -> pl.DataFrame:
        """Compute YoY inflation from CPI."""
        if "CPIAUCSL" not in macro_data.columns:
            return macro_data

        return macro_data.with_columns(
            ((pl.col("CPIAUCSL") / pl.col("CPIAUCSL").shift(4) - 1) * 100)
            .alias("inflation_yoy")
        )

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """
        Calculate variance decomposition for all banks.

        Returns panel with decomposition components and summary statistics.
        """
        bank_panel = data.get("bank_panel", pl.DataFrame())
        macro_data = data.get("macro_data", pl.DataFrame())

        if bank_panel.height == 0:
            return IndicatorResult(
                indicator_name="variance_decomposition",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No bank data available"},
            )

        # Prepare macro data
        if macro_data.height > 0:
            macro_data = self._compute_output_gap(macro_data)
            macro_data = self._compute_inflation(macro_data)

        # Decompose each bank
        decomposition_results = []
        variance_shares = {}

        for bank in bank_panel["ticker"].unique().to_list():
            try:
                decomp = self.decompose(
                    {"bank_panel": bank_panel, "macro_data": macro_data},
                    entity=bank,
                )

                if decomp.height > 0:
                    self._decompositions[bank] = decomp
                    shares = self.compute_variance_shares(decomp)
                    variance_shares[bank] = shares
                    decomposition_results.append(decomp)

            except Exception as e:
                print(f"Warning: Could not decompose {bank}: {e}")
                continue

        # Combine all decompositions
        if decomposition_results:
            combined = pl.concat(decomposition_results)
        else:
            combined = pl.DataFrame()

        return IndicatorResult(
            indicator_name="variance_decomposition",
            calculation_date=datetime.now(),
            data=combined,
            metadata={
                "variance_shares": variance_shares,
                "n_banks": len(variance_shares),
                "banks_analyzed": list(variance_shares.keys()),
            },
        )

    def decompose(
        self,
        data: dict[str, pl.DataFrame],
        entity: str,
    ) -> pl.DataFrame:
        """
        Decompose loan growth for a single bank into components.

        Components:
        - M (Macro): Loan growth explained by macro conditions
        - S (Size): Loan growth from proportional balance sheet scaling
        - A (Allocation): Loan growth from portfolio rebalancing
        - ε (Idiosyncratic): Residual unexplained by above

        Args:
            data: Dictionary with bank_panel and macro_data
            entity: Bank ticker

        Returns:
            DataFrame with decomposition components by date
        """
        bank_panel = data.get("bank_panel", pl.DataFrame())
        macro_data = data.get("macro_data", pl.DataFrame())

        # Filter to bank
        bank_df = bank_panel.filter(pl.col("ticker") == entity).sort("date")

        if bank_df.height < 8:
            return pl.DataFrame()

        # Compute loan growth
        bank_df = bank_df.with_columns(
            (pl.col("total_loans") - pl.col("total_loans").shift(1)).alias("delta_loans"),
            (pl.col("total_loans").shift(1)).alias("loans_lag"),
        )

        # Merge with macro data
        if macro_data.height > 0:
            bank_df = bank_df.join(macro_data, on="date", how="left")

        # Component 1: Macro Effect (M)
        bank_df = self._compute_macro_effect(bank_df, entity)

        # Component 2: Size Effect (S)
        bank_df = self._compute_size_effect(bank_df)

        # Component 3: Allocation Effect (A)
        bank_df = self._compute_allocation_effect(bank_df)

        # Component 4: Idiosyncratic (ε) = Δ Loans - M - S - A
        bank_df = bank_df.with_columns(
            (
                pl.col("delta_loans")
                - pl.col("macro_effect").fill_null(0)
                - pl.col("size_effect").fill_null(0)
                - pl.col("allocation_effect").fill_null(0)
            ).alias("idiosyncratic_effect")
        )

        # Add ticker column
        bank_df = bank_df.with_columns(pl.lit(entity).alias("ticker"))

        # Select decomposition columns
        decomp_cols = [
            "date", "ticker", "delta_loans",
            "macro_effect", "size_effect", "allocation_effect", "idiosyncratic_effect",
        ]

        available_cols = [c for c in decomp_cols if c in bank_df.columns]
        return bank_df.select(available_cols).drop_nulls(subset=["delta_loans"])

    def _compute_macro_effect(
        self,
        bank_df: pl.DataFrame,
        bank: str,
    ) -> pl.DataFrame:
        """
        Compute macro effect using ARDL estimation.

        M_{bank,t} = β_macro,bank' * X_macro,t-1
        """
        # Macro variables to use (lagged)
        macro_vars = []
        for var in ["output_gap", "inflation_yoy", "GS10", "BAA10Y"]:
            if var in bank_df.columns:
                lag_col = f"{var}_lag1"
                bank_df = bank_df.with_columns(pl.col(var).shift(1).alias(lag_col))
                macro_vars.append(lag_col)

        if not macro_vars or "delta_loans" not in bank_df.columns:
            bank_df = bank_df.with_columns(pl.lit(0.0).alias("macro_effect"))
            return bank_df

        # Simple OLS estimation for macro effect
        df_est = bank_df.drop_nulls(subset=["delta_loans"] + macro_vars)

        if df_est.height < 10:
            bank_df = bank_df.with_columns(pl.lit(0.0).alias("macro_effect"))
            return bank_df

        # Extract arrays
        y = df_est.select("delta_loans").to_numpy().flatten()
        X = np.column_stack([
            np.ones(len(y)),
            *[df_est.select(v).to_numpy().flatten() for v in macro_vars]
        ])

        try:
            # OLS estimation
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

            # Store coefficients
            self._macro_models[bank] = {
                "intercept": beta[0],
                "coefficients": dict(zip(macro_vars, beta[1:])),
            }

            # Compute fitted macro effect for all observations
            # Prepare full X matrix
            X_full = np.column_stack([
                np.ones(bank_df.height),
                *[bank_df.select(v).fill_null(0).to_numpy().flatten() for v in macro_vars]
            ])

            macro_effect = X_full @ beta

            bank_df = bank_df.with_columns(
                pl.Series("macro_effect", macro_effect)
            )

        except Exception:
            bank_df = bank_df.with_columns(pl.lit(0.0).alias("macro_effect"))

        return bank_df

    def _compute_size_effect(self, bank_df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute size effect from balance sheet scaling.

        S_{bank,t} = w̄_{bank,t} * ΔAssets_{bank,t}

        Where w̄ = Loans / Assets (loan share of balance sheet)
        """
        # If we have total_assets, use it; otherwise estimate from loans
        if "total_assets" in bank_df.columns:
            bank_df = bank_df.with_columns([
                (pl.col("total_loans").shift(1) / pl.col("total_assets").shift(1))
                .alias("loan_share_lag"),
                (pl.col("total_assets") - pl.col("total_assets").shift(1))
                .alias("delta_assets"),
            ])

            bank_df = bank_df.with_columns(
                (pl.col("loan_share_lag") * pl.col("delta_assets"))
                .alias("size_effect")
            )
        else:
            # Approximate: assume loans grow at same rate as "system" or use loan growth
            # For now, use a simplified version
            bank_df = bank_df.with_columns(
                (pl.col("loans_lag") * pl.col("loan_growth_yoy").fill_null(0) / 100 * 0.5)
                .alias("size_effect")
            )

        return bank_df

    def _compute_allocation_effect(self, bank_df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute allocation effect from portfolio rebalancing.

        A_{bank,t} = Δw_{loans,t} * Assets_{bank,t-1}

        Where w_{loans} = Loans / Assets
        """
        if "total_assets" in bank_df.columns:
            bank_df = bank_df.with_columns([
                (pl.col("total_loans") / pl.col("total_assets")).alias("loan_share"),
            ])

            bank_df = bank_df.with_columns([
                (pl.col("loan_share") - pl.col("loan_share").shift(1))
                .alias("delta_loan_share"),
            ])

            bank_df = bank_df.with_columns(
                (pl.col("delta_loan_share") * pl.col("total_assets").shift(1))
                .alias("allocation_effect")
            )
        else:
            # Simplified: allocation effect is residual between loan growth and size effect
            bank_df = bank_df.with_columns(
                pl.lit(0.0).alias("allocation_effect")
            )

        return bank_df

    def compute_variance_shares(
        self,
        decomposition: pl.DataFrame,
    ) -> dict[str, float]:
        """
        Compute variance contribution shares for each component.

        Following Paper Tables 5-6:
        Var(ΔLoans) = Var(M) + Var(S) + Var(A) + Var(ε) + 2*Cov(...)
        """
        components = ["macro_effect", "size_effect", "allocation_effect", "idiosyncratic_effect"]
        available = [c for c in components if c in decomposition.columns]

        if not available or decomposition.height < 5:
            return {}

        # Drop nulls
        df = decomposition.drop_nulls(subset=available)

        if df.height < 5:
            return {}

        # Compute variances
        variances = {}
        for comp in available:
            var = df.select(pl.col(comp).var()).item()
            variances[comp] = var if var is not None else 0

        total_variance = df.select(pl.col("delta_loans").var()).item() or 1e-10

        # Compute covariance contribution
        cov_total = 0
        for i, comp1 in enumerate(available):
            for comp2 in available[i + 1:]:
                vals1 = df.select(comp1).to_numpy().flatten()
                vals2 = df.select(comp2).to_numpy().flatten()
                cov = np.cov(vals1, vals2)[0, 1]
                cov_total += 2 * cov

        # Compute percentage contributions
        result = {
            "total_variance": total_variance,
            "n_observations": df.height,
        }

        for comp in available:
            short_name = comp.replace("_effect", "")
            result[f"{short_name}_variance"] = variances[comp]
            result[f"{short_name}_pct"] = (variances[comp] / total_variance * 100) if total_variance > 0 else 0

        result["covariance_pct"] = (cov_total / total_variance * 100) if total_variance > 0 else 0

        return result

    def classify_bank(self, variance_shares: dict[str, float]) -> BankArchetype:
        """
        Classify bank archetype based on variance decomposition.

        Returns the archetype based on dominant variance component.
        """
        macro_pct = variance_shares.get("macro_pct", 0)
        size_pct = variance_shares.get("size_pct", 0)
        allocation_pct = variance_shares.get("allocation_pct", 0)
        idio_pct = variance_shares.get("idiosyncratic_pct", 0)

        # Classification rules
        if macro_pct > 50:
            return ARCHETYPES["macro_follower"]
        elif allocation_pct > 30 and allocation_pct > size_pct:
            return ARCHETYPES["strategic_allocator"]
        elif size_pct > 40:
            return ARCHETYPES["steady_grower"]
        elif idio_pct > 40:
            return ARCHETYPES["idiosyncratic_specialist"]
        else:
            # Default to macro follower if no clear dominant
            return ARCHETYPES["macro_follower"]

    def nowcast(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """
        Nowcast current quarter decomposition using high-frequency data.

        Uses weekly H.8 data to estimate current quarter loan growth,
        and monthly/weekly macro data for the macro component.
        """
        system_h8 = data.get("system_h8", pl.DataFrame())
        macro_data = data.get("macro_data", pl.DataFrame())

        if system_h8.height == 0:
            return IndicatorResult(
                indicator_name="variance_decomposition_nowcast",
                calculation_date=datetime.now(),
                data=pl.DataFrame(),
                metadata={"error": "No H.8 data available for nowcast"},
            )

        # Compute current quarter credit growth from weekly data
        current_quarter_start = datetime.now().replace(
            month=((datetime.now().month - 1) // 3) * 3 + 1,
            day=1,
        )

        # Filter to current quarter
        if "date" in system_h8.columns:
            current_data = system_h8.filter(
                pl.col("date") >= current_quarter_start
            )
        else:
            current_data = system_h8

        # Compute aggregate growth
        if current_data.height > 0 and "TOTLL" in current_data.columns:
            first_val = current_data.head(1).select("TOTLL").item()
            last_val = current_data.tail(1).select("TOTLL").item()

            if first_val and last_val:
                qtd_growth = (last_val / first_val - 1) * 100

                return IndicatorResult(
                    indicator_name="variance_decomposition_nowcast",
                    calculation_date=datetime.now(),
                    data=current_data,
                    metadata={
                        "qtd_credit_growth": qtd_growth,
                        "current_quarter": current_quarter_start.strftime("%Y-Q%q").replace(
                            "%q", str((current_quarter_start.month - 1) // 3 + 1)
                        ),
                        "methodology": "Quarter-to-date H.8 credit growth",
                    },
                )

        return IndicatorResult(
            indicator_name="variance_decomposition_nowcast",
            calculation_date=datetime.now(),
            data=pl.DataFrame(),
            metadata={"error": "Insufficient data for nowcast"},
        )

    def get_dashboard_components(self) -> dict[str, Any]:
        """Return dashboard configuration for this indicator."""
        return {
            "tabs": [
                {"name": "Variance Decomposition", "icon": "bar_chart"},
                {"name": "Bank Archetypes", "icon": "category"},
                {"name": "Time Evolution", "icon": "timeline"},
            ],
            "primary_metric": "macro_pct",
            "component_colors": {
                "macro": "#3182ce",    # Blue
                "size": "#38a169",     # Green
                "allocation": "#dd6b20",  # Orange
                "idiosyncratic": "#718096",  # Gray
                "covariance": "#9f7aea",  # Purple
            },
        }

    def create_summary_table(
        self,
        variance_shares: dict[str, dict[str, float]],
    ) -> pl.DataFrame:
        """
        Create summary table of variance contributions across banks.

        Mirrors Paper's Tables 5-6 format.
        """
        rows = []
        for bank, shares in variance_shares.items():
            archetype = self.classify_bank(shares)
            rows.append({
                "Bank": bank,
                "Total Var": shares.get("total_variance", 0),
                "Macro %": shares.get("macro_pct", 0),
                "Size %": shares.get("size_pct", 0),
                "Allocation %": shares.get("allocation_pct", 0),
                "Idiosyncratic %": shares.get("idiosyncratic_pct", 0),
                "Covariance %": shares.get("covariance_pct", 0),
                "Archetype": archetype.name,
                "N": shares.get("n_observations", 0),
            })

        return pl.DataFrame(rows).sort("Macro %", descending=True)
