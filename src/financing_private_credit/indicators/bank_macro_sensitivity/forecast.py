"""
Monte Carlo Forecasting for Bank Macro Sensitivity

Implements joint distribution simulation for:
1. Industry-level NIM forecasts with uncertainty quantification
2. Bank-by-bank attribution showing contribution to industry response
3. Regime scenario analysis (rising rates, recession, etc.)

Key methodology:
- Estimate historical covariance structure of macro variables
- Generate correlated macro scenarios via Cholesky decomposition
- Predict each bank's NIM response using fitted APLR models
- Aggregate preserving bank-level attribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np
import polars as pl
from scipy import stats

from .indicator import MacroSensitivitySpec, get_sensitivity_model


@dataclass
class MacroScenario:
    """A single macro scenario for simulation."""

    rate_spread: float
    output_gap: float
    inflation_yoy: float
    term_spread: float = 0.0
    credit_spread: float = 1.5
    vix: float = 20.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for model prediction."""
        return {
            "rate_spread": self.rate_spread,
            "output_gap": self.output_gap,
            "inflation_yoy": self.inflation_yoy,
            "term_spread": self.term_spread,
            "credit_spread": self.credit_spread,
            "vix": self.vix,
        }


@dataclass
class ForecastResult:
    """Results from Monte Carlo forecast."""

    forecast_date: datetime
    horizon: int

    # Industry-level forecasts
    industry_mean: float
    industry_std: float
    industry_percentiles: dict[int, float]  # {5: val, 25: val, 50: val, 75: val, 95: val}

    # Bank-level forecasts
    bank_forecasts: pl.DataFrame  # [ticker, mean, std, p5, p50, p95]

    # Attribution
    bank_contributions: pl.DataFrame  # [ticker, contribution_to_industry_mean]

    # Simulation details
    n_simulations: int
    simulations: Optional[np.ndarray] = None  # (n_sims, n_banks) matrix

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioForecast:
    """Forecast under a specific scenario."""

    scenario_name: str
    scenario: MacroScenario
    bank_nims: pl.DataFrame  # [ticker, predicted_nim]
    industry_nim: float
    vs_baseline: float  # Change from baseline


class MonteCarloSimulator:
    """
    Monte Carlo simulator for joint macro-bank response distribution.

    Methodology:
    1. Estimate macro variable means, stds, and correlations from history
    2. Generate N correlated draws using Cholesky decomposition
    3. For each draw, predict all banks' NIM using fitted models
    4. Compute industry aggregate and percentiles

    This preserves:
    - Correlation structure of macro variables
    - Heterogeneous bank responses
    - Joint distribution of bank outcomes
    """

    def __init__(
        self,
        model,  # APLRSensitivityModel or FallbackLinearModel
        macro_data: pl.DataFrame,
        n_simulations: int = 1000,
        seed: int = 42,
    ):
        """
        Initialize simulator.

        Args:
            model: Fitted sensitivity model
            macro_data: Historical macro data for estimating distributions
            n_simulations: Number of Monte Carlo draws
            seed: Random seed for reproducibility
        """
        self.model = model
        self.macro_data = macro_data
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

        # Variables to simulate
        self.sim_vars = [
            "rate_spread", "output_gap", "inflation_yoy",
            "term_spread", "credit_spread", "vix"
        ]

        # Estimate distribution parameters
        self._estimate_distributions()

    def _estimate_distributions(self) -> None:
        """Estimate means, stds, and correlation matrix from historical data."""
        # Filter to available variables
        available = [v for v in self.sim_vars if v in self.macro_data.columns]

        if len(available) < 2:
            raise ValueError("Insufficient macro variables for simulation")

        self.sim_vars = available

        # Get numeric matrix
        df_clean = self.macro_data.select(available).drop_nulls()
        data_matrix = df_clean.to_numpy()

        # Estimate parameters
        self.means = np.mean(data_matrix, axis=0)
        self.stds = np.std(data_matrix, axis=0)

        # Correlation matrix
        self.corr_matrix = np.corrcoef(data_matrix, rowvar=False)

        # Cholesky decomposition for correlated sampling
        # Add small diagonal for numerical stability
        self.corr_matrix = self.corr_matrix + np.eye(len(available)) * 1e-6
        try:
            self.cholesky = np.linalg.cholesky(self.corr_matrix)
        except np.linalg.LinAlgError:
            # Fall back to eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(self.corr_matrix)
            eigvals = np.maximum(eigvals, 1e-6)
            self.cholesky = eigvecs @ np.diag(np.sqrt(eigvals))

    def generate_scenarios(
        self,
        base_scenario: Optional[MacroScenario] = None,
        shock_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Generate correlated macro scenarios.

        Args:
            base_scenario: Central scenario (uses historical mean if None)
            shock_scale: Scale factor for shocks (1.0 = historical volatility)

        Returns:
            Array of shape (n_simulations, n_variables)
        """
        # Generate independent standard normal draws
        z = self.rng.standard_normal((self.n_simulations, len(self.sim_vars)))

        # Transform to correlated draws
        correlated = z @ self.cholesky.T

        # Scale to actual distributions
        if base_scenario is not None:
            base_values = np.array([
                getattr(base_scenario, v, self.means[i])
                for i, v in enumerate(self.sim_vars)
            ])
        else:
            base_values = self.means

        scenarios = base_values + shock_scale * correlated * self.stds

        return scenarios

    def simulate(
        self,
        horizon: int = 4,
        base_scenario: Optional[MacroScenario] = None,
        shock_scale: float = 1.0,
    ) -> ForecastResult:
        """
        Run Monte Carlo simulation.

        Args:
            horizon: Forecast horizon in quarters
            base_scenario: Central macro scenario
            shock_scale: Volatility scaling factor

        Returns:
            ForecastResult with industry and bank-level forecasts
        """
        # Generate scenarios
        scenarios = self.generate_scenarios(base_scenario, shock_scale)

        # Get list of banks with fitted models
        banks = list(self.model.models.keys()) if hasattr(self.model, 'models') else list(self.model.coefficients.keys())

        if not banks:
            raise ValueError("No fitted models available for simulation")

        # Predict NIM for each bank under each scenario
        # Shape: (n_simulations, n_banks)
        predictions = np.zeros((self.n_simulations, len(banks)))

        for i, scenario_values in enumerate(scenarios):
            # Convert to scenario dict
            scenario_dict = {
                self.sim_vars[j]: scenario_values[j]
                for j in range(len(self.sim_vars))
            }

            for j, bank in enumerate(banks):
                try:
                    predictions[i, j] = self.model.predict(bank, scenario_dict)
                except Exception:
                    predictions[i, j] = np.nan

        # Handle any NaN predictions (use column mean)
        for j in range(len(banks)):
            mask = np.isnan(predictions[:, j])
            if mask.any():
                predictions[mask, j] = np.nanmean(predictions[:, j])

        # Compute industry aggregate (equal-weighted for now)
        industry_sims = np.mean(predictions, axis=1)

        # Industry statistics
        industry_mean = float(np.mean(industry_sims))
        industry_std = float(np.std(industry_sims))

        percentile_values = {
            p: float(np.percentile(industry_sims, p))
            for p in [5, 25, 50, 75, 95]
        }

        # Bank-level statistics
        bank_stats = []
        for j, bank in enumerate(banks):
            bank_sims = predictions[:, j]
            bank_stats.append({
                "ticker": bank,
                "mean": float(np.mean(bank_sims)),
                "std": float(np.std(bank_sims)),
                "p5": float(np.percentile(bank_sims, 5)),
                "p50": float(np.percentile(bank_sims, 50)),
                "p95": float(np.percentile(bank_sims, 95)),
            })

        bank_forecasts = pl.DataFrame(bank_stats)

        # Bank contributions to industry mean
        contributions = []
        for j, bank in enumerate(banks):
            # Contribution = bank's share of total
            contribution = np.mean(predictions[:, j]) / (industry_mean * len(banks)) * 100
            contributions.append({
                "ticker": bank,
                "contribution_pct": contribution,
                "mean_nim": float(np.mean(predictions[:, j])),
            })

        bank_contributions = pl.DataFrame(contributions).sort("contribution_pct", descending=True)

        return ForecastResult(
            forecast_date=datetime.now(),
            horizon=horizon,
            industry_mean=industry_mean,
            industry_std=industry_std,
            industry_percentiles=percentile_values,
            bank_forecasts=bank_forecasts,
            bank_contributions=bank_contributions,
            n_simulations=self.n_simulations,
            simulations=predictions,
            metadata={
                "n_banks": len(banks),
                "shock_scale": shock_scale,
                "variables_simulated": self.sim_vars,
            },
        )


class MacroSensitivityForecaster:
    """
    High-level forecaster for bank macro sensitivity.

    Provides:
    1. Monte Carlo simulation for uncertainty quantification
    2. Scenario analysis (rising rates, recession, etc.)
    3. Bank-level forecasts with attribution
    """

    # Predefined scenarios
    SCENARIOS = {
        "baseline": MacroScenario(
            rate_spread=1.5,
            output_gap=0.0,
            inflation_yoy=2.5,
            term_spread=1.0,
            credit_spread=1.5,
            vix=18.0,
        ),
        "rising_rates": MacroScenario(
            rate_spread=3.0,
            output_gap=0.5,
            inflation_yoy=3.5,
            term_spread=1.5,
            credit_spread=1.2,
            vix=15.0,
        ),
        "falling_rates": MacroScenario(
            rate_spread=-0.5,
            output_gap=-0.5,
            inflation_yoy=1.5,
            term_spread=0.5,
            credit_spread=2.0,
            vix=22.0,
        ),
        "recession": MacroScenario(
            rate_spread=0.0,
            output_gap=-3.0,
            inflation_yoy=1.0,
            term_spread=2.0,
            credit_spread=4.0,
            vix=35.0,
        ),
        "expansion": MacroScenario(
            rate_spread=2.0,
            output_gap=2.0,
            inflation_yoy=3.0,
            term_spread=1.2,
            credit_spread=1.0,
            vix=12.0,
        ),
        "stagflation": MacroScenario(
            rate_spread=1.0,
            output_gap=-1.5,
            inflation_yoy=5.0,
            term_spread=0.5,
            credit_spread=3.0,
            vix=28.0,
        ),
    }

    def __init__(
        self,
        model,
        macro_data: pl.DataFrame,
        n_simulations: int = 1000,
    ):
        """
        Initialize forecaster.

        Args:
            model: Fitted sensitivity model
            macro_data: Historical macro data
            n_simulations: Number of Monte Carlo simulations
        """
        self.model = model
        self.macro_data = macro_data
        self.n_simulations = n_simulations
        self.simulator = MonteCarloSimulator(model, macro_data, n_simulations)

    def forecast(
        self,
        horizon: int = 4,
        scenario: Optional[str | MacroScenario] = None,
    ) -> ForecastResult:
        """
        Generate Monte Carlo forecast.

        Args:
            horizon: Forecast horizon in quarters
            scenario: Scenario name or MacroScenario object

        Returns:
            ForecastResult with simulations
        """
        if scenario is None:
            base = None
        elif isinstance(scenario, str):
            if scenario not in self.SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario}")
            base = self.SCENARIOS[scenario]
        else:
            base = scenario

        return self.simulator.simulate(horizon=horizon, base_scenario=base)

    def scenario_analysis(
        self,
        scenarios: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """
        Compare bank responses across predefined scenarios.

        Args:
            scenarios: List of scenario names (uses all if None)

        Returns:
            DataFrame comparing banks across scenarios
        """
        if scenarios is None:
            scenarios = list(self.SCENARIOS.keys())

        banks = list(self.model.models.keys()) if hasattr(self.model, 'models') else list(self.model.coefficients.keys())

        results = []
        baseline = self.SCENARIOS.get("baseline", self.SCENARIOS[scenarios[0]])

        for bank in banks:
            row = {"ticker": bank}

            # Baseline prediction
            try:
                base_pred = self.model.predict(bank, baseline.to_dict())
                row["baseline"] = base_pred

                # Each scenario
                for scenario_name in scenarios:
                    if scenario_name == "baseline":
                        continue

                    scenario = self.SCENARIOS[scenario_name]
                    pred = self.model.predict(bank, scenario.to_dict())
                    row[scenario_name] = pred
                    row[f"{scenario_name}_vs_base"] = pred - base_pred

            except Exception as e:
                row["baseline"] = None
                for scenario_name in scenarios:
                    row[scenario_name] = None

            results.append(row)

        return pl.DataFrame(results)

    def regime_advantage_forecast(self) -> pl.DataFrame:
        """
        Forecast which banks have advantages in each regime.

        Returns:
            DataFrame ranking banks by advantage in each scenario
        """
        scenario_df = self.scenario_analysis()

        # For each scenario, compute relative advantage
        banks = scenario_df["ticker"].to_list()
        baseline_col = scenario_df["baseline"].to_numpy()

        advantages = []
        for bank, base_val in zip(banks, baseline_col):
            if base_val is None:
                continue

            row = {"ticker": bank}

            for scenario_name in ["rising_rates", "falling_rates", "recession", "expansion"]:
                diff_col = f"{scenario_name}_vs_base"
                if diff_col in scenario_df.columns:
                    bank_row = scenario_df.filter(pl.col("ticker") == bank)
                    if bank_row.height > 0:
                        diff = bank_row[diff_col][0]
                        if diff is not None:
                            row[f"{scenario_name}_advantage"] = diff

            advantages.append(row)

        return pl.DataFrame(advantages)

    def joint_distribution_stats(
        self,
        horizon: int = 4,
    ) -> dict[str, Any]:
        """
        Compute statistics of the joint distribution of bank NIMs.

        Returns:
            Dictionary with correlation matrix, tail dependencies, etc.
        """
        result = self.forecast(horizon)

        if result.simulations is None:
            return {}

        sims = result.simulations  # (n_sims, n_banks)

        # Correlation matrix of bank NIMs
        corr_matrix = np.corrcoef(sims, rowvar=False)

        # Tail dependence (lower and upper 10%)
        lower_tail = sims < np.percentile(sims, 10, axis=0)
        upper_tail = sims > np.percentile(sims, 90, axis=0)

        # Systemic risk: probability all banks below their 10th percentile
        all_low = np.all(lower_tail, axis=1)
        systemic_risk = np.mean(all_low)

        banks = list(self.model.models.keys()) if hasattr(self.model, 'models') else list(self.model.coefficients.keys())

        return {
            "correlation_matrix": corr_matrix,
            "bank_order": banks,
            "systemic_risk_lower_tail": systemic_risk,
            "avg_correlation": np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),
            "min_correlation": np.min(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),
            "max_correlation": np.max(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),
        }
