"""
Funding Stability Forecasting with ARDL and Monte Carlo Simulation

Forecasts the Funding Resilience Score and its components using:
1. ARDL (Autoregressive Distributed Lag) models for each component
2. Monte Carlo simulation on joint distribution of macro variables
3. Bank-specific elasticity estimation

Key macro variables:
- Fed funds rate (affects rate beta, FHLB usage)
- Yield curve slope (affects AOCI)
- Credit spreads (stress indicator)
- Deposit growth (industry-wide)

The goal is to project each funding component (e.g., FHLB advances, uninsured deposits)
as a function of macro conditions, then aggregate into the composite score.

This allows scenario analysis: "What happens to Bank X's funding stability if rates rise 200bp?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import polars as pl


@dataclass
class ARDLSpec:
    """
    Specification for ARDL model.

    ARDL(p, q) model:
    y_t = c + Σᵢ φᵢ y_{t-i} + Σⱼ βⱼ x_{t-j} + ε_t

    Where:
    - p: number of autoregressive lags
    - q: number of distributed lags for each exogenous variable
    """

    name: str
    description: str
    target: str  # Target variable (e.g., "fhlb_advance_ratio")

    # Lag structure
    ar_lags: int = 4  # Autoregressive lags (p)
    dist_lags: int = 4  # Distributed lags for each X (q)

    # Exogenous variables
    exog_vars: list[str] = field(default_factory=lambda: [
        "fed_funds_rate",
        "yield_curve_slope",
        "credit_spread",
    ])

    # Estimation settings
    include_constant: bool = True
    include_trend: bool = False

    # Error correction (for bounds testing)
    include_ecm: bool = False
    ecm_lags: int = 1

    @classmethod
    def from_json(cls, path: str | Path) -> "ARDLSpec":
        """Load specification from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        """Save specification to JSON file."""
        spec_dict = {
            "name": self.name,
            "description": self.description,
            "target": self.target,
            "ar_lags": self.ar_lags,
            "dist_lags": self.dist_lags,
            "exog_vars": self.exog_vars,
            "include_constant": self.include_constant,
            "include_trend": self.include_trend,
            "include_ecm": self.include_ecm,
            "ecm_lags": self.ecm_lags,
        }
        with open(path, "w") as f:
            json.dump(spec_dict, f, indent=2)


@dataclass
class ARDLResult:
    """Results from ARDL model estimation."""

    spec: ARDLSpec
    ticker: str
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    t_stats: dict[str, float]
    p_values: dict[str, float]

    # Model fit
    r_squared: float
    adj_r_squared: float
    aic: float
    bic: float
    durbin_watson: float

    # Sample info
    n_obs: int
    start_date: datetime
    end_date: datetime

    # Long-run multipliers (from ARDL)
    long_run_effects: dict[str, float] = field(default_factory=dict)

    # Residuals for simulation
    residuals: Optional[np.ndarray] = None
    residual_std: float = 0.0

    def get_forecast_equation(self) -> str:
        """Return human-readable forecast equation."""
        terms = []
        if "const" in self.coefficients:
            terms.append(f"{self.coefficients['const']:.4f}")

        for var, coef in self.coefficients.items():
            if var != "const":
                terms.append(f"{coef:+.4f}*{var}")

        return f"{self.spec.target} = " + " ".join(terms)


class ARDLModel:
    """
    ARDL model implementation for funding stability forecasting.

    Uses statsmodels ARDL or OLS fallback.
    """

    def __init__(self, spec: ARDLSpec):
        """
        Initialize ARDL model.

        Args:
            spec: Model specification
        """
        self.spec = spec
        self._results: dict[str, ARDLResult] = {}  # By ticker
        self._is_fitted = False

    def fit(
        self,
        data: pl.DataFrame,
        ticker: Optional[str] = None,
    ) -> dict[str, ARDLResult]:
        """
        Fit ARDL model(s).

        Args:
            data: Panel data with date, ticker, target, and exog variables
            ticker: Optional specific ticker (fits all if None)

        Returns:
            Dictionary of results by ticker
        """
        if data.height == 0:
            return {}

        tickers = [ticker] if ticker else data["ticker"].unique().to_list()

        for t in tickers:
            bank_data = data.filter(pl.col("ticker") == t).sort("date")

            if bank_data.height < self.spec.ar_lags + self.spec.dist_lags + 10:
                print(f"  Warning: Insufficient data for {t}, skipping")
                continue

            result = self._fit_single(bank_data, t)
            if result:
                self._results[t] = result

        self._is_fitted = len(self._results) > 0
        return self._results

    def _fit_single(
        self,
        data: pl.DataFrame,
        ticker: str,
    ) -> Optional[ARDLResult]:
        """Fit ARDL model for a single bank."""
        try:
            # Try statsmodels ARDL
            return self._fit_with_statsmodels(data, ticker)
        except ImportError:
            # Fall back to manual OLS
            return self._fit_with_ols(data, ticker)

    def _fit_with_statsmodels(
        self,
        data: pl.DataFrame,
        ticker: str,
    ) -> Optional[ARDLResult]:
        """Fit using statsmodels ARDL."""
        try:
            from statsmodels.tsa.ardl import ARDL
            import pandas as pd
        except ImportError:
            raise ImportError("statsmodels required for ARDL estimation")

        # Convert to pandas for statsmodels
        df_pd = data.to_pandas()
        df_pd = df_pd.set_index("date")

        # Check target variable exists
        if self.spec.target not in df_pd.columns:
            print(f"  Warning: Target {self.spec.target} not in data for {ticker}")
            return None

        # Prepare endog and exog
        endog = df_pd[self.spec.target].dropna()

        # Filter available exog variables
        available_exog = [v for v in self.spec.exog_vars if v in df_pd.columns]
        if not available_exog:
            # Fit AR-only model
            exog = None
            lags = {"order": self.spec.ar_lags}
        else:
            exog = df_pd[available_exog].dropna()
            # Align indices
            common_idx = endog.index.intersection(exog.index)
            endog = endog.loc[common_idx]
            exog = exog.loc[common_idx]
            lags = {v: self.spec.dist_lags for v in available_exog}

        if len(endog) < 20:
            return None

        try:
            model = ARDL(
                endog,
                lags=self.spec.ar_lags,
                exog=exog,
                order=lags if exog is not None else None,
                trend="c" if self.spec.include_constant else "n",
            )
            fit = model.fit()
        except Exception as e:
            print(f"  Warning: ARDL fit failed for {ticker}: {e}")
            return self._fit_with_ols(data, ticker)

        # Extract coefficients
        coefficients = dict(fit.params)
        std_errors = dict(fit.bse)
        t_stats = dict(fit.tvalues)
        p_values = dict(fit.pvalues)

        # Calculate long-run effects
        long_run = {}
        ar_sum = sum(coefficients.get(f"L{i}.{self.spec.target}", 0)
                     for i in range(1, self.spec.ar_lags + 1))
        if abs(1 - ar_sum) > 0.01:  # Avoid division by near-zero
            for var in available_exog:
                var_sum = sum(coefficients.get(f"L{i}.{var}", 0)
                              for i in range(self.spec.dist_lags + 1))
                long_run[var] = var_sum / (1 - ar_sum)

        return ARDLResult(
            spec=self.spec,
            ticker=ticker,
            coefficients=coefficients,
            std_errors=std_errors,
            t_stats=t_stats,
            p_values=p_values,
            r_squared=fit.rsquared,
            adj_r_squared=fit.rsquared_adj,
            aic=fit.aic,
            bic=fit.bic,
            durbin_watson=float(fit.durbin_watson) if hasattr(fit, 'durbin_watson') else 0.0,
            n_obs=len(endog),
            start_date=endog.index[0],
            end_date=endog.index[-1],
            long_run_effects=long_run,
            residuals=fit.resid.values,
            residual_std=float(fit.resid.std()),
        )

    def _fit_with_ols(
        self,
        data: pl.DataFrame,
        ticker: str,
    ) -> Optional[ARDLResult]:
        """Fallback: fit using simple OLS with lagged variables."""
        # Build lagged design matrix
        result_df = data.select(["date", self.spec.target]).clone()

        # Add AR lags
        for lag in range(1, self.spec.ar_lags + 1):
            result_df = result_df.with_columns(
                pl.col(self.spec.target).shift(lag).alias(f"L{lag}.{self.spec.target}")
            )

        # Add exog and their lags
        available_exog = [v for v in self.spec.exog_vars if v in data.columns]
        for var in available_exog:
            result_df = result_df.with_columns(pl.col(var).alias(var) if var in result_df.columns else data[var])
            for lag in range(1, self.spec.dist_lags + 1):
                result_df = result_df.with_columns(
                    data[var].shift(lag).alias(f"L{lag}.{var}")
                )

        # Drop NaN rows
        result_df = result_df.drop_nulls()

        if result_df.height < 10:
            return None

        # Prepare for OLS
        y = result_df[self.spec.target].to_numpy()

        # Build X matrix
        x_cols = [f"L{i}.{self.spec.target}" for i in range(1, self.spec.ar_lags + 1)]
        for var in available_exog:
            x_cols.append(var)
            for lag in range(1, self.spec.dist_lags + 1):
                x_cols.append(f"L{lag}.{var}")

        X = result_df.select(x_cols).to_numpy()

        # Add constant
        if self.spec.include_constant:
            X = np.column_stack([np.ones(len(y)), X])
            x_cols = ["const"] + x_cols

        # OLS estimation
        try:
            XtX = X.T @ X
            XtX_inv = np.linalg.inv(XtX + 1e-8 * np.eye(XtX.shape[0]))  # Regularization
            beta = XtX_inv @ X.T @ y

            # Residuals and statistics
            y_hat = X @ beta
            residuals = y - y_hat
            n, k = X.shape
            ssr = np.sum(residuals ** 2)
            sst = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ssr / sst
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

            # Standard errors
            sigma2 = ssr / (n - k)
            se = np.sqrt(np.diag(XtX_inv) * sigma2)
            t_stats = beta / (se + 1e-10)

            coefficients = dict(zip(x_cols, beta))
            std_errors = dict(zip(x_cols, se))
            t_statistics = dict(zip(x_cols, t_stats))

        except Exception as e:
            print(f"  Warning: OLS failed for {ticker}: {e}")
            return None

        return ARDLResult(
            spec=self.spec,
            ticker=ticker,
            coefficients=coefficients,
            std_errors=std_errors,
            t_stats=t_statistics,
            p_values={k: 0.05 for k in coefficients},  # Placeholder
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            aic=n * np.log(ssr / n) + 2 * k,
            bic=n * np.log(ssr / n) + k * np.log(n),
            durbin_watson=0.0,
            n_obs=n,
            start_date=result_df["date"][0],
            end_date=result_df["date"][-1],
            long_run_effects={},
            residuals=residuals,
            residual_std=float(np.std(residuals)),
        )

    def forecast(
        self,
        ticker: str,
        horizon: int,
        exog_forecast: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        Generate point forecasts.

        Args:
            ticker: Bank ticker
            horizon: Forecast horizon (periods)
            exog_forecast: DataFrame with future exog values

        Returns:
            DataFrame with forecast values
        """
        if ticker not in self._results:
            return pl.DataFrame()

        result = self._results[ticker]

        # Simple forecast using coefficients
        # (In practice, would iterate forward using lags)
        forecasts = []
        for h in range(1, horizon + 1):
            # Placeholder forecast
            fc_value = result.coefficients.get("const", 0)
            forecasts.append({
                "horizon": h,
                "ticker": ticker,
                f"{self.spec.target}_forecast": fc_value,
                "forecast_std": result.residual_std,
            })

        return pl.DataFrame(forecasts)


@dataclass
class MonteCarloScenario:
    """A single scenario from Monte Carlo simulation."""

    scenario_id: int
    macro_values: dict[str, float]
    component_forecasts: dict[str, float]
    funding_resilience_score: float


class MonteCarloSimulator:
    """
    Monte Carlo simulation for funding stability forecasting.

    Simulates joint distribution of macro variables, then propagates
    through ARDL models to get distribution of funding resilience scores.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Initialize simulator.

        Args:
            n_simulations: Number of Monte Carlo draws
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def simulate_macro_scenarios(
        self,
        baseline: dict[str, float],
        covariance: np.ndarray,
        var_names: list[str],
    ) -> pl.DataFrame:
        """
        Simulate macro scenarios from joint distribution.

        Uses Cholesky decomposition for correlated draws.

        Args:
            baseline: Baseline values for each macro variable
            covariance: Covariance matrix of macro variables
            var_names: Names of variables (matching covariance order)

        Returns:
            DataFrame with simulated scenarios
        """
        n_vars = len(var_names)

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(covariance + 1e-8 * np.eye(n_vars))
        except np.linalg.LinAlgError:
            # Fallback to diagonal
            L = np.diag(np.sqrt(np.diag(covariance)))

        # Generate correlated draws
        z = self._rng.standard_normal((self.n_simulations, n_vars))
        x = z @ L.T

        # Add baseline
        baseline_vec = np.array([baseline.get(v, 0) for v in var_names])
        scenarios = x + baseline_vec

        # Create DataFrame
        scenario_data = {
            "scenario_id": list(range(self.n_simulations)),
        }
        for i, var in enumerate(var_names):
            scenario_data[var] = scenarios[:, i]

        return pl.DataFrame(scenario_data)

    def run_simulation(
        self,
        ardl_models: dict[str, ARDLModel],
        macro_scenarios: pl.DataFrame,
        scoring_weights: dict[str, float],
    ) -> pl.DataFrame:
        """
        Run Monte Carlo simulation through ARDL models.

        Args:
            ardl_models: Dictionary of fitted ARDL models by component
            macro_scenarios: Simulated macro scenarios
            scoring_weights: Weights for each component in final score

        Returns:
            DataFrame with simulated funding resilience scores
        """
        results = []

        for row in macro_scenarios.iter_rows(named=True):
            scenario_id = row["scenario_id"]
            macro_values = {k: v for k, v in row.items() if k != "scenario_id"}

            # Forecast each component
            component_forecasts = {}
            for component, model in ardl_models.items():
                # Simplified: use long-run effects
                for ticker, result in model._results.items():
                    fc = result.coefficients.get("const", 0)
                    for var, effect in result.long_run_effects.items():
                        if var in macro_values:
                            fc += effect * macro_values[var]

                    # Add residual noise
                    fc += self._rng.normal(0, result.residual_std)
                    component_forecasts[f"{component}_{ticker}"] = fc

            # Calculate composite score (simplified)
            score = sum(
                scoring_weights.get(comp.split("_")[0], 0.1) * fc
                for comp, fc in component_forecasts.items()
            )

            results.append({
                "scenario_id": scenario_id,
                **macro_values,
                **component_forecasts,
                "funding_resilience_score": score,
            })

        return pl.DataFrame(results)

    def summarize_results(
        self,
        simulation_results: pl.DataFrame,
        percentiles: list[float] = [5, 25, 50, 75, 95],
    ) -> pl.DataFrame:
        """
        Summarize Monte Carlo results with confidence intervals.

        Args:
            simulation_results: Raw simulation results
            percentiles: Percentiles to calculate

        Returns:
            Summary DataFrame with statistics
        """
        # Calculate statistics for funding resilience score
        score_col = "funding_resilience_score"

        if score_col not in simulation_results.columns:
            return pl.DataFrame()

        stats = {
            "mean": simulation_results[score_col].mean(),
            "std": simulation_results[score_col].std(),
            "min": simulation_results[score_col].min(),
            "max": simulation_results[score_col].max(),
        }

        for p in percentiles:
            stats[f"p{p}"] = np.percentile(
                simulation_results[score_col].to_numpy(),
                p
            )

        return pl.DataFrame([stats])


class FundingStabilityForecaster:
    """
    Main forecasting class for Funding Stability Score.

    Orchestrates ARDL estimation and Monte Carlo simulation.
    """

    # Default macro variable covariance (historical estimate)
    DEFAULT_MACRO_COVARIANCE = np.array([
        [1.0, 0.3, 0.5, -0.2],   # Fed funds
        [0.3, 1.0, 0.4, 0.1],    # Yield slope
        [0.5, 0.4, 1.0, 0.3],    # Credit spread
        [-0.2, 0.1, 0.3, 1.0],   # Deposit growth
    ])

    DEFAULT_MACRO_VARS = [
        "fed_funds_rate",
        "yield_curve_slope",
        "credit_spread",
        "deposit_growth",
    ]

    def __init__(
        self,
        funding_data: pl.DataFrame,
        spec: Optional["FundingStabilitySpec"] = None,
    ):
        """
        Initialize forecaster.

        Args:
            funding_data: Historical funding stability data
            spec: Model specification
        """
        from .indicator import FundingStabilitySpec
        self.funding_data = funding_data
        self.spec = spec or FundingStabilitySpec(
            name="default",
            description="Default funding stability spec"
        )

        # ARDL models for each component
        self._component_models: dict[str, ARDLModel] = {}

        # Monte Carlo simulator
        self._simulator = MonteCarloSimulator(n_simulations=1000, seed=42)

    def fit_component_models(
        self,
        macro_data: pl.DataFrame,
        components: Optional[list[str]] = None,
    ) -> dict[str, dict[str, ARDLResult]]:
        """
        Fit ARDL models for each funding component.

        Args:
            macro_data: Macro variable time series
            components: Components to model (default: all)

        Returns:
            Nested dict of results by component and ticker
        """
        if components is None:
            components = [
                "fhlb_advance_ratio",
                "uninsured_deposit_ratio",
                "brokered_deposit_ratio",
                "aoci_impact_ratio",
                "wholesale_funding_ratio",
            ]

        # Merge funding data with macro data
        merged = self.funding_data.join(
            macro_data,
            on="date",
            how="left"
        )

        results = {}

        for component in components:
            if component not in merged.columns:
                print(f"  Warning: Component {component} not in data")
                continue

            # Create ARDL spec for this component
            ardl_spec = ARDLSpec(
                name=f"ardl_{component}",
                description=f"ARDL model for {component}",
                target=component,
                ar_lags=2,  # Conservative for quarterly data
                dist_lags=2,
                exog_vars=[v for v in self.DEFAULT_MACRO_VARS if v in merged.columns],
            )

            model = ARDLModel(ardl_spec)
            component_results = model.fit(merged)

            self._component_models[component] = model
            results[component] = component_results

        return results

    def forecast_score(
        self,
        ticker: str,
        horizon: int,
        macro_scenario: dict[str, float],
    ) -> dict[str, float]:
        """
        Forecast funding resilience score for a specific macro scenario.

        Args:
            ticker: Bank ticker
            horizon: Forecast horizon
            macro_scenario: Macro variable values

        Returns:
            Dictionary with component and aggregate forecasts
        """
        forecasts = {}

        for component, model in self._component_models.items():
            if ticker not in model._results:
                continue

            result = model._results[ticker]

            # Calculate forecast using long-run effects
            fc = result.coefficients.get("const", 0)
            for var, effect in result.long_run_effects.items():
                if var in macro_scenario:
                    fc += effect * macro_scenario[var]

            forecasts[component] = fc

        # Calculate aggregate score
        # (Would use proper weights from spec)
        aggregate = np.mean(list(forecasts.values())) if forecasts else 50.0
        forecasts["funding_resilience_score"] = aggregate

        return forecasts

    def monte_carlo_forecast(
        self,
        baseline_macro: dict[str, float],
        n_simulations: int = 1000,
    ) -> pl.DataFrame:
        """
        Run Monte Carlo forecast for all banks.

        Args:
            baseline_macro: Baseline macro scenario
            n_simulations: Number of simulations

        Returns:
            DataFrame with simulation results
        """
        self._simulator.n_simulations = n_simulations

        # Generate macro scenarios
        available_vars = [v for v in self.DEFAULT_MACRO_VARS if v in baseline_macro]
        n_vars = len(available_vars)

        if n_vars == 0:
            return pl.DataFrame()

        # Use subset of covariance matrix
        var_indices = [self.DEFAULT_MACRO_VARS.index(v) for v in available_vars]
        cov_subset = self.DEFAULT_MACRO_COVARIANCE[np.ix_(var_indices, var_indices)]

        scenarios = self._simulator.simulate_macro_scenarios(
            baseline=baseline_macro,
            covariance=cov_subset,
            var_names=available_vars,
        )

        # Get scoring weights from spec
        scoring_weights = {
            "fhlb_advance_ratio": self.spec.weight_fhlb_advances,
            "uninsured_deposit_ratio": self.spec.weight_uninsured_deposits,
            "brokered_deposit_ratio": self.spec.weight_brokered_deposits,
            "aoci_impact_ratio": self.spec.weight_aoci_impact,
            "wholesale_funding_ratio": self.spec.weight_wholesale_funding,
        }

        # Run simulation
        results = self._simulator.run_simulation(
            ardl_models=self._component_models,
            macro_scenarios=scenarios,
            scoring_weights=scoring_weights,
        )

        return results

    def scenario_analysis(
        self,
        scenarios: dict[str, dict[str, float]],
    ) -> pl.DataFrame:
        """
        Run scenario analysis for named macro scenarios.

        Args:
            scenarios: Dictionary of named scenarios with macro values
                e.g., {"baseline": {...}, "stress": {...}}

        Returns:
            DataFrame with scores under each scenario
        """
        tickers = self.funding_data["ticker"].unique().to_list()
        results = []

        for scenario_name, macro_values in scenarios.items():
            for ticker in tickers:
                fc = self.forecast_score(ticker, horizon=4, macro_scenario=macro_values)
                results.append({
                    "scenario": scenario_name,
                    "ticker": ticker,
                    **fc,
                })

        return pl.DataFrame(results)

    def get_model_summary(self) -> pl.DataFrame:
        """Get summary of fitted component models."""
        summaries = []

        for component, model in self._component_models.items():
            for ticker, result in model._results.items():
                summaries.append({
                    "component": component,
                    "ticker": ticker,
                    "r_squared": result.r_squared,
                    "adj_r_squared": result.adj_r_squared,
                    "n_obs": result.n_obs,
                    "residual_std": result.residual_std,
                })

        return pl.DataFrame(summaries)


# Predefined scenarios for stress testing
PREDEFINED_SCENARIOS = {
    "baseline": {
        "fed_funds_rate": 5.25,
        "yield_curve_slope": 0.0,
        "credit_spread": 1.5,
        "deposit_growth": 0.02,
    },
    "rate_hike_100bp": {
        "fed_funds_rate": 6.25,
        "yield_curve_slope": -0.25,
        "credit_spread": 1.75,
        "deposit_growth": 0.01,
    },
    "rate_cut_100bp": {
        "fed_funds_rate": 4.25,
        "yield_curve_slope": 0.5,
        "credit_spread": 1.25,
        "deposit_growth": 0.03,
    },
    "recession": {
        "fed_funds_rate": 3.00,
        "yield_curve_slope": 1.5,
        "credit_spread": 3.5,
        "deposit_growth": -0.02,
    },
    "svb_stress": {
        "fed_funds_rate": 5.50,
        "yield_curve_slope": -0.75,  # Inverted
        "credit_spread": 2.5,
        "deposit_growth": -0.05,  # Deposit outflows
    },
    "normalization": {
        "fed_funds_rate": 3.50,
        "yield_curve_slope": 1.0,
        "credit_spread": 1.0,
        "deposit_growth": 0.04,
    },
}
