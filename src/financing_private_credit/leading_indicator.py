"""
Credit Boom Leading Indicator Model.

Implements predictive models based on Boyarchenko & Elias (2024) findings:
- Banks that expand credit aggressively today will have higher provisions 3-4 years later
- Bank credit is more procyclical than nonbank credit
- Lender composition predicts crisis probability

Models:
1. Lending Intensity Score (LIS): Bank lending vs. system average
2. ARDL: Autoregressive Distributed Lag panel model
3. SARIMAX: Bank-specific time series forecasting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal

import numpy as np
import polars as pl
from scipy import stats

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


@dataclass
class LISResult:
    """Lending Intensity Score result for a bank."""

    ticker: str
    date: datetime
    lis: float  # Standard deviations from system mean
    bank_growth: float
    system_growth: float
    system_std: float
    percentile: float  # Rank among peers


@dataclass
class ARDLResult:
    """Results from ARDL estimation."""

    coefficients: dict[str, float]
    std_errors: dict[str, float]
    pvalues: dict[str, float]
    r_squared: float
    adj_r_squared: float
    n_obs: int
    aic: float
    bic: float
    lis_effect_3yr: float  # LIS coefficient at 12-quarter lag
    lis_effect_4yr: float  # LIS coefficient at 16-quarter lag


@dataclass
class ForecastResult:
    """Forecast result for a bank."""

    ticker: str
    forecast_date: datetime
    horizon_quarters: int
    point_forecast: float
    ci_lower: float
    ci_upper: float
    risk_level: Literal["low", "medium", "high"]
    model_used: str


class LendingIntensityScore:
    """
    Compute Lending Intensity Score (LIS) for each bank.

    LIS = (Bank Loan Growth - System Loan Growth) / σ(System Loan Growth)

    Interpretation:
    - LIS > 0: Bank is lending more aggressively than peers
    - LIS > 1: Bank is > 1 SD above system average
    - LIS > 2: Warning sign of aggressive lending
    """

    def __init__(
        self,
        bank_data: pl.DataFrame,
        system_data: pl.DataFrame,
        growth_col: str = "loan_growth_yoy",
    ):
        """
        Initialize LIS calculator.

        Args:
            bank_data: Panel of bank-level data with ticker, date, growth
            system_data: System-wide growth data with date and growth
            growth_col: Name of growth column
        """
        self.bank_data = bank_data
        self.system_data = system_data
        self.growth_col = growth_col

    def compute_lis(self) -> pl.DataFrame:
        """
        Compute LIS for all banks at all dates.

        Returns:
            DataFrame with ticker, date, lis, and components
        """
        # Merge bank and system data
        df = self.bank_data.join(
            self.system_data.select(["date", self.growth_col]).rename(
                {self.growth_col: "system_growth"}
            ),
            on="date",
            how="left",
        )

        # Rename bank growth column
        df = df.with_columns(
            pl.col(self.growth_col).alias("bank_growth")
        )

        # Compute rolling system statistics (20-quarter window)
        df = df.with_columns([
            pl.col("system_growth").rolling_mean(window_size=20).alias("system_mean"),
            pl.col("system_growth").rolling_std(window_size=20).alias("system_std"),
        ])

        # Compute LIS
        df = df.with_columns(
            ((pl.col("bank_growth") - pl.col("system_growth")) / pl.col("system_std"))
            .alias("lis")
        )

        # Compute cumulative LIS (sum over 12 quarters)
        df = df.with_columns(
            pl.col("lis").rolling_sum(window_size=12).over("ticker").alias("lis_cumulative_12q")
        )

        # Compute percentile rank among peers at each date
        df = df.with_columns(
            pl.col("bank_growth").rank().over("date").alias("growth_rank")
        )
        df = df.with_columns(
            (pl.col("growth_rank") / pl.col("growth_rank").max().over("date") * 100)
            .alias("growth_percentile")
        )

        return df

    def get_current_signals(self, threshold: float = 1.0) -> pl.DataFrame:
        """
        Get current LIS signals for all banks.

        Args:
            threshold: LIS threshold for flagging (default: 1 SD)

        Returns:
            DataFrame with current signals sorted by risk
        """
        df = self.compute_lis()

        # Get most recent date for each bank
        latest = df.group_by("ticker").agg(
            pl.col("date").max().alias("date")
        )

        # Filter to latest observations
        current = df.join(latest, on=["ticker", "date"], how="inner")

        # Add risk flags
        current = current.with_columns([
            (pl.col("lis") > threshold).alias("elevated_lis"),
            (pl.col("lis_cumulative_12q") > threshold * 4).alias("sustained_elevation"),
        ])

        return current.sort("lis", descending=True)


class ARDLModel:
    """
    Autoregressive Distributed Lag model for panel data.

    Specification:
    Provision_Rate_{i,t} = α_i + Σ β_j * Provision_{i,t-j} + Σ γ_h * LIS_{i,t-h} + Controls + ε

    Tests the paper's hypothesis that LIS at t-12 to t-16 predicts provisions at t.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        dep_var: str = "provision_rate",
        lis_var: str = "lis",
        ar_lags: int = 4,
        lis_lags: list[int] = None,
    ):
        """
        Initialize ARDL model.

        Args:
            data: Panel data with ticker, date, dep_var, lis_var, controls
            dep_var: Dependent variable name
            lis_var: LIS variable name
            ar_lags: Number of AR lags (default: 4 for quarterly data)
            lis_lags: LIS lag horizons (default: 12, 14, 16, 18, 20 quarters)
        """
        self.data = data
        self.dep_var = dep_var
        self.lis_var = lis_var
        self.ar_lags = ar_lags
        self.lis_lags = lis_lags or [12, 14, 16, 18, 20]
        self._result: Optional[ARDLResult] = None

    def prepare_data(self) -> pl.DataFrame:
        """
        Prepare data with lags for estimation.

        Returns:
            DataFrame with all lagged variables
        """
        df = self.data.clone()

        # Create AR lags of dependent variable
        for lag in range(1, self.ar_lags + 1):
            df = df.with_columns(
                pl.col(self.dep_var).shift(lag).over("ticker").alias(f"{self.dep_var}_lag{lag}")
            )

        # Create LIS lags
        for lag in self.lis_lags:
            df = df.with_columns(
                pl.col(self.lis_var).shift(lag).over("ticker").alias(f"{self.lis_var}_lag{lag}")
            )

        # Drop rows with missing values
        lag_cols = (
            [f"{self.dep_var}_lag{i}" for i in range(1, self.ar_lags + 1)] +
            [f"{self.lis_var}_lag{i}" for i in self.lis_lags]
        )
        df = df.drop_nulls(subset=[self.dep_var] + lag_cols)

        return df

    def estimate(
        self,
        controls: Optional[list[str]] = None,
        fixed_effects: bool = True,
    ) -> ARDLResult:
        """
        Estimate ARDL model using OLS.

        Args:
            controls: Additional control variables
            fixed_effects: Include bank fixed effects

        Returns:
            ARDLResult with coefficients and diagnostics
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required for ARDL estimation")

        df = self.prepare_data()

        # Build regressor list
        regressors = []

        # AR lags
        for lag in range(1, self.ar_lags + 1):
            regressors.append(f"{self.dep_var}_lag{lag}")

        # LIS lags
        for lag in self.lis_lags:
            regressors.append(f"{self.lis_var}_lag{lag}")

        # Controls
        if controls:
            regressors.extend([c for c in controls if c in df.columns])

        # Fixed effects (bank dummies)
        if fixed_effects:
            tickers = df["ticker"].unique().to_list()
            for ticker in tickers[1:]:  # Omit first for identification
                df = df.with_columns(
                    (pl.col("ticker") == ticker).cast(pl.Int32).alias(f"fe_{ticker}")
                )
                regressors.append(f"fe_{ticker}")

        # Convert to numpy for estimation
        y = df.select(self.dep_var).to_numpy().flatten()
        X = df.select(regressors).to_numpy()

        # Add constant
        X = add_constant(X)
        var_names = ["const"] + regressors

        # OLS estimation
        model = OLS(y, X, missing="drop")
        result = model.fit()

        # Extract LIS coefficients for key lags
        lis_12_col = f"{self.lis_var}_lag12"
        lis_16_col = f"{self.lis_var}_lag16"

        lis_effect_3yr = 0.0
        lis_effect_4yr = 0.0

        if lis_12_col in var_names:
            idx = var_names.index(lis_12_col)
            lis_effect_3yr = result.params[idx]

        if lis_16_col in var_names:
            idx = var_names.index(lis_16_col)
            lis_effect_4yr = result.params[idx]

        self._result = ARDLResult(
            coefficients=dict(zip(var_names, result.params)),
            std_errors=dict(zip(var_names, result.bse)),
            pvalues=dict(zip(var_names, result.pvalues)),
            r_squared=result.rsquared,
            adj_r_squared=result.rsquared_adj,
            n_obs=int(result.nobs),
            aic=result.aic,
            bic=result.bic,
            lis_effect_3yr=lis_effect_3yr,
            lis_effect_4yr=lis_effect_4yr,
        )

        return self._result

    def get_summary(self) -> str:
        """Get model summary."""
        if self._result is None:
            return "Model not yet estimated"

        r = self._result
        lines = [
            "=" * 60,
            "ARDL Model Results",
            "=" * 60,
            f"Observations: {r.n_obs}",
            f"R-squared: {r.r_squared:.4f}",
            f"Adj R-squared: {r.adj_r_squared:.4f}",
            f"AIC: {r.aic:.2f}",
            f"BIC: {r.bic:.2f}",
            "",
            "Key LIS Coefficients:",
            f"  LIS at t-12 (3yr): {r.lis_effect_3yr:.4f}",
            f"  LIS at t-16 (4yr): {r.lis_effect_4yr:.4f}",
            "",
            "Interpretation:",
            f"  1 SD increase in LIS at t-12 predicts {r.lis_effect_3yr*100:.2f} bp",
            f"  increase in provision rate 3 years later.",
            "=" * 60,
        ]

        return "\n".join(lines)


class SARIMAXForecaster:
    """
    SARIMAX model for bank-specific provision forecasting.

    Uses LIS and macro variables as exogenous regressors to forecast
    future provision rates 12-20 quarters ahead.
    """

    def __init__(
        self,
        bank_data: pl.DataFrame,
        ticker: str,
        dep_var: str = "provision_rate",
        exog_vars: Optional[list[str]] = None,
    ):
        """
        Initialize SARIMAX forecaster.

        Args:
            bank_data: Panel data
            ticker: Bank ticker to forecast
            dep_var: Variable to forecast
            exog_vars: Exogenous variables (LIS, macro controls)
        """
        self.bank_data = bank_data.filter(pl.col("ticker") == ticker)
        self.ticker = ticker
        self.dep_var = dep_var
        self.exog_vars = exog_vars or []
        self._model = None
        self._fit = None

    def identify_order(self) -> tuple[int, int, int]:
        """
        Identify ARIMA order using ACF/PACF analysis.

        Returns:
            Tuple of (p, d, q) order
        """
        if not HAS_STATSMODELS:
            return (1, 0, 1)  # Default

        y = self.bank_data.select(self.dep_var).drop_nulls().to_numpy().flatten()

        if len(y) < 20:
            return (1, 0, 1)

        # Test for stationarity
        adf_result = adfuller(y, maxlag=8)
        d = 0 if adf_result[1] < 0.05 else 1

        # Use differenced series if needed
        if d == 1:
            y_diff = np.diff(y)
        else:
            y_diff = y

        # ACF and PACF for p and q
        acf_vals = acf(y_diff, nlags=8)
        pacf_vals = pacf(y_diff, nlags=8)

        # Simple heuristic: count significant lags
        p = sum(1 for i in range(1, 5) if abs(pacf_vals[i]) > 0.2)
        q = sum(1 for i in range(1, 5) if abs(acf_vals[i]) > 0.2)

        p = max(1, min(p, 4))
        q = max(0, min(q, 2))

        return (p, d, q)

    def fit(
        self,
        order: Optional[tuple[int, int, int]] = None,
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 4),
    ):
        """
        Fit SARIMAX model.

        Args:
            order: ARIMA order (p, d, q). If None, auto-identify.
            seasonal_order: Seasonal order (P, D, Q, s)
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required for SARIMAX")

        if order is None:
            order = self.identify_order()

        # Prepare data
        df = self.bank_data.drop_nulls(subset=[self.dep_var])
        y = df.select(self.dep_var).to_numpy().flatten()

        # Exogenous variables
        if self.exog_vars:
            available = [v for v in self.exog_vars if v in df.columns]
            if available:
                X = df.select(available).to_numpy()
            else:
                X = None
        else:
            X = None

        # Fit model
        self._model = SARIMAX(
            y,
            exog=X,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._fit = self._model.fit(disp=False)

        return self

    def forecast(
        self,
        horizon: int = 12,
        exog_forecast: Optional[np.ndarray] = None,
        confidence: float = 0.95,
    ) -> list[ForecastResult]:
        """
        Generate forecasts.

        Args:
            horizon: Forecast horizon in quarters
            exog_forecast: Future values of exogenous variables
            confidence: Confidence level for intervals

        Returns:
            List of ForecastResult objects
        """
        if self._fit is None:
            raise ValueError("Model must be fit before forecasting")

        # Get forecast
        forecast = self._fit.get_forecast(steps=horizon, exog=exog_forecast)
        pred = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=1 - confidence)

        # Get last date
        last_date = self.bank_data["date"].max()

        results = []
        for h in range(horizon):
            # Forecast date
            forecast_date = last_date + pl.duration(days=91 * (h + 1))

            # Risk level based on forecast magnitude
            if pred[h] > np.percentile(pred, 75):
                risk = "high"
            elif pred[h] > np.percentile(pred, 50):
                risk = "medium"
            else:
                risk = "low"

            results.append(ForecastResult(
                ticker=self.ticker,
                forecast_date=forecast_date,
                horizon_quarters=h + 1,
                point_forecast=pred[h],
                ci_lower=conf_int.iloc[h, 0],
                ci_upper=conf_int.iloc[h, 1],
                risk_level=risk,
                model_used="SARIMAX",
            ))

        return results


class CreditBoomIndicator:
    """
    Main class for credit boom early warning system.

    Combines:
    1. LIS calculation for each bank
    2. ARDL estimation for historical validation
    3. SARIMAX forecasting for forward signals
    4. Dashboard generation
    """

    def __init__(
        self,
        bank_data: pl.DataFrame,
        system_data: pl.DataFrame,
        macro_data: Optional[pl.DataFrame] = None,
    ):
        """
        Initialize the credit boom indicator.

        Args:
            bank_data: Panel of bank-level data
            system_data: System-wide aggregates
            macro_data: Macro control variables
        """
        self.bank_data = bank_data
        self.system_data = system_data
        self.macro_data = macro_data

        self.lis_calculator: Optional[LendingIntensityScore] = None
        self.ardl_model: Optional[ARDLModel] = None
        self._signals: Optional[pl.DataFrame] = None

    def compute_lis(self) -> pl.DataFrame:
        """Compute LIS for all banks."""
        self.lis_calculator = LendingIntensityScore(
            self.bank_data,
            self.system_data,
            growth_col="loan_growth_yoy",
        )
        return self.lis_calculator.compute_lis()

    def estimate_ardl(
        self,
        controls: Optional[list[str]] = None,
    ) -> ARDLResult:
        """Estimate ARDL model on historical data."""
        # Merge LIS with bank data
        lis_data = self.compute_lis()

        # Merge with macro if available
        if self.macro_data is not None:
            lis_data = lis_data.join(
                self.macro_data,
                on="date",
                how="left",
            )

        self.ardl_model = ARDLModel(
            lis_data,
            dep_var="provision_rate",
            lis_var="lis",
        )

        return self.ardl_model.estimate(controls=controls)

    def generate_forecasts(
        self,
        horizon: int = 16,
    ) -> pl.DataFrame:
        """
        Generate provision forecasts for all banks.

        Args:
            horizon: Forecast horizon in quarters

        Returns:
            DataFrame with forecasts for all banks
        """
        lis_data = self.compute_lis()
        tickers = lis_data["ticker"].unique().to_list()

        all_forecasts = []

        for ticker in tickers:
            try:
                forecaster = SARIMAXForecaster(
                    lis_data,
                    ticker=ticker,
                    dep_var="provision_rate",
                    exog_vars=["lis"],
                )
                forecaster.fit()
                forecasts = forecaster.forecast(horizon=horizon)
                all_forecasts.extend(forecasts)
            except Exception as e:
                print(f"Warning: Could not forecast {ticker}: {e}")

        # Convert to DataFrame
        if not all_forecasts:
            return pl.DataFrame()

        records = [
            {
                "ticker": f.ticker,
                "forecast_date": f.forecast_date,
                "horizon_quarters": f.horizon_quarters,
                "point_forecast": f.point_forecast,
                "ci_lower": f.ci_lower,
                "ci_upper": f.ci_upper,
                "risk_level": f.risk_level,
            }
            for f in all_forecasts
        ]

        return pl.DataFrame(records)

    def generate_early_warning_signals(self) -> pl.DataFrame:
        """
        Generate comprehensive early warning signals.

        Combines:
        - Current LIS levels
        - Historical pattern (cumulative LIS)
        - Forward forecasts
        - Risk classification

        Returns:
            DataFrame with signals for each bank
        """
        # Current LIS
        lis_current = self.lis_calculator.get_current_signals() if self.lis_calculator else self.compute_lis()
        current = lis_current.group_by("ticker").agg(
            pl.col("date").max().alias("date")
        )
        current_signals = lis_current.join(current, on=["ticker", "date"], how="inner")

        # ARDL-based risk assessment
        if self.ardl_model and self.ardl_model._result:
            lis_effect = self.ardl_model._result.lis_effect_3yr
        else:
            lis_effect = 0.01  # Default assumption

        # Compute expected provision increase
        current_signals = current_signals.with_columns(
            (pl.col("lis") * lis_effect * 100).alias("expected_provision_impact_bp")
        )

        # Overall risk classification
        current_signals = current_signals.with_columns(
            pl.when(
                (pl.col("lis") > 2.0) | (pl.col("lis_cumulative_12q") > 8.0)
            ).then(pl.lit("HIGH"))
            .when(
                (pl.col("lis") > 1.0) | (pl.col("lis_cumulative_12q") > 4.0)
            ).then(pl.lit("MEDIUM"))
            .otherwise(pl.lit("LOW"))
            .alias("risk_classification")
        )

        self._signals = current_signals
        return current_signals

    def get_summary_table(self) -> pl.DataFrame:
        """
        Get summary table for dashboard display.

        Returns:
            DataFrame formatted for display
        """
        if self._signals is None:
            self.generate_early_warning_signals()

        return self._signals.select([
            "ticker",
            "date",
            "lis",
            "lis_cumulative_12q",
            "bank_growth",
            "system_growth",
            "expected_provision_impact_bp",
            "risk_classification",
        ]).sort("risk_classification", descending=True)


if __name__ == "__main__":
    # Test with synthetic data
    from .bank_data import SyntheticBankData

    print("Generating synthetic data...")
    synth = SyntheticBankData()
    panel = synth.generate_panel(n_banks=5)

    # Create system data (average across banks)
    system = panel.group_by("date").agg(
        pl.col("loan_growth_yoy").mean().alias("loan_growth_yoy")
    ).sort("date")

    print("\nComputing Lending Intensity Score...")
    lis_calc = LendingIntensityScore(panel, system)
    lis_data = lis_calc.compute_lis()
    print(lis_data.select(["date", "ticker", "lis", "lis_cumulative_12q"]).tail(10))

    print("\nCurrent signals:")
    current = lis_calc.get_current_signals()
    print(current.select(["ticker", "lis", "lis_cumulative_12q", "elevated_lis"]))

    print("\nEstimating ARDL model...")
    ardl = ARDLModel(lis_data, dep_var="provision_rate", lis_var="lis")
    try:
        result = ardl.estimate()
        print(ardl.get_summary())
    except Exception as e:
        print(f"ARDL estimation failed: {e}")

    print("\nCredit Boom Indicator complete.")
