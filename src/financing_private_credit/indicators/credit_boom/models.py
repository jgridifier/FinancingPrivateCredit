"""
Credit Boom Prediction Models.

Implements:
1. ARDL: Autoregressive Distributed Lag panel model
2. SARIMAX: Bank-specific time series forecasting

These models test the paper's hypothesis that LIS at t-12 to t-16
predicts provision rates at t.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal

import numpy as np
import polars as pl

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


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
