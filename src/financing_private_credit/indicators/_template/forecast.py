"""
Template Forecaster

This template shows how to implement forecasting for your indicator.
Forecasters predict future values based on historical data and macro scenarios.

The framework supports any model type through BaseForecastModel:
- sklearn models (RandomForest, XGBoost, etc.)
- statsmodels (ARIMA, VAR, ARDL, etc.)
- PyTorch/TensorFlow neural networks
- Custom implementations

Copy this file and adapt for your indicator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import polars as pl

# Import the base class and result container
from ..base import BaseForecastModel, ForecastResult


@dataclass
class TemplateForecasterSpec:
    """Configuration for the template forecaster."""

    name: str = "default"
    description: str = "Default template forecaster"

    # Model parameters
    ar_lags: int = 2  # Autoregressive lags
    horizon: int = 4  # Forecast horizon (quarters)
    confidence_level: float = 0.95  # For prediction intervals

    # Feature configuration
    include_macro: bool = True
    macro_features: tuple[str, ...] = ("fed_funds_rate", "yield_curve_slope")

    @classmethod
    def from_dict(cls, d: dict) -> "TemplateForecasterSpec":
        """Create spec from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Example 1: Using sklearn
# -----------------------
# Uncomment and adapt if using sklearn models

# from sklearn.ensemble import RandomForestRegressor
#
# class TemplateSklearnForecaster(BaseForecastModel[RandomForestRegressor]):
#     """
#     Example forecaster using sklearn RandomForest.
#     """
#
#     def __init__(self, spec: Optional[TemplateForecasterSpec] = None, **rf_params):
#         super().__init__()
#         self.spec = spec or TemplateForecasterSpec()
#         self._model = RandomForestRegressor(**rf_params)
#
#     def fit(
#         self,
#         data: pl.DataFrame,
#         target: str,
#         features: list[str],
#         **kwargs,
#     ) -> dict[str, Any]:
#         """Fit the random forest model."""
#         self._target = target
#         self._features = features
#
#         X = data.select(features).to_numpy()
#         y = data[target].to_numpy()
#
#         self._model.fit(X, y)
#         self._is_fitted = True
#
#         return {
#             "r2_train": float(self._model.score(X, y)),
#             "n_samples": len(y),
#             "n_features": len(features),
#         }
#
#     def predict(
#         self,
#         data: pl.DataFrame,
#         horizon: int = 1,
#         **kwargs,
#     ) -> ForecastResult:
#         """Generate predictions."""
#         if not self._is_fitted:
#             raise ValueError("Model must be fitted before prediction")
#
#         X = data.select(self._features).to_numpy()
#         predictions = self._model.predict(X)
#
#         result_df = pl.DataFrame({
#             "date": data["date"],
#             "prediction": predictions,
#         })
#
#         return ForecastResult(
#             target=self._target,
#             horizon=horizon,
#             predictions=result_df,
#             metadata={"model": "RandomForest"},
#         )
#
#     def get_feature_importance(self) -> Optional[dict[str, float]]:
#         """Return feature importances."""
#         if not self._is_fitted:
#             return None
#         return dict(zip(self._features, self._model.feature_importances_))


# Example 2: Using statsmodels
# ---------------------------
# Uncomment and adapt if using statsmodels

# from statsmodels.tsa.ar_model import AutoReg
#
# class TemplateARForecaster(BaseForecastModel):
#     """
#     Example forecaster using statsmodels AutoReg.
#     """
#
#     def __init__(self, spec: Optional[TemplateForecasterSpec] = None):
#         super().__init__()
#         self.spec = spec or TemplateForecasterSpec()
#         self._ar_model = None
#
#     def fit(
#         self,
#         data: pl.DataFrame,
#         target: str,
#         features: list[str],
#         **kwargs,
#     ) -> dict[str, Any]:
#         """Fit the AR model."""
#         self._target = target
#         self._features = features
#
#         y = data[target].to_numpy()
#
#         self._ar_model = AutoReg(y, lags=self.spec.ar_lags)
#         self._model = self._ar_model.fit()
#         self._is_fitted = True
#
#         return {
#             "aic": float(self._model.aic),
#             "bic": float(self._model.bic),
#             "n_obs": len(y),
#         }
#
#     def predict(
#         self,
#         data: pl.DataFrame,
#         horizon: int = 1,
#         **kwargs,
#     ) -> ForecastResult:
#         """Generate predictions."""
#         if not self._is_fitted:
#             raise ValueError("Model must be fitted before prediction")
#
#         start = len(data)
#         end = start + horizon - 1
#         predictions = self._model.predict(start=start, end=end)
#
#         # Create future dates
#         last_date = data["date"].max()
#         future_dates = pl.date_range(
#             last_date,
#             last_date + pl.duration(days=90 * horizon),
#             "3mo",
#             eager=True,
#         )[1:horizon+1]
#
#         result_df = pl.DataFrame({
#             "date": future_dates,
#             "prediction": predictions,
#         })
#
#         return ForecastResult(
#             target=self._target,
#             horizon=horizon,
#             predictions=result_df,
#             metadata={"model": "AutoReg", "lags": self.spec.ar_lags},
#         )
#
#     def get_coefficients(self) -> Optional[dict[str, float]]:
#         """Return AR coefficients."""
#         if not self._is_fitted:
#             return None
#         return {f"ar.L{i+1}": c for i, c in enumerate(self._model.params[1:])}


# Example 3: Custom implementation (no external model)
# ---------------------------------------------------
class TemplateForecaster(BaseForecastModel[None]):
    """
    Template forecaster with custom implementation.

    This example shows a simple moving average forecast.
    Replace with your actual forecasting logic.
    """

    def __init__(self, spec: Optional[TemplateForecasterSpec] = None):
        """
        Initialize the forecaster.

        Args:
            spec: Forecaster configuration
        """
        super().__init__()
        self.spec = spec or TemplateForecasterSpec()
        self._last_values: Optional[pl.DataFrame] = None

    def fit(
        self,
        data: pl.DataFrame,
        target: str,
        features: list[str],
        **kwargs,
    ) -> dict[str, Any]:
        """
        Fit the model to training data.

        Args:
            data: Training data with target and feature columns
            target: Name of the target variable
            features: List of feature column names
            **kwargs: Additional parameters

        Returns:
            Dictionary with fit metrics
        """
        self._target = target
        self._features = features

        # Store recent values for simple moving average forecast
        self._last_values = data.tail(self.spec.ar_lags).select([
            "date", target
        ])

        self._is_fitted = True

        # Calculate in-sample metrics
        mean_value = data[target].mean()
        std_value = data[target].std()

        return {
            "n_observations": data.height,
            "target_mean": float(mean_value) if mean_value is not None else None,
            "target_std": float(std_value) if std_value is not None else None,
            "lags_used": self.spec.ar_lags,
        }

    def predict(
        self,
        data: pl.DataFrame,
        horizon: int = 1,
        **kwargs,
    ) -> ForecastResult:
        """
        Generate predictions for future periods.

        Args:
            data: Input data (may include exogenous forecasts)
            horizon: Number of periods to forecast
            **kwargs: Additional parameters (e.g., scenarios)

        Returns:
            ForecastResult with predictions
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        horizon = horizon or self.spec.horizon

        # Simple moving average forecast (replace with your logic)
        last_mean = self._last_values[self._target].mean()

        # Generate future dates
        last_date = data["date"].max()
        future_dates = []
        current_date = last_date

        for i in range(horizon):
            # Add approximately 3 months (quarterly)
            if hasattr(current_date, 'offset_by'):
                current_date = current_date.offset_by("3mo")
            else:
                # Fallback for datetime
                from datetime import timedelta
                current_date = current_date + timedelta(days=91)
            future_dates.append(current_date)

        predictions_df = pl.DataFrame({
            "date": future_dates,
            "prediction": [float(last_mean)] * horizon,
            "prediction_lower": [float(last_mean) * 0.9] * horizon,
            "prediction_upper": [float(last_mean) * 1.1] * horizon,
        })

        confidence_df = predictions_df.select([
            "date",
            pl.col("prediction_lower").alias("lower"),
            pl.col("prediction_upper").alias("upper"),
        ])

        return ForecastResult(
            target=self._target,
            horizon=horizon,
            predictions=predictions_df.select(["date", "prediction"]),
            confidence_intervals=confidence_df,
            metadata={
                "model": "SimpleMovingAverage",
                "confidence_level": self.spec.confidence_level,
            },
        )

    def get_diagnostics(self) -> dict[str, Any]:
        """Return model diagnostics."""
        base = super().get_diagnostics()
        base.update({
            "spec_name": self.spec.name,
            "ar_lags": self.spec.ar_lags,
            "horizon": self.spec.horizon,
        })
        return base


class TemplateScenarioForecaster:
    """
    Optional: Scenario-based forecaster for stress testing.

    Generates forecasts under different macro scenarios.
    """

    def __init__(
        self,
        base_forecaster: BaseForecastModel,
        scenarios: Optional[dict[str, dict[str, float]]] = None,
    ):
        """
        Initialize scenario forecaster.

        Args:
            base_forecaster: Fitted forecaster to use
            scenarios: Dictionary of scenario name -> macro variable values
        """
        self.forecaster = base_forecaster
        self.scenarios = scenarios or {
            "baseline": {"fed_funds_rate": 5.25, "yield_curve_slope": 0.0},
            "rate_cut": {"fed_funds_rate": 4.00, "yield_curve_slope": 0.50},
            "recession": {"fed_funds_rate": 3.00, "yield_curve_slope": 1.50},
            "stagflation": {"fed_funds_rate": 6.50, "yield_curve_slope": -0.50},
        }

    def forecast_scenarios(
        self,
        data: pl.DataFrame,
        horizon: int = 4,
    ) -> dict[str, ForecastResult]:
        """
        Generate forecasts for all scenarios.

        Args:
            data: Input data
            horizon: Forecast horizon

        Returns:
            Dictionary mapping scenario names to ForecastResults
        """
        results = {}

        for scenario_name, scenario_values in self.scenarios.items():
            # Create scenario data by adjusting macro variables
            scenario_data = data.clone()

            for var_name, var_value in scenario_values.items():
                if var_name in scenario_data.columns:
                    scenario_data = scenario_data.with_columns(
                        pl.lit(var_value).alias(var_name)
                    )

            # Generate forecast
            result = self.forecaster.predict(scenario_data, horizon)
            result.metadata["scenario"] = scenario_name
            result.metadata["scenario_values"] = scenario_values
            results[scenario_name] = result

        return results
