"""
Credit Boom Leading Indicator Package

This package implements the Credit Boom Leading Indicator from NY Fed Staff Report 1111.
It identifies banks with aggressive lending behavior that may precede credit losses.

Components:
- indicator.py: Main CreditBoomIndicator class
- lis.py: Lending Intensity Score calculation
- models.py: ARDL and SARIMAX prediction models
- nowcast.py: High-frequency nowcasting using H.8 data
- forecast.py: APLR-based provision rate forecasting
"""

from .indicator import CreditBoomIndicator
from .lis import LendingIntensityScore, LISResult
from .models import ARDLModel, ARDLResult, SARIMAXForecaster, ForecastResult
from .nowcast import CreditNowcaster, NowcastResult, FinancialConditionsMonitor, NOWCAST_PROXY_SERIES
from .forecast import (
    ModelSpecification,
    APLRForecaster,
    FallbackForecaster,
    SeasonalFeatureGenerator,
    BacktestResult,
    get_forecaster,
    APLR_AVAILABLE,
)

__all__ = [
    # Main indicator
    "CreditBoomIndicator",
    # LIS
    "LendingIntensityScore",
    "LISResult",
    # Models
    "ARDLModel",
    "ARDLResult",
    "SARIMAXForecaster",
    "ForecastResult",
    # Nowcasting
    "CreditNowcaster",
    "NowcastResult",
    "FinancialConditionsMonitor",
    "NOWCAST_PROXY_SERIES",
    # Forecasting
    "ModelSpecification",
    "APLRForecaster",
    "FallbackForecaster",
    "SeasonalFeatureGenerator",
    "BacktestResult",
    "get_forecaster",
    "APLR_AVAILABLE",
]
