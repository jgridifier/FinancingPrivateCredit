"""
Indicators Package

This package contains implementations of various credit and financing indicators
based on NY Fed Staff Report 1111 methodology.

Available Indicators:
- credit_boom: Credit Boom Leading Indicator (LIS-based)
- variance_decomposition: Cross-Bank Variance Decomposition
- bank_macro_sensitivity: Bank-Specific Macro Sensitivity (NIM elasticities)
- duration_mismatch: Duration Mismatch as Predictive Signal (volatility prediction)

Usage:
    from financing_private_credit.indicators import get_indicator, list_indicators

    # List available indicators
    print(list_indicators())

    # Get an indicator instance
    indicator = get_indicator("variance_decomposition")
    data = indicator.fetch_data("2015-01-01")
    result = indicator.calculate(data)

    # Bank Macro Sensitivity example
    sensitivity = get_indicator("bank_macro_sensitivity")
    data = sensitivity.fetch_data("2010-01-01")
    result = sensitivity.calculate(data)
    print(result.data)  # Sensitivity rankings by bank

    # Duration Mismatch example
    duration = get_indicator("duration_mismatch")
    data = duration.fetch_data("2010-01-01")
    result = duration.calculate(data)
    print(result.data)  # Vulnerability rankings
"""

from .base import (
    BaseDecomposition,
    BaseForecastModel,
    BaseIndicator,
    IndicatorMetadata,
    IndicatorResult,
    get_indicator,
    list_indicators,
    register_indicator,
)

# Import indicator implementations to register them
from . import credit_boom
from . import variance_decomposition
from . import bank_macro_sensitivity
from . import duration_mismatch

__all__ = [
    "BaseIndicator",
    "BaseDecomposition",
    "BaseForecastModel",
    "IndicatorMetadata",
    "IndicatorResult",
    "get_indicator",
    "list_indicators",
    "register_indicator",
]
