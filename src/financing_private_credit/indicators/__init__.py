"""
Indicators Package

This package contains implementations of various credit and financing indicators
based on NY Fed Staff Report 1111 methodology.

Available Indicators:
- credit_boom: Credit Boom Leading Indicator (LIS-based)
- variance_decomposition: Cross-Bank Variance Decomposition

Usage:
    from financing_private_credit.indicators import get_indicator, list_indicators

    # List available indicators
    print(list_indicators())

    # Get an indicator instance
    indicator = get_indicator("variance_decomposition")
    data = indicator.fetch_data("2015-01-01")
    result = indicator.calculate(data)
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
