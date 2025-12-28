"""
Indicators Package

A modular indicator framework for credit and financing analysis.
Originally based on NY Fed Staff Report 1111 methodology, now extended
to support general financial indicator development.

Available Indicators:
- demand_system: Original paper replication (credit decomposition, elasticities)
- credit_boom: Credit Boom Leading Indicator (LIS-based)
- variance_decomposition: Cross-Bank Variance Decomposition
- bank_macro_sensitivity: Bank-Specific Macro Sensitivity (NIM elasticities)
- duration_mismatch: Duration Mismatch as Predictive Signal (volatility prediction)
- funding_stability: Funding Stability Score (procyclical behavior prediction)

Usage:
    from financing_private_credit.indicators import get_indicator, list_indicators

    # List available indicators
    print(list_indicators())

    # Get an indicator instance
    indicator = get_indicator("demand_system")
    data = indicator.fetch_data("2015-01-01")
    result = indicator.calculate(data)

    # Bank Macro Sensitivity with custom spec
    from financing_private_credit.indicators.bank_macro_sensitivity import (
        BankMacroSensitivityIndicator,
        MacroSensitivitySpec
    )
    spec = MacroSensitivitySpec.from_json("config/model_specs/my_spec.json")
    indicator = BankMacroSensitivityIndicator()
    result = indicator.calculate(data, spec=spec)

    # Funding Stability with stress testing
    from financing_private_credit.indicators.funding_stability import (
        FundingStabilityForecaster,
        PREDEFINED_SCENARIOS
    )
    forecaster = FundingStabilityForecaster(result.data)
    scenarios = forecaster.scenario_analysis(PREDEFINED_SCENARIOS)
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
from . import demand_system
from . import credit_boom
from . import variance_decomposition
from . import bank_macro_sensitivity
from . import duration_mismatch
from . import funding_stability

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
