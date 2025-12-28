"""
Financial Indicator Framework
=============================

A modular framework for bank-level financial indicators.
Originally based on NY Fed Staff Report 1111, now extended for general use.

Indicators:
- demand_system: Original paper replication (bank vs nonbank credit)
- credit_boom: Lending Intensity Score (LIS) for early warning
- bank_macro_sensitivity: NIM elasticities to macro conditions
- duration_mismatch: Duration exposure and earnings vulnerability
- funding_stability: SVB-style run risk prediction
- variance_decomposition: Systematic vs idiosyncratic risk

See `indicators` subpackage for the registry API:
    from financing_private_credit.indicators import get_indicator, list_indicators
"""

from .data import FREDDataFetcher, PrivateCreditData
from .macro import MacroDataFetcher, BankSystemData
from .bank_data import BankDataCollector, SyntheticBankData, TARGET_BANKS

# Import from demand_system indicator (paper replication)
from .indicators.demand_system import (
    DemandSystemIndicator,
    CreditDecomposition,
    DemandSystemModel,
)

# Import from indicator packages
from .indicators.credit_boom import (
    CreditBoomIndicator,
    LendingIntensityScore,
    ARDLModel,
    SARIMAXForecaster,
    CreditNowcaster,
)
from .indicators.variance_decomposition import (
    VarianceDecompositionIndicator,
)
from .indicators.bank_macro_sensitivity import (
    BankMacroSensitivityIndicator,
    MacroSensitivityForecaster,
    MacroSensitivityNowcaster,
    MacroSensitivityBacktester,
)
from .indicators.duration_mismatch import (
    DurationMismatchIndicator,
    DurationMismatchForecaster,
    DurationMismatchNowcaster,
    DurationMismatchBacktester,
    DurationMismatchVisualizer,
)
from .indicators.funding_stability import (
    FundingStabilityIndicator,
    FundingStabilityForecaster,
    FundingStabilityNowcaster,
    FundingStabilityBacktester,
    FundingStabilityVisualizer,
)

# Registry API
from .indicators import get_indicator, list_indicators

__version__ = "0.1.0"
__all__ = [
    # Registry API
    "get_indicator",
    "list_indicators",
    # Core data
    "FREDDataFetcher",
    "PrivateCreditData",
    # Macro data
    "MacroDataFetcher",
    "BankSystemData",
    # Bank data
    "BankDataCollector",
    "SyntheticBankData",
    "TARGET_BANKS",
    # Demand System (Paper Replication)
    "DemandSystemIndicator",
    "CreditDecomposition",
    "DemandSystemModel",
    # Credit Boom
    "CreditBoomIndicator",
    "LendingIntensityScore",
    "ARDLModel",
    "SARIMAXForecaster",
    "CreditNowcaster",
    # Variance Decomposition
    "VarianceDecompositionIndicator",
    # Bank Macro Sensitivity
    "BankMacroSensitivityIndicator",
    "MacroSensitivityForecaster",
    "MacroSensitivityNowcaster",
    "MacroSensitivityBacktester",
    # Duration Mismatch
    "DurationMismatchIndicator",
    "DurationMismatchForecaster",
    "DurationMismatchNowcaster",
    "DurationMismatchBacktester",
    "DurationMismatchVisualizer",
    # Funding Stability
    "FundingStabilityIndicator",
    "FundingStabilityForecaster",
    "FundingStabilityNowcaster",
    "FundingStabilityBacktester",
    "FundingStabilityVisualizer",
]
