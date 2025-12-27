"""
Financing Private Credit
========================

Reproduction and extension of NY Fed Staff Report 1111.

This package provides tools to:
1. Fetch private credit data from FRED (Z.1 Financial Accounts)
2. Decompose credit by lender type (banks vs nonbanks)
3. Analyze the demand system for credit supply/demand elasticities
4. Nowcast credit conditions using higher-frequency proxy data
5. Build credit boom leading indicators (ARDL, SARIMAX)
6. Generate early warning signals for bank credit stress
"""

from .data import FREDDataFetcher, PrivateCreditData
from .analysis import CreditDecomposition, DemandSystemModel
from .nowcast import CreditNowcaster
from .macro import MacroDataFetcher, BankSystemData
from .bank_data import BankDataCollector, SyntheticBankData, TARGET_BANKS
from .leading_indicator import (
    LendingIntensityScore,
    ARDLModel,
    SARIMAXForecaster,
    CreditBoomIndicator,
)

__version__ = "0.1.0"
__all__ = [
    # Core data
    "FREDDataFetcher",
    "PrivateCreditData",
    # Analysis
    "CreditDecomposition",
    "DemandSystemModel",
    # Nowcasting
    "CreditNowcaster",
    # Macro data
    "MacroDataFetcher",
    "BankSystemData",
    # Bank data
    "BankDataCollector",
    "SyntheticBankData",
    "TARGET_BANKS",
    # Leading indicators
    "LendingIntensityScore",
    "ARDLModel",
    "SARIMAXForecaster",
    "CreditBoomIndicator",
]
