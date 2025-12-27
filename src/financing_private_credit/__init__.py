"""
Financing Private Credit
========================

Reproduction and extension of NY Fed Staff Report 1111.

This package provides tools to:
1. Fetch private credit data from FRED (Z.1 Financial Accounts)
2. Decompose credit by lender type (banks vs nonbanks)
3. Analyze the demand system for credit supply/demand elasticities
4. Nowcast credit conditions using higher-frequency proxy data
"""

from .data import FREDDataFetcher, PrivateCreditData
from .analysis import CreditDecomposition, DemandSystemModel
from .nowcast import CreditNowcaster

__version__ = "0.1.0"
__all__ = [
    "FREDDataFetcher",
    "PrivateCreditData",
    "CreditDecomposition",
    "DemandSystemModel",
    "CreditNowcaster",
]
