"""
Demand System Indicator - Paper Replication

Implements the demand system approach from Boyarchenko & Elias (2024):
"Financing Private Credit: The Role of Lender Type in Credit Booms"

This is the original paper replication indicator, providing:
1. Credit decomposition by lender type (bank vs nonbank)
2. Supply elasticity estimation
3. Crisis probability computation
4. Schularick-Taylor credit expansion predictor

Key finding: Credit expansions financed primarily by banks are
associated with higher crisis probability than those financed by nonbanks.
"""

from .indicator import (
    DemandSystemIndicator,
    DemandSystemSpec,
    DemandSystemModel,
    CreditDecomposition,
    ElasticityResults,
    compute_schularick_taylor_predictor,
)
from .viz import DemandSystemVisualizer

__all__ = [
    "DemandSystemIndicator",
    "DemandSystemSpec",
    "DemandSystemModel",
    "CreditDecomposition",
    "ElasticityResults",
    "DemandSystemVisualizer",
    "compute_schularick_taylor_predictor",
]
