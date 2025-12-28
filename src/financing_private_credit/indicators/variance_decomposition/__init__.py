"""
Variance Decomposition Indicator Package

This package implements the Cross-Bank Variance Decomposition from NY Fed Staff Report 1111.
It decomposes bank loan growth into macro, size, allocation, and idiosyncratic components.

Components:
- indicator.py: Main VarianceDecompositionIndicator class
"""

from .indicator import (
    VarianceDecompositionIndicator,
    DecompositionResult,
    BankArchetype,
    ARCHETYPES,
)

__all__ = [
    "VarianceDecompositionIndicator",
    "DecompositionResult",
    "BankArchetype",
    "ARCHETYPES",
]
