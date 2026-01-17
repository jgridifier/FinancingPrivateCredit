"""
Prime Brokerage Leverage Lead Indicator V2 (PB-LLI)

A forecasting-centric indicator suite designed to predict prime brokerage
performance 1-2 quarters ahead by combining:
1. Hedge fund balance-sheet leverage demand (quarterly Fed Z.1)
2. Dealer balance-sheet supply (quarterly Fed Z.1)
3. Weekly nowcast layer using CFTC COT and NY Fed PD data

Design principle: Optimize for cycle and run-rate forecasting in normal market
conditions; treat stress as a secondary overlay rather than the objective function.
"""

from .indicator import (
    PrimeLeverageV2Indicator,
    PBLLISpec,
)
from .shadow_nowcast import (
    ShadowNowcaster,
    ShadowNowcastConfig,
    ShadowEstimate,
    BankDisclosureExtractor,
)

__all__ = [
    "PrimeLeverageV2Indicator",
    "PBLLISpec",
    "ShadowNowcaster",
    "ShadowNowcastConfig",
    "ShadowEstimate",
    "BankDisclosureExtractor",
]
