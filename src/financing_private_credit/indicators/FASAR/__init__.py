"""
FASAR - Flex-Adjusted Syndicate Absorption Ratio

Measures the mismatch between a bank's contractual inability to escape a deal
and the market's inability to absorb that deal.

Key components:
- Rigidity Score: NLP-derived metric from 8-K text quantifying how easily
  a bank can walk away or re-price a deal (1.0 = trapped, 0.0 = can exit)
- CLO Velocity: Flow-based metric measuring if CLO buyers are absorbing debt

Reference:
- Ivashina & Scharfstein (2010): Loan Syndication and Credit Cycles
- Gatev & Strahan (2006): Liquidity Risk and Syndicate Structure
"""

from .indicator import FASARIndicator, FASARSpec, CommitmentDeal, RigidityResult
from .rigidity import (
    RigidityScorer,
    RigidityClassification,
    RigidityEvidence,
    extract_rigidity_evidence,
    compute_preliminary_rigidity,
)
from .clo_velocity import CLOVelocityCalculator, CLOVelocityResult
from .nowcast import FASARNowcaster, NowcastSignal, WarehouseStressIndicator
from .forecast import (
    FASARForecaster,
    ScarTissueForecaster,
    ScarTissueSpec,
    MarketShareForecast,
)

__all__ = [
    # Core indicator
    "FASARIndicator",
    "FASARSpec",
    "CommitmentDeal",
    "RigidityResult",
    # Rigidity scoring
    "RigidityScorer",
    "RigidityClassification",
    "RigidityEvidence",
    "extract_rigidity_evidence",
    "compute_preliminary_rigidity",
    # CLO velocity
    "CLOVelocityCalculator",
    "CLOVelocityResult",
    # Nowcasting
    "FASARNowcaster",
    "NowcastSignal",
    "WarehouseStressIndicator",
    # Forecasting
    "FASARForecaster",
    "ScarTissueForecaster",
    "ScarTissueSpec",
    "MarketShareForecast",
]
