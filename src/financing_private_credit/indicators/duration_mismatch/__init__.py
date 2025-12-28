"""
Duration Mismatch as Predictive Signal Indicator

Novel Insight 2: The paper's finding on bond duration sensitivity (Figure 8b)
can predict bank earnings volatility and stock returns.

Hypothesis: Banks with more duration-sensitive portfolios have higher
earnings volatility when yields move.

Methodology:
1. Estimate each bank's duration exposure from securities portfolio (10-K/10-Q)
2. Calculate predicted earnings impact: duration_exposure × Δ(bond_yield)
3. Test if this predicts:
   - Actual earnings volatility
   - Stock return volatility
   - Future NIM compression/expansion

Data Sources (all public):
- SEC EDGAR: Securities portfolio details (AFS/HTM, duration)
- Yahoo Finance: Stock returns, earnings, volatility metrics
- FRED: Bond yields, term structure

Forecasting Approach:
- Primary: ARDL (Autoregressive Distributed Lag) for lead-lag dynamics
- Secondary: APLR for non-linear sensitivity analysis
"""

from .indicator import DurationMismatchIndicator
from .forecast import DurationMismatchForecaster, ARDLModel
from .nowcast import DurationMismatchNowcaster
from .backtest import DurationMismatchBacktester
from .viz import DurationMismatchVisualizer

__all__ = [
    "DurationMismatchIndicator",
    "DurationMismatchForecaster",
    "ARDLModel",
    "DurationMismatchNowcaster",
    "DurationMismatchBacktester",
    "DurationMismatchVisualizer",
]
