"""
Funding Stability Score Indicator

Novel Insight 3: Creates a "funding resilience" score based on the paper's
finding that banks respond more procyclically to macro conditions.

The score combines multiple stress indicators to predict which banks
will constrain credit most aggressively in downturns.

Reformulated Score Components:

Stability Factors (positive):
- Core Deposit Funding Ratio
- Duration Match Score

Risk Factors (inverted - lower raw value = higher score contribution):
- Uninsured Deposit Ratio (Run Risk - the "SVB" variable)
- FHLB Advance Utilization (Desperation Signal)
- Brokered Deposit Ratio (Hot Money)
- AOCI Impact Ratio (Trapped Capital)
- Wholesale Funding Ratio (Non-Core Dependence)
- Deposit Rate Beta (Rate Sensitivity)

Data Sources (all public):
- FFIEC Call Reports: RC-O, RC-M, RC-E, RC-B, RC-R schedules
- SEC EDGAR: 10-K/10-Q filings (fallback)
- FRED: Fed funds rate, macro variables
- UBPR: Pre-calculated regulatory ratios (optional)

Forecasting:
- ARDL models for each funding component
- Monte Carlo simulation on joint macro distribution
- Scenario analysis for stress testing
"""

from .indicator import (
    FundingStabilityIndicator,
    FundingStabilitySpec,
    FundingResilienceScorer,
    BankFundingProfile,
    DepositRateBetaCalculator,
)
from .forecast import (
    FundingStabilityForecaster,
    ARDLModel,
    ARDLSpec,
    ARDLResult,
    MonteCarloSimulator,
    PREDEFINED_SCENARIOS,
)
from .nowcast import (
    FundingStabilityNowcaster,
    NowcastResult,
    HighFrequencyProxyCalculator,
)
from .backtest import (
    FundingStabilityBacktester,
    BacktestConfig,
    BacktestResult,
    STRESS_PERIODS,
)
from .viz import FundingStabilityVisualizer
from .call_report_fetcher import (
    FundingMetrics,
    FFIECBulkDataFetcher,
    UBPRDataFetcher,
    SECFundingDataExtractor,
    MDRM_CODES,
)

__all__ = [
    # Main indicator
    "FundingStabilityIndicator",
    "FundingStabilitySpec",
    "FundingResilienceScorer",
    "BankFundingProfile",
    "DepositRateBetaCalculator",
    # Forecasting
    "FundingStabilityForecaster",
    "ARDLModel",
    "ARDLSpec",
    "ARDLResult",
    "MonteCarloSimulator",
    "PREDEFINED_SCENARIOS",
    # Nowcasting
    "FundingStabilityNowcaster",
    "NowcastResult",
    "HighFrequencyProxyCalculator",
    # Backtesting
    "FundingStabilityBacktester",
    "BacktestConfig",
    "BacktestResult",
    "STRESS_PERIODS",
    # Visualization
    "FundingStabilityVisualizer",
    # Data fetching
    "FundingMetrics",
    "FFIECBulkDataFetcher",
    "UBPRDataFetcher",
    "SECFundingDataExtractor",
    "MDRM_CODES",
]
