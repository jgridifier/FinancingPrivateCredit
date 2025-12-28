"""
Bank-Specific Macro Sensitivity Indicator

Novel Insight: Banks have heterogeneous elasticities to macroeconomic variables.
This indicator measures bank-specific sensitivity to:
- Interest rate spreads (bond yield - loan rate)
- Output gap
- Inflation
- Credit spreads
- Financial conditions

Methodology:
- Uses APLR (Automatic Piecewise Linear Regression) from interpretml
  to capture non-linear relationships and interactions
- Monte Carlo simulation on joint distribution for industry-level forecasts
- Bank-by-bank attribution for identifying structural advantages

Data Sources (all public):
- SEC EDGAR 10-Q: Bank-level NIM, loan data
- FRED: Fed funds rate, Treasury yields, output gap proxies, inflation
- CBO: Potential GDP for output gap calculation
"""

from .indicator import BankMacroSensitivityIndicator
from .forecast import MacroSensitivityForecaster, MonteCarloSimulator
from .nowcast import MacroSensitivityNowcaster
from .backtest import MacroSensitivityBacktester

__all__ = [
    "BankMacroSensitivityIndicator",
    "MacroSensitivityForecaster",
    "MonteCarloSimulator",
    "MacroSensitivityNowcaster",
    "MacroSensitivityBacktester",
]
