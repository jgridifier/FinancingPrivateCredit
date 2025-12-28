# Bank-Specific Macro Sensitivity Indicator

**Novel Insight 1:** Banks have heterogeneous elasticities to macroeconomic variables. This indicator measures bank-specific sensitivity to interest rates, output gap, and other macro factors to identify structurally advantaged banks in different macro regimes.

## Motivation

The NY Fed Staff Report 1111 shows aggregate effects of macro variables on bank profitability. However, no one has mapped this to individual bank heterogeneity. This indicator addresses the question:

> Do different banks have different elasticities to the macro variables identified in the paper, and does this explain performance divergence?

## Methodology

For each bank *i*, we estimate:

```
NIM_i,t = f(rate_spread_t, output_gap_t, inflation_t, ...) + bank_effects_i + ε
```

Where `f()` is modeled using **APLR (Automatic Piecewise Linear Regression)** from the `interpretml` package to capture:
- Non-linear relationships (e.g., NIM compression at zero lower bound)
- Regime-dependent effects (crisis vs. normal periods)
- Interaction effects between macro variables

### Key Outputs

| Output | Description |
|--------|-------------|
| `β_i_rates` | Bank *i*'s sensitivity to rate spread |
| `γ_i_macro` | Bank *i*'s sensitivity to output gap |
| Regime classification | Which banks outperform in rising rates, recessions, etc. |

## Data Sources (All Public)

| Source | Data |
|--------|------|
| SEC EDGAR 10-Q | Quarterly NIM, loan data for each bank |
| FRED | Fed funds rate, 10-year Treasury yield, output gap proxies |
| CBO | Potential GDP for output gap calculation |
| BLS | Inflation (CPI) |

## Module Structure

```
bank_macro_sensitivity/
├── __init__.py       # Package exports
├── indicator.py      # Core APLR model for NIM elasticities
├── forecast.py       # Monte Carlo joint distribution simulation
├── nowcast.py        # Real-time estimation using high-frequency data
└── backtest.py       # Model validation framework
```

## Usage

### Basic Usage

```python
from financing_private_credit.indicators import get_indicator

# Get the indicator
indicator = get_indicator("bank_macro_sensitivity")

# Fetch data (SEC EDGAR + FRED)
data = indicator.fetch_data("2010-01-01")

# Calculate sensitivities
result = indicator.calculate(data)

# View rankings
print(result.data)  # Banks ranked by rate sensitivity
```

### Monte Carlo Forecasting

```python
from financing_private_credit import MacroSensitivityForecaster

# Initialize forecaster with fitted model
forecaster = MacroSensitivityForecaster(
    model=indicator._model,
    macro_data=data["macro_data"],
    n_simulations=1000
)

# Generate forecast under different scenarios
rising_rates = forecaster.forecast(horizon=4, scenario="rising_rates")
recession = forecaster.forecast(horizon=4, scenario="recession")

# View industry distribution
print(f"Rising rates: {rising_rates.industry_percentiles}")

# Bank-by-bank attribution
print(rising_rates.bank_contributions)
```

### Scenario Analysis

```python
# Compare banks across predefined scenarios
scenario_comparison = forecaster.scenario_analysis(
    scenarios=["baseline", "rising_rates", "falling_rates", "recession"]
)
print(scenario_comparison)
```

### Backtesting

```python
from financing_private_credit import MacroSensitivityBacktester
from financing_private_credit.indicators.bank_macro_sensitivity.indicator import MacroSensitivitySpec

# Load specification
spec = MacroSensitivitySpec.from_json("config/model_specs/bank_macro_sensitivity.json")

# Run backtest
backtester = MacroSensitivityBacktester(
    spec=spec,
    method="expanding",  # or "rolling"
    min_train_periods=20,
    test_periods=4
)

result = backtester.backtest(data["bank_panel"], data["macro_data"])
print(result.summary())
```

## Model Specifications

Three pre-configured specifications are available in `config/model_specs/`:

| Specification | Description |
|---------------|-------------|
| `bank_macro_sensitivity.json` | Default full model with 6 macro features |
| `bank_macro_sensitivity_minimal.json` | Just rate spread + output gap (paper's original) |
| `bank_macro_sensitivity_extended.json` | Adds housing and labor market indicators |

## Banks Covered

### Original (G-SIBs & Large Regionals)
JPM, BAC, C, WFC, GS, MS, USB, PNC, TFC, COF

### Expanded Coverage
SCHW (Charles Schwab), BK (BNY Mellon), STT (State Street), NTRS (Northern Trust), RJF (Raymond James)

**Note:** TD Bank and Barclays were excluded due to limited SEC data availability (foreign parent companies).

## Trading Signal

Banks with high rate sensitivity (`β_i_rates > 0`) are expected to outperform when rates rise. The regime classification provides actionable signals:

| Regime | Advantaged Banks | Signal |
|--------|-----------------|--------|
| Rising rates | High `β_i_rates` | Overweight |
| Falling rates | Low/negative `β_i_rates` | Overweight |
| Expansion | High `γ_i_macro` | Overweight |
| Recession | Low `γ_i_macro` | Overweight |

## References

- NY Fed Staff Report 1111: "Financing Private Credit"
- interpretml APLR: https://interpret.ml/docs/aplr.html
