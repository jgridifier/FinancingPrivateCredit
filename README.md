# Financial Indicator Framework

A modular indicator framework for credit and financing analysis. Originally based on [NY Fed Staff Report 1111: Financing Private Credit](https://www.newyorkfed.org/research/staff_reports/sr1111), now extended to support general financial indicator development.

## Overview

This framework provides:
- **Modular Indicator Architecture**: Plug-and-play indicators with standardized interfaces
- **Enhanced Model Specification System**: Per-ticker and component-level model configurations
- **Monte Carlo Forecasting**: ARDL models with joint distribution simulation
- **Real-time Nowcasting**: High-frequency proxy updates between quarterly releases
- **Interactive Visualizations**: Numbered Vega-Altair charts for storytelling

## Available Indicators

| Indicator | Description | Key Insight |
|-----------|-------------|-------------|
| `demand_system` | Original paper replication | Bank-financed credit expansions → higher crisis probability |
| `credit_boom` | Credit Boom Leading Indicator | LIS predicts provisions 3-4 years ahead |
| `bank_macro_sensitivity` | Bank-specific macro elasticities | Heterogeneous NIM responses to rate changes |
| `duration_mismatch` | Duration exposure signal | Predicted earnings impact from yield changes |
| `funding_stability` | Funding resilience score | SVB-style run risk prediction |
| `variance_decomposition` | Cross-bank variance analysis | Systematic vs idiosyncratic credit risk |

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from financing_private_credit.indicators import get_indicator, list_indicators

# List available indicators
print(list_indicators())

# Get and run an indicator
indicator = get_indicator("funding_stability")
data = indicator.fetch_data("2015-01-01")
result = indicator.calculate(data)

# View rankings
print(result.data)
```

### Using Custom Model Specifications

```python
from financing_private_credit.core import ModelSpecRegistry

# Load multi-ticker spec file
registry = ModelSpecRegistry.from_json("config/model_specs/funding_stability_components.json")

# Get JPM-specific spec for uninsured deposits
spec = registry.get_component_spec("JPM", "uninsured_deposit_ratio")

# Fall back to default if ticker not defined
spec = registry.get_component_spec("UNKNOWN", "fhlb_advance_ratio")  # Uses "_" default
```

### Forecasting with Monte Carlo

```python
from financing_private_credit.indicators.funding_stability import (
    FundingStabilityForecaster,
    PREDEFINED_SCENARIOS
)

# Initialize forecaster
forecaster = FundingStabilityForecaster(result.data)

# Fit ARDL models for each component
fit_results = forecaster.fit_component_models(macro_data)

# Run Monte Carlo simulation
mc_results = forecaster.monte_carlo_forecast(
    baseline_macro=PREDEFINED_SCENARIOS["baseline"],
    n_simulations=5000
)
```

## Model Specification System

The framework supports flexible model specifications with per-ticker customization:

```json
{
    "_": {
        "uninsured_deposit_ratio": {
            "name": "default",
            "ar_lags": 2,
            "dist_lags": 2,
            "exog_vars": ["fed_funds_rate", "yield_curve_slope"]
        }
    },
    "JPM": {
        "uninsured_deposit_ratio": {
            "name": "jpm_specific",
            "ar_lags": 3,
            "exog_vars": ["fed_funds_rate", "yield_curve_slope", "vix"]
        }
    }
}
```

**Key Features:**
- `_` key defines the default specification
- Ticker-specific overrides (e.g., `"JPM"`, `"BAC"`)
- Component-level specs (e.g., `uninsured_deposit_ratio`, `fhlb_advance_ratio`)
- Grid search for hyperparameter optimization

## Bank Coverage

### Tier 1: G-SIBs
JPM, BAC, C, WFC

### Tier 2: Large Banks
GS, MS, BK, STT

### Tier 3: Regional & Specialty
USB, PNC, TFC, COF, SCHW, NTRS, RJF

## Data Sources

| Source | Data | Usage |
|--------|------|-------|
| **FFIEC Call Reports** | Schedule RC-O, RC-M, RC-E, RC-B, RC-R | Uninsured deposits, FHLB advances, securities |
| **SEC EDGAR** | 10-K/10-Q XBRL filings | NIM, loans, deposits, earnings |
| **FRED** | Macro series | Fed funds, yields, GDP, financial conditions |
| **Yahoo Finance** | Stock prices, earnings | Volatility, beta calculations |

## Project Structure

```
financing-private-credit/
├── src/financing_private_credit/
│   ├── core/                       # Core infrastructure
│   │   └── model_specs.py          # Enhanced spec system
│   │
│   ├── indicators/                 # Indicator implementations
│   │   ├── _template/              # Template for new indicators
│   │   ├── demand_system/          # Paper replication
│   │   ├── credit_boom/            # LIS-based indicator
│   │   ├── bank_macro_sensitivity/ # NIM elasticities
│   │   ├── duration_mismatch/      # Duration exposure
│   │   ├── funding_stability/      # Funding resilience
│   │   └── variance_decomposition/ # Cross-bank variance
│   │
│   ├── bank_data.py                # Bank-level data (TARGET_BANKS)
│   ├── cache.py                    # Data caching
│   ├── data.py                     # FRED data fetching
│   └── macro.py                    # Macro data
│
├── config/model_specs/             # Model specifications
├── tests/
├── CONTRIBUTING.md                 # How to add new indicators
└── README.md
```

## Adding New Indicators

See [CONTRIBUTING.md](CONTRIBUTING.md) for a complete guide.

Quick start:
```bash
# Copy the template
cp -r src/financing_private_credit/indicators/_template \
      src/financing_private_credit/indicators/my_indicator

# Edit indicator.py, register with @register_indicator("my_indicator")
# Add to indicators/__init__.py
```

## Key Concepts

### Funding Stability Score Components

| Component | Source | Interpretation |
|-----------|--------|----------------|
| Uninsured Deposits | RC-O Memo 2 | Run risk (>50% = high) |
| FHLB Advances | RC-M Item 5.a | Desperation signal (>10% = concern) |
| Brokered Deposits | RC-E Memo 1.b | Hot money (rate-sensitive) |
| AOCI Impact | RC-B + RC-R | Trapped capital (>50% TCE = critical) |

### Predefined Stress Scenarios

| Scenario | Fed Funds | Curve Slope | Credit Spread |
|----------|-----------|-------------|---------------|
| baseline | 5.25% | 0bp | 150bp |
| rate_hike_100bp | 6.25% | -25bp | 175bp |
| recession | 3.00% | +150bp | 350bp |
| svb_stress | 5.50% | -75bp | 250bp |

## References

- Boyarchenko, N., & Elias, L. (2024). [Financing Private Credit](https://www.newyorkfed.org/research/staff_reports/sr1111). NY Fed Staff Reports, No. 1111.
- Schularick, M., & Taylor, A. M. (2012). Credit booms gone bust. *American Economic Review*.
- FFIEC Call Report Instructions
- SVB Failure Analysis (FDIC, 2023)

## License

MIT
