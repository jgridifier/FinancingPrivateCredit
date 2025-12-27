# Financing Private Credit

Reproduction and extension of [NY Fed Staff Report 1111: Financing Private Credit](https://www.newyorkfed.org/research/staff_reports/sr1111) by Nina Boyarchenko and Leonardo Elias (August 2024).

## Overview

This project reproduces the methodology from the NY Fed paper and extends it with:
- **Credit Boom Leading Indicator**: Predict bank provisions 3-4 years ahead
- **Lending Intensity Score (LIS)**: Measure bank aggressiveness vs. peers
- **ARDL & SARIMAX Models**: Panel and time-series forecasting
- **Real-time nowcasting** using weekly bank credit data
- **Interactive visualizations** with Vega-Altair

### Key Finding from the Paper

> "The sectoral composition of lenders financing a credit expansion is a key determinant for subsequent real activity and crisis probability."

Banks that expand credit aggressively today will have higher provisions 3-4 years later.

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage: Credit Decomposition

```python
from financing_private_credit import PrivateCreditData, DemandSystemModel

# Fetch data from FRED
data = PrivateCreditData(start_date="1990-01-01")
raw = data.fetch_all()

# Compute bank vs nonbank decomposition
decomposed = data.compute_credit_decomposition()

# Estimate supply elasticities
model = DemandSystemModel(decomposed.drop_nulls())
results = model.estimate_full_system()

print(f"Bank elasticity: {results['supply_elasticities'].bank_elasticity:.3f}")
```

### Credit Boom Indicator

```python
from financing_private_credit import (
    SyntheticBankData, LendingIntensityScore, CreditBoomIndicator
)
import polars as pl

# Generate bank panel data (or use real SEC data)
synth = SyntheticBankData()
bank_panel = synth.generate_panel(n_banks=10)

# Compute system average
system_avg = bank_panel.group_by('date').agg(
    pl.col('loan_growth_yoy').mean().alias('loan_growth_yoy')
)

# Calculate Lending Intensity Score
lis = LendingIntensityScore(bank_panel, system_avg)
lis_data = lis.compute_lis()

# Get current signals
signals = lis.get_current_signals(threshold=1.0)
print(signals.select(['ticker', 'lis', 'elevated_lis']))
```

## Methodology

### 1. Lending Intensity Score (LIS)

Measures each bank's lending aggressiveness relative to the system:

```
LIS = (Bank_Loan_Growth - System_Loan_Growth) / σ(System_Growth)
```

- **LIS > 1**: Warning - Bank lending 1+ SDs above average
- **LIS > 2**: Alert - Bank lending 2+ SDs above average
- **Cumulative LIS**: Sum over 12 quarters for sustained exposure

### 2. ARDL Model

Autoregressive Distributed Lag model testing the paper's hypothesis:

```
Provision_{t} = α + Σ β_j Provision_{t-j} + Σ γ_h LIS_{t-h} + Controls + ε
```

Key lags: h = 12, 14, 16, 18, 20 quarters (3-5 years ahead)

### 3. SARIMAX Forecasting

Bank-specific time series forecasting with:
- Seasonal patterns (quarterly)
- Exogenous variables (LIS, macro conditions)
- Confidence intervals

### 4. Early Warning System

Combines LIS levels with ARDL coefficients to generate risk classifications:

| Risk Level | Criteria |
|------------|----------|
| HIGH | LIS > 2 OR Cumulative LIS > 8 |
| MEDIUM | LIS > 1 OR Cumulative LIS > 4 |
| LOW | LIS < 1 AND Cumulative LIS < 4 |

## Data Sources

### FRED (Federal Reserve Economic Data)

| Series | Description | Frequency |
|--------|-------------|-----------|
| CRDQUSAPABIS | Total Credit to Private Non-Financial Sector | Quarterly |
| TOTLL | Total Loans & Leases, Commercial Banks | Weekly |
| GDP, GDPPOT | GDP and Potential GDP | Quarterly |
| BAA10Y | Baa Corporate Spread | Daily |
| NFCI | Financial Conditions Index | Weekly |

### Bank-Level Data (SEC EDGAR / Synthetic)

- Total loans and loan growth
- Provision for credit losses
- Non-performing loans (NPL)
- Allowance for credit losses

## Project Structure

```
financing-private-credit/
├── src/financing_private_credit/
│   ├── data.py              # FRED data fetching & processing
│   ├── analysis.py          # Demand system & elasticity estimation
│   ├── nowcast.py           # High-frequency nowcasting
│   ├── macro.py             # Macro & H.8 system data
│   ├── bank_data.py         # Bank-level data collection
│   ├── leading_indicator.py # LIS, ARDL, SARIMAX models
│   └── viz.py               # Vega-Altair visualizations
├── notebooks/
│   ├── 01_reproduce_fed_methodology.ipynb
│   └── 02_credit_boom_indicator.ipynb
├── pyproject.toml
└── README.md
```

## Notebooks

1. **01_reproduce_fed_methodology.ipynb**: Reproduce the paper's core findings
2. **02_credit_boom_indicator.ipynb**: Full credit boom early warning system

## References

- Boyarchenko, N., & Elias, L. (2024). [Financing Private Credit](https://www.newyorkfed.org/research/staff_reports/sr1111). Federal Reserve Bank of New York Staff Reports, No. 1111.
- Schularick, M., & Taylor, A. M. (2012). Credit booms gone bust. *American Economic Review*.
- Greenwood, R., et al. (2022). Predictable financial crises. *Journal of Finance*.

## License

MIT
