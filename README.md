# Financing Private Credit

Reproduction and extension of [NY Fed Staff Report 1111: Financing Private Credit](https://www.newyorkfed.org/research/staff_reports/sr1111) by Nina Boyarchenko and Leonardo Elias (August 2024).

## Overview

This project reproduces the methodology from the NY Fed paper and extends it with:
- **Real-time nowcasting** using weekly bank credit data
- **Financial conditions monitoring** for credit environment assessment
- **Interactive visualizations** with Vega-Altair

### Key Finding from the Paper

> "The sectoral composition of lenders financing a credit expansion is a key determinant for subsequent real activity and crisis probability."

Bank credit is more sensitive to economic downturns than nonbank credit, implying that credit expansions financed primarily by banks carry higher crisis risk.

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from financing_private_credit import PrivateCreditData, DemandSystemModel

# Fetch data from FRED
data = PrivateCreditData(start_date="1990-01-01")
raw = data.fetch_all()

# Compute bank vs nonbank decomposition
decomposed = data.compute_credit_decomposition()
shares = data.get_lender_shares()

# Estimate supply elasticities (demand system approach)
model = DemandSystemModel(decomposed.drop_nulls())
results = model.estimate_full_system()

print(f"Bank elasticity: {results['supply_elasticities'].bank_elasticity:.3f}")
print(f"Nonbank elasticity: {results['supply_elasticities'].nonbank_elasticity:.3f}")
```

## Data Sources

All data is sourced from [FRED](https://fred.stlouisfed.org/) (Federal Reserve Economic Data):

| Series | Description | Frequency |
|--------|-------------|-----------|
| CRDQUSAPABIS | Total Credit to Private Non-Financial Sector (BIS) | Quarterly |
| CMDEBT | Household Debt | Quarterly |
| TBSDODNS | Nonfinancial Business Debt | Quarterly |
| TOTLL | Total Loans & Leases, Commercial Banks | Weekly |
| BUSLOANS | Commercial & Industrial Loans | Weekly |
| NFCI | Chicago Fed Financial Conditions Index | Weekly |

## Methodology

### 1. Credit Decomposition

Decompose total private credit by lender type:
- **Banks**: Commercial banks, credit unions, savings institutions
- **Nonbanks**: Shadow banks (MMFs, ABS issuers, broker-dealers, finance companies), insurance companies, pension funds

### 2. Demand System Approach

Following Boyarchenko & Elias (2024), we jointly model credit demand and supply:
- Estimate supply elasticities for each lender type
- Show banks are more procyclical than nonbanks
- Compute equilibrium elasticities

### 3. Crisis Probability Indicator

Based on the paper's finding:
- High credit growth + High bank share = Elevated crisis risk

### 4. Nowcasting Extension

Use weekly H.8 bank credit data to nowcast:
- Current quarter credit conditions
- Bank vs nonbank credit growth
- Financial stress indicators

## Project Structure

```
financing-private-credit/
├── src/financing_private_credit/
│   ├── data.py       # FRED data fetching & processing
│   ├── analysis.py   # Demand system & elasticity estimation
│   ├── nowcast.py    # High-frequency nowcasting
│   └── viz.py        # Vega-Altair visualizations
├── notebooks/
│   └── 01_reproduce_fed_methodology.ipynb
├── pyproject.toml
└── README.md
```

## References

- Boyarchenko, N., & Elias, L. (2024). [Financing Private Credit](https://www.newyorkfed.org/research/staff_reports/sr1111). Federal Reserve Bank of New York Staff Reports, No. 1111.
- Schularick, M., & Taylor, A. M. (2012). Credit booms gone bust: Monetary policy, leverage cycles, and financial crises, 1870-2008. *American Economic Review*.
- Greenwood, R., Hanson, S., Shleifer, A., & Sørensen, J. A. (2022). Predictable financial crises. *Journal of Finance*.

## License

MIT
