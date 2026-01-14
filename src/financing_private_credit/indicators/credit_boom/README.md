# Credit Boom Leading Indicator (LIS)

> Identifies banks with aggressive lending behavior that historically precede elevated credit losses.

**Version:** 1.0.0
**Status:** Production

## Overview

The Credit Boom Leading Indicator implements the Lending Intensity Score (LIS) methodology from NY Fed Staff Report 1111. This indicator identifies banks whose loan growth significantly exceeds the system average, standardized by cross-sectional volatility.

Banks with persistently high LIS scores have historically experienced elevated credit losses in subsequent periods. The indicator serves as an early warning system for credit risk buildup at individual banks.

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `lis` | (Bank Loan Growth - System Average) / System Std Dev | > 2.0 = Aggressive lending |
| `cumulative_lis` | Rolling 12-quarter sum of LIS | > 8.0 = Sustained aggression |
| `provision_forecast` | APLR-predicted provision rate | Higher = More expected losses |

## Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| SEC EDGAR | Bank-level loan and provision data (10-K/10-Q) | Quarterly |
| FRED H.8 | System-wide bank credit aggregates | Weekly |

## Methodology

### Step 1: Calculate Loan Growth

For each bank, calculate quarterly loan growth:
```
loan_growth_{bank,t} = (loans_{bank,t} - loans_{bank,t-1}) / loans_{bank,t-1}
```

### Step 2: Cross-Sectional Standardization

Compute the Lending Intensity Score:
```
LIS_{bank,t} = (loan_growth_{bank,t} - mean(loan_growth_t)) / std(loan_growth_t)
```

### Step 3: Cumulative Score

Rolling sum over 12 quarters to capture sustained patterns:
```
Cumulative_LIS_{bank,t} = sum(LIS_{bank,t-11:t})
```

## Usage

```python
from financing_private_credit.indicators import get_indicator

# Initialize and run
indicator = get_indicator("credit_boom")
data = indicator.fetch_data("2015-01-01")
result = indicator.calculate(data)

# View LIS scores
print(result.data)

# Check warning levels
warnings = result.data.filter(pl.col("lis") > 2.0)
```

## Configuration

Available in `config/model_specs/credit_boom.json`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lookback_quarters` | int | 12 | Quarters for cumulative LIS |
| `warning_threshold` | float | 2.0 | LIS warning threshold |
| `critical_threshold` | float | 3.0 | LIS critical threshold |

## Nowcasting

Nowcasting uses high-frequency proxies to adjust LIS between quarterly releases:

- **Stock Returns**: Bank stock performance vs peers
- **CDS Spreads**: Credit default swap movements
- **H.8 Data**: Weekly Fed bank credit data

## Interpretation Guide

### High LIS (> 2.0) Indicates

- Loan growth significantly above peers
- Potential credit quality deterioration ahead
- May reflect aggressive risk appetite

### Low LIS (< -2.0) Indicates

- Loan growth significantly below peers
- Conservative lending stance
- May indicate risk aversion or capacity constraints

### Warning Thresholds

| Level | LIS Threshold | Cumulative LIS | Action |
|-------|---------------|----------------|--------|
| Normal | -1.0 to 1.0 | < 4.0 | Monitor |
| Elevated | 1.0 to 2.0 | 4.0 - 6.0 | Increased scrutiny |
| Warning | 2.0 to 3.0 | 6.0 - 8.0 | Risk review |
| Critical | > 3.0 | > 8.0 | Immediate attention |

## References

- Boyarchenko, N., & Elias, L. (2024). "Financing Private Credit: The Role of Lender Type in Credit Booms." NY Fed Staff Report 1111.
- Schularick, M., & Taylor, A. M. (2012). "Credit Booms Gone Bust."

## Changelog

### v1.0.0 (2026-01-14)
- Initial release with LIS calculation
- Nowcasting support via stock and CDS proxies
- APLR-based provision forecasting
