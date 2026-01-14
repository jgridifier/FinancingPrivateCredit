# Cross-Bank Variance Decomposition

> Decomposes bank loan growth into macro, size, allocation, and idiosyncratic components to reveal lending strategy.

**Version:** 1.0.0
**Status:** Production

## Overview

The Cross-Bank Variance Decomposition indicator extends the methodology from NY Fed Staff Report 1111 (Tables 5-6) from country-level to individual banks. It decomposes each bank's quarterly loan growth into four distinct components:

1. **Macroeconomic (M)**: Sensitivity to economic conditions
2. **Size Growth (S)**: Proportional scaling with balance sheet
3. **Portfolio Allocation (A)**: Active rebalancing decisions
4. **Idiosyncratic (e)**: Bank-specific factors

This decomposition reveals whether a bank is "macro-driven" (high beta to cycle) versus "strategically-driven" (active portfolio management), enabling better risk assessment and peer comparison.

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `macro_pct` | % of variance from macro | > 50% = Macro follower |
| `allocation_pct` | % from allocation decisions | > 30% = Strategic allocator |
| `idiosyncratic_pct` | % from bank-specific factors | > 40% = Specialist |
| `archetype` | Bank classification | See archetypes below |

## Bank Archetypes

| Archetype | Dominant Component | Risk Profile | Investment Implication |
|-----------|-------------------|--------------|----------------------|
| Macro Follower | Macro > 50% | High beta, procyclical | Amplifies booms/busts |
| Strategic Allocator | Allocation > 30% | Unpredictable timing | Monitor management |
| Steady Grower | Size > 40% | Lower volatility | Defensive, lower beta |
| Idiosyncratic Specialist | Idio > 40% | Bank-specific risks | Deep fundamental analysis |

## Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| SEC EDGAR | Bank-level loan portfolio data | Quarterly |
| FRED | Macro variables (GDP, CPI, rates) | Monthly/Quarterly |
| FRED H.8 | System-wide loan aggregates | Weekly |

## Methodology

### Step 1: Construct Components

For each bank and quarter:

**Macro Component (M):**
```
M_{bank,t} = beta_macro * macro_factor_t
```

**Size Component (S):**
```
S_{bank,t} = (assets_growth_{bank,t}) * portfolio_share_{bank}
```

**Allocation Component (A):**
```
A_{bank,t} = sum(category_weight_change_{bank,t} * category_return_t)
```

### Step 2: Residual as Idiosyncratic

```
e_{bank,t} = loan_growth_{bank,t} - M_{bank,t} - S_{bank,t} - A_{bank,t}
```

### Step 3: Variance Decomposition

Compute variance shares:
```
var_share_M = var(M) / var(loan_growth)
var_share_S = var(S) / var(loan_growth)
var_share_A = var(A) / var(loan_growth)
var_share_e = var(e) / var(loan_growth)
```

## Usage

```python
from financing_private_credit.indicators import get_indicator

# Initialize and run
indicator = get_indicator("variance_decomposition")
data = indicator.fetch_data("2015-01-01")
result = indicator.calculate(data)

# View decomposition summary
print(result.data)

# Get bank archetypes
archetypes = result.metadata["archetypes"]
for bank, archetype in archetypes.items():
    print(f"{bank}: {archetype['name']}")
```

## Configuration

Available in `config/model_specs/variance_decomposition.json`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `macro_threshold` | float | 0.5 | Threshold for Macro Follower |
| `allocation_threshold` | float | 0.3 | Threshold for Strategic Allocator |
| `idiosyncratic_threshold` | float | 0.4 | Threshold for Specialist |
| `lookback_quarters` | int | 20 | Quarters for variance calculation |

## Interpretation Guide

### High Macro Share (> 50%)

- Bank lending highly correlated with economic cycle
- Expect procyclical behavior
- Higher systematic risk

### High Allocation Share (> 30%)

- Active portfolio rebalancing
- Management decisions drive volatility
- Monitor strategy shifts

### High Idiosyncratic Share (> 40%)

- Bank-specific factors dominate
- Less predictable from macro models
- Requires fundamental analysis

## Limitations

- Requires sufficient history for variance estimation
- Covariance terms can complicate interpretation
- Classification thresholds are somewhat arbitrary
- Does not capture within-quarter dynamics

## References

- Boyarchenko, N., & Elias, L. (2024). "Financing Private Credit." NY Fed Staff Report 1111, Tables 5-6.
- Growth accounting literature

## Changelog

### v1.0.0 (2026-01-14)
- Initial release with four-component decomposition
- Bank archetype classification
- Aggregate variance shares
