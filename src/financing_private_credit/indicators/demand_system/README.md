# Demand System Indicator

> Replicates the core analysis from NY Fed Staff Report 1111: credit decomposition by lender type and crisis probability.

**Version:** 1.0.0
**Status:** Production

## Overview

The Demand System Indicator implements the core analytical framework from Boyarchenko & Elias (2024) "Financing Private Credit: The Role of Lender Type in Credit Booms."

The key finding: Credit expansions financed primarily by banks are associated with higher crisis probability than those financed by nonbanks. This indicator computes:

1. Credit decomposition by lender type (bank vs nonbank)
2. Supply elasticity estimation
3. Crisis probability computation
4. Schularick-Taylor credit expansion predictor

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `bank_share` | Bank financing share of total credit | > 60% = Bank-driven expansion |
| `credit_growth` | YoY total credit growth | > 10% = Credit boom |
| `crisis_probability` | Predicted crisis likelihood | > 20% = Elevated risk |
| `bank_elasticity` | Bank credit supply elasticity | Higher = More procyclical |

## Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| Flow of Funds (Z.1) | Credit by lender type | Quarterly |
| FRED | Macro indicators | Various |
| BIS | Cross-country credit data | Quarterly |

## Methodology

### Step 1: Credit Decomposition

Decompose total credit into bank and nonbank components:
```
Total_Credit = Bank_Credit + Nonbank_Credit
Bank_Share = Bank_Credit / Total_Credit
```

### Step 2: Supply Elasticity Estimation

Estimate supply elasticities using demand-supply system:
```
log(Credit_Bank) = alpha + beta_bank * log(GDP) + gamma * spread + e
log(Credit_Nonbank) = alpha + beta_nonbank * log(GDP) + gamma * spread + e
```

### Step 3: Crisis Probability

Following Schularick-Taylor methodology:
```
P(Crisis | Credit_Expansion) = f(credit_growth, bank_share, macro_conditions)
```

## Usage

```python
from financing_private_credit.indicators import get_indicator

# Initialize and run
indicator = get_indicator("demand_system")
data = indicator.fetch_data("2015-01-01")
result = indicator.calculate(data)

# View credit decomposition
print(result.data.select(["date", "bank_share", "credit_growth"]))

# Get elasticity results
elasticities = result.metadata["elasticities"]
print(f"Bank elasticity: {elasticities['bank_elasticity']:.2f}")
print(f"Nonbank elasticity: {elasticities['nonbank_elasticity']:.2f}")

# Check crisis probability
print(f"Crisis probability: {result.metadata['crisis_probability']:.1%}")
```

## Configuration

Available in `config/model_specs/demand_system.json`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `growth_periods` | int | 4 | Quarters for growth calculation |
| `cycle_window` | int | 40 | Quarters for trend (10 years) |
| `credit_growth_threshold` | float | 10.0 | YoY growth threshold (%) |
| `bank_share_threshold` | float | 60.0 | Bank share threshold (%) |
| `elasticity_method` | str | "ols" | Estimation method |

## Interpretation Guide

### Bank-Driven Expansion (Bank Share > 60%)

- Higher crisis probability
- More procyclical dynamics
- Watch for asset quality deterioration

### Nonbank-Driven Expansion (Bank Share < 40%)

- Lower crisis probability
- Different risk transmission channels
- Monitor shadow banking risks

### Credit Boom Conditions

| Indicator | Threshold | Risk Level |
|-----------|-----------|------------|
| Credit Growth | > 10% YoY | Elevated |
| Bank Share | > 60% | Higher risk |
| Combined | Both exceeded | Warning |

## Key Finding from Paper

> "Credit expansions financed primarily by banks are associated with
> a 3x higher crisis probability than those financed by nonbanks."

This asymmetry reflects:
- Bank leverage and deposit funding
- Regulatory capital constraints
- Procyclical lending behavior
- Interconnectedness with real economy

## Limitations

- Historical relationships may not hold in future
- Nonbank sector data less complete
- Crisis definition affects results
- Cross-country variation significant

## References

- Boyarchenko, N., & Elias, L. (2024). "Financing Private Credit: The Role of Lender Type in Credit Booms." NY Fed Staff Report 1111.
- Schularick, M., & Taylor, A. M. (2012). "Credit Booms Gone Bust." American Economic Review.
- Adrian, T., & Shin, H. S. (2010). "Liquidity and Leverage."

## Changelog

### v1.0.0 (2026-01-14)
- Initial release with paper replication
- Credit decomposition by lender type
- Elasticity estimation (OLS, IV, GMM)
- Crisis probability computation
