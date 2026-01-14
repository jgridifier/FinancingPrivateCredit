# FASAR - Flex-Adjusted Syndicate Absorption Ratio

> Measures "trapped volume" risk - bank commitments weighted by rigidity relative to CLO market absorption capacity.

**Version:** 1.0.0
**Status:** Production

## Overview

FASAR (Flex-Adjusted Syndicate Absorption Ratio) quantifies the risk that banks become "stuck" with bridge loan commitments they cannot syndicate. This occurs when banks have rigid commitments (limited market flex) and the CLO market's absorption capacity is low.

The indicator combines commitment-level rigidity analysis with market-wide syndication velocity to identify banks at risk of holding "hung loans" - commitments that cannot be distributed and must remain on balance sheet.

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `fasar` | (Trapped Volume) / CLO Velocity | > 2.0 = High hung loan risk |
| `rigidity_score` | Commitment flexibility (0-1) | 1.0 = SunGard/No flex |
| `clo_velocity` | Market absorption rate | < 0.5 = Distressed market |

## Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| SEC EDGAR | Bridge commitment fee letters (8-K) | Event-driven |
| LCD/Pitchbook | CLO issuance data | Weekly |
| FRED | Credit spreads, rates | Daily |

## Methodology

### Step 1: Extract Commitments

Parse SEC filings for bridge commitment terms:
- Commitment amount
- Fee letter provisions (market flex, caps)
- SunGard/Limited conditionality clauses

### Step 2: Calculate Rigidity Score

For each commitment:
```
rigidity = 1.0 if sungard_clause or limited_conditionality else
           0.0 if has_market_outs else
           1.0 - flex_cap_pct if has_capped_flex else
           0.5  # Default for redacted terms
```

### Step 3: Compute FASAR

```
Trapped_Volume = sum(commitment_amount * rigidity_score)
CLO_Velocity = weekly_clo_issuance / base_capacity
FASAR = Trapped_Volume / CLO_Velocity
```

## Usage

```python
from financing_private_credit.indicators import get_indicator

# Initialize and run
indicator = get_indicator("fasar")
data = indicator.fetch_data("2015-01-01")
result = indicator.calculate(data)

# View FASAR by bank
print(result.data.select(["ticker", "fasar", "risk_level"]))

# Identify high-risk banks
high_risk = result.data.filter(pl.col("fasar") > 2.0)
```

## Configuration

Available in `config/model_specs/fasar.json`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `high_risk_threshold` | float | 2.0 | FASAR threshold for "Trapped" |
| `elevated_threshold` | float | 1.5 | FASAR threshold for "Elevated" |
| `base_syndication_capacity` | float | 5000.0 | Base weekly CLO capacity ($M) |
| `clo_velocity_lookback_weeks` | int | 4 | Weeks for velocity calculation |

## Interpretation Guide

### High FASAR (> 2.0) Indicates

- Bank has significant rigid commitments
- CLO market absorption is insufficient
- High probability of holding hung loans
- Potential balance sheet stress

### Low FASAR (< 1.0) Indicates

- Commitments have adequate flexibility
- Market is absorbing syndication normally
- Low hung loan risk

### Risk Classification

| Level | FASAR | Description | Action |
|-------|-------|-------------|--------|
| Normal | < 1.0 | Adequate flexibility or absorption | Monitor |
| Watch | 1.0 - 1.5 | Some rigidity exposure | Track closely |
| Elevated | 1.5 - 2.0 | Significant trapped volume | Stress test |
| Trapped | > 2.0 | High hung loan risk | Immediate review |

## Limitations

- Fee letter terms often redacted in public filings
- CLO issuance data may have reporting lag
- Does not capture private credit absorption
- Rigidity scoring involves judgment calls

## References

- NY Fed Staff Report 1111: "Financing Private Credit"
- LCD Quarterly Leveraged Lending Review
- Bank regulatory filings (FR Y-14Q)

## Changelog

### v1.0.0 (2026-01-14)
- Initial release with commitment parsing
- Rigidity scoring methodology
- CLO velocity integration
