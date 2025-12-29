# Template Indicator

> Replace this with a one-line description of your indicator.

## Overview

Describe what this indicator measures and why it matters for credit analysis.

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `metric_1` | What it measures | High = X, Low = Y |
| `metric_2` | What it measures | Threshold at Z |

## Data Sources

- **SEC EDGAR**: [What data from 10-K/10-Q]
- **FRED**: [Which series]
- **Other**: [Any additional sources]

## Methodology

### Step 1: Data Preparation

Describe how raw data is processed.

### Step 2: Calculation

```
Formula or algorithm description
```

### Step 3: Scoring/Ranking

How results are interpreted or ranked.

## Usage

```python
from financing_private_credit.indicators import get_indicator

# Initialize and run
indicator = get_indicator("template")
data = indicator.fetch_data("2015-01-01")
result = indicator.calculate(data)

# View results
print(result.data)
```

## Model Specifications

Default configuration in `config/model_specs/template.json`:

```json
{
    "name": "template_default",
    "window": 20,
    "threshold": 0.5
}
```

## Visualizations

| Chart | Description |
|-------|-------------|
| Chart 1 | Overview across banks |
| Chart 2 | Time series detail |

## References

- Paper or source citations
- Related research
