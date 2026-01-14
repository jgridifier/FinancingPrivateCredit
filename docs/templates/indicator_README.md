# {Indicator Name}

> {One-line description of what this indicator measures}

**Author:** {Author Name}
**Version:** {Version}
**Status:** Production | Development | Experimental

## Overview

{2-3 paragraphs explaining what this indicator measures and why it matters for credit analysis. Include the economic intuition behind the indicator.}

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `{metric_1}` | {What it measures} | High = X, Low = Y |
| `{metric_2}` | {What it measures} | Threshold at Z |

## Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| SEC EDGAR | {What data from 10-K/10-Q} | Quarterly |
| FRED | {Which series} | {Frequency} |
| {Other} | {Description} | {Frequency} |

## Methodology

### Step 1: Data Preparation

{Describe how raw data is processed.}

### Step 2: Core Calculation

```
{Mathematical formula or algorithm description}
```

### Step 3: Interpretation/Scoring

{How results are interpreted, ranked, or classified.}

## Usage

```python
from financing_private_credit.indicators import get_indicator

# Initialize and run
indicator = get_indicator("{indicator_name}")
data = indicator.fetch_data("2015-01-01")
result = indicator.calculate(data)

# View results
print(result.data)
print(result.metadata)
```

## Configuration

Available spec parameters in `config/model_specs/{indicator_name}.json`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `{param_1}` | {type} | {default} | {description} |
| `{param_2}` | {type} | {default} | {description} |

Example spec file:

```json
{
    "name": "{indicator_name}_default",
    "description": "{description}",
    "{param_1}": {default},
    "{param_2}": {default}
}
```

## Forecasting

{If applicable, explain forecasting methodology:}
- Model type (ARDL, VAR, ML, etc.)
- Key features used
- Horizon supported
- Typical accuracy metrics

## Nowcasting

{If applicable, explain nowcasting approach:}
- Proxy variables used
- Update frequency
- Confidence decay methodology

## Visualizations

| Chart | Description |
|-------|-------------|
| Chart 1 | {Overview description} |
| Chart 2 | {Time series description} |

## Interpretation Guide

### High Values Indicate

- {Bullet point 1}
- {Bullet point 2}

### Low Values Indicate

- {Bullet point 1}
- {Bullet point 2}

### Warning Thresholds

| Level | Threshold | Action |
|-------|-----------|--------|
| Normal | < X | Monitor |
| Elevated | X - Y | Increased scrutiny |
| Warning | Y - Z | Risk review |
| Critical | > Z | Immediate attention |

## Limitations

- {Limitation 1}
- {Limitation 2}
- {Limitation 3}

## References

- {Paper citation}
- {Related research}
- {External documentation}

## Changelog

### v{version} ({date})
- {Change 1}
- {Change 2}
