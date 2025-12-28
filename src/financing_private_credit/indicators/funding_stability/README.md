# Funding Stability Score

**Novel Insight 3:** Creates a "funding resilience" score that predicts which banks will constrain credit most aggressively in downturns.

## Motivation

The NY Fed Staff Report 1111 shows that banks respond procyclically to macro conditions. This indicator operationalizes that finding by creating a composite score that identifies funding vulnerabilities before they become crises.

> Banks with lower funding stability scores should:
> - Cut lending more in downturns
> - Have higher loan loss provisions
> - Underperform during credit boom periods

## Reformulated Score Formula

```python
Funding_Resilience_Score =
    w1 * (deposit_funding_ratio) +            # Stability factor
    w2 * (1 - wholesale_funding_ratio) +      # Less non-core
    w3 * (1 - uninsured_deposit_ratio) +      # Less run risk
    w4 * (1 - fhlb_advance_ratio) +           # Less desperation
    w5 * (1 - brokered_deposit_ratio) +       # Less hot money
    w6 * (1 - aoci_impact_ratio) +            # Less trapped capital
    w7 * (duration_match_score) +             # Better ALM
    w8 * (1 - rate_beta_deposits)             # Stickier deposits
```

## Score Components

### 1. Run Risk: Uninsured Deposit Ratio (The "SVB" Variable)

**Source:** Schedule RC-O (Other Data for Deposit Insurance), Memorandum Item 2

| MDRM Code | Description |
|-----------|-------------|
| RCON5597 | Estimated amount of uninsured deposits |
| RCON2200 | Total domestic deposits |

**Formula:** `Uninsured Deposits / Total Domestic Deposits`

**Interpretation:**
- \>50%: High run risk (heavy penalty)
- 30-50%: Moderate risk
- <30%: Low risk

### 2. Desperation Signal: FHLB Advance Utilization

**Source:** Schedule RC-M (Memoranda), Item 5.a

| MDRM Code | Description |
|-----------|-------------|
| RCFD2950 | FHLB advances |
| RCON2365 | Total liabilities |

**Formula:** `FHLB Advances / Total Liabilities`

**Interpretation:**
- \>10%: Desperation (bank has exhausted customer deposits)
- 5-10%: Elevated reliance
- <5%: Normal

### 3. Hot Money: Brokered Deposit Ratio

**Source:** Schedule RC-E (Deposit Liabilities), Memorandum Item 1.b

| MDRM Code | Description |
|-----------|-------------|
| RCONHK04 | Total brokered deposits |

**Formula:** `Brokered Deposits / Total Deposits`

**Interpretation:**
- Brokered deposits have zero loyalty and 100% rate sensitivity
- Refines historical rate beta by identifying future sensitivity

### 4. Trapped Capital: AOCI Impact Ratio

**Source:** Schedule RC-B (Securities) & RC-R (Regulatory Capital)

| MDRM Code | Description |
|-----------|-------------|
| RCFD8434 | HTM securities amortized cost |
| RCFD8435 | HTM securities fair value |
| RCFD8439 | AFS securities amortized cost |
| RCFD1773 | AFS securities fair value |
| RCFDA222 | Tangible common equity |

**Formula:** `(HTM Unrealized Loss + AFS Unrealized Loss) / Tangible Common Equity`

**Interpretation:**
- \>50% of TCE: Critical (market sees bank as fragile)
- Acts as "confidence multiplier" on other metrics

### 5. UBPR Shortcut: Net Non-Core Funding Dependence

For efficiency, you can pull the UBPR pre-calculated metric:

| UBPR Code | Description |
|-----------|-------------|
| UBPRE003 | Net Non-Core Funding Dependence |

This aggregates large time deposits, foreign deposits, and brokered deposits into one "Non-Core" number that matches regulatory standards.

## Data Sources

| Source | Schedule | Key MDRM Codes |
|--------|----------|----------------|
| FFIEC Call Report | RC-O | RCON5597 (uninsured deposits) |
| FFIEC Call Report | RC-M | RCFD2950 (FHLB advances) |
| FFIEC Call Report | RC-E | RCONHK04 (brokered deposits) |
| FFIEC Call Report | RC-B | RCFD8434, RCFD8435 (securities) |
| FFIEC Call Report | RC-R | RCFDA222 (tangible equity) |
| UBPR | - | UBPRE003 (non-core dependence) |
| SEC EDGAR | 10-K/10-Q | Fallback source |
| FRED | - | FEDFUNDS (for rate beta) |

## Module Structure

```
funding_stability/
├── __init__.py              # Package exports
├── indicator.py             # Core score calculation
├── call_report_fetcher.py   # FFIEC data fetching
├── forecast.py              # ARDL + Monte Carlo forecasting
├── nowcast.py               # High-frequency proxy updates
├── backtest.py              # Predictive power validation
├── viz.py                   # Vega-Altair visualizations
└── README.md                # This file
```

## Usage

### Basic Usage

```python
from financing_private_credit.indicators import get_indicator

# Get the indicator
indicator = get_indicator("funding_stability")

# Fetch data
data = indicator.fetch_data("2015-01-01")

# Calculate scores
result = indicator.calculate(data)

# View rankings
print(indicator.get_resilience_ranking())

# Check for stress
print(indicator.get_stress_indicators())
```

### Custom Weights

```python
from financing_private_credit.indicators.funding_stability import (
    FundingStabilityIndicator,
    FundingStabilitySpec,
)

# Custom specification with higher weight on run risk
spec = FundingStabilitySpec(
    name="run_risk_focus",
    description="Higher weight on uninsured deposits",
    weight_uninsured_deposits=0.25,  # Increased from 0.15
    weight_fhlb_advances=0.20,       # Increased
    weight_deposit_funding=0.10,     # Decreased
    # ... other weights adjusted
)

indicator = FundingStabilityIndicator()
result = indicator.calculate(data, spec=spec)
```

### Forecasting with ARDL

```python
from financing_private_credit.indicators.funding_stability import (
    FundingStabilityForecaster,
    PREDEFINED_SCENARIOS,
)

# Initialize forecaster
forecaster = FundingStabilityForecaster(
    funding_data=result.data,
    spec=indicator._spec
)

# Fit ARDL models for each component
fit_results = forecaster.fit_component_models(macro_data)

# Scenario analysis
scenarios = forecaster.scenario_analysis(PREDEFINED_SCENARIOS)
print(scenarios)

# Monte Carlo simulation
mc_results = forecaster.monte_carlo_forecast(
    baseline_macro=PREDEFINED_SCENARIOS["baseline"],
    n_simulations=5000
)
```

### Stress Testing

```python
from financing_private_credit.indicators.funding_stability import (
    FundingStabilityBacktester,
    STRESS_PERIODS,
)

backtester = FundingStabilityBacktester()

# Test predictive power during historical stress
stress_results = backtester.test_stress_periods(
    funding_data=result.data,
    outcome_data=outcome_df,
    stress_periods=STRESS_PERIODS
)

# Full backtest
backtest_result = backtester.backtest(result.data, outcome_df)
print(backtest_result.summary())
```

## Visualization

The visualizer generates 14 numbered charts:

| Chart | Description |
|-------|-------------|
| `01_resilience_ranking` | Bar chart of banks by score |
| `02_score_distribution` | Histogram of score distribution |
| `03_score_over_time` | Time series by bank |
| `04_component_breakdown` | Stacked bar of contributions |
| `05_risk_tier_composition` | Pie chart of tiers |
| `06_stress_indicators` | Heatmap of flags |
| `07_uninsured_deposit_heatmap` | Run risk exposure |
| `08_fhlb_dependence` | Desperation signal |
| `09_aoci_impact` | Trapped capital scatter |
| `10_forecast_scenarios` | Scenario comparison |
| `11_monte_carlo_distribution` | Simulation histogram |
| `12_scenario_comparison` | Grouped bar by scenario |
| `13_backtest_performance` | Historical accuracy |
| `14_executive_summary` | Key metrics dashboard |

## Predefined Scenarios

| Scenario | Fed Funds | Curve Slope | Credit Spread | Deposit Growth |
|----------|-----------|-------------|---------------|----------------|
| `baseline` | 5.25% | 0bp | 150bp | +2% |
| `rate_hike_100bp` | 6.25% | -25bp | 175bp | +1% |
| `rate_cut_100bp` | 4.25% | +50bp | 125bp | +3% |
| `recession` | 3.00% | +150bp | 350bp | -2% |
| `svb_stress` | 5.50% | -75bp | 250bp | -5% |
| `normalization` | 3.50% | +100bp | 100bp | +4% |

## Trading Signal

| Score Range | Risk Tier | Interpretation |
|-------------|-----------|----------------|
| 75-100 | Low | Resilient funding, likely to expand credit |
| 50-74 | Moderate | Some vulnerabilities, watch FHLB/uninsured |
| 25-49 | High | Significant stress, likely to constrain lending |
| 0-24 | Critical | Severe vulnerabilities, potential run risk |

## Key Metrics

- **Funding Resilience Score**: Composite score (0-100)
- **Risk Tier**: Categorical classification
- **Uninsured Deposit Ratio**: Run risk metric
- **FHLB Advance Ratio**: Desperation signal
- **AOCI Impact Ratio**: Trapped capital
- **Rate Beta**: Deposit stickiness

## Stress Flags

- **is_fhlb_dependent**: FHLB ratio > 10%
- **is_run_vulnerable**: Uninsured ratio > 50%
- **has_aoci_stress**: AOCI impact > 50% of TCE

## References

- NY Fed Staff Report 1111: "Financing Private Credit"
- FFIEC Call Report Instructions
- UBPR User's Guide
- SVB Failure Analysis (FDIC, 2023)
