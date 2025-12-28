# Duration Mismatch as Predictive Signal

**Novel Insight 2:** The paper's finding on bond duration sensitivity (Figure 8b) predicts bank earnings volatility and stock returns.

## Motivation

The NY Fed Staff Report 1111 shows that duration decreases with bond yields at the aggregate level. This indicator tests whether this relationship translates to bank-specific risk:

> Do banks with more duration-sensitive portfolios have higher earnings volatility when yields move?

## Hypothesis

```
predicted_earnings_shock = duration_exposure × Δ(bond_yield)
```

Banks with higher predicted impact should experience:
- Higher earnings volatility
- More stock return volatility
- Larger NIM swings

## Methodology

### Step 1: Estimate Duration Exposure (Refined Approach)

Duration is estimated using **Schedule RC-B (Securities)** from FFIEC Call Reports (FFIEC 031/041) or **Schedule HC-B** from FR Y-9C for bank holding companies.

**Primary Method: Maturity Bucket Data (Memorandum Item 2)**

The Call Report provides securities by maturity/repricing bucket:

| Bucket | MDRM Code | Proxy Duration |
|--------|-----------|----------------|
| ≤1 year | RCFDA549 | 0.5 years |
| 1-5 years | RCFDA550 | 2.5 years |
| 5-10 years | RCFDA551 | 6.5 years |
| >10 years | RCFDA552 | 12.0 years |

**Weighted Average Duration Formula:**
```
Duration = (Bucket₁ × 0.5 + Bucket₂ × 2.5 + Bucket₃ × 6.5 + Bucket₄ × 12.0) / Total
```

**Example Calculation:**
For a bank with $100B securities (10% ≤1yr, 40% 1-5yr, 35% 5-10yr, 15% >10yr):
```
Duration = (10×0.5 + 40×2.5 + 35×6.5 + 15×12.0) / 100
         = (5 + 100 + 227.5 + 180) / 100
         = 5.125 years
```

**MBS Special Handling:**
MBS in Call Reports is already adjusted for prepayment/expected average life, so the bucket classification reflects true expected duration.

**Fallback Method:**
When Call Report data is unavailable, use AFS/HTM weighted averages:
- AFS securities: ~4.5 year average duration
- HTM securities: ~6.0 year average duration

### Step 2: Calculate Predicted Impact

```
DV01 = Duration × Portfolio_Value × 0.0001
predicted_impact = -Duration × Δ(10Y_yield) × Portfolio_Value
```

### Step 3: Test Predictive Relationship

Using **ARDL (Autoregressive Distributed Lag)** models:
- Lead-lag dynamics between duration and volatility
- Error correction mechanism
- Works with mixed I(0)/I(1) variables

## Forecasting Approach: ARDL vs APLR vs SARIMAX

| Model | When to Use | Pros | Cons |
|-------|-------------|------|------|
| **ARDL** | Lead-lag relationships | Captures "duration today → volatility next Q" | Assumes linearity |
| **APLR** | Non-linear sensitivity | Detects regime changes, hedging effects | Less explicit dynamics |
| **SARIMAX** | Seasonal patterns | Handles autocorrelation | Limited benefit for quarterly data |

**Recommendation:** ARDL as primary model, APLR for robustness checks.

## Data Sources (All Public)

| Source | Data |
|--------|------|
| **FFIEC Call Reports** | Schedule RC-B maturity buckets (FFIEC 031/041) |
| **FR Y-9C** | Schedule HC-B for bank holding companies |
| SEC EDGAR 10-K/10-Q | Securities portfolio (AFS/HTM), backup source |
| Yahoo Finance | Stock returns, earnings, volatility metrics |
| FRED | Bond yields (DGS1, DGS2, DGS5, DGS10, DGS30) |

### FFIEC Call Report Data

The FFIEC Central Data Repository (CDR) provides bulk downloads of Call Report data:
- **URL**: https://cdr.ffiec.gov/public/
- **Format**: ZIP files containing fixed-width or CSV data
- **Frequency**: Quarterly
- **Key MDRM Codes**: RCFDA549-A552 (maturity buckets)

## Module Structure

```
duration_mismatch/
├── __init__.py           # Package exports
├── indicator.py          # Core duration extraction and impact calculation
├── call_report_data.py   # FFIEC Call Report data fetcher (Schedule RC-B/HC-B)
├── forecast.py           # ARDL-based volatility forecasting
├── nowcast.py            # Real-time exposure estimation
├── backtest.py           # Model validation framework
├── viz.py                # Vega-Altair visualizations
└── README.md             # This file
```

## Usage

### Basic Usage

```python
from financing_private_credit.indicators import get_indicator

# Get the indicator
indicator = get_indicator("duration_mismatch")

# Fetch data (SEC EDGAR + Yahoo Finance + FRED)
data = indicator.fetch_data("2010-01-01")

# Calculate duration exposure and vulnerability
result = indicator.calculate(data)

# View vulnerability rankings
print(result.data)
```

### ARDL Forecasting

```python
from financing_private_credit.indicators.duration_mismatch import DurationMismatchForecaster

# Initialize forecaster
forecaster = DurationMismatchForecaster(
    duration_data=result.data,
    spec=indicator._spec
)

# Fit models
fit_stats = forecaster.fit(target="earnings_volatility")

# Generate forecasts
for ticker in ["JPM", "BAC", "C"]:
    forecasts = forecaster.forecast(ticker, horizon=4)
    print(f"{ticker}: {[f.point_forecast for f in forecasts]}")

# Scenario analysis (100bp rate shock)
impact = forecaster.scenario_analysis(rate_change=1.0, horizon=4)
print(impact)
```

### Visualization

```python
from financing_private_credit.indicators.duration_mismatch import DurationMismatchVisualizer

# Create visualizer
viz = DurationMismatchVisualizer(result.data)

# Generate all charts
charts = viz.generate_all_charts(
    forecast_data=forecaster.cross_bank_comparison(),
    scenario_data=impact
)

# Charts are numbered for storytelling:
# 01_duration_by_bank
# 02_duration_over_time
# 03_portfolio_composition
# 04_vulnerability_ranking
# ... etc.

# Save charts
viz.save_all_charts("output/charts/", format="html")
```

### Backtesting

```python
from financing_private_credit.indicators.duration_mismatch import DurationMismatchBacktester

# Initialize backtester
backtester = DurationMismatchBacktester(
    target="earnings_volatility",
    method="expanding",
    min_train_periods=12,
    test_periods=4
)

# Run backtest
result = backtester.backtest(result.data)
print(result.summary())

# Test at different horizons
horizon_results = backtester.test_predictive_power(
    result.data,
    horizons=[1, 2, 4, 8]
)
print(horizon_results)
```

## Visualization Outputs

The visualizer generates numbered charts for storytelling:

| Chart | Description |
|-------|-------------|
| `01_duration_by_bank` | Bar chart of current duration exposure |
| `02_duration_over_time` | Time series of duration changes |
| `03_portfolio_composition` | AFS vs HTM breakdown |
| `04_vulnerability_ranking` | Banks ranked by vulnerability |
| `05_duration_vs_volatility_scatter` | Scatter with trend line |
| `06_dv01_comparison` | Dollar value of 1bp by bank |
| `07_yield_sensitivity_heatmap` | Sensitivity to yield curve points |
| `08_scenario_impact` | Impact under rate scenarios |
| `09_predicted_vs_actual` | Model validation scatter |
| `10_forecast_comparison` | Multi-bank forecast lines |
| `11_forecast_uncertainty` | Error bars for forecasts |
| `12_backtest_performance` | Rolling MAE over time |
| `13_executive_summary` | Key metrics summary |

## Banks Covered

### Original (G-SIBs & Large Regionals)
JPM, BAC, C, WFC, GS, MS, USB, PNC, TFC, COF

### Expanded Coverage
SCHW (Charles Schwab), BK (BNY Mellon), STT (State Street), NTRS (Northern Trust), RJF (Raymond James)

## Trading Signal

Banks with high duration exposure are vulnerable to rate volatility:

| Scenario | Signal |
|----------|--------|
| Rising rates expected | Underweight high-duration banks |
| Falling rates expected | Overweight high-duration banks |
| Rate volatility increasing | Reduce exposure to vulnerable banks |

## Model Specifications

Three pre-configured specifications in `config/model_specs/`:

| Specification | Description |
|---------------|-------------|
| `duration_mismatch.json` | Default indicator settings |
| `duration_mismatch_ardl.json` | ARDL model for earnings volatility |
| `duration_mismatch_stock_vol.json` | ARDL model for stock volatility |

## Key Metrics

- **Estimated Duration**: Modified duration in years (from bucket methodology)
- **DV01**: Dollar value impact of 1bp rate move (millions)
- **Predicted Impact**: Earnings impact from rate changes
- **Vulnerability Score**: Combined duration × volatility metric
- **Duration Method**: "bucket" (Call Report), "fallback" (AFS/HTM avg), or "synthetic"
- **Barbell Strategy Flag**: True if high short-term AND high long-term allocation
- **Bucket Distribution**: % in each maturity bucket (≤1yr, 1-5yr, 5-10yr, >10yr)

## References

- NY Fed Staff Report 1111: "Financing Private Credit" (Figure 8b)
- Hamilton, J.D. (1994). "Time Series Analysis" (ARDL methodology)
