# Prime Brokerage Leverage Lead Indicator (PB-LLI) Market Report
## Q2 2025 Assessment with Shadow Nowcast Extension | Report Date: January 16, 2026

---

## Executive Summary

**Current Signal: ACCELERATING (Positive)**

The PB-LLI composite signal stands at **+3.28%** with a z-score of **+0.37**, indicating Improving prime balances/revenue momentum likely 1-2 quarters ahead.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| PB_Lead | +3.28% | Above-average growth signal |
| Z-Score | +0.37 | Positive |
| Regime | Accelerating | Upper tercile |
| Stress Flag | OFF | No acute stress |

---

## 1. Official Data Analysis (Q2 2025)

### 1.1 Quarterly Anchor Index

The Fed Z.1 data through Q2 2025 shows:

- **PB Intensity (HF Borrowing/HF Equity)**: **30.8%**
- **PB Intensity Growth**: **+9.61% QoQ**
- **Dealer Supply Growth**: **+6.01% QoQ**

### 1.2 Attribution Decomposition

| Component | Contribution | Share |
|-----------|-------------|-------|
| HF Demand (Nowcast) | +1.27% | 38.8% |
| Dealer Supply | +1.50% | 45.9% |
| Lagged Intensity | +0.50% | 15.4% |

---

## 2. Shadow Nowcast Extension

### 2.1 Shadow Model Diagnostics

The shadow nowcast model extends official Z.1 data using weekly leverage appetite proxies.

| Metric | Value |
|--------|-------|
| Model R² | 0.041 |
| Residual Std | 6.30% |
| Training Observations | 50 |
| Ridge Penalty (λ) | 5.0 |

### 2.2 Shadow/Nowcast Estimates

These estimates extend the indicator beyond the official Z.1 release:

| Quarter | Type | PB Intensity Growth | 95% CI | Completeness |
|---------|------|---------------------|--------|--------------|
| Q3'25 | Shadow | +1.53% | [-15.95%, +19.00%] | 100% weekly |
| Q4'25 | Nowcast | +0.76% | [-17.56%, +19.09%] | 100% weekly |
| Q1'26 | Nowcast | +0.65% | [-31.81%, +33.10%] | 8% weekly |

**Interpretation**: Shadow estimates bridge the gap between the last official Z.1 release and the current quarter. These estimates use weekly CFTC positioning and NY Fed dealer statistics as leading indicators. Uncertainty bands are wider for quarters with less data coverage.

### 2.3 Bank Disclosure Pulse (GS / MS / JPM)

The disclosure pulse incorporates prime-relevant metrics from major bank earnings:

| Bank | Metric | Source | Prime Attribution |
|------|--------|--------|-------------------|
| **Goldman Sachs** | Equities Financing Net Revenues | SEC Exhibit 99.2 | "Prime and portfolio financing" |
| **Morgan Stanley** | Equity Net Revenues | Earnings PDF | "Financing revenues from higher client balances in prime brokerage" |
| **JPMorgan** | Equity Markets Revenue | Earnings PDF | Revenue driven "particularly in Prime" |

**Current Disclosure Status**:
- Coverage completeness (c_t): Pending next earnings cycle
- Disclosure pulse (p_t): Will update as banks report quarterly results

*Note: Bank disclosures provide higher information content than weekly proxies. When GS/MS/JPM report, the nowcast uncertainty bands tighten significantly.*

---

## 3. Forward Outlook (2 Quarters)

### 3.1 PB_Lead Forecasts

| Quarter | PB_Lead Forecast | 95% CI Lower | 95% CI Upper |
|---------|------------------|--------------|--------------|
| Q2'26 | +2.55% | -5.02% | +10.12% |
| Q3'26 | +2.55% | -5.02% | +10.12% |

### 3.2 Outlook Interpretation

The model projects **continued positive momentum** over the next two quarters, though with natural mean-reversion from current levels. Key implications:

1. **Prime brokerage balances** expected to grow faster than trend
2. **Revenue momentum** for equity prime businesses should remain positive
3. **Risk factors**: Equity market corrections, funding stress, or regulatory shocks could shift outlook

---

## 4. Historical Context

### 4.1 Sample Statistics (2012 - 2025)

| Statistic | PB_Lead | PB Intensity Growth | Dealer Supply Growth |
|-----------|---------|---------------------|---------------------|
| Mean | +0.85% | +0.89% | +1.71% |
| Std Dev | 3.86% | 6.93% | 7.07% |

### 4.2 Regime Distribution

- **Accelerating**: 31.4% of quarters
- **Stable**: 37.3% of quarters
- **Decelerating**: 31.4% of quarters

### 4.3 Recent Trend (Last 8 Quarters)

| Quarter | PB_Lead | Regime |
|---------|---------|--------|
| Q3'23 | +3.06% | Accelerating |
| Q4'23 | -2.10% | Decelerating |
| Q1'24 | +4.06% | Accelerating |
| Q2'24 | +1.95% | Stable |
| Q3'24 | -1.95% | Decelerating |
| Q4'24 | +3.49% | Accelerating |
| Q1'25 | +1.60% | Stable |
| Q2'25 | +3.28% | Accelerating |

---

## 5. Methodology & Data Quality

### 5.1 Three-Layer Publication System

1. **Official Layer** (through Q2 2025): Fed Z.1 HF sector data
2. **Shadow Backfill Layer**: Ridge regression estimates for completed unreported quarters
3. **Progressive Nowcast Layer**: Weekly-updated estimates for current quarter

### 5.2 Data Sources

| Source | Series | Frequency | Role |
|--------|--------|-----------|------|
| FRED (Fed Z.1) | HF balance sheet | Quarterly | Official anchor |
| CFTC COT (TFF) | Leveraged Funds | Weekly | Nowcast input |
| NY Fed PD Stats | Dealer repo/fails | Weekly | Nowcast input |

### 5.3 Shadow Model Specification

```
y_t = α + β₁·x̄_t + β₂·D_t + ε_t

Where:
- y_t = Δln(PB_Intensity)
- x̄_t = Quarterly mean of weekly leverage appetite factor
- D_t = Dealer supply growth
- Ridge penalty λ = 5.0
```

### 5.4 Limitations

- Shadow estimates are model-dependent and should be interpreted with wider confidence bands
- Weekly CFTC data sourced from CFTC Public Reporting API (Traders in Financial Futures)
- Estimates will be replaced with official Z.1 values when released

---

*Report generated by PB-LLI V2.0.0 with Shadow Nowcast Extension*
*Data through Q2 2025 (official) | Shadow estimates as of 2026-01-16*
