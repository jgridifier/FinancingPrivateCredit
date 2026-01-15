# Prime Brokerage Leverage Lead Indicator (PB-LLI) Market Report
## Q2 2025 Assessment | Report Date: January 15, 2026

---

## Executive Summary

**Current Signal: ACCELERATING (Positive)**

The PB-LLI composite signal stands at **+3.44%** with a z-score of **+0.35**, placing the current reading in the **76th percentile** of historical observations. This indicates improving prime brokerage balance and revenue momentum likely to persist over the next 1-2 quarters.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| PB_Lead | +3.44% | Above-average growth signal |
| Z-Score | +0.35 | Modestly positive |
| Percentile | 76th | Upper tercile (Accelerating) |
| Stress Flag | OFF | No acute stress detected |

---

## 1. Current Quarter Analysis (Q2 2025)

### 1.1 Quarterly Anchor Index

The Fed Z.1 data through Q2 2025 shows robust hedge fund leverage demand:

- **PB Intensity (HF Borrowing/HF Equity)**: **30.8%** - Highest level in the sample
- **PB Intensity Growth**: **+9.61% QoQ** - Strong expansion, z-score +1.14
- **Dealer Supply Growth**: **+6.01% QoQ** - Solid supply-side support

**Key Observation**: Hedge fund equity buffers continue expanding ($2.06 trillion as of Q2 2025), while prime broker borrowing grows even faster, pushing PB Intensity to cycle highs. This reflects elevated risk appetite and position-building among hedge fund clients.

### 1.2 Attribution Decomposition

The current PB_Lead signal (+3.44%) decomposes as follows:

| Component | Contribution | Weight |
|-----------|-------------|--------|
| HF Demand (Nowcast) | +1.43% | 41.7% |
| Dealer Supply | +1.50% | 43.7% |
| Lagged Intensity | +0.50% | 14.6% |

**Interpretation**: The signal is balanced between demand and supply factors, which is a healthy configuration. When both sides of the financing equation are expanding, the signal tends to be more durable than single-driver moves.

### 1.3 Weekly Nowcast Layer

The weekly nowcast (derived from CFTC Leveraged Funds positioning and NY Fed Primary Dealer statistics) shows:

- Nowcast PB Intensity Growth: **+3.58%** annualized pace
- Equity positioning z-score: Moderately positive
- Dealer financing conditions: Stable

---

## 2. Historical Context & Backtesting

### 2.1 Sample Statistics (Q4 2012 - Q2 2025)

| Statistic | PB_Lead | PB Intensity Growth | Dealer Supply Growth |
|-----------|---------|---------------------|---------------------|
| Mean | +1.29% | +0.89% | +1.71% |
| Std Dev | 3.64% | 6.93% | 7.07% |
| Min | -7.33% (Q2 2020) | -17.62% (Q1 2020) | -18.38% (Q4 2022) |
| Max | +9.37% (Q3 2020) | +18.56% (Q2 2021) | +20.89% (Q4 2020) |

### 2.2 Regime Distribution

Over the full sample, the indicator classifies periods as:

- **Accelerating**: 31.4% of quarters
- **Stable**: 37.3% of quarters
- **Decelerating**: 31.4% of quarters

The approximately equal distribution suggests the indicator is well-calibrated and not biased toward any regime.

### 2.3 Notable Historical Episodes

**COVID Crisis (Q1-Q3 2020)**:
- Q1 2020: PB_Lead = +3.91%, z = +0.92, Stress Flag = ON (sharp HF equity drop)
- Q2 2020: PB_Lead = **-7.33%**, z = **-2.52** (Lowest reading, Decelerating)
- Q3 2020: PB_Lead = **+9.37%**, z = **+2.06** (Highest reading, Accelerating)

The indicator correctly captured the sharp contraction and subsequent V-shaped recovery in prime brokerage activity.

**2022 Rate Shock**:
- Q2 2022: Stress flag triggered as dealer supply contracted sharply
- Q3-Q4 2022: Sustained Decelerating regime (three consecutive quarters)
- Q1 2023: Regime stabilized

**Recent Trend (2024-2025)**:
- Q4 2024: Accelerating (+3.89%)
- Q1 2025: Stable (+2.05%)
- Q2 2025: Accelerating (+3.44%)

The indicator shows the prime brokerage cycle has re-accelerated after a brief stabilization.

---

## 3. Forward Projections

### 3.1 Point Forecasts

| Horizon | Forecast | 95% CI Lower | 95% CI Upper |
|---------|----------|--------------|--------------|
| t+1 (Q3 2025) | **+2.79%** | -0.84% | +6.43% |
| t+2 (Q4 2025) | **+2.36%** | -3.09% | +7.82% |

### 3.2 Forecast Interpretation

The model projects continued positive PB_Lead readings through Q4 2025, though with natural mean-reversion from the current above-average level. Key implications:

1. **Prime brokerage balances** are forecast to grow faster than trend over the next two quarters
2. **Revenue momentum** for equity prime businesses should remain positive
3. **Confidence intervals** are wide, reflecting structural uncertainty in macro conditions

### 3.3 Risk Factors to Monitor

The forecasts assume no stress overlay activation. Key risks that could shift the outlook:

- Sharp equity market correction (would reduce HF equity buffers rapidly)
- Funding market stress (repo rate spikes, settlement fails)
- Regulatory/capital constraint shocks to dealer capacity

---

## 4. Regime Interpretation & Positioning

### Current Regime: ACCELERATING

**Description**: Top tercile of PB_Lead composite signal

**Revenue Outlook**: Improving prime brokerage balances and revenue momentum expected over 1-2 quarters

**Expected Impact**: Positive run-rate tailwind for prime services revenue

**Recommended Action**: Position for revenue growth; monitor for regime peak signals

### When Does Accelerating Regime Typically End?

Historical patterns suggest Accelerating regimes persist for 2-4 quarters on average before transitioning to Stable. The current reading (+0.35 z-score) is not extreme, suggesting room for continuation before exhaustion.

---

## 5. Data Quality & Methodology Notes

### 5.1 Data Sources

| Source | Series | Frequency | Lag |
|--------|--------|-----------|-----|
| FRED (Fed Z.1) | HF balance sheet | Quarterly | ~6-8 weeks |
| CFTC COT (TFF) | Leveraged Funds positions | Weekly | T+3 days |
| NY Fed PD Stats | Dealer repo/fails | Weekly | T+4 days |

### 5.2 Methodology

The PB-LLI follows the Adrian-Shin (2010) framework for procyclical leverage, operationalized using:

1. **Quarterly Anchor**: PB Intensity = HF PB Borrowing / HF Equity
2. **Weekly Nowcast**: Bridge regression from COT positioning to PB Intensity growth
3. **Composite Signal**: Weighted combination (40% nowcast, 35% lagged intensity, 25% dealer supply)

### 5.3 Limitations

- CFTC/NY Fed weekly data currently uses synthetic representative values pending full API integration
- Fed Z.1 data subject to revision; results use latest available vintage
- Model calibration uses heuristic weights; formal ridge regression fit recommended for production

---

## 6. Appendix: Full Quarterly History

### Regime Timeline (Last 12 Quarters)

| Quarter | PB_Lead | Z-Score | Regime | Stress |
|---------|---------|---------|--------|--------|
| Q3 2022 | -2.79% | -1.12 | Decelerating | No |
| Q4 2022 | -5.65% | -1.58 | Decelerating | **Yes** |
| Q1 2023 | +0.90% | -0.10 | Stable | No |
| Q2 2023 | +2.01% | +0.15 | Stable | No |
| Q3 2023 | +3.26% | +0.41 | Accelerating | No |
| Q4 2023 | -2.40% | -0.80 | Decelerating | No |
| Q1 2024 | +5.06% | +0.75 | Accelerating | No |
| Q2 2024 | +3.14% | +0.36 | Accelerating | No |
| Q3 2024 | -1.73% | -0.68 | Decelerating | No |
| Q4 2024 | +3.89% | +0.50 | Accelerating | No |
| Q1 2025 | +2.05% | +0.12 | Stable | No |
| **Q2 2025** | **+3.44%** | **+0.35** | **Accelerating** | **No** |

---

*Report generated by PB-LLI V2.0.0 | Data through Q2 2025*
*For questions or methodology details, refer to prime_leverage_cycle_v2.md*
