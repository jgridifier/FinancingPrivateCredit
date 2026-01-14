# LEVERAGE CYCLE POSITIONING**

## **2.1 THEORETICAL FOUNDATION**

### **Academic Literature**

**Primary References:**

1. **Adrian, T., & Shin, H. S. (2010).** "Liquidity and Leverage." *Journal of Financial Intermediation*, 19(3), 418-437.
   - Documents procyclical leverage in financial intermediaries
   - Shows leverage expands in booms, contracts in busts

2. **BIS (2024).** "The prime broker–hedge fund nexus: recent evolution and implications for bank risks." *BIS Quarterly Review*, March 2024.
   - **Key finding:** "Hedge fund credit quality deteriorates during weak market conditions... This positive correlation between default probability and net credit exposure constitutes wrong-way risk."
   - Shows secured borrowing correlates with stock valuations (Graph C2.A)

3. **Brunnermeier, M. K., & Pedersen, L. H. (2009).** "Market Liquidity and Funding Liquidity." *Review of Financial Studies*, 22(6), 2201-2238.
   - Theoretical foundation for liquidity spirals
   - Links market volatility to funding constraints

4. **Eren, E. (2018).** "Prime Brokerage Business Models." *GRF Policy Paper No. 6.*
   - Documents run-prone nature of prime brokerage
   - Connects leverage to systemic risk

### **Core Mechanism**

**Leverage Procyclicality:**
1. **Expansion phase:** Markets ↑ → Collateral values ↑ → Hedge funds can borrow more → Leverage ↑
2. **Contraction phase:** Markets ↓ → Collateral values ↓ → Margin calls → Forced deleveraging
3. **Amplification:** Deleveraging → Asset sales → Prices ↓ further → More deleveraging (liquidity spiral)

**Wrong-Way Risk:**
- Prime broker exposure to hedge fund ↑ precisely when hedge fund credit quality ↓
- Correlation = systemic risk indicator

**Prime Brokerage Revenue Implications:**
- **Peak leverage** = Peak revenue potential BUT maximum risk
- **Low leverage** = Bottom of cycle, recovery opportunity
- **Rapid deleveraging** = Revenue collapse + potential losses

---

## **2.2 MATHEMATICAL FORMULATION**

### **Core Indicator: Leverage Cycle Index (LCI)**

$$\text{LCI}_t = \underbrace{\frac{\text{HF\_Margin\_Loans}_t}{\text{HF\_AUM\_Proxy}_t}}_{\text{Leverage Ratio}} \times \underbrace{\frac{\text{SP500}_t}{\text{MA}_{252}(\text{SP500})_t}}_{\text{Market Valuation}}$$

Where:
- **HF_Margin_Loans** = Hedge fund borrowing via prime brokerages
- **HF_AUM_Proxy** = Estimated hedge fund assets under management
- **SP500** = S&P 500 price level
- **MA_252(SP500)** = 252-day (1-year) moving average of S&P 500

### **Component Breakdown**

**1. Leverage Ratio Component**

$$\text{Leverage\_Ratio}_t = \frac{\text{HF\_Margin\_Loans}_t}{\text{HF\_AUM\_Proxy}_t}$$

**Purpose:** Measures how levered hedge funds are relative to their equity base

**2. Market Valuation Component**

$$\text{Market\_Valuation}_t = \frac{\text{SP500}_t}{\text{MA}_{252}(\text{SP500})_t}$$

**Purpose:** Normalizes for whether markets are above/below trend
- Value > 1.0: Markets above trend (potential overvaluation)
- Value < 1.0: Markets below trend (potential undervaluation)

**3. Combined Index**

Product captures **embedded leverage risk**:
- High leverage × High valuations = Maximum systemic risk
- Low leverage × Low valuations = Recovery/opportunity zone

### **Derivative Indicators**

**Leverage Cycle Velocity:**

$$\text{LCV}_t = \frac{\text{LCI}_t - \text{LCI}_{t-1}}{\text{LCI}_{t-1}} \times 100$$

**Purpose:** Rate of change in cycle position. Sharp declines = deleveraging events.

**Regime Classification:**

$$\text{Regime}_t = \begin{cases}
\text{Expansion} & \text{if } \text{LCI}_t > \text{Percentile}_{75}(\text{LCI}_{\text{hist}}) \\
\text{Normal} & \text{if } \text{Percentile}_{25} \leq \text{LCI}_t \leq \text{Percentile}_{75} \\
\text{Contraction} & \text{if } \text{LCI}_t < \text{Percentile}_{25}(\text{LCI}_{\text{hist}})
\end{cases}$$

**Z-Score (for early warning):**

$$\text{LCI\_ZScore}_t = \frac{\text{LCI}_t - \mu(\text{LCI}_{\text{rolling\_8Q}})}{\sigma(\text{LCI}_{\text{rolling\_8Q}})}$$

**Purpose:** Identifies when leverage cycle is >2 standard deviations from recent mean (warning signal).

---

## **2.3 DATA REQUIREMENTS**

### **Primary Data Sources**

| Variable | Source | Series Code | Frequency | Description |
|----------|--------|-------------|-----------|-------------|
| **HF Margin Loans** | FRED | `BOGZ1FL624123035Q` | Quarterly | Hedge Funds; Loans, Total Secured Borrowing Via Prime Brokerages (Margin Accounts); Liability, Level |
| **S&P 500** | FRED | `SP500` | Daily | S&P 500 Index |
| **Total Equity Market Cap** | FRED | `WILL5000PRFC` | Weekly/Monthly | Wilshire 5000 Total Market Full Cap Index (alternative for AUM proxy) |

### **HF AUM Proxy Construction**

Since direct hedge fund AUM is not in FRED, construct proxy:

**Method 1: Using Equity Market Cap**

$$\text{HF\_AUM\_Proxy}_t = \text{Total\_Equity\_Mkt\_Cap}_t \times \alpha$$

Where α = hedge fund share of equity market (calibrate to ~3-4% based on industry reports showing HF AUM of ~$5T vs. equity market of ~$130T)

**Method 2: Using Margin Loans with Historical Leverage**

$$\text{HF\_AUM\_Proxy}_t = \frac{\text{HF\_Margin\_Loans}_t}{\text{Historical\_Leverage\_Ratio}}$$

Where Historical_Leverage_Ratio ≈ 1.5-2.0 (from academic literature, e.g., Ang et al. 2011)

**Recommended: Hybrid Approach**

```python
# Use both methods and average
aum_proxy = (equity_market_cap * 0.035 + margin_loans / 1.75) / 2
```

### **Derived Variables**

| Variable | Calculation | Purpose |
|----------|-------------|---------|
| **SP500_MA252** | Rolling mean of SP500 over 252 trading days | Market trend baseline |
| **Market_Valuation** | SP500 / SP500_MA252 | Overvaluation metric |
| **Leverage_Ratio** | HF_Margin_Loans / HF_AUM_Proxy | Direct leverage measure |

---

## **2.4 IMPLEMENTATION STEPS**

### **Step 1: Data Collection**

```python
import polars as pl
import numpy as np
from fredapi import Fred

# Initialize
fred = Fred(api_key='your_api_key_here')

# Download data
hf_margin = fred.get_series('BOGZ1FL624123035Q', observation_start='2000-01-01')
sp500_daily = fred.get_series('SP500', observation_start='2000-01-01')
wilshire = fred.get_series('WILL5000PRFC', observation_start='2000-01-01')

# Convert to DataFrames
df_margin = pl.DataFrame({
    'date': hf_margin.index,
    'hf_margin_loans': hf_margin.values
})

df_sp500 = pl.DataFrame({
    'date': sp500_daily.index,
    'sp500': sp500_daily.values
})

df_wilshire = pl.DataFrame({
    'date': wilshire.index,
    'total_equity_mcap': wilshire.values
})
```

### **Step 2: Calculate Market Valuation Component**

```python
# Calculate 252-day (1-year) moving average of S&P 500
df_sp500 = df_sp500.with_columns([
    pl.col('sp500').rolling_mean(window_size=252).alias('sp500_ma252'),
]).with_columns([
    (pl.col('sp500') / pl.col('sp500_ma252')).alias('market_valuation')
])

# Aggregate to quarterly (use quarter-end values)
df_sp500_quarterly = df_sp500.with_columns([
    pl.col('date').dt.quarter().alias('quarter'),
    pl.col('date').dt.year().alias('year')
]).group_by(['year', 'quarter']).agg([
    pl.col('market_valuation').last().alias('market_valuation'),  # Quarter-end value
    pl.col('sp500').last().alias('sp500_qtr_end')
]).with_columns([
    pl.date(pl.col('year'), pl.col('quarter') * 3, 1).dt.offset_by('1mo').dt.offset_by('-1d').alias('date')
])
```

### **Step 3: Construct HF AUM Proxy**

```python
# Align Wilshire to quarterly
df_wilshire_quarterly = df_wilshire.with_columns([
    pl.col('date').dt.quarter().alias('quarter'),
    pl.col('date').dt.year().alias('year')
]).group_by(['year', 'quarter']).agg([
    pl.col('total_equity_mcap').last().alias('total_equity_mcap')
]).with_columns([
    pl.date(pl.col('year'), pl.col('quarter') * 3, 1).dt.offset_by('1mo').dt.offset_by('-1d').alias('date')
])

# Merge with margin loans
df_leverage = df_margin.join(df_wilshire_quarterly, on='date', how='inner')

# Construct AUM proxy (hybrid method)
HF_SHARE_OF_EQUITY = 0.035  # 3.5% calibration
HISTORICAL_LEVERAGE = 1.75   # 1.75x leverage calibration

df_leverage = df_leverage.with_columns([
    # Method 1: Market cap based
    (pl.col('total_equity_mcap') * HF_SHARE_OF_EQUITY).alias('aum_proxy_method1'),
    
    # Method 2: Leverage ratio based
    (pl.col('hf_margin_loans') / HISTORICAL_LEVERAGE).alias('aum_proxy_method2'),
    
    # Hybrid: Average of both methods
    ((pl.col('total_equity_mcap') * HF_SHARE_OF_EQUITY + 
      pl.col('hf_margin_loans') / HISTORICAL_LEVERAGE) / 2).alias('hf_aum_proxy')
])
```

### **Step 4: Calculate Leverage Ratio**

```python
df_leverage = df_leverage.with_columns([
    (pl.col('hf_margin_loans') / pl.col('hf_aum_proxy')).alias('leverage_ratio')
])
```

### **Step 5: Merge and Calculate LCI**

```python
# Merge with market valuation
df_lci = df_leverage.join(
    df_sp500_quarterly.select(['date', 'market_valuation']), 
    on='date', 
    how='inner'
)

# Calculate Leverage Cycle Index
df_lci = df_lci.with_columns([
    (pl.col('leverage_ratio') * pl.col('market_valuation')).alias('lci')
])
```

### **Step 6: Calculate Derivative Indicators**

```python
df_lci = df_lci.with_columns([
    # Velocity (QoQ % change)
    (pl.col('lci').pct_change() * 100).alias('lci_velocity'),
    
    # Rolling 8-quarter mean and std for Z-score
    pl.col('lci').rolling_mean(window_size=8).alias('lci_rolling_mean'),
    pl.col('lci').rolling_std(window_size=8).alias('lci_rolling_std')
]).with_columns([
    # Z-score
    ((pl.col('lci') - pl.col('lci_rolling_mean')) / pl.col('lci_rolling_std')).alias('lci_zscore')
])

# Calculate historical percentiles for regime classification
p25 = df_lci['lci'].quantile(0.25)
p75 = df_lci['lci'].quantile(0.75)

df_lci = df_lci.with_columns([
    pl.when(pl.col('lci') > p75).then(pl.lit('Expansion'))
      .when(pl.col('lci') < p25).then(pl.lit('Contraction'))
      .otherwise(pl.lit('Normal'))
      .alias('regime')
])
```

---

## **2.5 INTERPRETATION & USE CASES**

### **What the Indicator Represents**

| Metric | Interpretation | Normal Range |
|--------|----------------|--------------|
| **Leverage_Ratio** | How much HFs borrow relative to equity | 1.3 - 2.0x |
| **Market_Valuation** | Markets vs. trend (>1 = above, <1 = below) | 0.9 - 1.1 |
| **LCI** | Combined leverage & valuation risk | 1.5 - 2.2 |
| **LCI_Velocity** | Speed of leverage cycle changes | -10% to +10% QoQ |
| **LCI_ZScore** | Standard deviations from recent mean | -2 to +2 |

### **Regime Characteristics**

**Expansion (LCI > 75th percentile):**
- High leverage + elevated valuations
- Revenue near peak
- **Risk:** Sudden reversal (2000, 2007, 2021)
- **Action:** Prepare for deleveraging

**Normal (25th - 75th percentile):**
- Moderate leverage, moderate valuations
- Stable revenue environment
- **Action:** Business as usual

**Contraction (LCI < 25th percentile):**
- Low leverage + depressed valuations
- Revenue trough
- **Opportunity:** Recovery positioning (2009, 2020)
- **Action:** Anticipate re-leveraging

### **Critical Thresholds & Signals**

| Signal | Threshold | Interpretation | Historical Examples |
|--------|-----------|----------------|---------------------|
| **Peak Warning** | LCI_ZScore > 2.0 | Extreme leverage conditions | Q2 2007, Q4 2021 |
| **Deleveraging Event** | LCI_Velocity < -15% | Rapid unwinding | Q4 2008, Q1 2020 |
| **Recovery Signal** | LCI_ZScore < -1.5 AND increasing | Bottoming process | Q2 2009, Q3 2020 |
| **Normal Range Exit** | LCI moves outside [p25, p75] | Regime change underway | Monitor closely |

---

## **2.6 INTEGRATION WITH REVENUE MODELS**

### **Revenue Forecasting Framework**

```python
# Regime-dependent revenue model
"""
Revenue[t] = β₀ + 
             β₁ * LCI[t] +                        # Direct leverage effect
             β₂ * LCI[t]² +                       # Non-linearity (inverted U)
             β₃ * LCI_Velocity[t-1] +             # Leading indicator
             β₄ * I(Expansion) * LCI[t] +         # Regime interaction
             β₅ * I(Contraction) * LCI[t] +       # Regime interaction
             Controls + ε[t]
"""

# Expected coefficient signs:
# β₁ > 0: Higher leverage → higher revenue (up to a point)
# β₂ < 0: Diminishing returns/risk at extreme leverage
# β₃: Leading indicator (negative = upcoming contraction)
# β₄ > β₁: Expansion regime has steeper leverage-revenue relationship
# β₅ < β₁: Contraction regime has flatter relationship
```

### **Early Warning System**

```python
# Predict revenue turning points
def leverage_cycle_signal(lci_current, lci_zscore, lci_velocity):
    """
    Returns signal: 1 (bullish), 0 (neutral), -1 (bearish)
    """
    if lci_zscore > 2.0 or lci_velocity < -15:
        return -1  # Bearish: Peak or deleveraging
    elif lci_zscore < -1.5 and lci_velocity > 5:
        return 1   # Bullish: Bottom and re-leveraging
    else:
        return 0   # Neutral: Normal range
```

---

## **2.7 VALIDATION & ROBUSTNESS**

### **Historical Backtesting**

Test against known leverage events:

| Event | Date | Expected LCI Behavior | Validation Check |
|-------|------|----------------------|------------------|
| **Dot-com Peak** | Q1 2000 | LCI_ZScore > 2, then collapse | Check FRED data Q1 2000 - Q1 2001 |
| **Financial Crisis** | Q3 2007 - Q2 2009 | Peak, then -50%+ decline | Verify LCI_Velocity < -20% |
| **COVID Crash** | Q1 2020 | Sharp drop then V-recovery | LCI should recover by Q3 2020 |
| **Archegos/Meme Stocks** | Q1 2021 | LCI spike to elevated levels | Check Q4 2020 - Q2 2021 |

### **Sensitivity Analysis**

Test robustness to calibration assumptions:

```python
# Alternative AUM proxy assumptions
sensitivity_test = pl.DataFrame({
    'scenario': ['Conservative', 'Base', 'Aggressive'],
    'hf_share': [0.03, 0.035, 0.04],
    'hist_leverage': [2.0, 1.75, 1.5]
})

# Recalculate LCI under each scenario
# LCI should be directionally consistent even if levels differ
```

### **Cross-Validation with Alternative Metrics**

```python
# Compare with VIX (should be negatively correlated)
correlation_lci_vix = df_lci.select([
    pl.corr('lci', 'vix_avg')  # Merge VIX data from earlier
])
# Expected: r < -0.5 (leverage high when volatility low)

# Compare with broker-dealer leverage (Adrian-Shin measure)
# Available in academic papers, not FRED, but conceptually similar
```

---

## **2.8 OUTPUT SPECIFICATIONS**

### **Recommended Output Structure**

```python
output_df = df_lci.select([
    'date',
    'hf_margin_loans',           # millions USD
    'hf_aum_proxy',              # millions USD (constructed)
    'leverage_ratio',             # ratio (e.g., 1.75)
    'sp500_qtr_end',             # index level
    'market_valuation',          # ratio (e.g., 1.05 = 5% above trend)
    'lci',                       # index level
    'lci_velocity',              # percent change QoQ
    'lci_zscore',                # standard deviations
    'regime'                     # categorical: Expansion/Normal/Contraction
]).sort('date')

# Save
output_df.write_csv('leverage_cycle_index.csv')
```

### **Summary Statistics**

```python
summary = output_df.select([
    pl.col('lci').mean().alias('lci_mean'),
    pl.col('lci').std().alias('lci_std'),
    pl.col('lci').min().alias('lci_min'),
    pl.col('lci').max().alias('lci_max'),
    pl.col('lci').quantile(0.25).alias('lci_p25'),
    pl.col('lci').quantile(0.75).alias('lci_p75'),
    
    # Regime distribution
    (pl.col('regime') == 'Expansion').mean().alias('pct_expansion'),
    (pl.col('regime') == 'Normal').mean().alias('pct_normal'),
    (pl.col('regime') == 'Contraction').mean().alias('pct_contraction')
])
```
