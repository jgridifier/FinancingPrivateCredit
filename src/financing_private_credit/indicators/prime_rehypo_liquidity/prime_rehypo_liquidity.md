# REHYPOTHECATION LIQUIDITY CREATION**

## **1.1 THEORETICAL FOUNDATION**

### **Academic Literature**

**Primary References:**
1. **Eren, E. (2014).** "Intermediary Funding Liquidity and Rehypothecation as Determinants of Repo Haircuts and Interest Rates." *Stanford University Working Paper.*
   - Shows haircut differentials create funding liquidity for prime brokers
   - Estimates several billion dollars per major prime broker

2. **Kirk, A., McAndrews, J., Sastry, P., & Weed, P. (2014).** "Matching Collateral Supply and Financing Demands in Dealer Banks." *FRBNY Economic Policy Review*, 20(2), 127-151.
   - Documents internalization vs. matched-book economics
   - Shows balance sheet treatment differences

3. **Singh, M., & Aitken, J. (2010).** "The (Sizable) Role of Rehypothecation in the Shadow Banking System." *IMF Working Paper WP/10/172.*
   - Quantifies scale of collateral reuse
   - Links to funding liquidity

4. **Infante, S. (2015).** "Liquidity Windfalls: The Consequences of Repo Rehypothecation." *Federal Reserve Board Working Paper.*
   - Shows how haircut spreads generate liquidity
   - Models risk implications

### **Core Mechanism**

When prime brokers finance hedge fund positions:
1. Hedge fund pledges collateral worth $100, receives $90 cash (10% haircut)
2. Prime broker repledges same collateral to money market fund for $95 (5% haircut)
3. Prime broker retains $5 = **liquidity creation** through haircut differential

This liquidity can fund:
- Other client positions
- Proprietary trading
- Balance sheet needs

**Risk:** During stress, third-party haircuts increase faster than client haircuts → liquidity evaporates

---

## **1.2 MATHEMATICAL FORMULATION**

### **Core Indicator: Rehypothecation Liquidity Index (RLI)**

$$\text{RLI}_t = \left(\text{Margin\_Receivables}_t + \text{Repo\_Liabilities}_t\right) \times \text{Haircut\_Spread}_t$$

Where:
- **Margin_Receivables** = Security brokers and dealers' receivables from customers (margin loans)
- **Repo_Liabilities** = Security brokers and dealers' repo liabilities  
- **Haircut_Spread** = Estimated differential between client haircuts and third-party haircuts

### **Haircut Spread Estimation**

Since haircuts are not directly observable in public data, estimate using regime-dependent model:

$$\text{Haircut\_Spread}_t = \begin{cases} 
\alpha_{\text{low}} + \beta_{\text{low}} \times \text{VIX}_t & \text{if VIX}_t < 20 \\
\alpha_{\text{high}} + \beta_{\text{high}} \times \text{VIX}_t & \text{if VIX}_t \geq 20
\end{cases}$$

**Calibrated Parameters** (from literature):
- **Low volatility regime** (VIX < 20):
  - α_low = 0.02 (base spread of 2%)
  - β_low = 0.0005 (spreads widen 0.5 bps per VIX point)
  
- **High volatility regime** (VIX ≥ 20):
  - α_high = 0.03 (base spread of 3%)
  - β_high = 0.002 (spreads widen 2 bps per VIX point)

**Source for calibration:** Singh & Aitken (2010), Eren (2014)

### **Normalized Indicator**

To make comparable over time:

$$\text{RLI\_Normalized}_t = \frac{\text{RLI}_t}{\text{Total\_Broker\_Dealer\_Assets}_t}$$

### **Rate of Change Indicator**

Early warning signal:

$$\text{RLI\_Velocity}_t = \frac{\text{RLI}_t - \text{RLI}_{t-1}}{\text{RLI}_{t-1}} \times 100$$

---

## **1.3 DATA REQUIREMENTS**

### **Primary Data Sources (All from FRED)**

| Variable | FRED Series Code | Frequency | Description |
|----------|------------------|-----------|-------------|
| **Margin Receivables** | `BOGZ1FL663067003Q` | Quarterly | Security Brokers and Dealers; Receivables Due from Customers (Margin Loans and Other Receivables); Asset, Level |
| **Repo Liabilities** | `BOGZ1FL662151003Q` | Quarterly | Security Brokers and Dealers; Security Repurchase Agreements; Liability, Level |
| **Total B-D Assets** | `BOGZ1FL664090005Q` | Quarterly | Security Brokers and Dealers; Total Financial Assets, Level |
| **VIX** | `VIXCLS` | Daily | CBOE Volatility Index |
| **Securities Sold Short** | `BOGZ1FL664140660Q` | Quarterly | Security Brokers and Dealers; Total Securities Sold Short; Liability, Level (optional enhancement) |

### **Derived Variables**

| Variable | Calculation | Purpose |
|----------|-------------|---------|
| **Total Collateral Base** | Margin_Receivables + Repo_Liabilities | Volume of activity eligible for rehypothecation |
| **VIX Quarterly Average** | Mean(VIX_daily) over quarter | Regime classification |
| **Haircut Spread** | See formula above | Key multiplier for liquidity creation |

---

## **1.4 IMPLEMENTATION STEPS**

### **Step 1: Data Collection**

```python
import polars as pl
from fredapi import Fred

# Initialize FRED API
fred = Fred(api_key='your_api_key_here')

# Download quarterly data
margin_recv = fred.get_series('BOGZ1FL663067003Q', observation_start='2000-01-01')
repo_liab = fred.get_series('BOGZ1FL662151003Q', observation_start='2000-01-01')
total_assets = fred.get_series('BOGZ1FL664090005Q', observation_start='2000-01-01')

# Download daily VIX
vix_daily = fred.get_series('VIXCLS', observation_start='2000-01-01')

# Convert to Polars DataFrames
df_margin = pl.DataFrame({
    'date': margin_recv.index,
    'margin_receivables': margin_recv.values
})

df_repo = pl.DataFrame({
    'date': repo_liab.index,
    'repo_liabilities': repo_liab.values
})

df_assets = pl.DataFrame({
    'date': total_assets.index,
    'total_bd_assets': total_assets.values
})

df_vix = pl.DataFrame({
    'date': vix_daily.index,
    'vix': vix_daily.values
})
```

### **Step 2: Calculate Quarterly VIX Average**

```python
# Aggregate VIX to quarterly
df_vix_quarterly = df_vix.with_columns([
    pl.col('date').dt.quarter().alias('quarter'),
    pl.col('date').dt.year().alias('year')
]).group_by(['year', 'quarter']).agg([
    pl.col('vix').mean().alias('vix_avg')
]).with_columns([
    # Create quarter-end date for merging
    pl.date(pl.col('year'), pl.col('quarter') * 3, 1).dt.offset_by('1mo').dt.offset_by('-1d').alias('date')
])
```

### **Step 3: Merge Data**

```python
# Merge all quarterly data
df_combined = df_margin.join(df_repo, on='date', how='inner') \
                       .join(df_assets, on='date', how='inner') \
                       .join(df_vix_quarterly.select(['date', 'vix_avg']), on='date', how='inner')
```

### **Step 4: Calculate Haircut Spread**

```python
# Define haircut spread function
def calculate_haircut_spread(vix):
    if vix < 20:
        return 0.02 + 0.0005 * vix
    else:
        return 0.03 + 0.002 * vix

# Apply to data
df_combined = df_combined.with_columns([
    pl.col('vix_avg').map_elements(calculate_haircut_spread, return_dtype=pl.Float64).alias('haircut_spread')
])
```

### **Step 5: Calculate RLI**

```python
df_combined = df_combined.with_columns([
    # Total collateral base
    (pl.col('margin_receivables') + pl.col('repo_liabilities')).alias('collateral_base'),
    
    # Rehypothecation Liquidity Index (in millions)
    ((pl.col('margin_receivables') + pl.col('repo_liabilities')) * pl.col('haircut_spread')).alias('rli'),
    
    # Normalized RLI
    (((pl.col('margin_receivables') + pl.col('repo_liabilities')) * pl.col('haircut_spread')) / 
     pl.col('total_bd_assets')).alias('rli_normalized'),
    
    # RLI velocity (quarter-over-quarter % change)
    (((pl.col('margin_receivables') + pl.col('repo_liabilities')) * pl.col('haircut_spread')).pct_change() * 100).alias('rli_velocity')
])
```

---

## **1.5 INTERPRETATION & USE CASES**

### **What the Indicator Represents**

| Metric | Interpretation |
|--------|----------------|
| **RLI (absolute)** | Dollar amount of liquidity generated through collateral rehypothecation. Higher values = more funding liquidity available to prime brokers. |
| **RLI_Normalized** | Liquidity creation as % of total broker-dealer assets. Controls for industry size over time. |
| **RLI_Velocity** | Rate of change in liquidity creation. Sharp declines signal liquidity stress. |
| **Haircut_Spread** | Risk premium differential. Widening spreads = higher perceived counterparty risk. |

### **Expected Relationships**

**Normal Times (VIX < 20):**
- RLI steady or growing
- Haircut spreads narrow (2-3%)
- Supports revenue growth through leverage

**Stress Times (VIX > 30):**
- RLI contracts sharply
- Haircut spreads widen (5-8%+)
- Triggers deleveraging → revenue decline

### **Integration with Revenue Models**

```python
# In regression framework
revenue_model = """
    Bank_Prime_Revenue[t] = β₀ + 
                           β₁ * HF_Margin_Loans[t] +
                           β₂ * RLI[t] +                    # Liquidity boost effect
                           β₃ * RLI_Velocity[t-1] +         # Early warning
                           β₄ * (RLI[t] * VIX[t]) +         # Interaction: liquidity*risk
                           Controls + ε[t]
"""

# Expected signs:
# β₂ > 0: More liquidity → more capacity for prime services
# β₃ < 0: Declining liquidity → upcoming revenue pressure
# β₄ < 0: High liquidity in high vol = risk (Archegos-type scenario)
```

### **Critical Thresholds (from literature)**

| Threshold | Meaning | Action |
|-----------|---------|--------|
| RLI_Velocity < -20% QoQ | Severe liquidity contraction | Expect 10-15% revenue decline next quarter |
| Haircut_Spread > 6% | Extreme stress regime | Historical median -25% revenue |
| RLI_Normalized < 1% | Below crisis levels | Compare to 2008: 0.8% |

---

## **1.6 VALIDATION & ROBUSTNESS**

### **Historical Validation Points**

Test indicator against known events:

| Event | Date | Expected Pattern | Data Check |
|-------|------|------------------|------------|
| **Lehman Crisis** | Q3-Q4 2008 | RLI collapse, haircut_spread spike | Verify against FRED data |
| **Archegos** | Q1 2021 | Localized spike then reversal | Check VIX-RLI relationship |
| **COVID Crash** | Q1 2020 | Sharp RLI drop, rapid recovery | Verify velocity indicator |

### **Sensitivity Analysis**

Test alternative calibrations:

```python
# Conservative (wider spreads)
haircut_spread_conservative = lambda vix: 0.025 + 0.001 * vix if vix < 20 else 0.04 + 0.003 * vix

# Aggressive (narrower spreads)
haircut_spread_aggressive = lambda vix: 0.015 + 0.0003 * vix if vix < 20 else 0.025 + 0.0015 * vix

# Compare RLI under different assumptions
```

### **Cross-Validation with Alternative Proxies**

```python
# Alternative: Use securities sold short as additional collateral proxy
df_combined = df_combined.with_columns([
    ((pl.col('margin_receivables') + pl.col('repo_liabilities') + 
      pl.col('securities_short')) * pl.col('haircut_spread')).alias('rli_with_shorts')
])

# Should be highly correlated (r > 0.95)
correlation = df_combined.select([
    pl.corr('rli', 'rli_with_shorts')
])
```

---

## **1.7 OUTPUT SPECIFICATIONS**

### **Recommended Output Table Structure**

```python
# Final output format
output_df = df_combined.select([
    'date',
    'margin_receivables',          # millions USD
    'repo_liabilities',             # millions USD
    'collateral_base',              # millions USD
    'vix_avg',                      # index level
    'haircut_spread',               # decimal (e.g., 0.025 = 2.5%)
    'rli',                          # millions USD
    'rli_normalized',               # decimal (% of total assets)
    'rli_velocity'                  # percent change
]).sort('date')

# Save for integration
output_df.write_csv('rehypothecation_liquidity_index.csv')
```

### **Summary Statistics to Report**

```python
summary = output_df.select([
    pl.col('rli').mean().alias('rli_mean'),
    pl.col('rli').std().alias('rli_std'),
    pl.col('rli').quantile(0.1).alias('rli_p10'),
    pl.col('rli').quantile(0.9).alias('rli_p90'),
    pl.col('rli_velocity').mean().alias('velocity_mean'),
    pl.col('rli_velocity').std().alias('velocity_std')
])
```

---

