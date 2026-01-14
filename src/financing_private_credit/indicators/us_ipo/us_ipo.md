# IPO Banking Volume Estimation & Forecasting System (REVISED)

**Here is a implemenation plane for the IPO banking volume estimator. It was originally designed for a different system but we want to integrate it into this folders indicator system.** Adapt the instructions to fit into the current framework and test it with the last 3 months data.

## 1. Data Strategy

### 1.1 Data Source: SEC EDGAR 424B4/424B5 Filings

**Primary data extraction workflow:**

```
SEC EDGAR API → 424B4 Filings → Parse HTML/XML → Extract structured fields → Store as Parquet
```

**Key fields to extract:**

1. **Deal Identification**
   - Company name
   - Ticker symbol
   - Filing date (proxy for IPO pricing date)
   - CIK number

2. **Deal Economics**
   - Number of shares offered (primary + secondary if applicable)
   - Offering price per share
   - **Gross proceeds** = shares × price
   - **Underwriting discount** (usually expressed as % and $ amount)
   - **Total underwriting fees** = gross proceeds × discount %

3. **Syndicate Structure**
   - Full list of underwriters in order (critical: order matters)
   - Distinguish between:
     - Lead bookrunners (typically listed first, sometimes labeled)
     - Co-managers
     - Syndicate members

### 1.2 Technical Implementation

**Architecture:**
```
1. Scheduler (daily/weekly cron job)
   ↓
2. EDGAR Scraper (Python)
   - Query SEC full-text search API for 424B4/424B5 filings
   - Filter by date range (new filings since last run)
   ↓
3. HTML/XML Parser
   - Extract underwriter table (usually in first 10 pages)
   - Parse "Underwriting" section for fee structure
   ↓
4. Data Validation & Storage
   - Validate required fields present
   - Append to Parquet files (partitioned by year/quarter)
   ↓
5. Aggregation Layer
   - Read Parquet files
   - Generate daily/weekly/monthly/quarterly aggregates by bank
```

**Python libraries needed:**
- `requests` - SEC API calls
- `beautifulsoup4` or `lxml` - HTML parsing
- `polars` - Data manipulation and Parquet I/O
- `sec-edgar-downloader` - Simplified EDGAR access
- `python-dateutil` - Date handling
- `pyarrow` - Parquet backend for Polars

### 1.3 Parquet Storage Schema

**File structure:**
```
data/
├── raw_filings/
│   ├── year=2024/
│   │   ├── quarter=Q1/
│   │   │   └── filings.parquet
│   │   ├── quarter=Q2/
│   │   │   └── filings.parquet
│   │   └── quarter=Q4/
│   │       └── filings.parquet
│   └── year=2025/
│       └── quarter=Q1/
│           └── filings.parquet
├── bank_allocations/
│   ├── year=2024/
│   │   └── allocations.parquet
│   └── year=2025/
│       └── allocations.parquet
└── aggregated/
    ├── daily_metrics.parquet
    ├── weekly_metrics.parquet
    ├── monthly_metrics.parquet
    └── quarterly_metrics.parquet
```

**Schema for `raw_filings/*.parquet`:**

```python
RAW_FILINGS_SCHEMA = {
    'deal_id': pl.Utf8,              # Unique identifier (e.g., CIK + filing date)
    'filing_date': pl.Date,
    'company_name': pl.Utf8,
    'ticker': pl.Utf8,
    'cik': pl.Utf8,
    'shares_offered': pl.Int64,
    'offering_price': pl.Float64,
    'gross_proceeds': pl.Float64,
    'underwriting_discount_pct': pl.Float64,
    'total_fees': pl.Float64,
    'underwriters_list': pl.List(pl.Utf8),  # Ordered list of bank names
    'scraped_at': pl.Datetime,
    'filing_url': pl.Utf8
}
```

**Schema for `bank_allocations/*.parquet`:**

```python
BANK_ALLOCATIONS_SCHEMA = {
    'deal_id': pl.Utf8,
    'filing_date': pl.Date,
    'bank_name': pl.Utf8,
    'rank_position': pl.Int32,
    'estimated_fee': pl.Float64,
    'fee_share_pct': pl.Float64,
    'total_deal_fees': pl.Float64,
    'gross_proceeds': pl.Float64,
    'company_name': pl.Utf8
}
```

**Schema for aggregated metrics:**

```python
AGGREGATED_METRICS_SCHEMA = {
    'bank_name': pl.Utf8,
    'period_start': pl.Date,
    'period_end': pl.Date,
    'period_type': pl.Utf8,  # 'daily', 'weekly', 'monthly', 'quarterly'
    'total_fee_volume': pl.Float64,
    'deal_count': pl.Int32,
    'avg_fee_per_deal': pl.Float64,
    'lead_bookrunner_count': pl.Int32,
    'total_gross_proceeds': pl.Float64,
    'market_share_pct': pl.Float64,
    'calculated_at': pl.Datetime
}
```

### 1.4 Implementation with Parquet

```python
import polars as pl
from pathlib import Path
from datetime import datetime

class IPODataStore:
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
    def save_raw_filing(self, deal_data: dict):
        """
        Save a single IPO filing to partitioned Parquet
        """
        filing_date = deal_data['filing_date']
        year = filing_date.year
        quarter = f"Q{(filing_date.month - 1) // 3 + 1}"
        
        # Create partition path
        partition_path = self.base_path / "raw_filings" / f"year={year}" / f"quarter={quarter}"
        partition_path.mkdir(parents=True, exist_ok=True)
        
        file_path = partition_path / "filings.parquet"
        
        # Convert to DataFrame
        df = pl.DataFrame([deal_data])
        
        # Append or create
        if file_path.exists():
            existing = pl.read_parquet(file_path)
            combined = pl.concat([existing, df])
            # Deduplicate by deal_id
            combined = combined.unique(subset=['deal_id'], keep='last')
            combined.write_parquet(file_path)
        else:
            df.write_parquet(file_path)
    
    def save_bank_allocations(self, allocations_df: pl.DataFrame):
        """
        Save bank fee allocations
        """
        for year in allocations_df['filing_date'].dt.year().unique():
            year_data = allocations_df.filter(pl.col('filing_date').dt.year() == year)
            
            partition_path = self.base_path / "bank_allocations" / f"year={year}"
            partition_path.mkdir(parents=True, exist_ok=True)
            
            file_path = partition_path / "allocations.parquet"
            
            if file_path.exists():
                existing = pl.read_parquet(file_path)
                combined = pl.concat([existing, year_data])
                # Deduplicate by deal_id + bank_name
                combined = combined.unique(subset=['deal_id', 'bank_name'], keep='last')
                combined.write_parquet(file_path)
            else:
                year_data.write_parquet(file_path)
    
    def read_all_filings(self, start_date=None, end_date=None) -> pl.DataFrame:
        """
        Read all raw filings across partitions with optional date filter
        """
        filings_path = self.base_path / "raw_filings"
        
        if not filings_path.exists():
            return pl.DataFrame(schema=RAW_FILINGS_SCHEMA)
        
        # Read all parquet files recursively
        all_files = list(filings_path.rglob("*.parquet"))
        
        if not all_files:
            return pl.DataFrame(schema=RAW_FILINGS_SCHEMA)
        
        df = pl.concat([pl.read_parquet(f) for f in all_files])
        
        # Apply date filters if provided
        if start_date:
            df = df.filter(pl.col('filing_date') >= start_date)
        if end_date:
            df = df.filter(pl.col('filing_date') <= end_date)
        
        return df
    
    def read_all_allocations(self, start_date=None, end_date=None) -> pl.DataFrame:
        """
        Read all bank allocations with optional date filter
        """
        allocations_path = self.base_path / "bank_allocations"
        
        if not allocations_path.exists():
            return pl.DataFrame(schema=BANK_ALLOCATIONS_SCHEMA)
        
        all_files = list(allocations_path.rglob("*.parquet"))
        
        if not all_files:
            return pl.DataFrame(schema=BANK_ALLOCATIONS_SCHEMA)
        
        df = pl.concat([pl.read_parquet(f) for f in all_files])
        
        if start_date:
            df = df.filter(pl.col('filing_date') >= start_date)
        if end_date:
            df = df.filter(pl.col('filing_date') <= end_date)
        
        return df
    
    def save_aggregated_metrics(self, metrics_df: pl.DataFrame, period_type: str):
        """
        Save aggregated metrics (daily/weekly/monthly/quarterly)
        """
        agg_path = self.base_path / "aggregated"
        agg_path.mkdir(exist_ok=True)
        
        file_path = agg_path / f"{period_type}_metrics.parquet"
        
        if file_path.exists():
            existing = pl.read_parquet(file_path)
            combined = pl.concat([existing, metrics_df])
            # Deduplicate by bank_name + period
            combined = combined.unique(subset=['bank_name', 'period_start', 'period_end'], keep='last')
            combined.write_parquet(file_path)
        else:
            metrics_df.write_parquet(file_path)
    
    def read_aggregated_metrics(self, period_type: str, start_date=None, end_date=None) -> pl.DataFrame:
        """
        Read aggregated metrics for a specific period type
        """
        file_path = self.base_path / "aggregated" / f"{period_type}_metrics.parquet"
        
        if not file_path.exists():
            return pl.DataFrame(schema=AGGREGATED_METRICS_SCHEMA)
        
        df = pl.read_parquet(file_path)
        
        if start_date:
            df = df.filter(pl.col('period_start') >= start_date)
        if end_date:
            df = df.filter(pl.col('period_end') <= end_date)
        
        return df
```

**Parsing and storage workflow:**

```python
def daily_scraper_job():
    """
    Run daily to scrape new IPO filings and update Parquet files
    """
    store = IPODataStore()
    
    # Get filings from yesterday
    yesterday = datetime.now().date() - timedelta(days=1)
    filings = get_recent_ipo_filings(yesterday, yesterday)
    
    for filing in filings:
        # Parse filing
        deal_data = extract_ipo_data(filing['filing_url'])
        
        if deal_data:
            # Save raw filing
            store.save_raw_filing(deal_data)
            
            # Allocate fees to banks
            allocations = allocate_fees_to_banks(deal_data)
            store.save_bank_allocations(allocations)
    
    # Update aggregated metrics
    update_all_aggregations(store)

def update_all_aggregations(store: IPODataStore):
    """
    Recalculate aggregations for all period types
    """
    allocations = store.read_all_allocations()
    
    if allocations.is_empty():
        return
    
    # Daily aggregation
    daily_metrics = calculate_metrics(allocations, 'daily')
    store.save_aggregated_metrics(daily_metrics, 'daily')
    
    # Weekly aggregation
    weekly_metrics = calculate_metrics(allocations, 'weekly')
    store.save_aggregated_metrics(weekly_metrics, 'weekly')
    
    # Monthly aggregation
    monthly_metrics = calculate_metrics(allocations, 'monthly')
    store.save_aggregated_metrics(monthly_metrics, 'monthly')
    
    # Quarterly aggregation
    quarterly_metrics = calculate_metrics(allocations, 'quarterly')
    store.save_aggregated_metrics(quarterly_metrics, 'quarterly')

def calculate_metrics(allocations: pl.DataFrame, period_type: str) -> pl.DataFrame:
    """
    Calculate aggregated metrics for a given period type
    """
    # Define period grouping
    if period_type == 'daily':
        period_col = pl.col('filing_date')
    elif period_type == 'weekly':
        period_col = pl.col('filing_date').dt.truncate('1w')
    elif period_type == 'monthly':
        period_col = pl.col('filing_date').dt.truncate('1mo')
    else:  # quarterly
        period_col = (
            pl.col('filing_date').dt.year().cast(pl.Utf8) + 
            "-Q" + 
            ((pl.col('filing_date').dt.month() - 1) // 3 + 1).cast(pl.Utf8)
        )
    
    # Group and aggregate
    metrics = allocations.group_by(['bank_name', period_col.alias('period_key')]).agg([
        pl.col('estimated_fee').sum().alias('total_fee_volume'),
        pl.col('deal_id').n_unique().alias('deal_count'),
        pl.col('estimated_fee').mean().alias('avg_fee_per_deal'),
        (pl.col('rank_position') == 1).sum().alias('lead_bookrunner_count'),
        pl.col('gross_proceeds').sum().alias('total_gross_proceeds'),
        pl.col('filing_date').min().alias('period_start'),
        pl.col('filing_date').max().alias('period_end')
    ])
    
    # Calculate market share
    total_market = metrics.group_by('period_key').agg(
        pl.col('total_fee_volume').sum().alias('market_total')
    )
    
    metrics = metrics.join(total_market, on='period_key')
    metrics = metrics.with_columns([
        (pl.col('total_fee_volume') / pl.col('market_total') * 100).alias('market_share_pct'),
        pl.lit(period_type).alias('period_type'),
        pl.lit(datetime.now()).alias('calculated_at')
    ])
    
    return metrics.select([
        'bank_name', 'period_start', 'period_end', 'period_type',
        'total_fee_volume', 'deal_count', 'avg_fee_per_deal',
        'lead_bookrunner_count', 'total_gross_proceeds', 'market_share_pct',
        'calculated_at'
    ]).sort(['period_start', 'total_fee_volume'], descending=[False, True])
```

---

## 2. IPO Volume Estimation by Bank (Directional Indicator)

### 2.1 Fee Allocation Model

Since actual fee splits are not disclosed, estimate based on syndicate position using industry norms:

**Assumed fee allocation by rank:**

```python
FEE_ALLOCATION_ASSUMPTIONS = {
    1: 0.35,   # Lead bookrunner gets 35% of total fees
    2: 0.25,   # Second position gets 25%
    3: 0.15,   # Third position gets 15%
    4: 0.10,   # Fourth gets 10%
    5: 0.08,   # Fifth gets 8%
    6: 0.07,   # Sixth+ split remaining
}

# Adjustment for multiple lead bookrunners
# If 2 banks both rank 1, they split the rank 1+2 allocation (60% total = 30% each)
```

### 2.2 Implementation

```python
import polars as pl

def allocate_fees_to_banks(deal_data: dict) -> pl.DataFrame:
    """
    Allocate total deal fees to individual banks based on syndicate position
    
    Returns DataFrame with bank-level allocations
    """
    total_fees = deal_data['total_fees']
    underwriters = deal_data['underwriters_list']  # Ordered list
    
    allocations = []
    
    # Handle co-lead bookrunners (multiple rank 1s)
    rank_counts = {}
    for idx, bank in enumerate(underwriters):
        rank = idx + 1
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    for idx, bank in enumerate(underwriters):
        rank = idx + 1
        
        # Get base allocation percentage
        if rank <= 6:
            base_pct = FEE_ALLOCATION_ASSUMPTIONS[rank]
        else:
            # Remaining banks split leftover
            remaining_pct = 1 - sum(FEE_ALLOCATION_ASSUMPTIONS.values())
            num_remaining = len(underwriters) - 6
            base_pct = remaining_pct / num_remaining if num_remaining > 0 else 0
        
        # Adjust if multiple banks at same rank
        if rank_counts[rank] > 1:
            base_pct = base_pct / rank_counts[rank]
        
        estimated_fee = total_fees * base_pct
        
        allocations.append({
            'deal_id': deal_data['deal_id'],
            'filing_date': deal_data['filing_date'],
            'bank_name': normalize_bank_name(bank),
            'rank_position': rank,
            'estimated_fee': estimated_fee,
            'fee_share_pct': base_pct * 100,
            'total_deal_fees': total_fees,
            'gross_proceeds': deal_data['gross_proceeds'],
            'company_name': deal_data['company_name']
        })
    
    return pl.DataFrame(allocations)

def normalize_bank_name(raw_name: str) -> str:
    """
    Standardize bank names for aggregation
    """
    name_mapping = {
        'goldman sachs': 'Goldman Sachs',
        'morgan stanley': 'Morgan Stanley',
        'jp morgan': 'JPMorgan',
        'jpmorgan': 'JPMorgan',
        'j.p. morgan': 'JPMorgan',
        'bank of america': 'BofA Securities',
        'bofa': 'BofA Securities',
        'citigroup': 'Citi',
        'barclays': 'Barclays',
        'deutsche bank': 'Deutsche Bank',
        'wells fargo': 'Wells Fargo',
        'credit suisse': 'Credit Suisse',
        'ubs': 'UBS',
        # Add more as needed
    }
    
    normalized = raw_name.lower()
    for pattern, standard in name_mapping.items():
        if pattern in normalized:
            return standard
    
    return raw_name  # Return original if no match
```

---

## 3. Momentum-Based Forecast Model

### 3.1 Model Architecture

Use **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) configured for momentum:

**Model specification:**

```python
# SARIMAX(p,d,q)(P,D,Q)s configuration

DAILY_CONFIG = {
    'order': (7, 1, 1),        # 7 day lookback for momentum
    'seasonal_order': (0, 0, 0, 0),  # No seasonality for daily
    'trend': None
}

WEEKLY_CONFIG = {
    'order': (4, 1, 1),        # 4 week momentum
    'seasonal_order': (1, 0, 0, 52),  # 52-week seasonal pattern
    'trend': None
}

MONTHLY_CONFIG = {
    'order': (2, 1, 1),        # 2 month momentum lags
    'seasonal_order': (1, 0, 1, 12),  # 12-month seasonal cycle
    'trend': 't'
}

QUARTERLY_CONFIG = {
    'order': (3, 1, 1),        # More AR lags for quarterly
    'seasonal_order': (1, 0, 1, 4),   # 4-quarter seasonal cycle
    'trend': 't'
}
```

### 3.2 Implementation

```python
import polars as pl
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from datetime import timedelta

class BankIPOForecaster:
    def __init__(self, bank_name: str, period_type: str = 'monthly', data_store: IPODataStore = None):
        self.bank_name = bank_name
        self.period_type = period_type
        self.data_store = data_store or IPODataStore()
        self.model = None
        self.fitted_model = None
        
    def prepare_data(self) -> pl.DataFrame:
        """
        Load historical metrics for the bank from Parquet
        """
        metrics = self.data_store.read_aggregated_metrics(self.period_type)
        
        bank_data = metrics.filter(
            pl.col('bank_name') == self.bank_name
        ).sort('period_start')
        
        if bank_data.is_empty():
            raise ValueError(f"No data found for {self.bank_name}")
        
        # Create complete date range (fill gaps with zeros)
        if self.period_type == 'daily':
            date_range = pl.date_range(
                bank_data['period_start'].min(),
                bank_data['period_start'].max(),
                interval='1d',
                eager=True
            )
        elif self.period_type == 'weekly':
            date_range = pl.date_range(
                bank_data['period_start'].min(),
                bank_data['period_start'].max(),
                interval='1w',
                eager=True
            )
        elif self.period_type == 'monthly':
            date_range = pl.date_range(
                bank_data['period_start'].min(),
                bank_data['period_start'].max(),
                interval='1mo',
                eager=True
            )
        else:  # quarterly
            # Need custom quarterly range
            start = bank_data['period_start'].min()
            end = bank_data['period_start'].max()
            date_range = []
            current = start
            while current <= end:
                date_range.append(current)
                current = current + timedelta(days=91)  # Approximate quarter
            date_range = pl.Series(date_range)
        
        complete_df = pl.DataFrame({'period_start': date_range})
        
        # Join and fill missing with zeros
        ts_data = complete_df.join(
            bank_data.select(['period_start', 'total_fee_volume']),
            on='period_start',
            how='left'
        ).with_columns(
            pl.col('total_fee_volume').fill_null(0)
        )
        
        return ts_data
    
    def fit(self, exog_vars: pl.DataFrame = None):
        """
        Fit SARIMAX model on historical IPO fee volume
        """
        ts_data = self.prepare_data()
        
        # Select config based on period type
        config = {
            'daily': DAILY_CONFIG,
            'weekly': WEEKLY_CONFIG,
            'monthly': MONTHLY_CONFIG,
            'quarterly': QUARTERLY_CONFIG
        }[self.period_type]
        
        # Fit model
        self.model = SARIMAX(
            ts_data['total_fee_volume'].to_numpy(),
            order=config['order'],
            seasonal_order=config['seasonal_order'],
            trend=config['trend'],
            exog=exog_vars.to_numpy() if exog_vars is not None else None,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = self.model.fit(disp=False)
        
        return self.fitted_model.summary()
    
    def forecast(self, periods: int, exog_future: pl.DataFrame = None) -> pl.DataFrame:
        """
        Generate forecast for next N periods
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.fitted_model.forecast(
            steps=periods,
            exog=exog_future.to_numpy() if exog_future is not None else None
        )
        
        # Get confidence intervals
        forecast_ci = self.fitted_model.get_forecast(
            steps=periods,
            exog=exog_future.to_numpy() if exog_future is not None else None
        ).conf_int()
        
        # Generate future dates
        ts_data = self.prepare_data()
        last_date = ts_data['period_start'].max()
        
        if self.period_type == 'daily':
            future_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        elif self.period_type == 'weekly':
            future_dates = [last_date + timedelta(weeks=i+1) for i in range(periods)]
        elif self.period_type == 'monthly':
            future_dates = pl.date_range(
                last_date,
                last_date + timedelta(days=30*periods),
                interval='1mo',
                eager=True
            )[1:periods+1].to_list()
        else:  # quarterly
            future_dates = [last_date + timedelta(days=91*(i+1)) for i in range(periods)]
        
        forecast_df = pl.DataFrame({
            'period': future_dates[:periods],
            'bank_name': self.bank_name,
            'forecast_fee_volume': forecast_result.values,
            'lower_ci_95': forecast_ci.iloc[:, 0].values,
            'upper_ci_95': forecast_ci.iloc[:, 1].values,
            'period_type': self.period_type
        })
        
        return forecast_df
    
    def backtest(self, test_periods: int = 12):
        """
        Rolling window backtest to evaluate forecast accuracy
        """
        ts_data = self.prepare_data()
        
        if len(ts_data) < test_periods + 20:  # Need sufficient history
            raise ValueError(f"Insufficient data for backtesting. Need at least {test_periods + 20} periods")
        
        config = {
            'daily': DAILY_CONFIG,
            'weekly': WEEKLY_CONFIG,
            'monthly': MONTHLY_CONFIG,
            'quarterly': QUARTERLY_CONFIG
        }[self.period_type]
        
        errors = []
        actuals = []
        
        for i in range(test_periods):
            # Split data
            train_end = len(ts_data) - test_periods + i
            train = ts_data[:train_end]
            actual = ts_data[train_end]['total_fee_volume']
            actuals.append(actual)
            
            # Fit and forecast
            temp_model = SARIMAX(
                train['total_fee_volume'].to_numpy(),
                order=config['order'],
                seasonal_order=config['seasonal_order'],
                trend=config['trend']
            ).fit(disp=False)
            
            forecast = temp_model.forecast(steps=1)[0]
            errors.append(actual - forecast)
        
        # Calculate metrics
        actuals_array = np.array(actuals)
        errors_array = np.array(errors)
        
        mape = np.mean(np.abs(errors_array) / (actuals_array + 1)) * 100
        rmse = np.sqrt(np.mean(errors_array**2))
        mae = np.mean(np.abs(errors_array))
        
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae,
            'mean_actual': np.mean(actuals_array),
            'test_periods': test_periods
        }
```

---

## 4. Nowcasting Implementation

**Nowcasting for daily/weekly data = latest available aggregated metrics**

```python
from datetime import datetime, timedelta

class IPONowcaster:
    def __init__(self, data_store: IPODataStore = None):
        self.data_store = data_store or IPODataStore()
    
    def get_latest_metrics(self, period_type: str = 'daily', bank_name: str = None, lookback_periods: int = 30) -> pl.DataFrame:
        """
        Get the most recent metrics (nowcast)
        
        period_type: 'daily' or 'weekly' (nowcasting doesn't make sense for monthly/quarterly)
        bank_name: Optional filter for specific bank
        lookback_periods: How many recent periods to return
        """
        metrics = self.data_store.read_aggregated_metrics(period_type)
        
        if metrics.is_empty():
            return pl.DataFrame()
        
        # Filter by bank if specified
        if bank_name:
            metrics = metrics.filter(pl.col('bank_name') == bank_name)
        
        # Get most recent N periods
        latest = metrics.sort('period_start', descending=True).head(lookback_periods)
        
        return latest.sort('period_start')
    
    def get_current_period_snapshot(self, period_type: str = 'daily') -> pl.DataFrame:
        """
        Get snapshot of current incomplete period (e.g., today's deals so far, this week's deals)
        """
        today = datetime.now().date()
        
        if period_type == 'daily':
            period_start = today
        elif period_type == 'weekly':
            # Start of current week (Monday)
            period_start = today - timedelta(days=today.weekday())
        else:
            raise ValueError("Nowcasting only supports 'daily' or 'weekly' period types")
        
        # Read raw allocations for current period
        allocations = self.data_store.read_all_allocations(
            start_date=period_start,
            end_date=today
        )
        
        if allocations.is_empty():
            return pl.DataFrame()
        
        # Calculate current metrics
        snapshot = allocations.group_by('bank_name').agg([
            pl.col('estimated_fee').sum().alias('total_fee_volume'),
            pl.col('deal_id').n_unique().alias('deal_count'),
            (pl.col('rank_position') == 1).sum().alias('lead_bookrunner_count'),
            pl.col('gross_proceeds').sum().alias('total_gross_proceeds')
        ])
        
        # Calculate market share
        market_total = snapshot['total_fee_volume'].sum()
        snapshot = snapshot.with_columns([
            (pl.col('total_fee_volume') / market_total * 100).alias('market_share_pct'),
            pl.lit(period_start).alias('period_start'),
            pl.lit(today).alias('period_end'),
            pl.lit(f'{period_type}_incomplete').alias('period_type'),
            pl.lit(datetime.now()).alias('calculated_at')
        ])
        
        return snapshot.sort('total_fee_volume', descending=True)
    
    def get_trailing_metrics(self, period_type: str, trailing_periods: int, bank_name: str = None) -> pl.DataFrame:
        """
        Calculate rolling sum for trailing N periods
        
        Example: trailing_periods=4 for weekly gives you last 4 weeks aggregate
        """
        metrics = self.data_store.read_aggregated_metrics(period_type)
        
        if bank_name:
            metrics = metrics.filter(pl.col('bank_name') == bank_name)
        
        if metrics.is_empty():
            return pl.DataFrame()
        
        # Sort by date
        metrics = metrics.sort(['bank_name', 'period_start'])
        
        # Calculate rolling sum
        rolling_metrics = metrics.group_by('bank_name').agg([
            pl.col('total_fee_volume').rolling_sum(window_size=trailing_periods).alias('trailing_fee_volume'),
            pl.col('deal_count').rolling_sum(window_size=trailing_periods).alias('trailing_deal_count'),
            pl.col('period_start')
        ])
        
        # Explode to get one row per period
        rolling_metrics = rolling_metrics.explode(['trailing_fee_volume', 'trailing_deal_count', 'period_start'])
        
        # Filter to only complete windows
        rolling_metrics = rolling_metrics.filter(
            pl.col('trailing_fee_volume').is_not_null()
        )
        
        return rolling_metrics.sort(['period_start', 'trailing_fee_volume'], descending=[False, True])
    
    def generate_nowcast_report(self, period_type: str = 'weekly') -> dict:
        """
        Generate comprehensive nowcast report
        
        Returns dict with:
        - current_period_snapshot: Incomplete current period data
        - last_complete_period: Most recent complete period
        - trailing_4_periods: Rolling 4-period aggregate
        - top_banks: Current rankings
        """
        # Current incomplete period
        current = self.get_current_period_snapshot(period_type)
        
        # Last complete period
        latest = self.get_latest_metrics(period_type, lookback_periods=1)
        
        # Trailing periods
        trailing = self.get_trailing_metrics(period_type, trailing_periods=4)
        latest_trailing = trailing.filter(
            pl.col('period_start') == trailing['period_start'].max()
        )
        
        return {
            'current_period_incomplete': current,
            'last_complete_period': latest,
            'trailing_4_periods': latest_trailing,
            'period_type': period_type,
            'generated_at': datetime.now()
        }
```

**Example usage:**

```python
# Daily nowcasting
nowcaster = IPONowcaster()

# Get today's deals so far
today_snapshot = nowcaster.get_current_period_snapshot('daily')
print("Today's IPO activity:")
print(today_snapshot)

# Get last 30 days of daily metrics
recent_daily = nowcaster.get_latest_metrics('daily', lookback_periods=30)

# Weekly nowcasting
weekly_report = nowcaster.generate_nowcast_report('weekly')
print("\nThis week so far:")
print(weekly_report['current_period_incomplete'])
print("\nLast complete week:")
print(weekly_report['last_complete_period'])
print("\nTrailing 4 weeks:")
print(weekly_report['trailing_4_periods'])
```

---

## 5. System Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                  DAILY/WEEKLY SCHEDULER                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  1. Scrape new 424B4 filings from SEC EDGAR            │ │
│  │  2. Parse and extract IPO data                         │ │
│  │  3. Allocate fees to banks using position assumptions  │ │
│  │  4. Append to Parquet files (partitioned)              │ │
│  │  5. Regenerate aggregations (daily/weekly/monthly/qtr) │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    PARQUET STORAGE                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  data/                                                  │ │
│  │  ├── raw_filings/        (partitioned by year/quarter) │ │
│  │  ├── bank_allocations/   (partitioned by year)         │ │
│  │  └── aggregated/                                        │ │
│  │      ├── daily_metrics.parquet                          │ │
│  │      ├── weekly_metrics.parquet                         │ │
│  │      ├── monthly_metrics.parquet                        │ │
│  │      └── quarterly_metrics.parquet                      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    ANALYTICS LAYER                           │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │   NOWCASTING     │  │   FORECASTING    │                │
│  │   (Daily/Weekly) │  │   (Monthly/Qtr)  │                │
│  │                  │  │                  │                │
│  │  • Current period│  │  • SARIMAX model │                │
│  │    snapshot      │  │  • 3-6 period    │                │
│  │  • Trailing N    │  │    forecast      │                │
│  │    periods       │  │  • Confidence    │                │
│  │  • Read directly │  │    intervals     │                │
│  │    from Parquet  │  │  • Backtesting   │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT/REPORTING                          │
│  • Bank league tables (by period)                           │
│  • Market share trends                                      │
│  • Nowcast dashboards (daily/weekly)                        │
│  • Forecast dashboards (monthly/quarterly)                  │
│  • Alert system (e.g., "Goldman lost #1 position")         │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Developer Handoff Checklist

**Phase 1: Data Pipeline (Weeks 1-2)**
- [ ] Set up SEC EDGAR scraper with daily/weekly scheduler
- [ ] Build HTML parser for 424B4 filings
- [ ] Implement underwriter extraction logic
- [ ] Create Parquet storage structure (IPODataStore class)
- [ ] Build bank name normalization function
- [ ] Implement data validation rules
- [ ] Test with historical filings (sample 100+ IPOs)
- [ ] Set up partition strategy (year/quarter for raw, year for allocations)

**Phase 2: Fee Allocation & Aggregation (Week 3)**
- [ ] Implement fee allocation algorithm
- [ ] Create rank-based assumption configuration
- [ ] Build aggregation functions (daily/weekly/monthly/quarterly)
- [ ] Calculate market share metrics
- [ ] Create unit tests for edge cases
- [ ] Verify Parquet read/write performance

**Phase 3: Nowcasting (Week 4)**
- [ ] Build IPONowcaster class
- [ ] Implement current period snapshot logic
- [ ] Create trailing metrics calculations
- [ ] Build comprehensive nowcast report generator
- [ ] Test with live data
- [ ] Create visualization outputs

**Phase 4: Forecasting (Week 5)**
- [ ] Install and configure SARIMAX (statsmodels)
- [ ] Build BankIPOForecaster class with configs for each period type
- [ ] Implement backtesting framework
- [ ] Test on historical data (2020-2024)
- [ ] Tune hyperparameters per bank and period type
- [ ] Document forecast accuracy metrics by bank


**Key Files to Create:**
1. `scraper.py` - SEC EDGAR data extraction
2. `parser.py` - HTML/XML parsing logic
3. `allocator.py` - Fee allocation to banks
4. `data_store.py` - Parquet I/O (IPODataStore class)
5. `aggregator.py` - Calculate metrics by period
6. `nowcaster.py` - Real-time metrics (IPONowcaster class)
7. `forecaster.py` - SARIMAX modeling (BankIPOForecaster class)
8. `config.py` - Assumptions and parameters (fee allocation, model configs)
9. `utils.py` - Bank name normalization, validation
10. `scheduler.py` - Daily/weekly job orchestration

**Data Files should only be included if the functionality cannot be intergrated into the existing data pulling framework**

**Parquet Performance Tips:**
- Use snappy compression for good balance of speed/size
- Partition raw filings by year/quarter for efficient queries
- Use column projection when reading (only select needed columns)
- Consider row group size (~100MB) for optimal performance
- Use `pl.scan_parquet()` for lazy evaluation on large datasets

This revised design uses Parquet for storage and focuses nowcasting on daily/weekly granularity where it makes the most sense.