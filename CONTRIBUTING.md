# Contributing to the Financial Indicator Framework

This guide explains how to add new indicators to the framework.

## Quick Start

1. Copy the template: `cp -r src/financing_private_credit/indicators/_template src/financing_private_credit/indicators/my_indicator`
2. Implement the required methods in `indicator.py`
3. Register your indicator with `@register_indicator("my_indicator")`
4. Add exports to `__init__.py`
5. Create a model spec in `config/model_specs/`

## Indicator Architecture

Each indicator is a self-contained package with a standard structure:

```
indicators/
└── my_indicator/
    ├── __init__.py       # Package exports
    ├── indicator.py      # Core indicator class (required)
    ├── forecast.py       # Forecasting models (optional)
    ├── nowcast.py        # High-frequency updates (optional)
    ├── backtest.py       # Model validation (optional)
    ├── viz.py            # Visualizations (optional)
    └── README.md         # Indicator documentation (recommended)
```

## Step-by-Step Guide

### 1. Create the Indicator Class

Your indicator must inherit from `BaseIndicator` and implement three methods:

```python
# indicators/my_indicator/indicator.py

from ..base import BaseIndicator, IndicatorMetadata, IndicatorResult, register_indicator

@register_indicator("my_indicator")
class MyIndicator(BaseIndicator):
    """One-line description of what this indicator measures."""

    def get_metadata(self) -> IndicatorMetadata:
        """Return indicator metadata."""
        return IndicatorMetadata(
            name="My Indicator Full Name",
            short_name="MyInd",
            description="Detailed description of what this measures and why it matters.",
            version="1.0.0",
            paper_reference="Optional citation",
            data_sources=["SEC EDGAR", "FRED"],  # List your data sources
            update_frequency="quarterly",  # or "daily", "weekly", "monthly"
            lookback_periods=20,  # How much history needed
        )

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch all required data."""
        # Use existing data fetchers
        from ...bank_data import BankDataCollector, TARGET_BANKS
        from ...cache import CachedFREDFetcher

        collector = BankDataCollector(start_date=start_date)
        bank_panel = collector.fetch_all_banks()

        return {
            "bank_panel": bank_panel,
            # Add other data as needed
        }

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """Calculate the indicator values."""
        # Your calculation logic here
        result_df = ...

        return IndicatorResult(
            indicator_name="my_indicator",
            calculation_date=datetime.now(),
            data=result_df,
            metadata={"key": "value"},
        )
```

### 2. Register in `__init__.py`

```python
# indicators/my_indicator/__init__.py

from .indicator import MyIndicator

__all__ = ["MyIndicator"]
```

Then add to the parent `indicators/__init__.py`:

```python
from .my_indicator import MyIndicator

# Update the imports dict
_INDICATOR_IMPORTS = {
    # ... existing indicators ...
    "my_indicator": ("my_indicator", "MyIndicator"),
}
```

### 3. Create Model Specifications

Model specs configure your indicator's parameters:

```json
// config/model_specs/my_indicator.json
{
    "name": "my_indicator_default",
    "description": "Default configuration for my indicator",
    "target": "my_metric",
    "parameters": {
        "window": 20,
        "threshold": 0.5
    }
}
```

For per-ticker configurations, use the `_` default pattern:

```json
{
    "_": {
        "component_a": {"param": 1},
        "component_b": {"param": 2}
    },
    "JPM": {
        "component_a": {"param": 3}
    }
}
```

### 4. Add Forecasting (Optional)

```python
# indicators/my_indicator/forecast.py

from dataclasses import dataclass
import polars as pl

@dataclass
class MyForecastResult:
    ticker: str
    forecast_date: str
    point_forecast: float
    confidence_lower: float
    confidence_upper: float

class MyForecaster:
    """Forecast my indicator using ARDL or other models."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.models = {}

    def fit(self, data: pl.DataFrame, macro_data: pl.DataFrame) -> dict:
        """Fit forecasting models for each bank."""
        # Implementation
        pass

    def predict(self, ticker: str, horizon: int = 4) -> MyForecastResult:
        """Generate forecasts."""
        pass
```

### 5. Add Nowcasting (Optional)

For high-frequency updates between quarterly releases:

```python
# indicators/my_indicator/nowcast.py

class MyNowcaster:
    """Update indicator estimates using high-frequency proxies."""

    def __init__(self, quarterly_data: pl.DataFrame):
        self.quarterly_data = quarterly_data

    def update(self, proxy_data: pl.DataFrame) -> pl.DataFrame:
        """Produce nowcast using proxy variables."""
        pass
```

### 6. Add Backtesting (Optional)

```python
# indicators/my_indicator/backtest.py

@dataclass
class BacktestResult:
    spec_name: str
    mae: float
    rmse: float
    directional_accuracy: float

class MyBacktester:
    """Validate indicator forecasting performance."""

    def run_backtest(
        self,
        data: pl.DataFrame,
        initial_window: int = 20,
        step: int = 1,
    ) -> BacktestResult:
        """Rolling-window backtest."""
        pass
```

### 7. Add Visualizations (Optional)

Use Vega-Altair with numbered charts for storytelling:

```python
# indicators/my_indicator/viz.py

import altair as alt

class MyVisualizer:
    """Numbered charts for indicator analysis."""

    def __init__(self, data: pl.DataFrame):
        self.data = data

    def chart_1_overview(self) -> alt.Chart:
        """Chart 1: Overview of indicator values across banks."""
        pass

    def chart_2_time_series(self, ticker: str) -> alt.Chart:
        """Chart 2: Time series for a specific bank."""
        pass
```

## Data Sources

### Using Existing Fetchers

```python
# Bank-level SEC data
from ...bank_data import BankDataCollector, TARGET_BANKS

collector = BankDataCollector(start_date="2015-01-01")
panel = collector.fetch_all_banks()

# FRED macro data
from ...cache import CachedFREDFetcher

fred = CachedFREDFetcher(max_age_hours=6)
data = fred.fetch_multiple_series(["FEDFUNDS", "DGS10"], start_date="2015-01-01")

# Macro data with derived variables
from ...macro import MacroDataFetcher

macro = MacroDataFetcher(start_date="2015-01-01")
df = macro.compute_derived_variables()
```

### Bank Coverage

All indicators should use `TARGET_BANKS` from `bank_data.py`:

| Tier | Banks |
|------|-------|
| 1 (G-SIBs) | JPM, BAC, C, WFC |
| 2 (Large) | GS, MS, BK, STT |
| 3 (Regional) | USB, PNC, TFC, COF, SCHW, NTRS, RJF |

## Testing Your Indicator

```python
# Basic usage test
from financing_private_credit.indicators import get_indicator

indicator = get_indicator("my_indicator")
data = indicator.fetch_data("2015-01-01")
result = indicator.calculate(data)

print(result.data)
```

Add unit tests in `tests/test_indicators.py`:

```python
def test_my_indicator():
    from financing_private_credit.indicators import get_indicator

    indicator = get_indicator("my_indicator")
    assert indicator is not None
    assert indicator.get_metadata().name == "My Indicator Full Name"
```

## Code Style

- Use type hints for all public methods
- Use Polars (not Pandas) for DataFrames
- Use dataclasses for structured results
- Follow existing naming conventions
- Add docstrings with Args/Returns sections

## Checklist

Before submitting:

- [ ] Indicator class inherits from `BaseIndicator`
- [ ] Registered with `@register_indicator("name")`
- [ ] `get_metadata()` returns complete `IndicatorMetadata`
- [ ] `fetch_data()` uses `TARGET_BANKS` for all banks
- [ ] `calculate()` returns `IndicatorResult`
- [ ] Added to `indicators/__init__.py`
- [ ] Created model spec in `config/model_specs/`
- [ ] Added README.md documenting the indicator
- [ ] Tests pass
