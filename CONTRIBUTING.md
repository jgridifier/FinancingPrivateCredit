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
    ├── indicator.py      # Core indicator class (REQUIRED)
    ├── forecast.py       # Forecasting models (optional)
    ├── nowcast.py        # High-frequency updates (optional)
    ├── backtest.py       # Model validation (optional)
    ├── viz.py            # Visualizations (optional)
    └── README.md         # Indicator documentation (recommended)
```

## Temporal Pipeline

Understand when each component runs:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEMPORAL PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   T-8 ←─────→ T-1         T (now)          T+1 ←─────→ T+4     │
│   ───────────────         ────────         ────────────────     │
│   calculate()             nowcast()         forecast()          │
│                                                                  │
│   Historical data         Adjust for        Predict future      │
│   (quarterly SEC          current market    under macro         │
│   filings, FRED)          using proxies     scenarios           │
│                           (stocks, CDS)                          │
│                                                                  │
│   Updates: Quarterly      Updates: Daily    Updates: On-demand  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

- **calculate()**: Core indicator using quarterly data (REQUIRED)
- **nowcast()**: High-frequency proxy-based updates (OPTIONAL)
- **forecast()**: Future predictions under scenarios (OPTIONAL)

## Required vs Optional Methods

### Required Methods

Every indicator must implement these three methods:

| Method | Purpose |
|--------|---------|
| `get_metadata()` | Describe your indicator (name, sources, frequency) |
| `fetch_data()` | Gather required data using DataRegistry |
| `calculate()` | Compute indicator values from data |

### Optional Methods (Have Sensible Defaults)

| Method | Default Behavior | Override When |
|--------|------------------|---------------|
| `nowcast()` | Raises NotImplementedError | You have high-frequency proxies |
| `get_dashboard_components()` | Returns minimal config | You need custom dashboard |
| `get_required_data_sources()` | Returns empty list | You want to document dependencies |
| `validate_data()` | Checks for empty DataFrames | You need custom validation |

## Step-by-Step Guide

### 1. Create the Indicator Class

Your indicator must inherit from `BaseIndicator` and implement three methods:

```python
# indicators/my_indicator/indicator.py

from ..base import BaseIndicator, IndicatorMetadata, IndicatorResult, register_indicator
from ...core import DataRegistry

@register_indicator("my_indicator")
class MyIndicator(BaseIndicator):
    """One-line description of what this indicator measures."""

    # Set to True if you implement nowcast()
    supports_nowcast: bool = False

    def get_metadata(self) -> IndicatorMetadata:
        """Return indicator metadata."""
        return IndicatorMetadata(
            name="My Indicator Full Name",
            short_name="MyInd",
            description="Detailed description of what this measures and why it matters.",
            version="1.0.0",
            paper_reference="Optional citation",
            data_sources=["SEC EDGAR", "FRED"],
            update_frequency="quarterly",
            lookback_periods=20,
        )

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch all required data using DataRegistry."""
        registry = DataRegistry.get_instance()

        # Shared data (cached automatically)
        bank_panel = registry.get_bank_panel(start_date)
        macro_data = registry.get_macro_series(["FEDFUNDS", "DGS10"], start_date)

        return {
            "bank_panel": bank_panel,
            "macro_data": macro_data,
        }

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """Calculate the indicator values."""
        result_df = ...  # Your calculation logic

        return IndicatorResult(
            indicator_name="my_indicator",
            calculation_date=datetime.now(),
            data=result_df,
            metadata={"key": "value"},
        )
```

### 2. Using DataRegistry (Recommended)

The `DataRegistry` provides centralized data fetching with smart caching:

```python
from financing_private_credit.core import DataRegistry

# Get singleton instance
registry = DataRegistry.get_instance()

# Shared data sources (fetched once, cached in Arrow format)
bank_panel = registry.get_bank_panel("2015-01-01")
macro_data = registry.get_macro_series(["FEDFUNDS", "DGS10", "BAA10Y"], "2015-01-01")

# Register custom data source
def fetch_call_reports(start_date: str, schedule: str = "RC-O") -> pl.DataFrame:
    # Your custom fetching logic
    ...

registry.register_source("call_reports", fetch_call_reports, ttl_hours=48)
call_data = registry.get("call_reports", start_date="2015-01-01", schedule="RC-O")

# Cache management
registry.invalidate("bank_panel")  # Clear specific source
registry.invalidate()  # Clear all cached data
registry.force_refresh()  # Bypass cache on next fetch
```

**Cache Configuration:**
- Bank panel data: 24 hours (quarterly updates)
- FRED daily series: 6 hours
- FRED weekly series (H.8): 24 hours
- Custom sources: 12 hours (configurable)

### 3. Register in `__init__.py`

```python
# indicators/my_indicator/__init__.py

from .indicator import MyIndicator

__all__ = ["MyIndicator"]
```

Then add to the parent `indicators/__init__.py`:

```python
from .my_indicator import MyIndicator

_INDICATOR_IMPORTS = {
    # ... existing indicators ...
    "my_indicator": ("my_indicator", "MyIndicator"),
}
```

### 4. Create Model Specifications

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

### 5. Add Forecasting (Optional)

Create `forecast.py` using `BaseForecastModel`:

```python
# indicators/my_indicator/forecast.py

from ..base import BaseForecastModel, ForecastResult

class MyForecaster(BaseForecastModel[None]):
    """Forecast my indicator using custom logic."""

    def fit(
        self,
        data: pl.DataFrame,
        target: str,
        features: list[str],
        **kwargs,
    ) -> dict[str, Any]:
        """Fit the model."""
        self._target = target
        self._features = features
        self._is_fitted = True
        return {"n_observations": data.height}

    def predict(
        self,
        data: pl.DataFrame,
        horizon: int = 4,
        **kwargs,
    ) -> ForecastResult:
        """Generate predictions."""
        # Your prediction logic
        predictions_df = ...

        return ForecastResult(
            target=self._target,
            horizon=horizon,
            predictions=predictions_df,
        )
```

`BaseForecastModel` is model-agnostic - use it with sklearn, statsmodels, PyTorch, or custom implementations:

```python
# With sklearn
class RandomForestForecaster(BaseForecastModel[RandomForestRegressor]):
    ...

# With statsmodels
class ARDLForecaster(BaseForecastModel[AutoReg]):
    ...

# Custom implementation
class CustomForecaster(BaseForecastModel[None]):
    ...
```

### 6. Add Nowcasting (Optional)

For high-frequency updates between quarterly releases:

```python
# indicators/my_indicator/nowcast.py

class MyNowcaster:
    """Update indicator estimates using high-frequency proxies."""

    def nowcast(
        self,
        quarterly_data: pl.DataFrame,
        proxy_data: dict[str, pl.DataFrame],
    ) -> IndicatorResult:
        """Produce nowcast using proxy variables."""
        # Use stock prices, CDS spreads, etc. to adjust
        ...
```

Then enable in your indicator:

```python
class MyIndicator(BaseIndicator):
    supports_nowcast = True  # Enable nowcasting

    def nowcast(self, data, **kwargs) -> IndicatorResult:
        from .nowcast import MyNowcaster
        nowcaster = MyNowcaster()
        return nowcaster.nowcast(...)
```

### 7. Add Backtesting (Optional)

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

### 8. Add Visualizations (Optional)

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

### Using DataRegistry (Recommended)

```python
from financing_private_credit.core import DataRegistry

registry = DataRegistry.get_instance()

# Bank-level SEC data (cached)
bank_panel = registry.get_bank_panel("2015-01-01")

# FRED macro data (cached)
macro_data = registry.get_macro_series(
    ["FEDFUNDS", "DGS10", "UNRATE"],
    "2015-01-01"
)

# Data quality summary
quality = registry.get_data_quality_summary()
```

### Using Fetchers Directly (Legacy)

```python
# Bank-level SEC data
from ...bank_data import BankDataCollector, TARGET_BANKS

collector = BankDataCollector(start_date="2015-01-01")
panel = collector.fetch_all_banks()

# FRED macro data
from ...cache import CachedFREDFetcher

fred = CachedFREDFetcher(max_age_hours=6)
data = fred.fetch_multiple_series(["FEDFUNDS", "DGS10"], start_date="2015-01-01")
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
    assert indicator.supports_nowcast == False  # or True if implemented
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
- [ ] `fetch_data()` uses DataRegistry for shared data
- [ ] `calculate()` returns `IndicatorResult`
- [ ] Added to `indicators/__init__.py`
- [ ] Created model spec in `config/model_specs/`
- [ ] Added README.md documenting the indicator
- [ ] Tests pass

## Common Patterns

### Indicator with Nowcasting

```python
@register_indicator("my_indicator")
class MyIndicator(BaseIndicator):
    supports_nowcast = True

    def nowcast(self, data, **kwargs) -> IndicatorResult:
        # Implement high-frequency updates
        ...
```

### Indicator with Decomposition

```python
from ..base import BaseDecomposition

class MyDecomposition(BaseDecomposition):
    def decompose(self, data, entity) -> pl.DataFrame:
        # Implement variance/growth decomposition
        ...

    def compute_variance_shares(self, decomposition) -> dict[str, float]:
        # Compute contribution shares
        ...
```

### Custom Dashboard

```python
def get_dashboard_components(self) -> dict[str, Any]:
    return {
        "tabs": [
            {"name": "Resilience Scores", "icon": "shield"},
            {"name": "Risk Factors", "icon": "warning"},
        ],
        "primary_metric": "resilience_score",
        "alert_fields": ["is_stressed", "needs_review"],
    }
```
