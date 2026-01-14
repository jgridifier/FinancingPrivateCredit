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

### Quick Start

```python
# Basic usage test
from financing_private_credit.indicators import get_indicator

indicator = get_indicator("my_indicator")
data = indicator.fetch_data("2015-01-01")
result = indicator.calculate(data)

print(result.data)
```

### CLI Validation (Recommended)

Use the validation tool to check your indicator:

```bash
# Validate a specific indicator
financing-private-credit validate my_indicator

# Validate all indicators
financing-private-credit validate --all

# Output as JSON for CI/CD
financing-private-credit validate my_indicator --json
```

### Test File Structure

Create `tests/indicators/test_my_indicator.py`:

```python
"""
Tests for MyIndicator

Run with: pytest tests/indicators/test_my_indicator.py -v
"""

import pytest
import polars as pl
from datetime import datetime

from financing_private_credit.indicators import get_indicator
from financing_private_credit.indicators.base import IndicatorResult


# ============================================================================
# Fixtures - Reusable test data
# ============================================================================

@pytest.fixture
def mock_bank_panel():
    """Create mock bank panel data for testing."""
    return pl.DataFrame({
        "ticker": ["JPM", "BAC", "C", "WFC"] * 4,
        "date": (
            ["2024-Q1"] * 4 + ["2024-Q2"] * 4 +
            ["2024-Q3"] * 4 + ["2024-Q4"] * 4
        ),
        "total_assets": [3000, 2500, 2000, 1800] * 4,
        "total_loans": [1000, 900, 800, 700] * 4,
        "deposits": [2000, 1800, 1500, 1300] * 4,
        "total_equity": [300, 250, 200, 180] * 4,
        "net_income": [30, 25, 20, 18] * 4,
    })


@pytest.fixture
def mock_macro_data():
    """Create mock macro data for testing."""
    return pl.DataFrame({
        "date": ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"],
        "FEDFUNDS": [5.25, 5.50, 5.25, 5.00],
        "DGS10": [4.20, 4.30, 4.10, 4.00],
        "BAA10Y": [1.50, 1.60, 1.55, 1.45],
    })


@pytest.fixture
def mock_data(mock_bank_panel, mock_macro_data):
    """Combined mock data dictionary."""
    return {
        "bank_panel": mock_bank_panel,
        "macro_data": mock_macro_data,
    }


# ============================================================================
# Registration Tests
# ============================================================================

class TestMyIndicatorRegistration:
    """Test indicator registration and instantiation."""

    def test_registration(self):
        """Test that indicator is properly registered."""
        indicator = get_indicator("my_indicator")
        assert indicator is not None
        assert indicator.__class__.__name__ == "MyIndicator"

    def test_metadata_complete(self):
        """Test that metadata is complete and valid."""
        indicator = get_indicator("my_indicator")
        metadata = indicator.get_metadata()

        assert metadata.name is not None and len(metadata.name) > 0
        assert metadata.short_name is not None
        assert metadata.description is not None
        assert len(metadata.data_sources) > 0
        assert metadata.update_frequency in ["daily", "weekly", "monthly", "quarterly"]
        assert metadata.lookback_periods > 0
        assert metadata.version is not None

    def test_supports_nowcast_flag(self):
        """Test nowcast support flag is properly set."""
        indicator = get_indicator("my_indicator")
        assert isinstance(indicator.supports_nowcast, bool)


# ============================================================================
# Calculation Tests
# ============================================================================

class TestMyIndicatorCalculation:
    """Test indicator calculation logic."""

    def test_calculate_returns_result(self, mock_data):
        """Test that calculate returns an IndicatorResult."""
        indicator = get_indicator("my_indicator")
        result = indicator.calculate(mock_data)

        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "my_indicator"
        assert isinstance(result.calculation_date, datetime)

    def test_calculate_with_mock_data(self, mock_data):
        """Test calculation produces expected output."""
        indicator = get_indicator("my_indicator")
        result = indicator.calculate(mock_data)

        # Check result has expected structure
        assert result.data.height > 0
        # Add indicator-specific assertions:
        # assert "my_metric" in result.data.columns
        # assert result.data["my_metric"].min() >= 0

    def test_calculate_metadata(self, mock_data):
        """Test that calculation includes useful metadata."""
        indicator = get_indicator("my_indicator")
        result = indicator.calculate(mock_data)

        assert isinstance(result.metadata, dict)
        # Check for expected metadata keys:
        # assert "n_banks" in result.metadata


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestMyIndicatorEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test graceful handling of empty data."""
        indicator = get_indicator("my_indicator")
        empty_data = {
            "bank_panel": pl.DataFrame(),
            "macro_data": pl.DataFrame(),
        }
        result = indicator.calculate(empty_data)

        assert isinstance(result, IndicatorResult)
        # Should handle gracefully, possibly with error in metadata

    def test_missing_columns(self, mock_bank_panel):
        """Test handling of missing columns."""
        indicator = get_indicator("my_indicator")
        # Remove a required column
        incomplete_data = {
            "bank_panel": mock_bank_panel.drop("total_loans"),
            "macro_data": pl.DataFrame(),
        }
        # Should either handle gracefully or raise informative error
        # result = indicator.calculate(incomplete_data)

    def test_single_bank(self, mock_macro_data):
        """Test with single bank data."""
        indicator = get_indicator("my_indicator")
        single_bank = pl.DataFrame({
            "ticker": ["JPM"] * 4,
            "date": ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"],
            "total_assets": [3000, 3100, 3200, 3300],
            "total_loans": [1000, 1050, 1100, 1150],
            "deposits": [2000, 2100, 2200, 2300],
        })
        data = {"bank_panel": single_bank, "macro_data": mock_macro_data}
        result = indicator.calculate(data)

        assert isinstance(result, IndicatorResult)

    def test_single_period(self, mock_macro_data):
        """Test with single time period."""
        indicator = get_indicator("my_indicator")
        single_period = pl.DataFrame({
            "ticker": ["JPM", "BAC", "C", "WFC"],
            "date": ["2024-Q1"] * 4,
            "total_assets": [3000, 2500, 2000, 1800],
            "total_loans": [1000, 900, 800, 700],
            "deposits": [2000, 1800, 1500, 1300],
        })
        data = {"bank_panel": single_period, "macro_data": mock_macro_data}
        result = indicator.calculate(data)

        assert isinstance(result, IndicatorResult)


# ============================================================================
# Data Validation Tests
# ============================================================================

class TestMyIndicatorValidation:
    """Test data validation."""

    def test_validate_data_empty(self):
        """Test validation catches empty DataFrames."""
        indicator = get_indicator("my_indicator")
        errors = indicator.validate_data({"bank_panel": pl.DataFrame()})
        assert len(errors) > 0
        assert any("Empty" in e for e in errors)

    def test_validate_data_missing_date(self):
        """Test validation catches missing date column."""
        indicator = get_indicator("my_indicator")
        no_date = pl.DataFrame({"ticker": ["JPM"], "value": [100]})
        errors = indicator.validate_data({"bank_panel": no_date})
        assert any("date" in e.lower() for e in errors)

    def test_required_data_sources(self):
        """Test that required data sources are documented."""
        indicator = get_indicator("my_indicator")
        sources = indicator.get_required_data_sources()
        assert isinstance(sources, list)


# ============================================================================
# Nowcast Tests (if supports_nowcast=True)
# ============================================================================

class TestMyIndicatorNowcast:
    """Test nowcasting functionality (if enabled)."""

    def test_nowcast_not_supported(self):
        """Test that nowcast raises if not supported."""
        indicator = get_indicator("my_indicator")
        if not indicator.supports_nowcast:
            with pytest.raises(NotImplementedError):
                indicator.nowcast({})

    # Uncomment if supports_nowcast=True:
    # def test_nowcast_returns_result(self, mock_data):
    #     """Test that nowcast returns valid result."""
    #     indicator = get_indicator("my_indicator")
    #     mock_data["stock_data"] = pl.DataFrame({
    #         "ticker": ["JPM", "BAC"],
    #         "date": ["2024-01-15", "2024-01-15"],
    #         "close": [150.0, 35.0],
    #     })
    #     result = indicator.nowcast(mock_data)
    #     assert isinstance(result, IndicatorResult)


# ============================================================================
# Forecast Tests (if forecast.py exists)
# ============================================================================

class TestMyIndicatorForecast:
    """Test forecasting functionality (if implemented)."""

    def test_forecaster_fit(self, mock_bank_panel):
        """Test forecaster fit method."""
        try:
            from financing_private_credit.indicators.my_indicator import MyForecaster
        except ImportError:
            pytest.skip("Forecaster not implemented")

        forecaster = MyForecaster()
        assert not forecaster.is_fitted

        fit_result = forecaster.fit(
            data=mock_bank_panel,
            target="total_loans",
            features=["total_assets", "deposits"],
        )
        assert forecaster.is_fitted
        assert isinstance(fit_result, dict)

    def test_forecaster_predict(self, mock_bank_panel):
        """Test forecaster predict method."""
        try:
            from financing_private_credit.indicators.my_indicator import MyForecaster
            from financing_private_credit.indicators.base import ForecastResult
        except ImportError:
            pytest.skip("Forecaster not implemented")

        forecaster = MyForecaster()
        forecaster.fit(
            data=mock_bank_panel,
            target="total_loans",
            features=["total_assets", "deposits"],
        )

        forecast = forecaster.predict(data=mock_bank_panel, horizon=4)
        assert isinstance(forecast, ForecastResult)
        assert forecast.predictions.height > 0
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific indicator tests
pytest tests/indicators/test_my_indicator.py -v

# Run with coverage
pytest tests/ --cov=src/financing_private_credit --cov-report=html

# Run only fast unit tests (no network)
pytest tests/ -m "not slow" -v
```

### Mocking Best Practices

**Avoid Network Calls in Tests:**

```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_registry():
    """Create a mock DataRegistry for testing."""
    with patch("financing_private_credit.core.DataRegistry.get_instance") as mock:
        registry = Mock()
        registry.get_bank_panel.return_value = pl.DataFrame({...})
        registry.get_macro_series.return_value = pl.DataFrame({...})
        mock.return_value = registry
        yield registry

def test_fetch_data_uses_registry(mock_registry):
    """Test that fetch_data uses DataRegistry correctly."""
    indicator = get_indicator("my_indicator")
    data = indicator.fetch_data("2015-01-01")

    mock_registry.get_bank_panel.assert_called_once()
```

### Test Markers

Use markers for test organization:

```python
import pytest

@pytest.mark.slow
def test_full_data_fetch():
    """Test with real data - requires network."""
    ...

@pytest.mark.integration
def test_end_to_end_workflow():
    """Test full workflow."""
    ...
```

Configure in `pytest.ini`:

```ini
[pytest]
markers =
    slow: marks tests as slow (requires network)
    integration: marks tests as integration tests
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
