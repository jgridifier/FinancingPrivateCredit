# Repository Structure Review & Enhancement Plan

**Date:** 2026-01-14
**Status:** Planning Phase
**Goal:** Improve framework usability for both human developers and AI coding agents

---

## Executive Summary

This is a **well-architected framework** with strong fundamentals. The repository demonstrates professional software engineering practices with clear abstractions, comprehensive documentation, and consistent patterns across indicators. However, there are opportunities to improve discoverability, reduce friction for new contributors, and provide better tooling for AI agents.

**Overall Assessment:**
- **Ease of Development:** 7.5/10
- **AI Agent Friendliness:** 8/10
- **Code Quality:** 9/10
- **Documentation:** 9/10

---

## Current State Analysis

### Repository Structure

```
FinancingPrivateCredit/
├── src/financing_private_credit/
│   ├── indicators/                         # 8 indicator implementations
│   │   ├── base.py                        # Abstract base classes & registry
│   │   ├── _template/                     # Template for new indicators
│   │   ├── FASAR/                         # Flex-Adjusted Syndicate Absorption Ratio
│   │   ├── bank_macro_sensitivity/        # Bank-specific macro elasticities
│   │   ├── credit_boom/                   # Credit Boom Leading Indicator (LIS)
│   │   ├── demand_system/                 # Paper replication (original)
│   │   ├── duration_mismatch/             # Duration exposure signal
│   │   ├── funding_stability/             # Funding resilience score
│   │   └── variance_decomposition/        # Cross-bank variance analysis
│   ├── core/                              # Infrastructure (7 modules)
│   │   ├── config.py, registry.py, model_specs.py
│   │   ├── data_registry.py               # Centralized data caching
│   │   ├── sec_edgar.py, llm_extractor.py
│   │   └── utils.py
│   ├── bank_data.py, cache.py, data.py, macro.py
│   └── dashboard.py                       # Streamlit dashboard
├── config/model_specs/                    # JSON specifications (10 files)
├── examples/                              # 9 example scripts
├── tests/                                 # Test suite
├── CONTRIBUTING.md, README.md
└── pyproject.toml
```

### Indicator Package Structure

**Required Files:**
- `__init__.py` - Package exports
- `indicator.py` - Core indicator class (must inherit from `BaseIndicator`)

**Optional Files (Common Patterns):**
- `forecast.py` - Forecasting models (present in 6/7 indicators)
- `nowcast.py` - High-frequency updates (present in 6/7 indicators)
- `backtest.py` - Model validation (present in 3/7 indicators)
- `viz.py` - Visualizations (present in 3/7 indicators)
- `README.md` - Documentation (present in 4/7 indicators)

### File Coverage by Indicator

| Indicator | indicator.py | forecast.py | nowcast.py | backtest.py | viz.py | README.md |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| FASAR | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| bank_macro_sensitivity | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| credit_boom | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| demand_system | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| duration_mismatch | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| funding_stability | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| variance_decomposition | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| _template | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |

---

## Strengths (What's Working Well)

### 1. Clear Abstraction Layer ⭐⭐⭐⭐⭐

**BaseIndicator Design:**
- Only 3 required methods: `get_metadata()`, `fetch_data()`, `calculate()`
- Optional methods have sensible defaults
- Generic base classes support different indicator types

**Code Example:**
```python
@register_indicator("my_indicator")
class MyIndicator(BaseIndicator):
    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(...)

    def fetch_data(self, start_date, end_date=None) -> dict[str, pl.DataFrame]:
        registry = DataRegistry.get_instance()
        return {"bank_panel": registry.get_bank_panel(start_date)}

    def calculate(self, data, **kwargs) -> IndicatorResult:
        return IndicatorResult(...)
```

### 2. Excellent Documentation ⭐⭐⭐⭐⭐

- CONTRIBUTING.md: 500+ lines with step-by-step guide
- Temporal pipeline diagram explaining when components run
- Template with inline comments
- 9 working examples

### 3. Consistent Implementation ⭐⭐⭐⭐

- All 7 indicators follow the same structure
- Registry pattern for discovery: `get_indicator("name")`
- Spec objects with JSON serialization
- DataRegistry for shared data with smart caching

### 4. Infrastructure Quality ⭐⭐⭐⭐⭐

- Arrow Feather format caching (fast, compressed)
- Centralized data fetching reduces API calls
- Type hints throughout
- Polars DataFrames for performance
- Configuration system with JSON specs

---

## Friction Points & Challenges

### For Human Developers

1. **Manual Setup** - Must manually copy template directory and edit multiple files
2. **No Validation** - No tool to check if indicator follows conventions before runtime
3. **Registration Steps** - Must remember to add to multiple `__init__.py` files
4. **Testing Unclear** - No guide for writing indicator tests
5. **Spec Patterns Vary** - Each indicator implements Spec slightly differently

### For AI Coding Agents

1. **Optional Patterns Not Explicit** - forecast.py/nowcast.py/backtest.py/viz.py are common but not formally specified
2. **Multi-File Updates** - Adding an indicator requires touching 3+ files in specific ways
3. **Documentation Scattered** - Some indicators have README.md (4/7), some don't
4. **Error Handling Unclear** - No standard approach for handling data fetching failures
5. **Hard-Coded Dependencies** - DataRegistry instantiation makes testing harder

---

## Enhancement Options

### Option 1: CLI Scaffolding Tool

**Priority:** P0 (Highest)
**Impact:** High
**Effort:** Low

#### Problem
Manual copy-paste of template is error-prone and requires multiple file edits.

#### Solution
Create a CLI command to scaffold new indicators automatically.

#### Proposed Usage
```bash
financing-private-credit new-indicator my_indicator \
  --author "Your Name" \
  --description "Brief description" \
  --supports-nowcast \
  --supports-forecast
```

#### What It Would Do
1. Copy `_template/` to `indicators/my_indicator/`
2. Replace all "Template" references with "MyIndicator"
3. Update `indicators/__init__.py` automatically
4. Generate spec file in `config/model_specs/`
5. Create placeholder test file
6. Create README.md from template
7. Optionally create forecast.py, nowcast.py, backtest.py, viz.py

#### Benefits
- Reduces setup time from ~15 minutes to 30 seconds
- Eliminates copy-paste errors
- Ensures all required files are created
- **AI agents can call this command directly**
- Enforces naming conventions

#### Implementation Plan
```python
# src/financing_private_credit/cli/scaffold.py

def scaffold_indicator(
    name: str,
    author: str,
    description: str,
    supports_nowcast: bool = False,
    supports_forecast: bool = True,
):
    # 1. Create directory
    # 2. Copy and transform template files
    # 3. Update imports
    # 4. Generate config files
    # 5. Create README
    pass
```

**Entry Point in pyproject.toml:**
```toml
[project.scripts]
financing-private-credit = "financing_private_credit.cli:main"
```

---

### Option 2: Indicator Validation Tool

**Priority:** P0 (Highest)
**Impact:** High
**Effort:** Medium

#### Problem
No way to verify an indicator follows framework conventions until runtime.

#### Solution
Create a validation CLI command and pytest plugin.

#### Proposed Usage
```bash
# Validate specific indicator
financing-private-credit validate my_indicator

# Validate all indicators
financing-private-credit validate --all

# Output as JSON for CI/CD
financing-private-credit validate --json
```

#### Validation Checks

**Required Elements:**
- [ ] Inherits from `BaseIndicator` (or `BaseDecomposition`)
- [ ] Has `@register_indicator("name")` decorator
- [ ] Implements `get_metadata()` returning `IndicatorMetadata`
- [ ] Implements `fetch_data()` returning `dict[str, pl.DataFrame]`
- [ ] Implements `calculate()` returning `IndicatorResult`
- [ ] Exported in `__init__.py`
- [ ] Has spec file in `config/model_specs/`

**Optional Elements:**
- [ ] If `supports_nowcast=True`, `nowcast()` is implemented
- [ ] If `forecast.py` exists, has class inheriting from `BaseForecastModel`
- [ ] If `nowcast.py` exists, has class with `nowcast()` method
- [ ] Has tests in `tests/indicators/test_{name}.py`
- [ ] Has README.md with required sections

**Code Quality:**
- [ ] Follows naming conventions (CamelCase for classes)
- [ ] Has type hints on public methods
- [ ] Has docstrings with Args/Returns
- [ ] No obvious security issues (SQL injection, command injection, etc.)

#### Benefits
- Immediate feedback during development
- **AI agents can self-validate their work**
- Catches common mistakes before runtime
- Can be integrated into CI/CD pipeline
- Provides quality assurance checklist

#### Implementation Plan
```python
# src/financing_private_credit/cli/validate.py

import ast
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ValidationResult:
    indicator_name: str
    passed: bool
    errors: list[str]
    warnings: list[str]

def validate_indicator(name: str) -> ValidationResult:
    # Use AST parsing to check structure
    # Import and instantiate to verify runtime behavior
    # Check file system for required files
    pass
```

---

### Option 3: Standardize Optional File Patterns

**Priority:** P1
**Impact:** Medium
**Effort:** Low

#### Problem
forecast.py, nowcast.py, backtest.py, viz.py patterns exist but aren't formally specified.

#### Solution
Create explicit base classes or protocols for optional modules.

#### Current State
```python
# forecast.py has base class ✓
class MyForecaster(BaseForecastModel):
    ...

# nowcast.py has no base class ✗
class MyNowcaster:
    def nowcast(self, ...):
        ...

# backtest.py has no base class ✗
class MyBacktester:
    def run_backtest(self, ...):
        ...

# viz.py has no base class ✗
class MyVisualizer:
    def chart_1_overview(self):
        ...
```

#### Proposed Enhancement
Add to `base.py`:

```python
from abc import ABC, abstractmethod

@dataclass
class BacktestResult:
    """Container for backtest results."""
    spec_name: str
    mae: float
    rmse: float
    directional_accuracy: float
    metadata: dict[str, Any] = field(default_factory=dict)

class BaseNowcaster(ABC):
    """
    Base class for high-frequency nowcasting.

    Nowcasters update indicator estimates between quarterly releases
    using proxy variables (stock prices, CDS spreads, etc.).
    """

    @abstractmethod
    def nowcast(
        self,
        quarterly_data: pl.DataFrame,
        proxy_data: dict[str, pl.DataFrame],
    ) -> IndicatorResult:
        """
        Produce nowcast using proxy variables.

        Args:
            quarterly_data: Latest quarterly indicator results
            proxy_data: High-frequency proxy variables

        Returns:
            IndicatorResult with nowcast values
        """
        pass

class BaseBacktester(ABC):
    """
    Base class for model backtesting.

    Backtests validate forecasting performance using historical data.
    """

    @abstractmethod
    def run_backtest(
        self,
        data: pl.DataFrame,
        initial_window: int = 20,
        step: int = 1,
    ) -> BacktestResult:
        """
        Run rolling-window backtest.

        Args:
            data: Historical data for backtesting
            initial_window: Initial training window size
            step: Step size for rolling window

        Returns:
            BacktestResult with performance metrics
        """
        pass

    def get_metrics(self) -> dict[str, float]:
        """Get available performance metrics."""
        return {}

class BaseVisualizer(ABC):
    """
    Base class for indicator visualizations.

    Visualizers create numbered charts using Vega-Altair for storytelling.
    Chart numbering helps with reproducibility and references.
    """

    def __init__(self, data: pl.DataFrame):
        """
        Initialize visualizer with data.

        Args:
            data: Indicator results to visualize
        """
        self.data = data

    @abstractmethod
    def chart_1_overview(self) -> alt.Chart:
        """
        Chart 1: High-level overview of indicator values.

        Typically shows indicator values across all banks or time periods.
        """
        pass

    def get_all_charts(self) -> dict[str, alt.Chart]:
        """
        Get all available charts.

        Returns:
            Dictionary mapping chart names to Chart objects
        """
        import inspect
        charts = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("chart_"):
                charts[name] = method()
        return charts
```

#### Benefits
- Makes optional patterns explicit and discoverable
- Enables type checking with mypy
- **AI agents can discover standard interfaces**
- Provides templates for common patterns
- Enforces consistent method signatures

#### Migration Plan
1. Add base classes to `base.py`
2. Update `_template/` to use base classes
3. Update CONTRIBUTING.md with examples
4. Gradually migrate existing indicators (non-breaking)

---

### Option 4: Enhanced Documentation Structure

**Priority:** P2
**Impact:** Medium
**Effort:** Medium

#### Problem
Documentation is comprehensive but spread across multiple locations. Only 4/7 indicators have README.md.

#### Solution
Standardize documentation structure for each indicator.

#### Required Files for Each Indicator

```
indicators/my_indicator/
├── README.md         # REQUIRED - indicator documentation
├── indicator.py      # REQUIRED - core implementation
├── forecast.py       # OPTIONAL but recommended
├── nowcast.py        # OPTIONAL but recommended
├── backtest.py       # OPTIONAL
├── viz.py            # OPTIONAL
└── tests/           # OPTIONAL but recommended
    └── test_my_indicator.py
```

#### README.md Template

```markdown
# {Indicator Name}

**Version:** 1.0.0
**Status:** Production
**Maintainer:** {Author Name}

## Overview

Brief description of what this indicator measures and why it matters.

## Methodology

Detailed explanation of the calculation methodology:
- Mathematical formula
- Data transformations
- Key assumptions

## Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| SEC EDGAR | 10-K/10-Q | Quarterly |
| FRED | FEDFUNDS, DGS10 | Daily |

## Usage

```python
from financing_private_credit.indicators import get_indicator

# Initialize indicator
indicator = get_indicator("my_indicator")

# Fetch data
data = indicator.fetch_data("2015-01-01")

# Calculate
result = indicator.calculate(data)

# View results
print(result.data)
```

## Interpretation

How to read and interpret the results:
- High values indicate...
- Low values indicate...
- Thresholds: X < moderate < Y < high

## Configuration

Available spec parameters:
- `window`: Rolling window size (default: 20)
- `threshold`: Alert threshold (default: 0.5)

Example spec file: `config/model_specs/my_indicator.json`

## Forecasting

If applicable, explain forecasting methodology.

## Nowcasting

If applicable, explain nowcasting approach and proxy variables used.

## References

- Paper citation
- Related indicators
- External documentation

## Changelog

### v1.0.0 (2026-01-14)
- Initial release
```

#### Benefits
- Consistent documentation across indicators
- **AI agents can find information predictably**
- Easier onboarding for human developers
- Better searchability
- Documentation stays close to code

#### Implementation Plan
1. Create `docs/templates/indicator_README.md` template
2. Add README.md generation to scaffolding tool
3. Create READMEs for existing indicators
4. Add validation check for README presence

---

### Option 5: Add Dependency Injection for DataRegistry

**Priority:** P3 (Lower)
**Impact:** Low
**Effort:** Medium

#### Problem
Indicators create their own DataRegistry instances, making testing harder.

#### Current Pattern
```python
def fetch_data(self, start_date, end_date=None):
    registry = DataRegistry.get_instance()  # Hard-coded singleton
    bank_panel = registry.get_bank_panel(start_date)
    ...
```

#### Proposed Pattern
```python
def __init__(self, config_path=None, data_registry=None):
    super().__init__(config_path)
    self._registry = data_registry or DataRegistry.get_instance()

def fetch_data(self, start_date, end_date=None):
    bank_panel = self._registry.get_bank_panel(start_date)
    ...
```

#### Benefits
- Easier to mock for testing
- Supports different data backends
- Better for integration testing
- More flexible architecture

#### Tradeoffs
- Adds complexity to constructor
- Requires updating all indicators
- Not critical for current use cases

#### Decision
**Defer to Phase 4** - Nice-to-have but not essential for initial improvements.

---

### Option 6: Create Indicator Testing Guide

**Priority:** P1
**Impact:** Medium
**Effort:** Low

#### Problem
CONTRIBUTING.md mentions tests but doesn't show how to write them.

#### Solution
Add comprehensive testing section to CONTRIBUTING.md with examples.

#### Topics to Cover

1. **Unit Tests** - Test each method independently
2. **Integration Tests** - Test full indicator workflow
3. **Mocking Data** - How to avoid real API calls in tests
4. **Fixtures** - Reusable test data
5. **Backtest Validation** - Verify forecasting accuracy
6. **Edge Cases** - Missing data, date ranges, etc.

#### Example Test Structure

```python
# tests/indicators/test_my_indicator.py

import pytest
import polars as pl
from financing_private_credit.indicators import get_indicator
from financing_private_credit.indicators.base import IndicatorResult

@pytest.fixture
def mock_bank_panel():
    """Mock bank panel data for testing."""
    return pl.DataFrame({
        "ticker": ["JPM", "BAC", "C"] * 4,
        "date": ["2024-Q1", "2024-Q1", "2024-Q1", "2024-Q2", ...],
        "total_assets": [3000, 2500, 2000] * 4,
        "total_loans": [1000, 900, 800] * 4,
    })

@pytest.fixture
def mock_macro_data():
    """Mock macro data for testing."""
    return pl.DataFrame({
        "date": ["2024-Q1", "2024-Q2"],
        "FEDFUNDS": [5.25, 5.50],
        "DGS10": [4.20, 4.30],
    })

@pytest.fixture
def mock_data(mock_bank_panel, mock_macro_data):
    """Combined mock data."""
    return {
        "bank_panel": mock_bank_panel,
        "macro_data": mock_macro_data,
    }

# Test 1: Registration
def test_indicator_registration():
    """Test that indicator is properly registered."""
    indicator = get_indicator("my_indicator")
    assert indicator is not None
    assert indicator.__class__.__name__ == "MyIndicator"

# Test 2: Metadata
def test_metadata():
    """Test that metadata is complete and valid."""
    indicator = get_indicator("my_indicator")
    metadata = indicator.get_metadata()

    assert metadata.name == "My Indicator"
    assert metadata.short_name == "MyInd"
    assert len(metadata.data_sources) > 0
    assert metadata.update_frequency in ["daily", "weekly", "monthly", "quarterly"]
    assert metadata.lookback_periods > 0

# Test 3: Calculate with mock data
def test_calculate_with_mock_data(mock_data):
    """Test calculation with mock data."""
    indicator = get_indicator("my_indicator")
    result = indicator.calculate(mock_data)

    assert isinstance(result, IndicatorResult)
    assert result.data.height > 0
    assert "ticker" in result.data.columns

# Test 4: Empty data handling
def test_calculate_with_empty_data():
    """Test graceful handling of empty data."""
    indicator = get_indicator("my_indicator")
    empty_data = {
        "bank_panel": pl.DataFrame(),
        "macro_data": pl.DataFrame(),
    }
    result = indicator.calculate(empty_data)

    assert isinstance(result, IndicatorResult)
    assert result.data.height == 0
    assert "error" in result.metadata

# Test 5: Data validation
def test_validate_data(mock_data):
    """Test data validation."""
    indicator = get_indicator("my_indicator")
    errors = indicator.validate_data(mock_data)

    assert len(errors) == 0

# Test 6: Nowcast support (if applicable)
def test_nowcast_support():
    """Test nowcast support flag."""
    indicator = get_indicator("my_indicator")

    if indicator.supports_nowcast:
        # Should not raise NotImplementedError
        data = {...}
        result = indicator.nowcast(data)
        assert isinstance(result, IndicatorResult)
    else:
        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            indicator.nowcast({})

# Test 7: Forecast model (if applicable)
def test_forecast_model(mock_data):
    """Test forecast model if available."""
    try:
        from financing_private_credit.indicators.my_indicator import MyForecaster

        forecaster = MyForecaster()
        assert not forecaster.is_fitted

        # Fit model
        fit_result = forecaster.fit(
            data=mock_data["bank_panel"],
            target="total_loans",
            features=["total_assets"],
        )
        assert forecaster.is_fitted
        assert "r2" in fit_result or "aic" in fit_result

        # Predict
        forecast = forecaster.predict(
            data=mock_data["bank_panel"],
            horizon=4,
        )
        assert forecast.predictions.height > 0

    except ImportError:
        pytest.skip("Forecaster not implemented")
```

#### Benefits
- Improves code reliability
- Provides examples for developers
- **AI agents can generate tests following patterns**
- Catches regressions early
- Documents expected behavior

#### Implementation Plan
1. Add testing section to CONTRIBUTING.md
2. Create `tests/indicators/test_template.py`
3. Add pytest fixtures to `tests/conftest.py`
4. Create mock data generators
5. Add to scaffolding tool

---

### Option 7: Error Handling & Data Quality Standards

**Priority:** P2
**Impact:** High
**Effort:** Medium

#### Problem
No standardized approach for handling data fetching failures or data quality issues.

#### Solution
Create error handling patterns and data quality checks.

#### Proposed Base Class Additions

```python
from enum import Enum
from dataclasses import dataclass

class DataQualitySeverity(Enum):
    """Severity levels for data quality issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class DataQualityIssue:
    """Data quality issue description."""
    severity: DataQualitySeverity
    source: str
    message: str
    affected_records: int = 0

class BaseIndicator(ABC):
    """Enhanced base indicator with error handling."""

    def fetch_data_with_fallback(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        strict: bool = False,
    ) -> tuple[dict[str, pl.DataFrame], list[DataQualityIssue]]:
        """
        Fetch data with graceful degradation.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date
            strict: If True, raise on any errors

        Returns:
            - data: Dictionary of DataFrames (may be partial if not strict)
            - issues: List of data quality issues
        """
        data = {}
        issues = []

        try:
            data = self.fetch_data(start_date, end_date)
        except Exception as e:
            issue = DataQualityIssue(
                severity=DataQualitySeverity.ERROR,
                source="fetch_data",
                message=f"Data fetch failed: {e}",
            )
            issues.append(issue)

            if strict:
                raise

            # Attempt fallback
            data = self._get_fallback_data()

        # Quality checks
        quality_issues = self._check_data_quality(data)
        issues.extend(quality_issues)

        return data, issues

    def _check_data_quality(
        self,
        data: dict[str, pl.DataFrame]
    ) -> list[DataQualityIssue]:
        """
        Check data quality and return issues.

        Checks:
        - Missing dates (gaps in time series)
        - Outliers (values beyond expected ranges)
        - Stale data (older than expected)
        - Incomplete coverage (missing banks)
        - Data consistency (cross-field validation)
        """
        issues = []

        for source_name, df in data.items():
            # Check 1: Empty data
            if df.height == 0:
                issues.append(DataQualityIssue(
                    severity=DataQualitySeverity.ERROR,
                    source=source_name,
                    message="DataFrame is empty",
                ))
                continue

            # Check 2: Missing required columns
            if "date" not in df.columns:
                issues.append(DataQualityIssue(
                    severity=DataQualitySeverity.ERROR,
                    source=source_name,
                    message="Missing 'date' column",
                ))

            # Check 3: Null values
            null_counts = df.null_count()
            for col in null_counts.columns:
                null_pct = null_counts[col][0] / df.height
                if null_pct > 0.1:  # >10% null
                    issues.append(DataQualityIssue(
                        severity=DataQualitySeverity.WARNING,
                        source=source_name,
                        message=f"Column '{col}' has {null_pct:.1%} null values",
                        affected_records=null_counts[col][0],
                    ))

            # Check 4: Date gaps
            if "date" in df.columns:
                dates = df["date"].unique().sort()
                if dates.height > 1:
                    # Simplified gap check
                    expected_count = self._expected_observation_count(
                        dates[0], dates[-1]
                    )
                    if dates.height < expected_count * 0.8:  # >20% missing
                        issues.append(DataQualityIssue(
                            severity=DataQualitySeverity.WARNING,
                            source=source_name,
                            message=f"Potential date gaps: {dates.height} observations, expected ~{expected_count}",
                        ))

            # Check 5: Stale data
            if "date" in df.columns:
                latest_date = df["date"].max()
                age_days = (datetime.now().date() - latest_date).days

                freq = self.get_metadata().update_frequency
                max_age = {
                    "daily": 7,
                    "weekly": 14,
                    "monthly": 45,
                    "quarterly": 120,
                }.get(freq, 30)

                if age_days > max_age:
                    issues.append(DataQualityIssue(
                        severity=DataQualitySeverity.WARNING,
                        source=source_name,
                        message=f"Data is {age_days} days old (expected <{max_age} days)",
                    ))

        return issues

    def _expected_observation_count(
        self,
        start_date: date,
        end_date: date,
    ) -> int:
        """Calculate expected number of observations."""
        freq = self.get_metadata().update_frequency
        days = (end_date - start_date).days

        return {
            "daily": days,
            "weekly": days // 7,
            "monthly": days // 30,
            "quarterly": days // 90,
        }.get(freq, days // 30)

    def _get_fallback_data(self) -> dict[str, pl.DataFrame]:
        """
        Get fallback data when primary fetch fails.

        Override to provide fallback data sources.
        Default returns empty DataFrames.
        """
        return {
            "bank_panel": pl.DataFrame(),
            "macro_data": pl.DataFrame(),
        }
```

#### Benefits
- Graceful degradation when data sources fail
- Early detection of data quality issues
- Better error messages for debugging
- Production-ready reliability
- **AI agents can handle edge cases better**

#### Implementation Plan
1. Add error handling to `base.py`
2. Update template to use new methods
3. Add data quality reporting to dashboard
4. Create monitoring/alerting for quality issues

---

## Prioritized Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

**Goal:** Reduce friction for new indicator development by 90%

#### 1.1 CLI Scaffolding Tool (Option 1)
- **Effort:** 2-3 days
- **Impact:** Immediate productivity boost
- **Tasks:**
  - [ ] Create `src/financing_private_credit/cli/scaffold.py`
  - [ ] Implement template copying and transformation
  - [ ] Add CLI entry point to pyproject.toml
  - [ ] Write tests for scaffolding logic
  - [ ] Update documentation with CLI usage

#### 1.2 Testing Guide (Option 6)
- **Effort:** 1-2 days
- **Impact:** Improves code quality
- **Tasks:**
  - [ ] Add testing section to CONTRIBUTING.md
  - [ ] Create example test file
  - [ ] Create pytest fixtures
  - [ ] Document mocking patterns

**Success Metrics:**
- Creating new indicator takes <5 minutes
- Test coverage >80% for new indicators

---

### Phase 2: Quality & Standards (2-3 weeks)

**Goal:** Establish quality gates and standardize patterns

#### 2.1 Validation Tool (Option 2)
- **Effort:** 3-5 days
- **Impact:** Automated quality assurance
- **Tasks:**
  - [ ] Create `src/financing_private_credit/cli/validate.py`
  - [ ] Implement AST-based structure checks
  - [ ] Add runtime validation tests
  - [ ] Create validation report formatter
  - [ ] Integrate with CI/CD pipeline

#### 2.2 Standardize Optional Patterns (Option 3)
- **Effort:** 2-3 days
- **Impact:** Better discoverability
- **Tasks:**
  - [ ] Add BaseNowcaster to base.py
  - [ ] Add BaseBacktester to base.py
  - [ ] Add BaseVisualizer to base.py
  - [ ] Update template to use base classes
  - [ ] Update CONTRIBUTING.md

**Success Metrics:**
- All new indicators pass validation
- Optional patterns have formal interfaces

---

### Phase 3: Polish & Production-Readiness (3-4 weeks)

**Goal:** Professional-grade documentation and error handling

#### 3.1 Documentation Structure (Option 4)
- **Effort:** 4-5 days
- **Impact:** Better onboarding
- **Tasks:**
  - [ ] Create README template
  - [ ] Add README generation to scaffolding
  - [ ] Create READMEs for existing indicators
  - [ ] Add validation check for README

#### 3.2 Error Handling Standards (Option 7)
- **Effort:** 3-4 days
- **Impact:** Production reliability
- **Tasks:**
  - [ ] Add error handling to base.py
  - [ ] Implement data quality checks
  - [ ] Add fallback mechanisms
  - [ ] Create monitoring/alerting

**Success Metrics:**
- All indicators have documentation
- Data quality issues are caught early

---

### Phase 4: Advanced (Optional)

**Goal:** Advanced features for power users

#### 4.1 Dependency Injection (Option 5)
- **Effort:** 2-3 days
- **Impact:** Better testing
- **Tasks:**
  - [ ] Refactor BaseIndicator constructor
  - [ ] Update all indicators
  - [ ] Create mock DataRegistry for tests

**Success Metrics:**
- Unit tests run without network calls

---

## Priority Matrix

| Option | Impact | Effort | Priority | Dependencies |
|--------|--------|--------|----------|--------------|
| **1. CLI Scaffolding** | High | Low | **P0** | None |
| **2. Validation Tool** | High | Medium | **P0** | None |
| **6. Testing Guide** | Medium | Low | **P1** | None |
| **3. Optional Patterns** | Medium | Low | **P1** | None |
| **4. Documentation** | Medium | Medium | **P2** | Option 1 |
| **7. Error Handling** | High | Medium | **P2** | None |
| **5. Dependency Injection** | Low | Medium | **P3** | Options 2, 6 |

---

## Success Criteria

### For Human Developers
- [ ] Time to create new indicator: <5 minutes (down from ~30 minutes)
- [ ] Clear validation feedback before runtime
- [ ] Comprehensive testing examples
- [ ] Consistent documentation across all indicators

### For AI Coding Agents
- [ ] Can scaffold indicators autonomously using CLI
- [ ] Can validate their work automatically
- [ ] Can discover patterns through base classes
- [ ] Can find information predictably (standard README locations)
- [ ] Can generate tests following documented patterns

### For Framework Quality
- [ ] Test coverage >80%
- [ ] All indicators pass validation
- [ ] Data quality issues caught early
- [ ] Production-ready error handling

---

## AI Agent Development Notes

### What Makes This Framework AI-Friendly ✓

1. **Consistent Patterns**
   - All indicators follow same structure
   - Registry pattern for discovery
   - Type hints and dataclasses

2. **Clear Abstractions**
   - Only 3 required methods
   - Optional methods have defaults
   - Generic base classes

3. **Excellent Documentation**
   - Comprehensive docstrings
   - Working examples
   - Template as reference

4. **Predictable Structure**
   - Standard file names (indicator.py, forecast.py, etc.)
   - Single responsibility per file
   - Clear separation of concerns

### Improvements That Would Help AI Agents ⚡

1. **Scaffolding CLI** - AI can invoke directly instead of manual file manipulation
2. **Validation Tool** - AI can verify its work automatically
3. **Standardized Interfaces** - AI can discover patterns through base classes
4. **Consistent README** - AI can find information predictably
5. **Testing Templates** - AI can generate tests following patterns

### Current Friction Points for AI ⚠️

1. Multi-file changes required (3+ files to add an indicator)
2. Optional patterns not formalized (no base classes)
3. No self-validation mechanism
4. Documentation scattered
5. Hard-coded dependencies

---

## Appendix: Code Examples

### Example: Scaffolded Indicator Structure

```python
# After running: financing-private-credit new-indicator credit_quality

indicators/credit_quality/
├── __init__.py                    # Auto-generated exports
├── indicator.py                   # Auto-generated from template
├── forecast.py                    # Optional, created if --supports-forecast
├── nowcast.py                     # Optional, created if --supports-nowcast
├── README.md                      # Auto-generated with placeholders
└── tests/
    └── test_credit_quality.py    # Auto-generated test template
```

### Example: Validation Output

```bash
$ financing-private-credit validate credit_quality

Validating indicator: credit_quality
=====================================

✓ Structure
  ✓ Inherits from BaseIndicator
  ✓ Has @register_indicator decorator
  ✓ Implements get_metadata()
  ✓ Implements fetch_data()
  ✓ Implements calculate()

✓ Registration
  ✓ Exported in __init__.py
  ✓ Available via get_indicator()

✓ Configuration
  ✓ Has spec file: config/model_specs/credit_quality.json
  ✓ Spec is valid JSON

⚠ Documentation
  ✓ Has README.md
  ⚠ Missing "Interpretation" section in README

✓ Testing
  ✓ Has test file: tests/indicators/test_credit_quality.py
  ✓ Tests pass

✓ Code Quality
  ✓ Has type hints on public methods
  ✓ Has docstrings

Overall: PASS with 1 warning
```

### Example: Data Quality Report

```python
indicator = get_indicator("credit_quality")
data, issues = indicator.fetch_data_with_fallback("2015-01-01")

for issue in issues:
    if issue.severity == DataQualitySeverity.ERROR:
        print(f"❌ {issue.source}: {issue.message}")
    elif issue.severity == DataQualitySeverity.WARNING:
        print(f"⚠️  {issue.source}: {issue.message}")
    else:
        print(f"ℹ️  {issue.source}: {issue.message}")

# Output:
# ⚠️  bank_panel: Column 'total_equity' has 15.3% null values (affected: 23 records)
# ⚠️  macro_data: Data is 45 days old (expected <30 days)
# ℹ️  bank_panel: Successfully fetched 150 records
```

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Prioritize phases** based on immediate needs
3. **Start with Phase 1** (CLI Scaffolding + Testing Guide)
4. **Iterate based on feedback** from developers and AI agents

---

**Document Version:** 1.0
**Last Updated:** 2026-01-14
**Status:** ✅ Ready for Implementation
