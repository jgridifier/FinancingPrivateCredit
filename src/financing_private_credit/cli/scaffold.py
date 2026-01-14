"""
Indicator scaffolding tool.

Creates new indicator packages from the _template directory with proper
naming, registration, and file structure.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ScaffoldConfig:
    """Configuration for scaffolding a new indicator."""

    name: str  # snake_case name (e.g., "credit_quality")
    author: str
    description: str
    supports_forecast: bool = True
    supports_nowcast: bool = False
    supports_backtest: bool = False
    supports_viz: bool = False

    @property
    def class_name(self) -> str:
        """Convert snake_case to PascalCase for class name."""
        return "".join(word.capitalize() for word in self.name.split("_")) + "Indicator"

    @property
    def short_name(self) -> str:
        """Generate a short name abbreviation."""
        words = self.name.split("_")
        if len(words) == 1:
            return self.name[:3].upper()
        return "".join(w[0].upper() for w in words)

    @property
    def spec_class_name(self) -> str:
        """Class name for the spec dataclass."""
        return "".join(word.capitalize() for word in self.name.split("_")) + "Spec"

    @property
    def forecaster_class_name(self) -> str:
        """Class name for the forecaster."""
        return "".join(word.capitalize() for word in self.name.split("_")) + "Forecaster"

    @property
    def nowcaster_class_name(self) -> str:
        """Class name for the nowcaster."""
        return "".join(word.capitalize() for word in self.name.split("_")) + "Nowcaster"


@dataclass
class ScaffoldResult:
    """Result of scaffolding operation."""

    success: bool
    indicator_path: Optional[Path] = None
    spec_path: Optional[Path] = None
    files_created: list[str] = None
    errors: list[str] = None

    def __post_init__(self):
        if self.files_created is None:
            self.files_created = []
        if self.errors is None:
            self.errors = []


def _get_project_root() -> Path:
    """Get the project root directory."""
    # Start from this file and go up to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


def _validate_name(name: str) -> list[str]:
    """Validate the indicator name."""
    errors = []

    if not name:
        errors.append("Indicator name cannot be empty")
        return errors

    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        errors.append(
            "Indicator name must be snake_case (lowercase letters, numbers, underscores, "
            "starting with a letter)"
        )

    if name.startswith("_"):
        errors.append("Indicator name cannot start with underscore")

    # Check for reserved names
    reserved = {"base", "template", "_template", "test", "tests"}
    if name in reserved:
        errors.append(f"'{name}' is a reserved name")

    return errors


def _transform_template_content(
    content: str,
    config: ScaffoldConfig,
) -> str:
    """Transform template content with indicator-specific values."""
    # Replace class names
    content = content.replace("TemplateIndicator", config.class_name)
    content = content.replace("TemplateSpec", config.spec_class_name)
    content = content.replace("TemplateForecaster", config.forecaster_class_name)
    content = content.replace("TemplateForecasterSpec", f"{config.forecaster_class_name}Spec")
    content = content.replace("TemplateNowcaster", config.nowcaster_class_name)
    content = content.replace("TemplateScenarioForecaster", f"{config.forecaster_class_name}Scenario")

    # Replace string references
    content = content.replace('"template"', f'"{config.name}"')
    content = content.replace("'template'", f"'{config.name}'")
    content = content.replace('"Template"', f'"{config.short_name}"')
    content = content.replace("'Template'", f"'{config.short_name}'")

    # Replace metadata placeholders
    content = content.replace(
        'name="Template Indicator"',
        f'name="{config.class_name.replace("Indicator", " Indicator")}"',
    )
    content = content.replace(
        'short_name="Template"',
        f'short_name="{config.short_name}"',
    )

    # Replace description placeholder
    content = content.replace(
        "Replace this with a detailed description of what "
        '"your indicator measures and why it matters."',
        f'"{config.description}"',
    )

    # Update supports_nowcast flag if enabled
    if config.supports_nowcast:
        content = content.replace(
            "supports_nowcast: bool = False",
            "supports_nowcast: bool = True",
        )

    # Uncomment the @register_indicator decorator
    content = content.replace(
        f'# @register_indicator("{config.name}")',
        f'@register_indicator("{config.name}")',
    )
    # Also handle the original template format
    content = content.replace(
        '# Uncomment and rename when ready to register\n# @register_indicator("template")',
        f'@register_indicator("{config.name}")',
    )

    return content


def _create_init_file(config: ScaffoldConfig, indicator_dir: Path) -> str:
    """Create the __init__.py file for the indicator package."""
    exports = [config.class_name, config.spec_class_name]
    imports = [
        f"from .indicator import {config.class_name}, {config.spec_class_name}",
    ]

    if config.supports_forecast:
        exports.extend([config.forecaster_class_name, f"{config.forecaster_class_name}Spec"])
        imports.append(
            f"from .forecast import {config.forecaster_class_name}, {config.forecaster_class_name}Spec"
        )

    if config.supports_nowcast:
        exports.extend([config.nowcaster_class_name, "NowcastSpec", "NowcastResult"])
        imports.append(
            f"from .nowcast import {config.nowcaster_class_name}, NowcastSpec, NowcastResult"
        )

    content = f'''"""
{config.class_name.replace("Indicator", " Indicator")} Package

{config.description}

Author: {config.author}
Created: {datetime.now().strftime("%Y-%m-%d")}
"""

{chr(10).join(imports)}

__all__ = [
    # Indicator
    "{config.class_name}",
    "{config.spec_class_name}",
'''

    if config.supports_forecast:
        content += f'''    # Forecast
    "{config.forecaster_class_name}",
    "{config.forecaster_class_name}Spec",
'''

    if config.supports_nowcast:
        content += f'''    # Nowcast
    "{config.nowcaster_class_name}",
    "NowcastSpec",
    "NowcastResult",
'''

    content += "]\n"

    return content


def _create_readme(config: ScaffoldConfig) -> str:
    """Create README.md for the indicator."""
    return f"""# {config.class_name.replace("Indicator", " Indicator")}

> {config.description}

**Author:** {config.author}
**Created:** {datetime.now().strftime("%Y-%m-%d")}
**Version:** 1.0.0

## Overview

{config.description}

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `metric_1` | What it measures | High = X, Low = Y |

## Data Sources

- **SEC EDGAR**: [What data from 10-K/10-Q]
- **FRED**: [Which series]

## Methodology

### Step 1: Data Preparation

Describe how raw data is processed.

### Step 2: Calculation

```
Formula or algorithm description
```

## Usage

```python
from financing_private_credit.indicators import get_indicator

# Initialize and run
indicator = get_indicator("{config.name}")
data = indicator.fetch_data("2015-01-01")
result = indicator.calculate(data)

# View results
print(result.data)
```

## Model Specifications

Default configuration in `config/model_specs/{config.name}.json`:

```json
{{
    "name": "{config.name}_default",
    "description": "{config.description}",
    "window": 20,
    "threshold": 0.5
}}
```

## Forecasting

{"Forecasting is supported. See `forecast.py` for implementation." if config.supports_forecast else "Forecasting not yet implemented."}

## Nowcasting

{"Nowcasting is supported. See `nowcast.py` for implementation." if config.supports_nowcast else "Nowcasting not yet implemented."}

## References

- Add paper citations
- Related research
"""


def _create_spec_file(config: ScaffoldConfig, spec_dir: Path) -> str:
    """Create the model spec JSON file."""
    spec = {
        "name": f"{config.name}_default",
        "description": config.description,
        "window": 20,
        "threshold": 0.5,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "author": config.author,
    }
    return json.dumps(spec, indent=2)


def _create_test_file(config: ScaffoldConfig) -> str:
    """Create a test file template for the indicator."""
    base_tests = f'''"""
Tests for {config.class_name}

Run with: pytest tests/indicators/test_{config.name}.py
"""

import pytest
import polars as pl

from financing_private_credit.indicators import get_indicator
from financing_private_credit.indicators.base import IndicatorResult


class Test{config.class_name}:
    """Test suite for {config.class_name}."""

    def test_registration(self):
        """Test that indicator is properly registered."""
        indicator = get_indicator("{config.name}")
        assert indicator is not None
        assert indicator.__class__.__name__ == "{config.class_name}"

    def test_metadata(self):
        """Test that metadata is complete and valid."""
        indicator = get_indicator("{config.name}")
        metadata = indicator.get_metadata()

        assert metadata.name is not None
        assert metadata.short_name == "{config.short_name}"
        assert len(metadata.data_sources) > 0
        assert metadata.update_frequency in ["daily", "weekly", "monthly", "quarterly"]
        assert metadata.lookback_periods > 0

    def test_required_data_sources(self):
        """Test that required data sources are documented."""
        indicator = get_indicator("{config.name}")
        sources = indicator.get_required_data_sources()
        assert isinstance(sources, list)

    def test_validate_data_empty(self):
        """Test validation catches empty DataFrames."""
        indicator = get_indicator("{config.name}")
        errors = indicator.validate_data({{"bank_panel": pl.DataFrame()}})
        assert len(errors) > 0
        assert any("Empty" in e for e in errors)


@pytest.fixture
def mock_bank_panel():
    """Create mock bank panel data for testing."""
    return pl.DataFrame({{
        "ticker": ["JPM", "BAC", "C", "WFC"] * 4,
        "date": (
            ["2024-Q1"] * 4 + ["2024-Q2"] * 4 +
            ["2024-Q3"] * 4 + ["2024-Q4"] * 4
        ),
        "total_assets": [3000, 2500, 2000, 1800] * 4,
        "total_loans": [1000, 900, 800, 700] * 4,
        "deposits": [2000, 1800, 1500, 1300] * 4,
    }})


@pytest.fixture
def mock_macro_data():
    """Create mock macro data for testing."""
    return pl.DataFrame({{
        "date": ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"],
        "FEDFUNDS": [5.25, 5.50, 5.25, 5.00],
        "DGS10": [4.20, 4.30, 4.10, 4.00],
    }})


@pytest.fixture
def mock_data(mock_bank_panel, mock_macro_data):
    """Combined mock data dictionary."""
    return {{
        "bank_panel": mock_bank_panel,
        "macro_data": mock_macro_data,
    }}


class Test{config.class_name}Calculation:
    """Tests for indicator calculation."""

    def test_calculate_returns_result(self, mock_data):
        """Test that calculate returns an IndicatorResult."""
        indicator = get_indicator("{config.name}")
        result = indicator.calculate(mock_data)

        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "{config.name}"

    def test_calculate_with_empty_data(self):
        """Test graceful handling of empty data."""
        indicator = get_indicator("{config.name}")
        empty_data = {{
            "bank_panel": pl.DataFrame(),
            "macro_data": pl.DataFrame(),
        }}
        result = indicator.calculate(empty_data)

        assert isinstance(result, IndicatorResult)
        # Should handle gracefully, possibly with error in metadata
'''

    # Add nowcast tests if supported
    if config.supports_nowcast:
        nowcast_class_name = config.class_name.replace("Indicator", "Nowcast")
        nowcast_tests = f'''

class Test{nowcast_class_name}:
    """Tests for nowcasting functionality."""

    def test_supports_nowcast_flag(self):
        """Test that nowcast support is properly configured."""
        indicator = get_indicator("{config.name}")
        assert indicator.supports_nowcast is True

    def test_nowcast_returns_result(self, mock_data):
        """Test that nowcast returns valid result."""
        indicator = get_indicator("{config.name}")
        # Add proxy data for nowcasting
        mock_data["stock_data"] = pl.DataFrame({{
            "ticker": ["JPM", "BAC"],
            "date": ["2024-01-15", "2024-01-15"],
            "close": [150.0, 35.0],
        }})
        result = indicator.nowcast(mock_data)
        assert isinstance(result, IndicatorResult)
'''
        base_tests += nowcast_tests
    else:
        base_tests += "\n\n# Nowcast tests not applicable - nowcasting not enabled\n"

    return base_tests


def _update_indicators_init(config: ScaffoldConfig, indicators_dir: Path) -> None:
    """Update indicators/__init__.py to import the new indicator."""
    init_path = indicators_dir / "__init__.py"
    content = init_path.read_text()

    # Find the last import line for indicators and add the new one
    import_line = f"from . import {config.name}"

    # Check if already imported
    if import_line in content:
        return

    # Find position to insert (after last "from . import" line)
    lines = content.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if line.startswith("from . import ") and not line.startswith("from .base"):
            insert_idx = i + 1

    if insert_idx is not None:
        lines.insert(insert_idx, import_line)
        init_path.write_text("\n".join(lines))


def scaffold_indicator(config: ScaffoldConfig) -> ScaffoldResult:
    """
    Scaffold a new indicator from template.

    Args:
        config: Scaffolding configuration

    Returns:
        ScaffoldResult with operation details
    """
    result = ScaffoldResult(success=False)

    # Validate name
    name_errors = _validate_name(config.name)
    if name_errors:
        result.errors = name_errors
        return result

    try:
        project_root = _get_project_root()
    except RuntimeError as e:
        result.errors = [str(e)]
        return result

    indicators_dir = project_root / "src" / "financing_private_credit" / "indicators"
    template_dir = indicators_dir / "_template"
    new_indicator_dir = indicators_dir / config.name
    spec_dir = project_root / "config" / "model_specs"
    tests_dir = project_root / "tests" / "indicators"

    # Check template exists
    if not template_dir.exists():
        result.errors = [f"Template directory not found: {template_dir}"]
        return result

    # Check indicator doesn't already exist
    if new_indicator_dir.exists():
        result.errors = [f"Indicator directory already exists: {new_indicator_dir}"]
        return result

    # Create indicator directory
    new_indicator_dir.mkdir(parents=True)
    result.indicator_path = new_indicator_dir

    # Copy and transform template files
    files_to_copy = [
        ("indicator.py", True),  # (filename, should_copy)
        ("forecast.py", config.supports_forecast),
        ("nowcast.py", config.supports_nowcast),
    ]

    for filename, should_copy in files_to_copy:
        if not should_copy:
            continue

        src_path = template_dir / filename
        if not src_path.exists():
            continue

        content = src_path.read_text()
        transformed = _transform_template_content(content, config)

        dst_path = new_indicator_dir / filename
        dst_path.write_text(transformed)
        result.files_created.append(str(dst_path.relative_to(project_root)))

    # Create __init__.py
    init_content = _create_init_file(config, new_indicator_dir)
    init_path = new_indicator_dir / "__init__.py"
    init_path.write_text(init_content)
    result.files_created.append(str(init_path.relative_to(project_root)))

    # Create README.md
    readme_content = _create_readme(config)
    readme_path = new_indicator_dir / "README.md"
    readme_path.write_text(readme_content)
    result.files_created.append(str(readme_path.relative_to(project_root)))

    # Create model spec file
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_content = _create_spec_file(config, spec_dir)
    spec_path = spec_dir / f"{config.name}.json"
    spec_path.write_text(spec_content)
    result.spec_path = spec_path
    result.files_created.append(str(spec_path.relative_to(project_root)))

    # Create test file
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_content = _create_test_file(config)
    test_path = tests_dir / f"test_{config.name}.py"
    test_path.write_text(test_content)
    result.files_created.append(str(test_path.relative_to(project_root)))

    # Update indicators/__init__.py
    _update_indicators_init(config, indicators_dir)
    result.files_created.append("src/financing_private_credit/indicators/__init__.py (updated)")

    result.success = True
    return result
