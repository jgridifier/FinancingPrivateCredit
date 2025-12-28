"""
Enhanced Model Specification System

Provides a flexible specification system that supports:
1. Multiple specifications per file
2. Per-ticker specifications with "_" as default
3. Component-level specifications (e.g., predict uninsured_deposit_ratio for JPM)
4. Model search and selection capabilities

File Format:
{
    "_": {
        "name": "default",
        "description": "Default specification for all tickers",
        "ar_lags": 2,
        "dist_lags": 2,
        ...
    },
    "JPM": {
        "name": "jpm_specific",
        "description": "JPM-optimized specification",
        "ar_lags": 4,
        "dist_lags": 3,
        ...
    },
    "BAC": {
        "name": "bac_specific",
        ...
    }
}

For component-level specs:
{
    "_": {
        "uninsured_deposit_ratio": {...},
        "fhlb_advance_ratio": {...},
        ...
    },
    "JPM": {
        "uninsured_deposit_ratio": {...},
        ...
    }
}

Usage:
    from financing_private_credit.core.model_specs import ModelSpecRegistry

    # Load specs
    registry = ModelSpecRegistry.from_json("config/model_specs/funding_stability_ardl.json")

    # Get spec for a ticker (falls back to "_" default)
    spec = registry.get_spec("JPM")

    # Get component-specific spec
    spec = registry.get_component_spec("JPM", "uninsured_deposit_ratio")

    # Search for best spec
    best_spec = registry.search("funding_stability", target="fhlb_advance_ratio")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Generic

T = TypeVar("T")


@dataclass
class ModelSpec:
    """
    Base model specification.

    All indicator-specific specs should inherit from this or include these fields.
    """

    name: str
    description: str

    # Target variable
    target: Optional[str] = None

    # Model type
    model_type: str = "ardl"  # "ardl", "aplr", "ols", "var", etc.

    # Lag structure
    ar_lags: int = 2
    dist_lags: int = 2

    # Exogenous variables
    exog_vars: list[str] = field(default_factory=list)

    # Estimation settings
    include_constant: bool = True
    include_trend: bool = False
    include_seasonality: bool = False
    seasonality_period: int = 4

    # Validation settings
    holdout_periods: int = 4
    min_train_periods: int = 16

    # Forecasting settings
    forecast_horizon: int = 4
    n_simulations: int = 1000

    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    version: str = "1.0.0"

    # Performance metrics (for tracking)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "target": self.target,
            "model_type": self.model_type,
            "ar_lags": self.ar_lags,
            "dist_lags": self.dist_lags,
            "exog_vars": self.exog_vars,
            "include_constant": self.include_constant,
            "include_trend": self.include_trend,
            "include_seasonality": self.include_seasonality,
            "seasonality_period": self.seasonality_period,
            "holdout_periods": self.holdout_periods,
            "min_train_periods": self.min_train_periods,
            "forecast_horizon": self.forecast_horizon,
            "n_simulations": self.n_simulations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            target=data.get("target"),
            model_type=data.get("model_type", "ardl"),
            ar_lags=data.get("ar_lags", 2),
            dist_lags=data.get("dist_lags", 2),
            exog_vars=data.get("exog_vars", []),
            include_constant=data.get("include_constant", True),
            include_trend=data.get("include_trend", False),
            include_seasonality=data.get("include_seasonality", False),
            seasonality_period=data.get("seasonality_period", 4),
            holdout_periods=data.get("holdout_periods", 4),
            min_train_periods=data.get("min_train_periods", 16),
            forecast_horizon=data.get("forecast_horizon", 4),
            n_simulations=data.get("n_simulations", 1000),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            version=data.get("version", "1.0.0"),
            metrics=data.get("metrics", {}),
        )


class ModelSpecRegistry:
    """
    Registry for model specifications.

    Supports:
    - Multiple specs per file (keyed by ticker)
    - Default specs (key "_")
    - Component-level specs (nested structure)
    - Search and selection
    """

    DEFAULT_KEY = "_"

    def __init__(self, name: str = "default"):
        """
        Initialize registry.

        Args:
            name: Name of this registry (for identification)
        """
        self.name = name
        self._specs: dict[str, dict[str, Any]] = {}
        self._source_path: Optional[Path] = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelSpecRegistry":
        """
        Load specifications from a JSON file.

        The file can contain:
        - A single spec (will be used as default)
        - Multiple specs keyed by ticker (with "_" as default)
        - Nested component specs

        Args:
            path: Path to JSON file

        Returns:
            ModelSpecRegistry instance
        """
        path = Path(path)
        registry = cls(name=path.stem)
        registry._source_path = path

        with open(path, "r") as f:
            data = json.load(f)

        # Detect format
        if isinstance(data, dict):
            if "name" in data and "description" in data:
                # Single spec format - use as default
                registry._specs[cls.DEFAULT_KEY] = data
            else:
                # Multi-spec format
                registry._specs = data

        return registry

    def get_spec(
        self,
        ticker: Optional[str] = None,
        as_dataclass: bool = False,
    ) -> dict[str, Any] | ModelSpec:
        """
        Get specification for a ticker.

        Falls back to default ("_") if ticker-specific not found.

        Args:
            ticker: Ticker symbol (None for default)
            as_dataclass: Return as ModelSpec dataclass

        Returns:
            Specification dictionary or ModelSpec
        """
        key = ticker if ticker and ticker in self._specs else self.DEFAULT_KEY

        if key not in self._specs:
            raise KeyError(
                f"No specification found for '{ticker}' and no default ('_') defined"
            )

        spec = self._specs[key]

        if as_dataclass:
            return ModelSpec.from_dict(spec)

        return spec

    def get_component_spec(
        self,
        ticker: Optional[str],
        component: str,
        as_dataclass: bool = False,
    ) -> dict[str, Any] | ModelSpec:
        """
        Get specification for a specific component (e.g., uninsured_deposit_ratio).

        Lookup order:
        1. {ticker}.{component}
        2. _.{component}
        3. Raise KeyError

        Args:
            ticker: Ticker symbol
            component: Component name
            as_dataclass: Return as ModelSpec dataclass

        Returns:
            Component specification
        """
        # Try ticker-specific first
        ticker_key = ticker if ticker else self.DEFAULT_KEY

        if ticker_key in self._specs:
            ticker_specs = self._specs[ticker_key]
            if isinstance(ticker_specs, dict) and component in ticker_specs:
                spec = ticker_specs[component]
                if as_dataclass:
                    return ModelSpec.from_dict(spec)
                return spec

        # Fall back to default
        if self.DEFAULT_KEY in self._specs:
            default_specs = self._specs[self.DEFAULT_KEY]
            if isinstance(default_specs, dict) and component in default_specs:
                spec = default_specs[component]
                if as_dataclass:
                    return ModelSpec.from_dict(spec)
                return spec

        raise KeyError(
            f"No specification found for component '{component}' "
            f"(ticker='{ticker}')"
        )

    def list_tickers(self) -> list[str]:
        """List all tickers with specific specs."""
        return [k for k in self._specs.keys() if k != self.DEFAULT_KEY]

    def list_components(self, ticker: Optional[str] = None) -> list[str]:
        """List all components with specs for a ticker."""
        key = ticker if ticker and ticker in self._specs else self.DEFAULT_KEY

        if key not in self._specs:
            return []

        spec = self._specs[key]
        if isinstance(spec, dict) and "name" not in spec:
            # Component-level structure
            return list(spec.keys())

        return []

    def has_ticker(self, ticker: str) -> bool:
        """Check if ticker has a specific spec."""
        return ticker in self._specs

    def has_component(self, ticker: str, component: str) -> bool:
        """Check if ticker has a specific component spec."""
        try:
            self.get_component_spec(ticker, component)
            return True
        except KeyError:
            return False

    def add_spec(
        self,
        spec: dict[str, Any] | ModelSpec,
        ticker: Optional[str] = None,
        component: Optional[str] = None,
    ) -> None:
        """
        Add or update a specification.

        Args:
            spec: Specification to add
            ticker: Ticker (None for default)
            component: Component (None for base spec)
        """
        if isinstance(spec, ModelSpec):
            spec = spec.to_dict()

        # Add timestamp
        spec["updated_at"] = datetime.now().isoformat()

        ticker_key = ticker if ticker else self.DEFAULT_KEY

        if component:
            # Component-level spec
            if ticker_key not in self._specs:
                self._specs[ticker_key] = {}
            self._specs[ticker_key][component] = spec
        else:
            # Base spec
            self._specs[ticker_key] = spec

    def save(self, path: Optional[str | Path] = None) -> None:
        """
        Save specifications to JSON file.

        Args:
            path: Output path (uses source path if not specified)
        """
        save_path = Path(path) if path else self._source_path
        if save_path is None:
            raise ValueError("No path specified and no source path available")

        with open(save_path, "w") as f:
            json.dump(self._specs, f, indent=2, default=str)

    def merge(self, other: "ModelSpecRegistry") -> "ModelSpecRegistry":
        """
        Merge another registry into this one.

        Other's specs take precedence on conflicts.

        Args:
            other: Registry to merge

        Returns:
            Self (for chaining)
        """
        for ticker, spec in other._specs.items():
            if isinstance(spec, dict):
                if ticker not in self._specs:
                    self._specs[ticker] = {}
                if isinstance(self._specs[ticker], dict):
                    self._specs[ticker].update(spec)
                else:
                    self._specs[ticker] = spec
            else:
                self._specs[ticker] = spec

        return self


class ModelSpecSearch:
    """
    Search and select optimal model specifications.

    Supports:
    - Grid search over spec parameters
    - Performance-based selection
    - Cross-validation
    """

    def __init__(self, base_spec: ModelSpec):
        """
        Initialize search.

        Args:
            base_spec: Base specification to start from
        """
        self.base_spec = base_spec
        self.search_results: list[dict[str, Any]] = []

    def grid_search(
        self,
        param_grid: dict[str, list[Any]],
        evaluate_fn: Callable[[ModelSpec], dict[str, float]],
        metric: str = "rmse",
        minimize: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Perform grid search over parameter combinations.

        Args:
            param_grid: Dictionary of parameter names to lists of values
                e.g., {"ar_lags": [1, 2, 4], "dist_lags": [1, 2]}
            evaluate_fn: Function that evaluates a spec and returns metrics
            metric: Metric to optimize
            minimize: Whether to minimize (True) or maximize (False)

        Returns:
            Sorted list of results with specs and metrics
        """
        from itertools import product

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        results = []

        for combo in product(*values):
            # Create spec with this combination
            spec_dict = self.base_spec.to_dict()
            for i, key in enumerate(keys):
                spec_dict[key] = combo[i]

            spec = ModelSpec.from_dict(spec_dict)

            # Evaluate
            try:
                metrics = evaluate_fn(spec)
                results.append({
                    "spec": spec,
                    "params": dict(zip(keys, combo)),
                    "metrics": metrics,
                    "score": metrics.get(metric, float("inf") if minimize else float("-inf")),
                })
            except Exception as e:
                print(f"Warning: Evaluation failed for {combo}: {e}")

        # Sort by metric
        results.sort(key=lambda x: x["score"], reverse=not minimize)

        self.search_results = results
        return results

    def get_best_spec(self) -> Optional[ModelSpec]:
        """Get the best specification from search results."""
        if not self.search_results:
            return None
        return self.search_results[0]["spec"]

    def get_top_n_specs(self, n: int = 5) -> list[ModelSpec]:
        """Get top N specifications from search results."""
        return [r["spec"] for r in self.search_results[:n]]


def get_spec_registry(
    indicator: str,
    spec_name: str = "default",
) -> ModelSpecRegistry:
    """
    Get specification registry for an indicator.

    Convenience function that searches in standard locations.

    Args:
        indicator: Indicator name (e.g., "funding_stability")
        spec_name: Specification name (e.g., "ardl", "run_risk")

    Returns:
        ModelSpecRegistry
    """
    from .config import get_config_dir

    # Try indicator-specific directory first
    indicator_path = get_config_dir() / "indicators" / indicator / f"{spec_name}.json"
    if indicator_path.exists():
        return ModelSpecRegistry.from_json(indicator_path)

    # Try model_specs directory
    model_specs_path = get_config_dir() / "model_specs" / f"{indicator}_{spec_name}.json"
    if model_specs_path.exists():
        return ModelSpecRegistry.from_json(model_specs_path)

    # Try just the spec name
    simple_path = get_config_dir() / "model_specs" / f"{spec_name}.json"
    if simple_path.exists():
        return ModelSpecRegistry.from_json(simple_path)

    raise FileNotFoundError(
        f"No specification found for indicator='{indicator}', spec='{spec_name}'"
    )


def create_multi_ticker_spec(
    default_spec: dict[str, Any],
    ticker_overrides: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Create a multi-ticker specification file structure.

    Args:
        default_spec: Default specification
        ticker_overrides: Ticker-specific overrides (merged with default)

    Returns:
        Dictionary suitable for JSON serialization
    """
    result = {ModelSpecRegistry.DEFAULT_KEY: default_spec}

    for ticker, overrides in ticker_overrides.items():
        # Start with default and apply overrides
        ticker_spec = default_spec.copy()
        ticker_spec.update(overrides)
        ticker_spec["name"] = f"{default_spec.get('name', 'unnamed')}_{ticker.lower()}"
        result[ticker] = ticker_spec

    return result


def create_component_spec(
    components: dict[str, dict[str, Any]],
    ticker_components: Optional[dict[str, dict[str, dict[str, Any]]]] = None,
) -> dict[str, Any]:
    """
    Create a component-level specification file structure.

    Args:
        components: Default component specifications
        ticker_components: Ticker-specific component overrides

    Returns:
        Dictionary suitable for JSON serialization
    """
    result = {ModelSpecRegistry.DEFAULT_KEY: components}

    if ticker_components:
        for ticker, comps in ticker_components.items():
            # Start with default components and apply overrides
            ticker_spec = {}
            for comp_name, default_spec in components.items():
                if comp_name in comps:
                    merged = default_spec.copy()
                    merged.update(comps[comp_name])
                    ticker_spec[comp_name] = merged
                else:
                    ticker_spec[comp_name] = default_spec.copy()

            # Add any new components specific to this ticker
            for comp_name, comp_spec in comps.items():
                if comp_name not in ticker_spec:
                    ticker_spec[comp_name] = comp_spec

            result[ticker] = ticker_spec

    return result
