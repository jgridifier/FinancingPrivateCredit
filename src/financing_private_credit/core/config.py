"""
Configuration Loading and Validation

Provides structured config loading with:
- JSON/YAML support
- Schema validation
- Environment variable interpolation
- Indicator-specific config paths
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypeVar, Type

T = TypeVar("T")


def get_project_root() -> Path:
    """Get the project root directory."""
    # Walk up from this file to find pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to current working directory
    return Path.cwd()


def get_config_dir() -> Path:
    """Get the config directory."""
    return get_project_root() / "config"


def get_indicator_config_path(indicator_name: str, config_name: str) -> Path:
    """
    Get path to an indicator-specific config file.

    Args:
        indicator_name: Name of the indicator (e.g., "credit_boom")
        config_name: Name of the config file (without extension)

    Returns:
        Path to config file
    """
    # Try indicator-specific directory first
    indicator_dir = get_config_dir() / "indicators" / indicator_name
    if indicator_dir.exists():
        config_path = indicator_dir / f"{config_name}.json"
        if config_path.exists():
            return config_path

    # Fall back to model_specs directory
    return get_config_dir() / "model_specs" / f"{config_name}.json"


@dataclass
class ConfigSchema:
    """
    Schema definition for config validation.

    Usage:
        schema = ConfigSchema(
            required=["name", "target"],
            optional={"ar_lags": [1, 2, 3, 4]},
            types={"ar_lags": list, "max_bins": int},
        )
        errors = schema.validate(config_dict)
    """

    required: list[str] = field(default_factory=list)
    optional: dict[str, Any] = field(default_factory=dict)
    types: dict[str, type] = field(default_factory=dict)
    validators: dict[str, callable] = field(default_factory=dict)

    def validate(self, config: dict[str, Any]) -> list[str]:
        """
        Validate a config against this schema.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required fields
        for field_name in self.required:
            if field_name not in config:
                errors.append(f"Missing required field: {field_name}")

        # Check types
        for field_name, expected_type in self.types.items():
            if field_name in config:
                value = config[field_name]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Field '{field_name}' expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

        # Run custom validators
        for field_name, validator in self.validators.items():
            if field_name in config:
                try:
                    if not validator(config[field_name]):
                        errors.append(f"Validation failed for field: {field_name}")
                except Exception as e:
                    errors.append(f"Validator error for '{field_name}': {e}")

        return errors

    def apply_defaults(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply default values for missing optional fields."""
        result = config.copy()
        for field_name, default_value in self.optional.items():
            if field_name not in result:
                result[field_name] = default_value
        return result


class ConfigLoader:
    """
    Load and validate configuration files.

    Supports:
    - JSON files
    - Environment variable interpolation (${VAR_NAME})
    - Schema validation
    - Caching
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize config loader.

        Args:
            base_dir: Base directory for config files (defaults to project config/)
        """
        self.base_dir = base_dir or get_config_dir()
        self._cache: dict[str, dict] = {}

    def load(
        self,
        path: str | Path,
        schema: Optional[ConfigSchema] = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Load a config file.

        Args:
            path: Path to config file (absolute or relative to base_dir)
            schema: Optional schema for validation
            use_cache: Whether to use cached config

        Returns:
            Loaded and validated config dict

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If validation fails
        """
        # Resolve path
        if not Path(path).is_absolute():
            path = self.base_dir / path

        path = Path(path)
        cache_key = str(path)

        # Check cache
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Load file
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config = json.load(f)

        # Interpolate environment variables
        config = self._interpolate_env_vars(config)

        # Validate and apply defaults
        if schema:
            errors = schema.validate(config)
            if errors:
                raise ValueError(
                    f"Config validation failed for {path}:\n" +
                    "\n".join(f"  - {e}" for e in errors)
                )
            config = schema.apply_defaults(config)

        # Cache result
        if use_cache:
            self._cache[cache_key] = config

        return config

    def _interpolate_env_vars(self, obj: Any) -> Any:
        """Recursively interpolate ${VAR} patterns with environment variables."""
        if isinstance(obj, str):
            # Look for ${VAR} patterns
            import re
            pattern = r"\$\{([^}]+)\}"

            def replace(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))

            return re.sub(pattern, replace, obj)

        elif isinstance(obj, dict):
            return {k: self._interpolate_env_vars(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self._interpolate_env_vars(item) for item in obj]

        return obj

    def clear_cache(self):
        """Clear the config cache."""
        self._cache.clear()


def validate_config(
    config: dict[str, Any],
    schema: ConfigSchema,
) -> dict[str, Any]:
    """
    Validate and apply defaults to a config dict.

    Convenience function for inline validation.
    """
    errors = schema.validate(config)
    if errors:
        raise ValueError(
            "Config validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )
    return schema.apply_defaults(config)


# Common schemas for reuse
MODEL_SPEC_SCHEMA = ConfigSchema(
    required=["name", "target"],
    optional={
        "ar_lags": [1, 2, 3, 4],
        "include_seasonality": True,
        "seasonality_period": 4,
        "holdout_periods": 4,
    },
    types={
        "name": str,
        "ar_lags": list,
        "include_seasonality": bool,
        "seasonality_period": int,
        "holdout_periods": int,
    },
)

INDICATOR_CONFIG_SCHEMA = ConfigSchema(
    required=["indicator"],
    optional={
        "start_date": "2015-01-01",
        "banks": None,
    },
    types={
        "indicator": str,
        "start_date": str,
    },
)
