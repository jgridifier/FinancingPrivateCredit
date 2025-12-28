"""
Core infrastructure for the indicator framework.

Provides:
- Configuration loading and validation
- Generic registry pattern
- Common utilities
"""

from .config import (
    ConfigLoader,
    ConfigSchema,
    validate_config,
    get_indicator_config_path,
)
from .registry import Registry
from .utils import (
    to_quarterly,
    compute_yoy_growth,
    compute_rolling_stats,
    format_pct,
    format_currency,
)

__all__ = [
    "ConfigLoader",
    "ConfigSchema",
    "validate_config",
    "get_indicator_config_path",
    "Registry",
    "to_quarterly",
    "compute_yoy_growth",
    "compute_rolling_stats",
    "format_pct",
    "format_currency",
]
