"""
Core infrastructure for the indicator framework.

Provides:
- Configuration loading and validation
- Generic registry pattern
- Model specification system (multi-spec, per-ticker, component-level)
- Data registry with smart caching
- SEC EDGAR filing access
- LLM extraction framework
- Common utilities
"""

from .config import (
    ConfigLoader,
    ConfigSchema,
    validate_config,
    get_indicator_config_path,
)
from .registry import Registry
from .model_specs import (
    ModelSpec,
    ModelSpecRegistry,
    ModelSpecSearch,
    get_spec_registry,
    create_multi_ticker_spec,
    create_component_spec,
)
from .data_registry import (
    DataRegistry,
    DataCache,
    CacheConfig,
    CacheEntry,
)
from .sec_edgar import (
    SECEdgarClient,
    Filing,
    CommitmentLetterExtract,
)
from .llm_extractor import (
    LLMExtractor,
    LLMProvider,
    ExtractionPrompt,
    ExtractionResult,
    COMMITMENT_LETTER_PROMPT,
    DEAL_TERMS_PROMPT,
    extract_commitment_letter,
)
from .utils import (
    to_quarterly,
    compute_yoy_growth,
    compute_rolling_stats,
    format_pct,
    format_currency,
)

__all__ = [
    # Config
    "ConfigLoader",
    "ConfigSchema",
    "validate_config",
    "get_indicator_config_path",
    # Registry
    "Registry",
    # Model Specs
    "ModelSpec",
    "ModelSpecRegistry",
    "ModelSpecSearch",
    "get_spec_registry",
    "create_multi_ticker_spec",
    "create_component_spec",
    # Data Registry
    "DataRegistry",
    "DataCache",
    "CacheConfig",
    "CacheEntry",
    # SEC EDGAR
    "SECEdgarClient",
    "Filing",
    "CommitmentLetterExtract",
    # LLM Extraction
    "LLMExtractor",
    "LLMProvider",
    "ExtractionPrompt",
    "ExtractionResult",
    "COMMITMENT_LETTER_PROMPT",
    "DEAL_TERMS_PROMPT",
    "extract_commitment_letter",
    # Utils
    "to_quarterly",
    "compute_yoy_growth",
    "compute_rolling_stats",
    "format_pct",
    "format_currency",
]
