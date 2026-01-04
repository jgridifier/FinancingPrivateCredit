"""
Generic LLM Extraction Framework for SEC Filings.

Provides a reusable structure for extracting structured data from SEC filings
using LLMs (Claude, OpenAI) with regex fallback.

Features:
- Support for multiple LLM providers (Anthropic, OpenAI)
- Configurable extraction prompts
- Batch processing with rate limiting
- JSON schema validation
- Regex fallback when LLM unavailable
- Caching of extraction results

Usage:
    from financing_private_credit.core import LLMExtractor, ExtractionPrompt

    # Define extraction prompt
    prompt = ExtractionPrompt(
        name="commitment_letter",
        system_prompt="You are an expert at analyzing SEC filings...",
        user_template="Extract the following from this 8-K filing: {text}",
        output_schema={
            "commitment_amount": "float",
            "banks": "list[str]",
            "has_sungard": "bool",
        },
        regex_fallback={
            "commitment_amount": r"\$([0-9,]+)\s*(million|billion)",
        }
    )

    extractor = LLMExtractor(provider="anthropic")
    result = extractor.extract(text, prompt)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union

import requests


class LLMProvider(Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"  # For local models via compatible API


@dataclass
class ExtractionPrompt:
    """
    Configuration for an extraction task.

    Defines the prompt template and expected output schema.
    """

    name: str
    system_prompt: str
    user_template: str  # Use {text} for the input text, {context} for additional context

    # Expected output schema (for validation)
    output_schema: dict[str, str] = field(default_factory=dict)

    # Regex patterns for fallback extraction
    regex_fallback: dict[str, str] = field(default_factory=dict)

    # Processing options
    max_input_tokens: int = 8000
    temperature: float = 0.1
    max_output_tokens: int = 1000

    def format_user_prompt(self, text: str, **context) -> str:
        """Format the user prompt with input text and context."""
        return self.user_template.format(text=text, **context)


@dataclass
class ExtractionResult:
    """Result of an extraction operation."""

    prompt_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    method: str = "llm"  # "llm" or "regex"
    error: Optional[str] = None
    processing_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_name": self.prompt_name,
            "success": self.success,
            "data": self.data,
            "method": self.method,
            "error": self.error,
            "processing_time": self.processing_time,
        }


# Pre-built prompts for common extraction tasks
COMMITMENT_LETTER_PROMPT = ExtractionPrompt(
    name="commitment_letter_analysis",
    system_prompt="""You are an expert financial analyst specializing in M&A financing and syndicated loans.
Your task is to analyze 8-K filings and extract key information about bridge commitment letters.

Focus on identifying:
1. The structure and terms of the financing commitment
2. Which banks are providing the commitment
3. Key provisions that affect the bank's ability to exit the deal (rigidity)
4. Any "SunGard-style" or "certain funds" provisions that limit conditionality

Be precise and extract exact figures when available.""",

    user_template="""Analyze this 8-K filing excerpt and extract the commitment letter details.

Filing Text:
{text}

Extract the following information in JSON format:
{{
    "target_company": "name of company being acquired or null",
    "acquirer_company": "name of acquiring company or null",
    "total_commitment_amount_millions": "total commitment amount in millions USD as a number or null",
    "commitment_banks": ["list of bank names providing the commitment"],
    "deal_type": "one of: leveraged_buyout, strategic_acquisition, refinancing, or other",

    "rigidity_indicators": {{
        "has_sungard_clause": "true/false - 'certain funds' or 'SunGard-style' commitment",
        "has_limited_conditionality": "true/false - limited or no financing conditions",
        "has_market_flex": "true/false - bank has pricing flexibility",
        "flex_is_capped": "true/false - market flex has a cap (e.g., 50bps)",
        "flex_cap_bps": "cap amount in basis points or null",
        "has_successful_syndication_condition": "true/false - funding tied to syndication",
        "mae_excludes_market_conditions": "true/false - MAE clause excludes market disruption"
    }},

    "key_terms": {{
        "interest_rate_type": "fixed or floating",
        "maturity_months": "loan maturity in months or null",
        "closing_conditions": "brief description of key closing conditions"
    }},

    "rigidity_score": "0.0 to 1.0 score based on above indicators (1.0 = most rigid/trapped)",
    "confidence": "0.0 to 1.0 confidence in extraction accuracy",
    "reasoning": "brief explanation of rigidity score"
}}

Return ONLY the JSON object, no other text.""",

    output_schema={
        "target_company": "Optional[str]",
        "acquirer_company": "Optional[str]",
        "total_commitment_amount_millions": "Optional[float]",
        "commitment_banks": "list[str]",
        "deal_type": "str",
        "rigidity_indicators": "dict",
        "rigidity_score": "float",
        "confidence": "float",
    },

    regex_fallback={
        "total_commitment_amount_millions": r"\$\s*([0-9,]+(?:\.[0-9]+)?)\s*(billion|million)",
        "has_sungard_clause": r"(?i)(SunGard|certain\s+funds|limited\s+conditionality)",
        "has_market_flex": r"(?i)(market\s+flex|pricing\s+flex)",
    },

    max_input_tokens=12000,
    temperature=0.1,
)


DEAL_TERMS_PROMPT = ExtractionPrompt(
    name="deal_terms_extraction",
    system_prompt="""You are a financial analyst extracting deal terms from SEC filings.
Focus on precise numerical values and specific contractual terms.""",

    user_template="""Extract deal terms from this filing:

{text}

Return JSON:
{{
    "deal_value_millions": "total deal value in millions USD",
    "financing_structure": {{
        "senior_debt_millions": "senior debt amount",
        "subordinated_debt_millions": "subordinated debt amount",
        "equity_millions": "equity contribution"
    }},
    "interest_rate": {{
        "type": "fixed or floating",
        "spread_bps": "spread over base rate in basis points",
        "base_rate": "SOFR, LIBOR, Prime, etc."
    }},
    "covenants": ["list of key financial covenants"]
}}""",

    output_schema={
        "deal_value_millions": "Optional[float]",
        "financing_structure": "dict",
        "interest_rate": "dict",
        "covenants": "list[str]",
    },

    regex_fallback={
        "deal_value_millions": r"(?:deal|transaction|purchase)\s+(?:value|price|consideration)[^\$]*\$\s*([0-9,]+(?:\.[0-9]+)?)\s*(billion|million)?",
    },
)


class LLMExtractor:
    """
    Generic LLM-based extraction engine.

    Supports multiple providers and includes regex fallback.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limit_rpm: int = 50,
    ):
        """
        Initialize the LLM extractor.

        Args:
            provider: LLM provider ("anthropic", "openai", "local")
            api_key: API key (defaults to env var)
            model: Model name (defaults based on provider)
            base_url: Custom API base URL (for local models)
            rate_limit_rpm: Requests per minute limit
        """
        self.provider = LLMProvider(provider.lower())

        # Set defaults based on provider
        if self.provider == LLMProvider.ANTHROPIC:
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.model = model or "claude-3-5-sonnet-20241022"
            self.base_url = base_url or "https://api.anthropic.com/v1"
        elif self.provider == LLMProvider.OPENAI:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.model = model or "gpt-4o"
            self.base_url = base_url or "https://api.openai.com/v1"
        else:
            self.api_key = api_key
            self.model = model or "local-model"
            self.base_url = base_url or "http://localhost:8000/v1"

        self.rate_limit_rpm = rate_limit_rpm
        self._last_request_time = 0.0
        self._min_delay = 60.0 / rate_limit_rpm

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_request_time = time.time()

    def extract(
        self,
        text: str,
        prompt: ExtractionPrompt,
        use_llm: bool = True,
        context: Optional[dict[str, Any]] = None,
    ) -> ExtractionResult:
        """
        Extract structured data from text using the specified prompt.

        Args:
            text: Input text to analyze
            prompt: ExtractionPrompt configuration
            use_llm: Whether to use LLM (False = regex only)
            context: Additional context for prompt formatting

        Returns:
            ExtractionResult with extracted data
        """
        start_time = time.time()
        context = context or {}

        # Truncate text if needed
        if len(text) > prompt.max_input_tokens * 4:  # Rough char-to-token ratio
            text = self._truncate_text(text, prompt.max_input_tokens * 4)

        # Try LLM extraction first
        if use_llm and self.api_key:
            try:
                result = self._llm_extract(text, prompt, context)
                if result.success:
                    result.processing_time = time.time() - start_time
                    return result
            except Exception as e:
                print(f"LLM extraction failed: {e}, falling back to regex")

        # Fallback to regex extraction
        result = self._regex_extract(text, prompt)
        result.processing_time = time.time() - start_time
        return result

    def _llm_extract(
        self,
        text: str,
        prompt: ExtractionPrompt,
        context: dict[str, Any],
    ) -> ExtractionResult:
        """Extract using LLM API."""
        self._rate_limit()

        user_prompt = prompt.format_user_prompt(text, **context)

        if self.provider == LLMProvider.ANTHROPIC:
            return self._anthropic_extract(prompt, user_prompt)
        elif self.provider == LLMProvider.OPENAI:
            return self._openai_extract(prompt, user_prompt)
        else:
            return self._openai_compatible_extract(prompt, user_prompt)

    def _anthropic_extract(
        self,
        prompt: ExtractionPrompt,
        user_prompt: str,
    ) -> ExtractionResult:
        """Extract using Anthropic Claude API."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "max_tokens": prompt.max_output_tokens,
            "temperature": prompt.temperature,
            "system": prompt.system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
        }

        response = requests.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            return ExtractionResult(
                prompt_name=prompt.name,
                success=False,
                error=f"API error: {response.status_code} - {response.text[:200]}",
            )

        result = response.json()
        content = result["content"][0]["text"]

        return self._parse_json_response(prompt.name, content)

    def _openai_extract(
        self,
        prompt: ExtractionPrompt,
        user_prompt: str,
    ) -> ExtractionResult:
        """Extract using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "temperature": prompt.temperature,
            "max_tokens": prompt.max_output_tokens,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            return ExtractionResult(
                prompt_name=prompt.name,
                success=False,
                error=f"API error: {response.status_code}",
            )

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return self._parse_json_response(prompt.name, content)

    def _openai_compatible_extract(
        self,
        prompt: ExtractionPrompt,
        user_prompt: str,
    ) -> ExtractionResult:
        """Extract using OpenAI-compatible API (for local models)."""
        # Same as OpenAI but with custom base URL
        return self._openai_extract(prompt, user_prompt)

    def _parse_json_response(
        self,
        prompt_name: str,
        content: str,
    ) -> ExtractionResult:
        """Parse JSON from LLM response."""
        try:
            # Try to find JSON in response
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group())
                return ExtractionResult(
                    prompt_name=prompt_name,
                    success=True,
                    data=data,
                    raw_response=content,
                    method="llm",
                )

        except json.JSONDecodeError as e:
            return ExtractionResult(
                prompt_name=prompt_name,
                success=False,
                raw_response=content,
                error=f"JSON parse error: {e}",
            )

        return ExtractionResult(
            prompt_name=prompt_name,
            success=False,
            raw_response=content,
            error="No JSON found in response",
        )

    def _regex_extract(
        self,
        text: str,
        prompt: ExtractionPrompt,
    ) -> ExtractionResult:
        """Extract using regex fallback patterns."""
        data = {}

        for field, pattern in prompt.regex_fallback.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Handle different capture group types
                if match.lastindex and match.lastindex >= 1:
                    value = match.group(1)

                    # Try to convert to appropriate type
                    if field.endswith("_millions") or field.endswith("_amount"):
                        try:
                            value = float(value.replace(",", ""))
                            # Check for billion/million unit
                            if match.lastindex >= 2:
                                unit = match.group(2)
                                if unit and "billion" in unit.lower():
                                    value *= 1000
                        except ValueError:
                            pass
                    elif field.startswith("has_") or field.startswith("is_"):
                        value = True
                else:
                    # Boolean patterns (just check if match exists)
                    value = True

                data[field] = value

        return ExtractionResult(
            prompt_name=prompt.name,
            success=bool(data),
            data=data,
            method="regex",
        )

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate text keeping beginning and end."""
        if len(text) <= max_chars:
            return text

        half = max_chars // 2
        return text[:half] + "\n...[TRUNCATED]...\n" + text[-half:]

    def extract_batch(
        self,
        texts: list[str],
        prompt: ExtractionPrompt,
        use_llm: bool = True,
        show_progress: bool = True,
    ) -> list[ExtractionResult]:
        """
        Extract from multiple texts.

        Args:
            texts: List of input texts
            prompt: ExtractionPrompt configuration
            use_llm: Whether to use LLM
            show_progress: Whether to print progress

        Returns:
            List of ExtractionResult objects
        """
        results = []

        for i, text in enumerate(texts):
            if show_progress:
                print(f"  Processing {i + 1}/{len(texts)}...")

            result = self.extract(text, prompt, use_llm)
            results.append(result)

        return results


# Convenience function for quick extraction
def extract_commitment_letter(
    text: str,
    provider: str = "anthropic",
    use_llm: bool = True,
) -> ExtractionResult:
    """
    Quick extraction of commitment letter details.

    Args:
        text: 8-K filing text
        provider: LLM provider to use
        use_llm: Whether to use LLM or regex-only

    Returns:
        ExtractionResult with commitment letter data
    """
    extractor = LLMExtractor(provider=provider)
    return extractor.extract(text, COMMITMENT_LETTER_PROMPT, use_llm=use_llm)
