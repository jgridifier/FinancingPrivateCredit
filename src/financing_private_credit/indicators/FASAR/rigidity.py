"""
Rigidity Score Extraction from 8-K Filings

Uses a two-stage approach:
1. Regex/keyword search for identification of relevant clauses
2. LLM (OpenAI-compatible) for reasoning-based scoring

Rigidity Score Interpretation:
- 1.0 = "SunGard" / Limited Conditionality (Bank MUST fund)
- 0.0 = Full Market Outs (Bank can cancel if market conditions change)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import requests


class RigidityClassification(Enum):
    """Classification of deal rigidity."""

    TRAPPED = "trapped"  # Bank must fund regardless of market
    CONSTRAINED = "constrained"  # Limited flex, bank has some risk
    FLEXIBLE = "flexible"  # Bank has market flex / outs
    UNKNOWN = "unknown"  # Cannot determine from text


@dataclass
class RigidityEvidence:
    """Evidence extracted from 8-K text."""

    # SunGard / Limited Conditionality indicators
    sungard_clause: bool = False
    limited_conditionality: bool = False
    certain_funds: bool = False

    # Market flex indicators
    market_flex_mentioned: bool = False
    flex_is_capped: bool = False
    flex_cap_amount: Optional[str] = None  # e.g., "200bps"

    # Conditions
    successful_syndication_condition: bool = False
    mae_excludes_market: bool = False

    # Raw evidence
    relevant_snippets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sungard_clause": self.sungard_clause,
            "limited_conditionality": self.limited_conditionality,
            "certain_funds": self.certain_funds,
            "market_flex_mentioned": self.market_flex_mentioned,
            "flex_is_capped": self.flex_is_capped,
            "flex_cap_amount": self.flex_cap_amount,
            "successful_syndication_condition": self.successful_syndication_condition,
            "mae_excludes_market": self.mae_excludes_market,
            "n_snippets": len(self.relevant_snippets),
        }


# Regex patterns for identifying relevant clauses
RIGIDITY_PATTERNS = {
    # High rigidity indicators
    "sungard": re.compile(
        r"sungard|sun\s*gard|certain\s+funds?|limited\s+conditionality",
        re.IGNORECASE,
    ),
    "must_fund": re.compile(
        r"(shall|must|will)\s+(be\s+)?(required\s+to\s+)?(fund|provide|advance)",
        re.IGNORECASE,
    ),
    "no_conditions": re.compile(
        r"without\s+(any\s+)?(conditions?|contingenc)",
        re.IGNORECASE,
    ),

    # Market flex indicators (lower rigidity)
    "market_flex": re.compile(
        r"market\s+flex|flex\s+provision|pricing\s+flex",
        re.IGNORECASE,
    ),
    "successful_syndication": re.compile(
        r"(condition|subject)\s+(to|of|upon)\s+successful\s+syndication",
        re.IGNORECASE,
    ),

    # Flex limitations
    "flex_capped": re.compile(
        r"flex\s+(is\s+)?(limited|capped|restricted)\s+to\s+(\d+)",
        re.IGNORECASE,
    ),
    "customary_flex": re.compile(
        r"customary\s+(market\s+)?flex",
        re.IGNORECASE,
    ),

    # MAE clauses
    "mae_clause": re.compile(
        r"material\s+adverse\s+(effect|change|event)",
        re.IGNORECASE,
    ),
    "mae_excludes_market": re.compile(
        r"(exclud|except)\w*\s+(from\s+)?(the\s+)?(definition\s+of\s+)?.*?(general\s+)?market\s+condition",
        re.IGNORECASE,
    ),

    # Commitment letter references
    "commitment_letter": re.compile(
        r"commitment\s+letter|bridge\s+commitment|financing\s+commitment",
        re.IGNORECASE,
    ),
    "fee_letter": re.compile(
        r"fee\s+letter",
        re.IGNORECASE,
    ),
}


def extract_rigidity_evidence(text: str) -> RigidityEvidence:
    """
    Stage 1: Extract rigidity evidence using regex patterns.

    Args:
        text: 8-K filing text (Item 1.01 or Exhibit 10)

    Returns:
        RigidityEvidence with detected patterns
    """
    evidence = RigidityEvidence()
    snippets = []

    # Normalize text
    text_normalized = " ".join(text.split())

    # Check each pattern
    if RIGIDITY_PATTERNS["sungard"].search(text_normalized):
        evidence.sungard_clause = True
        # Extract surrounding context
        for match in RIGIDITY_PATTERNS["sungard"].finditer(text_normalized):
            start = max(0, match.start() - 100)
            end = min(len(text_normalized), match.end() + 100)
            snippets.append(text_normalized[start:end])

    if RIGIDITY_PATTERNS["must_fund"].search(text_normalized):
        evidence.limited_conditionality = True

    if RIGIDITY_PATTERNS["market_flex"].search(text_normalized):
        evidence.market_flex_mentioned = True
        for match in RIGIDITY_PATTERNS["market_flex"].finditer(text_normalized):
            start = max(0, match.start() - 100)
            end = min(len(text_normalized), match.end() + 100)
            snippets.append(text_normalized[start:end])

    if RIGIDITY_PATTERNS["successful_syndication"].search(text_normalized):
        evidence.successful_syndication_condition = True

    # Check for flex caps
    flex_cap_match = RIGIDITY_PATTERNS["flex_capped"].search(text_normalized)
    if flex_cap_match:
        evidence.flex_is_capped = True
        evidence.flex_cap_amount = flex_cap_match.group(3) + "bps"

    if RIGIDITY_PATTERNS["customary_flex"].search(text_normalized):
        # Customary flex without cap = open flex = low rigidity
        if not evidence.flex_is_capped:
            evidence.market_flex_mentioned = True

    if RIGIDITY_PATTERNS["mae_excludes_market"].search(text_normalized):
        evidence.mae_excludes_market = True

    if RIGIDITY_PATTERNS["no_conditions"].search(text_normalized):
        evidence.certain_funds = True

    evidence.relevant_snippets = snippets[:5]  # Limit to 5 snippets

    return evidence


def compute_preliminary_rigidity(evidence: RigidityEvidence) -> float:
    """
    Compute preliminary rigidity score from evidence.

    This is used when LLM is not available or as a baseline.
    """
    score = 0.5  # Start at neutral

    # High rigidity indicators
    if evidence.sungard_clause:
        score += 0.3
    if evidence.limited_conditionality:
        score += 0.15
    if evidence.certain_funds:
        score += 0.2
    if evidence.mae_excludes_market:
        score += 0.15  # MAE excluding market = bank can't use market downturn as out

    # Low rigidity indicators
    if evidence.successful_syndication_condition:
        score -= 0.4  # Bank can walk if syndication fails
    if evidence.market_flex_mentioned and not evidence.flex_is_capped:
        score -= 0.2  # Open flex = bank has pricing power
    if evidence.flex_is_capped:
        score += 0.1  # Capped flex = limited adjustment ability

    return max(0.0, min(1.0, score))


# LLM prompt for rigidity scoring
RIGIDITY_SCORING_PROMPT = """You are a legal analyst specializing in leveraged finance and syndicated loan documentation.

Your task is to determine the "Rigidity Score" for a bank's bridge loan commitment based on the filing text. The Rigidity Score measures how trapped the bank is - their inability to exit or reprice the deal.

## Rigidity Score Scale:
- 1.0 = TRAPPED: Bank MUST fund regardless of market conditions (SunGard/Certain Funds terms)
- 0.7-0.9 = HIGH: Limited conditionality, bank has minimal flexibility
- 0.4-0.6 = MODERATE: Some flex provisions but with caps/limits
- 0.1-0.3 = LOW: Significant market flex, bank can reprice substantially
- 0.0 = FREE: Successful syndication condition OR full market outs

## Key Terms to Look For:

HIGH RIGIDITY (increases score):
- "SunGard" or "Certain Funds" = Bank must fund unconditionally
- "Limited Conditionality" = Few ways out for bank
- MAE definition EXCLUDES "general market conditions" = Can't use market downturn as excuse
- Flex is "capped" or "limited to X bps" = Restricted repricing ability

LOW RIGIDITY (decreases score):
- "Subject to successful syndication" = Bank can exit if deal doesn't sell
- "Customary market flex" without limits = Bank can reprice freely
- "Market out" provisions = Bank can cancel if markets deteriorate
- MAE definition INCLUDES market conditions = Broad exit rights

## Example:
Text: "The Commitment is subject to limited conditionality and provides for certain funds certainty, with flex limited to 50 basis points."
Reasoning: Limited conditionality + certain funds = high rigidity. Flex is capped at 50bps = limited repricing. No syndication out mentioned.
Score: 0.85

## Filing Text to Analyze:
{text}

## Previously Detected Evidence:
{evidence}

Based on the text and evidence above, provide your analysis in the following JSON format:
{{
    "reasoning": "Your step-by-step analysis of the key terms found",
    "rigidity_score": <float between 0.0 and 1.0>,
    "classification": "<trapped|constrained|flexible|unknown>",
    "confidence": <float between 0.0 and 1.0>,
    "key_findings": ["list", "of", "key", "findings"]
}}
"""


class RigidityScorer:
    """
    Two-stage rigidity scorer using regex + LLM.

    Stage 1: Regex patterns identify relevant clauses
    Stage 2: LLM reasons about the combined evidence to score
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        use_llm: bool = True,
    ):
        """
        Initialize the rigidity scorer.

        Args:
            api_base: OpenAI-compatible API base URL
            api_key: API key for the LLM service
            model: Model name to use
            use_llm: Whether to use LLM (False = regex-only scoring)
        """
        self.api_base = api_base or os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.use_llm = use_llm and self.api_key is not None

    def score(
        self,
        text: str,
        deal_id: Optional[str] = None,
    ) -> tuple[float, RigidityClassification, RigidityEvidence]:
        """
        Score the rigidity of a deal from 8-K text.

        Args:
            text: 8-K filing text
            deal_id: Optional identifier for logging

        Returns:
            Tuple of (rigidity_score, classification, evidence)
        """
        # Stage 1: Extract evidence with regex
        evidence = extract_rigidity_evidence(text)

        # Compute preliminary score
        preliminary_score = compute_preliminary_rigidity(evidence)

        if not self.use_llm:
            # Return regex-based scoring
            classification = self._classify(preliminary_score)
            return preliminary_score, classification, evidence

        # Stage 2: Use LLM for reasoning-based scoring
        try:
            llm_result = self._llm_score(text, evidence)

            if llm_result:
                score = llm_result.get("rigidity_score", preliminary_score)
                classification = RigidityClassification(
                    llm_result.get("classification", "unknown")
                )

                # Add LLM findings to evidence
                if "key_findings" in llm_result:
                    evidence.relevant_snippets.extend(llm_result["key_findings"])

                return score, classification, evidence

        except Exception as e:
            print(f"LLM scoring failed, using regex fallback: {e}")

        # Fallback to preliminary score
        classification = self._classify(preliminary_score)
        return preliminary_score, classification, evidence

    def _llm_score(
        self,
        text: str,
        evidence: RigidityEvidence,
    ) -> Optional[dict[str, Any]]:
        """
        Use LLM to reason about rigidity score.

        Args:
            text: Filing text (truncated if needed)
            evidence: Pre-extracted evidence

        Returns:
            Parsed LLM response or None if failed
        """
        # Truncate text to fit context window
        max_text_length = 8000
        if len(text) > max_text_length:
            # Keep beginning and end where key terms often appear
            text = text[:max_text_length // 2] + "\n...[TRUNCATED]...\n" + text[-max_text_length // 2:]

        prompt = RIGIDITY_SCORING_PROMPT.format(
            text=text,
            evidence=json.dumps(evidence.to_dict(), indent=2),
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # Low temperature for consistent scoring
            "max_tokens": 500,
        }

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            return None

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return None

    def _classify(self, score: float) -> RigidityClassification:
        """Classify based on score."""
        if score >= 0.7:
            return RigidityClassification.TRAPPED
        elif score >= 0.4:
            return RigidityClassification.CONSTRAINED
        elif score >= 0.1:
            return RigidityClassification.FLEXIBLE
        else:
            return RigidityClassification.UNKNOWN

    def score_batch(
        self,
        filings: list[dict[str, str]],
    ) -> list[tuple[str, float, RigidityClassification, RigidityEvidence]]:
        """
        Score multiple filings.

        Args:
            filings: List of dicts with 'deal_id' and 'text' keys

        Returns:
            List of (deal_id, score, classification, evidence) tuples
        """
        results = []
        for filing in filings:
            deal_id = filing.get("deal_id", "unknown")
            text = filing.get("text", "")

            score, classification, evidence = self.score(text, deal_id)
            results.append((deal_id, score, classification, evidence))

        return results


def apply_binary_fallback(evidence: RigidityEvidence) -> float:
    """
    Apply binary fallback logic for redacted fee letters.

    From the spec:
    - If "successful syndication" appears → Rigidity = 0 (Bank is safe)
    - If absent and "SunGard" language present → Rigidity = 1 (Bank is trapped)
    """
    if evidence.successful_syndication_condition:
        return 0.0

    if evidence.sungard_clause or evidence.certain_funds or evidence.limited_conditionality:
        return 1.0

    # Default to moderate when cannot determine
    return 0.5
