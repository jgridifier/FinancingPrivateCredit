"""
SEC EDGAR API Client for fetching 8-K filings.

Provides functionality to:
- Search for 8-K filings by date, company, or form type
- Download and parse filing documents
- Extract specific items (e.g., Item 1.01 Material Agreements)
- Handle rate limiting (SEC EDGAR limit: 10 requests/second)

SEC EDGAR API Documentation: https://www.sec.gov/developer

Usage:
    from financing_private_credit.core import SECEdgarClient

    client = SECEdgarClient()

    # Get 8-K filings for a date
    filings = client.get_8k_filings(date="2024-01-15")

    # Get commitment letters from filings
    letters = client.extract_commitment_letters(filings)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from pathlib import Path
import json

import requests
import polars as pl


# SEC EDGAR API endpoints
SEC_EDGAR_BASE = "https://www.sec.gov"
SEC_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SEC_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SEC_COMPANY_SEARCH = f"{SEC_EDGAR_BASE}/cgi-bin/browse-edgar"
SEC_FILINGS_API = "https://data.sec.gov/submissions"

# Rate limiting
SEC_RATE_LIMIT = 10  # requests per second
SEC_MIN_DELAY = 0.1  # minimum delay between requests


@dataclass
class Filing:
    """Represents an SEC filing."""

    accession_number: str
    form_type: str
    filing_date: datetime
    company_name: str
    cik: str
    document_url: str
    description: Optional[str] = None

    # Parsed content
    raw_text: Optional[str] = None
    items: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accession_number": self.accession_number,
            "form_type": self.form_type,
            "filing_date": self.filing_date.isoformat(),
            "company_name": self.company_name,
            "cik": self.cik,
            "document_url": self.document_url,
            "description": self.description,
            "items": self.items,
        }


@dataclass
class CommitmentLetterExtract:
    """Extracted commitment letter information from 8-K."""

    filing: Filing
    target_company: Optional[str] = None
    acquirer_company: Optional[str] = None
    commitment_amount: Optional[float] = None  # In millions
    commitment_banks: list[str] = field(default_factory=list)
    raw_text: str = ""

    # Key provisions detected
    has_sungard_clause: bool = False
    has_limited_conditionality: bool = False
    has_market_flex: bool = False
    has_flex_cap: bool = False
    has_mae_market_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accession_number": self.filing.accession_number,
            "filing_date": self.filing.filing_date.isoformat(),
            "company_name": self.filing.company_name,
            "cik": self.filing.cik,
            "target_company": self.target_company,
            "acquirer_company": self.acquirer_company,
            "commitment_amount": self.commitment_amount,
            "commitment_banks": self.commitment_banks,
            "has_sungard_clause": self.has_sungard_clause,
            "has_limited_conditionality": self.has_limited_conditionality,
            "has_market_flex": self.has_market_flex,
            "has_flex_cap": self.has_flex_cap,
            "has_mae_market_out": self.has_mae_market_out,
        }


class SECEdgarClient:
    """
    Client for fetching and parsing SEC EDGAR filings.

    Focuses on 8-K filings with Item 1.01 (Material Agreements)
    which contain commitment letters for M&A financing.
    """

    def __init__(
        self,
        user_agent: str = "FinancingPrivateCredit research@example.com",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize SEC EDGAR client.

        Args:
            user_agent: User agent string (SEC requires identification)
            cache_dir: Directory to cache downloaded filings
        """
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "application/json, text/html, */*",
        })

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self) -> None:
        """Enforce SEC rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < SEC_MIN_DELAY:
            time.sleep(SEC_MIN_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, **kwargs) -> requests.Response:
        """Make a rate-limited GET request."""
        self._rate_limit()
        response = self._session.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response

    def get_8k_filings(
        self,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cik: Optional[str] = None,
        company_name: Optional[str] = None,
        max_results: int = 100,
    ) -> list[Filing]:
        """
        Search for 8-K filings.

        Args:
            date: Specific date (YYYY-MM-DD)
            start_date: Start of date range
            end_date: End of date range (None = get latest without date filter)
            cik: Filter by company CIK
            company_name: Filter by company name (partial match)
            max_results: Maximum number of results

        Returns:
            List of Filing objects
        """
        # Build search query
        if date:
            start_date = date
            end_date = date

        # Build params for search
        # If no dates specified, don't filter by date (get latest filings)
        params = {
            "dateRange": "custom" if (start_date or end_date) else "all",
            "forms": "8-K",
        }

        if start_date:
            params["startdt"] = start_date
        if end_date:
            params["enddt"] = end_date

        if cik:
            params["ciks"] = cik

        if company_name:
            params["q"] = company_name

        try:
            # Use the atom feed API
            response = self._search_filings(params, max_results)
            return response
        except Exception as e:
            print(f"Search failed: {e}, trying alternative method...")
            return self._search_filings_alternative(start_date, end_date, max_results)

    def _search_filings(
        self,
        params: dict,
        max_results: int,
    ) -> list[Filing]:
        """Search filings using SEC atom feed API."""
        # Use the atom feed API which is more reliable
        feed_url = f"{SEC_EDGAR_BASE}/cgi-bin/browse-edgar"

        query_params = {
            "action": "getcurrent",
            "type": "8-K",
            "company": params.get("q", ""),
            "count": min(max_results, 100),
            "output": "atom",
        }

        try:
            response = self._get(feed_url, params=query_params)
            filings = self._parse_atom_feed(response.text, params, max_results)
            return filings

        except Exception as e:
            print(f"Atom feed API error: {e}")
            return []

    def _parse_atom_feed(
        self,
        xml_text: str,
        params: dict,
        max_results: int,
    ) -> list[Filing]:
        """Parse SEC EDGAR atom feed response."""
        filings = []

        # Parse using regex (avoiding XML library dependency issues)
        entry_pattern = r"<entry>(.*?)</entry>"
        entries = re.findall(entry_pattern, xml_text, re.DOTALL)

        start_date = params.get("startdt")
        end_date = params.get("enddt")

        for entry in entries[:max_results * 2]:  # Get more to filter by date
            try:
                # Extract title (company name and CIK)
                title_match = re.search(r"<title>8-K - (.+?) \((\d+)\)", entry)
                if not title_match:
                    continue

                company_name = title_match.group(1)
                cik = title_match.group(2)

                # Extract accession number
                acc_match = re.search(r"AccNo:.*?(\d{10}-\d{2}-\d{6})", entry)
                if not acc_match:
                    continue
                accession = acc_match.group(1)

                # Extract filing date
                date_match = re.search(r"Filed:.*?(\d{4}-\d{2}-\d{2})", entry)
                if date_match:
                    filing_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                else:
                    continue

                # Filter by date range if specified
                if start_date:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    if filing_date < start_dt:
                        continue
                if end_date:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    if filing_date > end_dt:
                        continue

                # Extract link
                link_match = re.search(r'<link rel="alternate".*?href="([^"]+)"', entry)
                doc_url = link_match.group(1) if link_match else ""

                # Extract description (items)
                desc_match = re.search(r"Item 1\.01.*?Agreement", entry)
                description = "Item 1.01" if desc_match else None

                filing = Filing(
                    accession_number=accession.replace("-", ""),
                    form_type="8-K",
                    filing_date=filing_date,
                    company_name=company_name,
                    cik=cik,
                    document_url=self._build_document_url(cik, accession),
                    description=description,
                )
                filings.append(filing)

                if len(filings) >= max_results:
                    break

            except Exception as e:
                continue

        return filings

    def _search_filings_alternative(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        max_results: int,
    ) -> list[Filing]:
        """
        Alternative search using the company filings API.

        Falls back to searching through recent filings index.
        """
        # Use SEC's daily filings index
        filings = []

        target_date = datetime.strptime(
            start_date or datetime.now().strftime("%Y-%m-%d"),
            "%Y-%m-%d"
        )
        end = datetime.strptime(
            end_date or datetime.now().strftime("%Y-%m-%d"),
            "%Y-%m-%d"
        )

        current_date = target_date
        while current_date <= end and len(filings) < max_results:
            day_filings = self._get_daily_filings(current_date)
            filings.extend(day_filings)
            current_date += timedelta(days=1)

        return filings[:max_results]

    def _get_daily_filings(self, date: datetime) -> list[Filing]:
        """Get 8-K filings from the daily index."""
        # SEC daily index URL format
        year = date.year
        quarter = (date.month - 1) // 3 + 1

        index_url = (
            f"{SEC_EDGAR_BASE}/Archives/edgar/daily-index/"
            f"{year}/QTR{quarter}/company.{date.strftime('%Y%m%d')}.idx"
        )

        try:
            response = self._get(index_url)
            lines = response.text.split("\n")

            filings = []
            for line in lines:
                if "8-K" in line and "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 5:
                        company_name = parts[0].strip()
                        form_type = parts[1].strip()
                        cik = parts[2].strip()
                        filing_date_str = parts[3].strip()
                        accession = parts[4].strip().replace(".txt", "")

                        if form_type == "8-K":
                            filing = Filing(
                                accession_number=accession.replace("-", ""),
                                form_type="8-K",
                                filing_date=datetime.strptime(filing_date_str, "%Y-%m-%d"),
                                company_name=company_name,
                                cik=cik,
                                document_url=self._build_document_url(cik, accession),
                            )
                            filings.append(filing)

            return filings

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                # No filings for this date (weekend/holiday)
                return []
            raise

    def _build_document_url(self, cik: str, accession: str) -> str:
        """Build the URL for the filing document."""
        cik = str(cik).lstrip("0")
        accession_formatted = accession.replace("-", "")
        return (
            f"{SEC_EDGAR_BASE}/Archives/edgar/data/"
            f"{cik}/{accession_formatted}/{accession}.txt"
        )

    def download_filing(self, filing: Filing) -> str:
        """
        Download the full filing document.

        Args:
            filing: Filing object with document URL

        Returns:
            Raw filing text
        """
        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"{filing.accession_number}.txt"
            if cache_file.exists():
                return cache_file.read_text(encoding="utf-8", errors="ignore")

        try:
            response = self._get(filing.document_url)
            text = response.text

            # Cache the result
            if self.cache_dir:
                cache_file.write_text(text, encoding="utf-8")

            return text

        except Exception as e:
            print(f"Failed to download filing {filing.accession_number}: {e}")
            return ""

    def parse_8k_items(self, text: str) -> dict[str, str]:
        """
        Parse 8-K filing to extract items.

        Args:
            text: Raw 8-K filing text

        Returns:
            Dictionary mapping item numbers to content
        """
        items = {}

        # Common 8-K item patterns
        item_patterns = [
            (r"Item\s*1\.01[:\s]*Entry into a Material Definitive Agreement", "1.01"),
            (r"Item\s*1\.02[:\s]*Termination of a Material Definitive Agreement", "1.02"),
            (r"Item\s*2\.01[:\s]*Completion of Acquisition", "2.01"),
            (r"Item\s*2\.03[:\s]*Creation of a Direct Financial Obligation", "2.03"),
            (r"Item\s*5\.02[:\s]*Departure of Directors", "5.02"),
            (r"Item\s*7\.01[:\s]*Regulation FD Disclosure", "7.01"),
            (r"Item\s*8\.01[:\s]*Other Events", "8.01"),
            (r"Item\s*9\.01[:\s]*Financial Statements and Exhibits", "9.01"),
        ]

        # Find all item sections
        for pattern, item_num in item_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = match.start()

                # Find the end (next item or end of document)
                end = len(text)
                for next_pattern, _ in item_patterns:
                    next_match = re.search(next_pattern, text[start + 100:], re.IGNORECASE)
                    if next_match:
                        potential_end = start + 100 + next_match.start()
                        if potential_end < end:
                            end = potential_end

                items[item_num] = text[start:end].strip()

        return items

    def extract_commitment_letters(
        self,
        filings: list[Filing],
        download: bool = True,
    ) -> list[CommitmentLetterExtract]:
        """
        Extract commitment letter information from 8-K filings.

        Looks for Item 1.01 entries that describe bridge commitments,
        acquisition financing, or similar material agreements.

        Args:
            filings: List of Filing objects
            download: Whether to download filing content

        Returns:
            List of CommitmentLetterExtract objects
        """
        extracts = []

        # Keywords indicating a commitment letter
        commitment_keywords = [
            r"commitment\s+letter",
            r"bridge\s+(loan|facility|commitment|financing)",
            r"acquisition\s+financing",
            r"term\s+loan\s+facility",
            r"credit\s+agreement",
            r"financing\s+commitment",
        ]
        commitment_pattern = "|".join(commitment_keywords)

        for filing in filings:
            if download and not filing.raw_text:
                filing.raw_text = self.download_filing(filing)

            if not filing.raw_text:
                continue

            # Parse items
            filing.items = self.parse_8k_items(filing.raw_text)

            # Check Item 1.01 for commitment language
            item_101 = filing.items.get("1.01", "")
            item_203 = filing.items.get("2.03", "")

            relevant_text = item_101 + "\n" + item_203

            if re.search(commitment_pattern, relevant_text, re.IGNORECASE):
                extract = self._extract_commitment_details(filing, relevant_text)
                if extract:
                    extracts.append(extract)

        return extracts

    def _extract_commitment_details(
        self,
        filing: Filing,
        text: str,
    ) -> Optional[CommitmentLetterExtract]:
        """
        Extract structured details from commitment letter text.

        Uses regex patterns to identify key provisions.
        """
        extract = CommitmentLetterExtract(
            filing=filing,
            raw_text=text[:10000],  # Limit stored text
        )

        # Extract target/acquirer companies
        merger_pattern = r"(?:acquisition|merger|purchase)\s+of\s+([A-Z][A-Za-z\s,\.]+?)(?:by|with|and)\s+([A-Z][A-Za-z\s,\.]+)"
        merger_match = re.search(merger_pattern, text, re.IGNORECASE)
        if merger_match:
            extract.target_company = merger_match.group(1).strip()
            extract.acquirer_company = merger_match.group(2).strip()

        # Extract commitment amount
        amount_patterns = [
            r"\$\s*([0-9,]+(?:\.[0-9]+)?)\s*(billion|million)",
            r"([0-9,]+(?:\.[0-9]+)?)\s*(billion|million)\s*(?:dollar|USD)",
            r"aggregate\s+commitment[s]?\s+of\s+\$?\s*([0-9,]+(?:\.[0-9]+)?)\s*(billion|million)?",
        ]

        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = float(match.group(1).replace(",", ""))
                unit = match.group(2).lower() if match.lastindex >= 2 and match.group(2) else "million"
                if unit == "billion":
                    amount *= 1000  # Convert to millions
                extract.commitment_amount = amount
                break

        # Extract commitment banks
        bank_patterns = [
            r"(JPMorgan\s*Chase|J\.?P\.?\s*Morgan)",
            r"(Goldman\s*Sachs)",
            r"(Morgan\s*Stanley)",
            r"(Bank\s+of\s+America|BofA)",
            r"(Citigroup|Citibank|Citi)",
            r"(Wells\s*Fargo)",
            r"(Credit\s*Suisse)",
            r"(Deutsche\s*Bank)",
            r"(Barclays)",
            r"(UBS)",
        ]

        for pattern in bank_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                bank_match = re.search(pattern, text, re.IGNORECASE)
                if bank_match:
                    bank_name = bank_match.group(1)
                    # Normalize bank names to tickers
                    bank_ticker = self._normalize_bank_name(bank_name)
                    if bank_ticker and bank_ticker not in extract.commitment_banks:
                        extract.commitment_banks.append(bank_ticker)

        # Detect key provisions
        extract.has_sungard_clause = bool(re.search(
            r"SunGard|certain\s+funds|limited\s+conditionality",
            text, re.IGNORECASE
        ))

        extract.has_limited_conditionality = bool(re.search(
            r"limited\s+conditionality|no\s+financing\s+condition",
            text, re.IGNORECASE
        ))

        extract.has_market_flex = bool(re.search(
            r"market\s+flex|pricing\s+flex|flex\s+provisions?",
            text, re.IGNORECASE
        ))

        extract.has_flex_cap = bool(re.search(
            r"flex\s+(?:is\s+)?(?:limited|capped)|(?:cap|limit)\s+(?:of|on)\s+\d+\s*(?:bp|basis\s+points)",
            text, re.IGNORECASE
        ))

        extract.has_mae_market_out = bool(re.search(
            r"material\s+adverse\s+(?:effect|change).*(?:exclud|except).*market",
            text, re.IGNORECASE
        ))

        # Only return if we found meaningful content
        if extract.commitment_amount or extract.commitment_banks:
            return extract

        return None

    def _normalize_bank_name(self, name: str) -> Optional[str]:
        """Normalize bank name to ticker symbol."""
        name_lower = name.lower().replace(" ", "").replace(".", "")

        mapping = {
            "jpmorganchase": "JPM",
            "jpmorgan": "JPM",
            "goldmansachs": "GS",
            "morganstanley": "MS",
            "bankofamerica": "BAC",
            "bofa": "BAC",
            "citigroup": "C",
            "citibank": "C",
            "citi": "C",
            "wellsfargo": "WFC",
            "creditsuisse": "CS",
            "deutschebank": "DB",
            "barclays": "BCS",
            "ubs": "UBS",
        }

        for key, ticker in mapping.items():
            if key in name_lower:
                return ticker

        return None

    def get_commitment_letters_for_date(
        self,
        date: str,
        max_filings: int = 50,
    ) -> list[CommitmentLetterExtract]:
        """
        Get all commitment letters filed on a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            max_filings: Maximum 8-K filings to check

        Returns:
            List of extracted commitment letters
        """
        print(f"Fetching 8-K filings for {date}...")
        filings = self.get_8k_filings(date=date, max_results=max_filings)
        print(f"  Found {len(filings)} 8-K filings")

        print("Extracting commitment letters...")
        extracts = self.extract_commitment_letters(filings)
        print(f"  Found {len(extracts)} commitment letters")

        return extracts

    def build_deals_dataframe(
        self,
        extracts: list[CommitmentLetterExtract],
    ) -> pl.DataFrame:
        """
        Build a DataFrame of deals from extracted commitment letters.

        Args:
            extracts: List of CommitmentLetterExtract objects

        Returns:
            DataFrame suitable for FASAR calculation
        """
        if not extracts:
            return pl.DataFrame()

        records = []
        for i, extract in enumerate(extracts):
            # Create a record for each bank in the commitment
            banks = extract.commitment_banks or ["UNKNOWN"]
            amount_per_bank = (
                extract.commitment_amount / len(banks)
                if extract.commitment_amount
                else 0
            )

            for bank in banks:
                records.append({
                    "deal_id": f"{extract.filing.accession_number}_{bank}",
                    "bank_ticker": bank,
                    "announcement_date": extract.filing.filing_date,
                    "commitment_amount": amount_per_bank,
                    "target_company": extract.target_company or extract.filing.company_name,
                    "acquirer_company": extract.acquirer_company or "Unknown",
                    "has_sungard_clause": extract.has_sungard_clause,
                    "has_limited_conditionality": extract.has_limited_conditionality,
                    "has_market_flex": extract.has_market_flex,
                    "flex_is_capped": extract.has_flex_cap,
                    "has_mae_market_out": extract.has_mae_market_out,
                    "filing_url": extract.filing.document_url,
                    "cik": extract.filing.cik,
                    "raw_text": extract.raw_text[:5000],  # Limit text size
                })

        return pl.DataFrame(records)
