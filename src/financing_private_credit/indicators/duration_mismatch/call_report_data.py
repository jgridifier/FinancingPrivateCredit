"""
FFIEC Call Report Data Fetcher

Fetches Schedule RC-B (Securities) data from FFIEC Call Reports (FFIEC 031/041)
and Schedule HC-B from FR Y-9C for bank holding companies.

Key data: Memorandum Item 2 - Maturity/Repricing buckets:
- 2.a: ≤1 year
- 2.b: 1-5 years
- 2.c: 5-10 years
- 2.d: >10 years

This enables precise duration estimation using weighted average of buckets
rather than flat averages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import polars as pl
import requests


@dataclass
class MaturityBucketData:
    """Securities data by maturity bucket from Call Report Schedule RC-B."""

    rssd_id: str
    report_date: datetime

    # Total securities by maturity bucket (Memorandum Item 2)
    bucket_1yr_or_less: float = 0.0       # 2.a
    bucket_1yr_to_5yr: float = 0.0        # 2.b
    bucket_5yr_to_10yr: float = 0.0       # 2.c  (some forms split 5-15, >15)
    bucket_over_10yr: float = 0.0         # 2.d

    # MBS-specific buckets (already adjusted for prepayment/expected life)
    mbs_bucket_1yr_or_less: float = 0.0
    mbs_bucket_1yr_to_5yr: float = 0.0
    mbs_bucket_5yr_to_10yr: float = 0.0
    mbs_bucket_over_10yr: float = 0.0

    # AFS vs HTM totals
    afs_amortized_cost: float = 0.0
    afs_fair_value: float = 0.0
    htm_amortized_cost: float = 0.0
    htm_fair_value: float = 0.0

    # Unrealized gains/losses (important for AOCI impact)
    afs_unrealized_gains: float = 0.0
    afs_unrealized_losses: float = 0.0
    htm_unrealized_gains: float = 0.0
    htm_unrealized_losses: float = 0.0


# FFIEC MDRM codes for Schedule RC-B items
# These are the variable codes used in the FFIEC bulk data
SCHEDULE_RCB_MDRM_CODES = {
    # Total securities by maturity (Memorandum Item 2)
    "bucket_1yr_or_less": "RCFDA549",      # M2.a - 1 year or less
    "bucket_1yr_to_5yr": "RCFDA550",       # M2.b - Over 1 through 5 years
    "bucket_5yr_to_10yr": "RCFDA551",      # M2.c - Over 5 through 10 years (or 5-15 in some forms)
    "bucket_over_10yr": "RCFDA552",        # M2.d - Over 10 years (or >15)

    # Alternative codes for more granular buckets
    "bucket_5yr_to_15yr": "RCFDA551",      # Over 5 through 15 years
    "bucket_over_15yr": "RCFDA552",        # Over 15 years

    # AFS securities
    "afs_amortized_cost": "RCFD1773",      # AFS - Amortized cost
    "afs_fair_value": "RCFD1772",          # AFS - Fair value

    # HTM securities
    "htm_amortized_cost": "RCFD1754",      # HTM - Amortized cost
    "htm_fair_value": "RCFD1771",          # HTM - Fair value

    # By security type (for MBS identification)
    "mbs_afs_amortized": "RCFDG379",       # MBS - AFS amortized cost
    "mbs_afs_fair": "RCFDG380",            # MBS - AFS fair value
    "mbs_htm_amortized": "RCFDG381",       # MBS - HTM amortized cost
    "mbs_htm_fair": "RCFDG382",            # MBS - HTM fair value

    # Treasuries and agencies
    "treasury_afs": "RCFD0211",            # US Treasury - AFS
    "treasury_htm": "RCFD0213",            # US Treasury - HTM
    "agency_afs": "RCFD1289",              # US Agency - AFS
    "agency_htm": "RCFD1294",              # US Agency - HTM

    # Unrealized gains/losses
    "afs_unrealized_gains": "RCFDA221",
    "afs_unrealized_losses": "RCFDA222",
}

# FR Y-9C codes for holding company Schedule HC-B
SCHEDULE_HCB_MDRM_CODES = {
    # Similar structure to RC-B but with BHCK/BHCT prefixes
    "bucket_1yr_or_less": "BHCKA549",
    "bucket_1yr_to_5yr": "BHCKA550",
    "bucket_5yr_to_10yr": "BHCKA551",
    "bucket_over_10yr": "BHCKA552",

    "afs_amortized_cost": "BHCK1773",
    "afs_fair_value": "BHCK1772",
    "htm_amortized_cost": "BHCK1754",
    "htm_fair_value": "BHCK1771",

    "mbs_afs_amortized": "BHCKG379",
    "mbs_htm_amortized": "BHCKG381",
}


class FFIECCallReportFetcher:
    """
    Fetch Call Report data from FFIEC Central Data Repository.

    The FFIEC CDR provides bulk downloads of Call Report data.
    For production use, you may need to download the bulk files
    or use the FFIEC's web services.

    This class also supports fetching from the Federal Reserve's
    FR Y-9C data for bank holding companies.
    """

    # FFIEC CDR bulk data URL
    FFIEC_BULK_URL = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"

    # Federal Reserve FR Y-9C URL
    FRB_Y9C_URL = "https://www.federalreserve.gov/apps/mdrm/"

    def __init__(self):
        self._cache: dict[str, pl.DataFrame] = {}

    def fetch_schedule_rcb(
        self,
        rssd_id: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch Schedule RC-B data for a bank.

        Args:
            rssd_id: Federal Reserve RSSD ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (optional)

        Returns:
            DataFrame with securities maturity bucket data
        """
        # In production, this would query the FFIEC CDR or bulk files
        # For now, we'll attempt to construct from SEC EDGAR data
        # which contains some of this information

        print(f"Fetching Schedule RC-B for RSSD {rssd_id}...")

        # Try to get from cached bulk data first
        cache_key = f"rcb_{rssd_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Attempt to fetch from FFIEC (may require authentication/download)
        try:
            df = self._fetch_from_ffiec_cdr(rssd_id, "031", start_date, end_date)
            if df.height > 0:
                self._cache[cache_key] = df
                return df
        except Exception as e:
            print(f"  FFIEC CDR fetch failed: {e}")

        # Fallback: return empty DataFrame (will use estimation)
        return pl.DataFrame()

    def fetch_schedule_hcb(
        self,
        rssd_id: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch Schedule HC-B data for a bank holding company (FR Y-9C).

        Args:
            rssd_id: Federal Reserve RSSD ID
            start_date: Start date
            end_date: End date (optional)

        Returns:
            DataFrame with securities maturity bucket data
        """
        print(f"Fetching Schedule HC-B (Y-9C) for RSSD {rssd_id}...")

        cache_key = f"hcb_{rssd_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            df = self._fetch_from_frb_y9c(rssd_id, start_date, end_date)
            if df.height > 0:
                self._cache[cache_key] = df
                return df
        except Exception as e:
            print(f"  FR Y-9C fetch failed: {e}")

        return pl.DataFrame()

    def _fetch_from_ffiec_cdr(
        self,
        rssd_id: str,
        form_type: str,  # "031" or "041"
        start_date: str,
        end_date: Optional[str],
    ) -> pl.DataFrame:
        """
        Fetch from FFIEC Central Data Repository.

        Note: The FFIEC CDR requires bulk data download or specific API access.
        This is a placeholder for the actual implementation.
        """
        # FFIEC CDR typically requires:
        # 1. Downloading bulk ZIP files by quarter
        # 2. Parsing the fixed-width or CSV format
        # 3. Filtering by RSSD ID

        # For now, return empty - in production, implement bulk file parsing
        return pl.DataFrame()

    def _fetch_from_frb_y9c(
        self,
        rssd_id: str,
        start_date: str,
        end_date: Optional[str],
    ) -> pl.DataFrame:
        """
        Fetch FR Y-9C data from Federal Reserve.

        The Fed provides this data in various formats including
        bulk downloads and API access.
        """
        # Similar to FFIEC, requires bulk download parsing
        return pl.DataFrame()

    def get_available_quarters(self, rssd_id: str) -> list[str]:
        """Get list of available reporting quarters for a bank."""
        return []


class CallReportParser:
    """
    Parse Call Report data files.

    Handles both FFIEC 031/041 (banks) and FR Y-9C (holding companies).
    """

    def parse_bulk_file(
        self,
        filepath: str,
        rssd_ids: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """
        Parse a bulk Call Report data file.

        Args:
            filepath: Path to the bulk data file (ZIP or CSV)
            rssd_ids: Optional list of RSSD IDs to filter

        Returns:
            DataFrame with parsed data
        """
        import zipfile
        from pathlib import Path

        path = Path(filepath)

        if path.suffix == ".zip":
            # Extract and parse
            with zipfile.ZipFile(path, 'r') as zf:
                # Find the main data file
                data_files = [f for f in zf.namelist() if f.endswith('.txt') or f.endswith('.csv')]
                if not data_files:
                    return pl.DataFrame()

                with zf.open(data_files[0]) as f:
                    # Parse as CSV/TSV
                    df = pl.read_csv(f, separator='\t', infer_schema_length=10000)

        elif path.suffix in ['.csv', '.txt']:
            df = pl.read_csv(path, separator='\t', infer_schema_length=10000)
        else:
            return pl.DataFrame()

        # Filter by RSSD if provided
        if rssd_ids and "RSSD" in df.columns:
            df = df.filter(pl.col("RSSD").is_in(rssd_ids))

        return df

    def extract_maturity_buckets(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract maturity bucket data from parsed Call Report.

        Args:
            df: Raw Call Report data

        Returns:
            DataFrame with maturity bucket columns
        """
        # Map MDRM codes to friendly names
        code_mapping = {v: k for k, v in SCHEDULE_RCB_MDRM_CODES.items()}

        # Pivot or select relevant columns
        result_cols = ["RSSD", "REPDTE"]  # Report date

        for code, name in code_mapping.items():
            if code in df.columns:
                result_cols.append(code)

        if len(result_cols) <= 2:
            return pl.DataFrame()

        result = df.select([c for c in result_cols if c in df.columns])

        # Rename columns
        rename_map = {code: name for code, name in code_mapping.items() if code in result.columns}
        result = result.rename(rename_map)

        return result


def estimate_duration_from_buckets(
    bucket_data: MaturityBucketData,
    method: str = "standard",
) -> dict[str, float]:
    """
    Calculate weighted average duration from maturity bucket data.

    This implements the refined methodology using Schedule RC-B/HC-B data
    rather than flat average assumptions.

    Args:
        bucket_data: MaturityBucketData from Call Report
        method: "standard" (midpoint) or "conservative" (lower bound)

    Returns:
        Dictionary with duration metrics
    """
    # Proxy durations by bucket
    if method == "conservative":
        # Lower bound estimates (more conservative)
        BUCKET_DURATIONS = {
            "1yr_or_less": 0.4,    # Weighted toward shorter end
            "1yr_to_5yr": 2.0,     # Conservative midpoint
            "5yr_to_10yr": 5.5,    # Skewed lower for prepayments
            "over_10yr": 10.0,     # Conservative for very long
        }
    else:
        # Standard midpoint estimates
        BUCKET_DURATIONS = {
            "1yr_or_less": 0.5,    # Midpoint of 0-1 year
            "1yr_to_5yr": 2.5,     # Midpoint, slightly conservative
            "5yr_to_10yr": 6.5,    # Midpoint, adjusted for convexity
            "over_10yr": 12.0,     # Assumes long-dated securities
        }

    # Calculate total and weighted duration
    total = (
        bucket_data.bucket_1yr_or_less +
        bucket_data.bucket_1yr_to_5yr +
        bucket_data.bucket_5yr_to_10yr +
        bucket_data.bucket_over_10yr
    )

    if total <= 0:
        return {"weighted_duration": 0.0, "total_securities": 0.0}

    weighted_duration = (
        bucket_data.bucket_1yr_or_less * BUCKET_DURATIONS["1yr_or_less"] +
        bucket_data.bucket_1yr_to_5yr * BUCKET_DURATIONS["1yr_to_5yr"] +
        bucket_data.bucket_5yr_to_10yr * BUCKET_DURATIONS["5yr_to_10yr"] +
        bucket_data.bucket_over_10yr * BUCKET_DURATIONS["over_10yr"]
    ) / total

    # Calculate DV01 (dollar value of 1bp)
    # DV01 = Duration * Value * 0.0001
    dv01 = weighted_duration * total * 0.0001

    # Convexity adjustment (rough estimate)
    # Higher duration = higher convexity impact
    convexity_factor = 1.0 + (weighted_duration / 100)

    # Bucket composition for risk assessment
    pct_short = bucket_data.bucket_1yr_or_less / total * 100 if total > 0 else 0
    pct_long = bucket_data.bucket_over_10yr / total * 100 if total > 0 else 0

    # "Barbell" indicator (high short + high long = barbell strategy)
    is_barbell = pct_short > 20 and pct_long > 20

    return {
        "weighted_duration": weighted_duration,
        "total_securities": total,
        "dv01": dv01,
        "convexity_factor": convexity_factor,
        "pct_short_term": pct_short,
        "pct_long_term": pct_long,
        "is_barbell_strategy": is_barbell,
        "bucket_1yr_or_less": bucket_data.bucket_1yr_or_less,
        "bucket_1yr_to_5yr": bucket_data.bucket_1yr_to_5yr,
        "bucket_5yr_to_10yr": bucket_data.bucket_5yr_to_10yr,
        "bucket_over_10yr": bucket_data.bucket_over_10yr,
    }


def adjust_mbs_duration(
    mbs_bucket_data: dict[str, float],
    prepayment_assumption: str = "cpr_6",
) -> float:
    """
    Adjust MBS duration using expected average life buckets.

    The Call Report already adjusts MBS for prepayments/expected life,
    so we trust the bucket classification. This function applies
    additional adjustments if needed.

    Args:
        mbs_bucket_data: MBS-specific bucket data
        prepayment_assumption: "cpr_6" (normal), "cpr_12" (fast), "cpr_3" (slow)

    Returns:
        Adjusted MBS duration
    """
    # MBS buckets from Call Report already reflect expected average life
    # Apply modest adjustment based on prepayment speed assumption

    base_durations = {
        "1yr_or_less": 0.5,
        "1yr_to_5yr": 2.5,
        "5yr_to_10yr": 6.0,  # Slightly lower than non-MBS due to prepayments
        "over_10yr": 10.0,
    }

    # Prepayment speed adjustments
    speed_multipliers = {
        "cpr_3": 1.15,   # Slow prepayments = longer duration
        "cpr_6": 1.0,    # Normal (baseline)
        "cpr_12": 0.85,  # Fast prepayments = shorter duration
        "cpr_20": 0.70,  # Very fast prepayments
    }

    multiplier = speed_multipliers.get(prepayment_assumption, 1.0)

    total = sum(mbs_bucket_data.values())
    if total <= 0:
        return 0.0

    weighted_duration = sum(
        mbs_bucket_data.get(bucket, 0) * base_durations[bucket] * multiplier
        for bucket in base_durations
    ) / total

    return weighted_duration
