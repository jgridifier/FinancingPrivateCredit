"""
Duration Mismatch Indicator

Measures bank-specific duration exposure and its relationship to earnings volatility.

Duration exposure is extracted from:
1. SEC EDGAR: Securities portfolio disclosures (AFS/HTM classification, maturities)
2. Derived metrics: Estimated modified duration based on portfolio composition

The key signal is:
    predicted_earnings_impact = duration_exposure × Δ(bond_yield)

Banks with higher predicted impact should experience:
- Higher earnings volatility
- More stock return volatility
- Larger NIM swings
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

from ..base import (
    BaseIndicator,
    IndicatorMetadata,
    IndicatorResult,
    register_indicator,
)


@dataclass
class DurationMismatchSpec:
    """Specification for duration mismatch model."""

    name: str
    description: str

    # Duration estimation method
    duration_method: str = "estimated"  # "estimated", "disclosed", "hybrid"

    # Securities to include
    include_afs: bool = True  # Available-for-sale
    include_htm: bool = True  # Held-to-maturity
    include_trading: bool = False  # Trading securities (mark-to-market)

    # Yield curve series
    yield_series: list[str] = field(default_factory=lambda: [
        "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"
    ])

    # Volatility windows
    earnings_vol_window: int = 8  # Quarters for rolling earnings volatility
    stock_vol_window: int = 60  # Trading days for stock volatility

    # Prediction settings
    forecast_horizon: int = 4  # Quarters ahead to predict
    min_observations: int = 16  # Minimum quarters for estimation

    @classmethod
    def from_json(cls, path: str | Path) -> "DurationMismatchSpec":
        """Load specification from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        """Save specification to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class BankDurationProfile:
    """Duration exposure profile for a single bank."""

    ticker: str
    name: str
    as_of_date: datetime

    # Securities portfolio composition
    total_securities: float  # $ millions
    afs_securities: float
    htm_securities: float
    trading_securities: float

    # Duration metrics
    estimated_duration: float  # Modified duration in years
    duration_gap: float  # Asset duration - liability duration

    # Sensitivity metrics
    dv01: float  # Dollar value of 01 (1bp move)
    predicted_nim_sensitivity: float  # NIM change per 100bp rate move

    # Historical volatility
    earnings_volatility: Optional[float] = None
    stock_volatility: Optional[float] = None
    nim_volatility: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            "total_securities": self.total_securities,
            "afs_securities": self.afs_securities,
            "htm_securities": self.htm_securities,
            "estimated_duration": self.estimated_duration,
            "duration_gap": self.duration_gap,
            "dv01": self.dv01,
            "predicted_nim_sensitivity": self.predicted_nim_sensitivity,
            "earnings_volatility": self.earnings_volatility,
            "stock_volatility": self.stock_volatility,
        }


class SecuritiesPortfolioExtractor:
    """
    Extract securities portfolio data from SEC filings.

    XBRL concepts for securities:
    - AvailableForSaleSecurities
    - HeldToMaturitySecurities
    - TradingSecurities
    - SecuritiesHeldAssets
    """

    # XBRL concepts for securities portfolio
    SECURITIES_CONCEPTS = {
        "afs_securities": [
            ("us-gaap", "AvailableForSaleSecuritiesDebtSecurities"),
            ("us-gaap", "AvailableForSaleSecurities"),
            ("us-gaap", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
            ("us-gaap", "MarketableSecuritiesCurrent"),
        ],
        "htm_securities": [
            ("us-gaap", "HeldToMaturitySecurities"),
            ("us-gaap", "HeldToMaturitySecuritiesDebtSecurities"),
            ("us-gaap", "HeldToMaturitySecuritiesAmortizedCostBeforeOtherThanTemporaryImpairment"),
        ],
        "trading_securities": [
            ("us-gaap", "TradingSecurities"),
            ("us-gaap", "TradingSecuritiesDebt"),
        ],
        "total_securities": [
            ("us-gaap", "MarketableSecurities"),
            ("us-gaap", "AvailableForSaleAndHeldToMaturitySecurities"),
            ("us-gaap", "InvestmentSecurities"),
        ],
        # Duration-related disclosures (rare in XBRL, usually in notes)
        "weighted_avg_life": [
            ("us-gaap", "DebtSecuritiesAvailableForSaleWeightedAverageLife"),
        ],
    }

    def __init__(self):
        from ...bank_data import SECEdgarFetcher
        self.sec_fetcher = SECEdgarFetcher()

    def extract_securities(self, cik: str, ticker: str) -> pl.DataFrame:
        """
        Extract securities portfolio data for a bank.

        Args:
            cik: SEC Central Index Key
            ticker: Bank ticker symbol

        Returns:
            DataFrame with quarterly securities data
        """
        facts = self.sec_fetcher.get_company_facts(cik)

        if not facts:
            return pl.DataFrame()

        # Extract each securities category
        dfs = {}

        for metric_name, concepts in self.SECURITIES_CONCEPTS.items():
            best_df = None
            best_date = None

            for taxonomy, concept in concepts:
                df = self.sec_fetcher.extract_metric(facts, taxonomy, concept)
                if df.height > 0:
                    latest_date = df.select(pl.col("date").max()).item()
                    if best_date is None or latest_date > best_date:
                        best_df = df.rename({"value": metric_name})
                        best_date = latest_date

            if best_df is not None:
                dfs[metric_name] = best_df

        if not dfs:
            return pl.DataFrame()

        # Merge all securities data
        result = None
        for name, df in dfs.items():
            if result is None:
                result = df
            else:
                result = result.join(df, on="date", how="outer_coalesce")

        if result is not None:
            result = result.with_columns(pl.lit(ticker).alias("ticker"))

        return result if result is not None else pl.DataFrame()


class MarketDataFetcher:
    """
    Fetch market data from Yahoo Finance.

    Data includes:
    - Stock prices and returns
    - Earnings (for volatility calculation)
    - Market cap
    """

    def __init__(self):
        self._cache: dict[str, pl.DataFrame] = {}

    def fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch stock price data from Yahoo Finance.

        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (optional)

        Returns:
            DataFrame with date, close, returns, volume
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if hist.empty:
                return pl.DataFrame()

            # Convert to polars
            df = pl.DataFrame({
                "date": hist.index.date,
                "close": hist["Close"].values,
                "volume": hist["Volume"].values,
            })

            # Calculate returns
            df = df.with_columns([
                (pl.col("close") / pl.col("close").shift(1) - 1).alias("daily_return"),
                (pl.col("close") / pl.col("close").shift(5) - 1).alias("weekly_return"),
            ])

            # Rolling volatility
            df = df.with_columns([
                pl.col("daily_return").rolling_std(window_size=20).alias("volatility_20d"),
                pl.col("daily_return").rolling_std(window_size=60).alias("volatility_60d"),
            ])

            df = df.with_columns(pl.lit(ticker).alias("ticker"))
            self._cache[ticker] = df

            return df

        except ImportError:
            print("Warning: yfinance not installed. Install with: pip install yfinance")
            return pl.DataFrame()
        except Exception as e:
            print(f"Warning: Could not fetch data for {ticker}: {e}")
            return pl.DataFrame()

    def fetch_earnings_data(
        self,
        ticker: str,
    ) -> pl.DataFrame:
        """
        Fetch earnings history from Yahoo Finance.

        Args:
            ticker: Stock ticker

        Returns:
            DataFrame with earnings dates and surprises
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)

            # Get earnings history
            try:
                earnings = stock.earnings_history
                if earnings is None or earnings.empty:
                    return pl.DataFrame()

                df = pl.DataFrame({
                    "date": earnings.index.date,
                    "eps_actual": earnings["epsActual"].values if "epsActual" in earnings.columns else None,
                    "eps_estimate": earnings["epsEstimate"].values if "epsEstimate" in earnings.columns else None,
                })

                # Calculate surprise
                if "eps_actual" in df.columns and "eps_estimate" in df.columns:
                    df = df.with_columns(
                        (pl.col("eps_actual") - pl.col("eps_estimate")).alias("eps_surprise")
                    )

                df = df.with_columns(pl.lit(ticker).alias("ticker"))
                return df

            except Exception:
                # Fallback to quarterly financials
                quarterly = stock.quarterly_financials
                if quarterly is None or quarterly.empty:
                    return pl.DataFrame()

                # Try to get Net Income
                if "Net Income" in quarterly.index:
                    net_income = quarterly.loc["Net Income"]
                    df = pl.DataFrame({
                        "date": [d.date() for d in net_income.index],
                        "net_income": net_income.values,
                    })
                    df = df.with_columns(pl.lit(ticker).alias("ticker"))
                    return df

                return pl.DataFrame()

        except ImportError:
            print("Warning: yfinance not installed")
            return pl.DataFrame()
        except Exception as e:
            print(f"Warning: Could not fetch earnings for {ticker}: {e}")
            return pl.DataFrame()

    def compute_earnings_volatility(
        self,
        earnings_df: pl.DataFrame,
        window: int = 8,
    ) -> pl.DataFrame:
        """
        Compute rolling earnings volatility.

        Args:
            earnings_df: DataFrame with earnings data
            window: Number of quarters for rolling window

        Returns:
            DataFrame with earnings volatility
        """
        if earnings_df.height == 0:
            return earnings_df

        value_col = "eps_actual" if "eps_actual" in earnings_df.columns else "net_income"

        if value_col not in earnings_df.columns:
            return earnings_df

        result = earnings_df.sort("date")

        # Compute YoY change and volatility
        result = result.with_columns([
            (pl.col(value_col) / pl.col(value_col).shift(4) - 1).alias("earnings_growth"),
        ])

        result = result.with_columns([
            pl.col("earnings_growth").rolling_std(window_size=window).alias("earnings_volatility"),
            pl.col("earnings_growth").rolling_mean(window_size=window).alias("earnings_growth_avg"),
        ])

        return result


class DurationEstimator:
    """
    Estimate duration exposure from securities portfolio.

    Since banks rarely disclose exact duration, we estimate based on:
    1. Securities composition (AFS vs HTM)
    2. Maturity bucketing (if available)
    3. Industry benchmarks for asset class durations
    """

    # Benchmark durations by asset class (in years)
    BENCHMARK_DURATIONS = {
        "treasury_short": 1.5,   # T-bills, short-term
        "treasury_medium": 4.0,  # 2-5 year
        "treasury_long": 12.0,   # 10+ year
        "mbs": 5.5,              # Mortgage-backed (prepayment adjusted)
        "corporate_ig": 6.0,     # Investment grade corporate
        "corporate_hy": 4.0,     # High yield (shorter)
        "muni": 7.0,             # Municipal bonds
        "afs_average": 4.5,      # Average for AFS portfolio
        "htm_average": 6.0,      # HTM typically longer duration
    }

    def estimate_portfolio_duration(
        self,
        securities_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Estimate duration for each bank's securities portfolio.

        Uses weighted average based on AFS/HTM composition.

        Args:
            securities_df: DataFrame with securities data

        Returns:
            DataFrame with estimated duration metrics
        """
        if securities_df.height == 0:
            return securities_df

        result = securities_df.clone()

        # Calculate total if not present
        if "total_securities" not in result.columns:
            cols_to_sum = ["afs_securities", "htm_securities", "trading_securities"]
            available = [c for c in cols_to_sum if c in result.columns]

            if available:
                result = result.with_columns(
                    sum(pl.col(c).fill_null(0) for c in available).alias("total_securities")
                )
            else:
                result = result.with_columns(pl.lit(0.0).alias("total_securities"))

        # Estimate weighted duration
        afs_dur = self.BENCHMARK_DURATIONS["afs_average"]
        htm_dur = self.BENCHMARK_DURATIONS["htm_average"]

        # Weight by composition
        afs_col = pl.col("afs_securities").fill_null(0) if "afs_securities" in result.columns else pl.lit(0)
        htm_col = pl.col("htm_securities").fill_null(0) if "htm_securities" in result.columns else pl.lit(0)
        total_col = pl.col("total_securities").fill_null(1)  # Avoid division by zero

        result = result.with_columns([
            # Portfolio weights
            (afs_col / total_col).alias("afs_weight"),
            (htm_col / total_col).alias("htm_weight"),
        ])

        result = result.with_columns([
            # Estimated duration (weighted average)
            (
                pl.col("afs_weight") * afs_dur +
                pl.col("htm_weight") * htm_dur
            ).alias("estimated_duration"),
        ])

        # DV01 = Duration * Portfolio Value * 0.0001
        result = result.with_columns(
            (pl.col("estimated_duration") * pl.col("total_securities") * 0.0001)
            .alias("dv01")
        )

        return result

    def calculate_predicted_impact(
        self,
        duration_df: pl.DataFrame,
        yield_changes: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Calculate predicted earnings impact from yield changes.

        Formula: predicted_impact = -duration × Δ(yield) × portfolio_value

        Args:
            duration_df: DataFrame with duration estimates
            yield_changes: DataFrame with yield changes

        Returns:
            DataFrame with predicted impacts
        """
        # Join duration with yield changes
        result = duration_df.join(
            yield_changes.select(["date", "yield_change"]),
            on="date",
            how="left"
        )

        # Calculate predicted impact
        result = result.with_columns(
            (
                -pl.col("estimated_duration") *
                pl.col("yield_change") / 100 *  # Convert to decimal
                pl.col("total_securities")
            ).alias("predicted_earnings_impact")
        )

        # Normalize by portfolio size for comparability
        result = result.with_columns(
            (pl.col("predicted_earnings_impact") / pl.col("total_securities"))
            .alias("predicted_impact_pct")
        )

        return result


@register_indicator("duration_mismatch")
class DurationMismatchIndicator(BaseIndicator):
    """
    Duration Mismatch Indicator.

    Measures bank-specific duration exposure and tests whether it
    predicts earnings volatility and stock returns.

    Key outputs:
    - Duration exposure by bank
    - Predicted earnings impact from yield changes
    - Correlation with actual volatility
    - Vulnerability ranking
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self._spec: Optional[DurationMismatchSpec] = None
        self._securities_extractor = SecuritiesPortfolioExtractor()
        self._market_fetcher = MarketDataFetcher()
        self._duration_estimator = DurationEstimator()
        self._duration_data: Optional[pl.DataFrame] = None
        self._market_data: Optional[dict[str, pl.DataFrame]] = None

    def get_metadata(self) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="Duration Mismatch Predictive Signal",
            short_name="DurationMismatch",
            description=(
                "Measures bank-specific duration exposure from securities portfolios "
                "and tests whether it predicts earnings volatility and stock returns. "
                "Banks with higher duration sensitivity are more vulnerable to rate moves."
            ),
            version="1.0.0",
            paper_reference="Extension of NY Fed Staff Report 1111 Figure 8b",
            data_sources=["SEC EDGAR (10-K/10-Q)", "Yahoo Finance", "FRED"],
            update_frequency="quarterly",
            lookback_periods=20,
        )

    def fetch_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch securities, market, and yield data."""
        from ...bank_data import TARGET_BANKS, BankDataCollector
        from ...cache import CachedFREDFetcher

        # 1. Fetch securities portfolio data from SEC
        securities_dfs = []
        for ticker, bank_info in TARGET_BANKS.items():
            print(f"Fetching securities data for {ticker}...")
            sec_df = self._securities_extractor.extract_securities(
                bank_info.cik, ticker
            )
            if sec_df.height > 0:
                securities_dfs.append(sec_df)

        securities_panel = (
            pl.concat(securities_dfs, how="diagonal").sort(["ticker", "date"])
            if securities_dfs else pl.DataFrame()
        )

        # 2. Fetch bank fundamentals (for NIM, earnings context)
        collector = BankDataCollector(start_date=start_date)
        bank_panel = collector.fetch_all_banks()
        bank_panel = collector.compute_derived_metrics(bank_panel)

        # 3. Fetch market data (stock prices, volatility)
        market_data = {}
        for ticker in TARGET_BANKS.keys():
            print(f"Fetching market data for {ticker}...")
            stock_df = self._market_fetcher.fetch_stock_data(ticker, start_date, end_date)
            earnings_df = self._market_fetcher.fetch_earnings_data(ticker)

            if stock_df.height > 0:
                market_data[f"{ticker}_stock"] = stock_df
            if earnings_df.height > 0:
                earnings_vol = self._market_fetcher.compute_earnings_volatility(earnings_df)
                market_data[f"{ticker}_earnings"] = earnings_vol

        # 4. Fetch yield curve data from FRED
        fetcher = CachedFREDFetcher(max_age_hours=6)
        yield_series = ["DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
        yields = fetcher.fetch_multiple_series(yield_series, start_date=start_date)

        # Calculate yield changes
        if yields is not None and not yields.is_empty():
            # Use 10-year as primary rate
            if "DGS10" in yields.columns:
                yields = yields.with_columns([
                    (pl.col("DGS10") - pl.col("DGS10").shift(1)).alias("yield_change"),
                    (pl.col("DGS10") - pl.col("DGS10").shift(4)).alias("yield_change_qtr"),
                    (pl.col("DGS10") - pl.col("DGS10").shift(252)).alias("yield_change_yoy"),
                ])

        return {
            "securities_panel": securities_panel,
            "bank_panel": bank_panel,
            "market_data": market_data,
            "yields": yields,
            "data_quality": collector.get_data_quality_summary(),
        }

    def calculate(
        self,
        data: dict[str, pl.DataFrame],
        spec: Optional[DurationMismatchSpec] = None,
        **kwargs,
    ) -> IndicatorResult:
        """
        Calculate duration mismatch metrics for all banks.

        Returns rankings by:
        - Duration exposure
        - Predicted earnings impact
        - Vulnerability score
        """
        securities_panel = data.get("securities_panel", pl.DataFrame())
        bank_panel = data.get("bank_panel", pl.DataFrame())
        yields = data.get("yields", pl.DataFrame())
        market_data = data.get("market_data", {})

        if spec is None:
            spec = DurationMismatchSpec(name="default", description="Default duration mismatch spec")
        self._spec = spec

        if securities_panel.height == 0:
            # Fall back to using bank panel for rough estimates
            if bank_panel.height == 0:
                return IndicatorResult(
                    indicator_name="duration_mismatch",
                    calculation_date=datetime.now(),
                    data=pl.DataFrame(),
                    metadata={"error": "No securities or bank data available"},
                )

            # Estimate from total assets
            securities_panel = self._estimate_securities_from_assets(bank_panel)

        # 1. Estimate duration for each bank
        duration_panel = self._duration_estimator.estimate_portfolio_duration(securities_panel)

        # 2. Calculate predicted impact from yield changes
        if yields.height > 0 and "yield_change_qtr" in yields.columns:
            # Aggregate yields to quarterly
            yields_qtr = yields.with_columns(
                pl.col("date").dt.truncate("1q").alias("quarter")
            ).group_by("quarter").agg([
                pl.col("DGS10").last().alias("DGS10"),
                pl.col("yield_change_qtr").last().alias("yield_change"),
            ]).rename({"quarter": "date"})

            duration_panel = self._duration_estimator.calculate_predicted_impact(
                duration_panel, yields_qtr
            )

        # 3. Merge with actual volatility data from market
        duration_with_vol = self._merge_volatility_data(duration_panel, market_data)

        # 4. Calculate vulnerability scores
        vulnerability_df = self._calculate_vulnerability_scores(duration_with_vol)

        self._duration_data = vulnerability_df

        return IndicatorResult(
            indicator_name="duration_mismatch",
            calculation_date=datetime.now(),
            data=vulnerability_df,
            metadata={
                "n_banks": vulnerability_df["ticker"].n_unique() if vulnerability_df.height > 0 else 0,
                "avg_duration": float(vulnerability_df["estimated_duration"].mean()) if "estimated_duration" in vulnerability_df.columns else None,
                "spec_name": spec.name,
            },
        )

    def _estimate_securities_from_assets(self, bank_panel: pl.DataFrame) -> pl.DataFrame:
        """
        Estimate securities portfolio from total assets when SEC data unavailable.

        Typical large banks hold 15-25% of assets in securities.
        """
        if "total_assets" not in bank_panel.columns:
            return pl.DataFrame()

        # Estimate: 20% of assets are securities, 60% AFS, 40% HTM
        result = bank_panel.select(["date", "ticker", "total_assets"]).with_columns([
            (pl.col("total_assets") * 0.20).alias("total_securities"),
            (pl.col("total_assets") * 0.20 * 0.60).alias("afs_securities"),
            (pl.col("total_assets") * 0.20 * 0.40).alias("htm_securities"),
        ])

        return result

    def _merge_volatility_data(
        self,
        duration_df: pl.DataFrame,
        market_data: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Merge duration data with market volatility metrics."""
        result = duration_df.clone()

        for ticker in result["ticker"].unique().to_list():
            stock_key = f"{ticker}_stock"
            earnings_key = f"{ticker}_earnings"

            # Get stock volatility
            if stock_key in market_data:
                stock_df = market_data[stock_key]
                if stock_df.height > 0 and "volatility_60d" in stock_df.columns:
                    # Get latest volatility
                    latest_vol = stock_df.filter(
                        pl.col("volatility_60d").is_not_null()
                    ).tail(1)

                    if latest_vol.height > 0:
                        vol_value = latest_vol["volatility_60d"][0]
                        result = result.with_columns(
                            pl.when(pl.col("ticker") == ticker)
                            .then(pl.lit(float(vol_value)))
                            .otherwise(pl.col("stock_volatility") if "stock_volatility" in result.columns else pl.lit(None))
                            .alias("stock_volatility")
                        )

            # Get earnings volatility
            if earnings_key in market_data:
                earnings_df = market_data[earnings_key]
                if earnings_df.height > 0 and "earnings_volatility" in earnings_df.columns:
                    latest_earn_vol = earnings_df.filter(
                        pl.col("earnings_volatility").is_not_null()
                    ).tail(1)

                    if latest_earn_vol.height > 0:
                        earn_vol = latest_earn_vol["earnings_volatility"][0]
                        result = result.with_columns(
                            pl.when(pl.col("ticker") == ticker)
                            .then(pl.lit(float(earn_vol)))
                            .otherwise(pl.col("earnings_volatility") if "earnings_volatility" in result.columns else pl.lit(None))
                            .alias("earnings_volatility")
                        )

        return result

    def _calculate_vulnerability_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate vulnerability score combining duration and volatility.

        Vulnerability = Duration × Historical Volatility × Rate Sensitivity
        """
        if df.height == 0:
            return df

        result = df.clone()

        # Normalize duration (z-score across banks at each date)
        result = result.with_columns([
            ((pl.col("estimated_duration") - pl.col("estimated_duration").mean().over("date")) /
             pl.col("estimated_duration").std().over("date")).alias("duration_zscore"),
        ])

        # Vulnerability score
        if "stock_volatility" in result.columns:
            vol_col = pl.col("stock_volatility").fill_null(pl.col("stock_volatility").mean())
        else:
            vol_col = pl.lit(0.02)  # Default 2% volatility

        result = result.with_columns(
            (pl.col("duration_zscore").abs() * vol_col * 100).alias("vulnerability_score")
        )

        # Rank banks by vulnerability
        result = result.with_columns(
            pl.col("vulnerability_score").rank(descending=True).over("date").alias("vulnerability_rank")
        )

        return result

    def nowcast(
        self,
        data: dict[str, pl.DataFrame],
        **kwargs,
    ) -> IndicatorResult:
        """Nowcast current duration exposure using latest available data."""
        from .nowcast import DurationMismatchNowcaster

        if self._duration_data is None:
            self.calculate(data)

        nowcaster = DurationMismatchNowcaster(
            duration_data=self._duration_data,
            spec=self._spec,
        )

        yields = data.get("yields", pl.DataFrame())
        nowcast_df = nowcaster.nowcast(yields)

        return IndicatorResult(
            indicator_name="duration_mismatch_nowcast",
            calculation_date=datetime.now(),
            data=nowcast_df,
            metadata={"methodology": "Latest duration with current yield changes"},
        )

    def get_vulnerability_ranking(self) -> pl.DataFrame:
        """Get current vulnerability rankings."""
        if self._duration_data is None:
            return pl.DataFrame()

        # Get latest date for each bank
        latest = self._duration_data.group_by("ticker").agg(
            pl.all().sort_by("date").last()
        )

        return latest.select([
            "ticker", "estimated_duration", "dv01",
            "vulnerability_score", "vulnerability_rank",
            "stock_volatility", "earnings_volatility"
        ]).sort("vulnerability_rank")

    def get_dashboard_components(self) -> dict[str, Any]:
        """Return dashboard configuration."""
        return {
            "tabs": [
                {"name": "Duration Exposure", "icon": "chart_with_upwards_trend"},
                {"name": "Vulnerability Rankings", "icon": "warning"},
                {"name": "Impact Simulation", "icon": "lightning"},
            ],
            "primary_metric": "vulnerability_score",
            "ranking_metric": "estimated_duration",
        }
