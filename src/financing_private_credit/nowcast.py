"""
Nowcasting module for private credit conditions.

Extends the Boyarchenko & Elias (2024) methodology with higher-frequency
proxy data to provide more timely estimates of credit conditions.

Key innovation: Use weekly/monthly bank credit data to nowcast
the quarterly Z.1 Financial Accounts data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl
from scipy import stats

from .data import FREDDataFetcher, PRIVATE_CREDIT_SERIES


@dataclass
class NowcastResult:
    """Results from nowcasting exercise."""

    estimate: float
    confidence_interval: tuple[float, float]
    last_observation_date: datetime
    nowcast_date: datetime
    proxy_used: str
    model_r_squared: float


# Weekly/Monthly proxy series for nowcasting
NOWCAST_PROXY_SERIES = {
    # Weekly bank credit data (H.8 release)
    "TOTLL": "Total Loans and Leases, All Commercial Banks (Weekly)",
    "BUSLOANS": "Commercial and Industrial Loans (Weekly)",
    "CONSUMER": "Consumer Loans (Weekly)",
    "REALLN": "Real Estate Loans (Weekly)",

    # Monthly data
    "DPCREDIT": "Depository Institutions Credit (Monthly)",

    # Financial conditions proxies
    "NFCI": "Chicago Fed National Financial Conditions Index (Weekly)",
    "STLFSI4": "St. Louis Fed Financial Stress Index (Weekly)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread (Daily)",
}


class CreditNowcaster:
    """
    Nowcast private credit conditions using high-frequency proxy data.

    Methodology:
    1. Estimate bridge equation relating quarterly Z.1 data to weekly/monthly proxies
    2. Use latest proxy data to nowcast current quarter credit conditions
    3. Decompose nowcast into bank vs nonbank components
    """

    def __init__(self, lookback_years: int = 10):
        """
        Initialize the nowcaster.

        Args:
            lookback_years: Years of historical data for bridge equation estimation
        """
        self.lookback_years = lookback_years
        self.fetcher = FREDDataFetcher()
        self._bridge_models: dict[str, dict] = {}

    def fetch_proxy_data(self) -> pl.DataFrame:
        """
        Fetch high-frequency proxy data for nowcasting.

        Returns:
            DataFrame with weekly/monthly proxy series
        """
        start_date = f"{datetime.now().year - self.lookback_years}-01-01"

        # Fetch weekly bank credit data
        proxy_ids = list(NOWCAST_PROXY_SERIES.keys())

        print(f"Fetching {len(proxy_ids)} proxy series...")
        return self.fetcher.fetch_multiple_series(
            proxy_ids,
            start_date=start_date,
        )

    def _aggregate_to_quarterly(self, df: pl.DataFrame, method: str = "mean") -> pl.DataFrame:
        """
        Aggregate high-frequency data to quarterly frequency.

        Args:
            df: DataFrame with date column and value columns
            method: Aggregation method ('mean', 'last', 'sum')

        Returns:
            Quarterly aggregated DataFrame
        """
        # Add quarter column
        df = df.with_columns(
            pl.col("date").dt.truncate("1q").alias("quarter")
        )

        # Aggregate by quarter
        value_cols = [c for c in df.columns if c not in ["date", "quarter"]]

        if method == "mean":
            agg_exprs = [pl.col(c).mean().alias(c) for c in value_cols]
        elif method == "last":
            agg_exprs = [pl.col(c).last().alias(c) for c in value_cols]
        else:  # sum
            agg_exprs = [pl.col(c).sum().alias(c) for c in value_cols]

        return df.group_by("quarter").agg(agg_exprs).sort("quarter").rename({"quarter": "date"})

    def estimate_bridge_equation(
        self,
        target_series: str,
        proxy_series: list[str],
        quarterly_data: pl.DataFrame,
    ) -> dict:
        """
        Estimate bridge equation relating quarterly target to proxy series.

        Bridge equation: Y_q = α + β * X_proxy_q + ε

        Args:
            target_series: Name of quarterly target series
            proxy_series: Names of proxy series to use
            quarterly_data: DataFrame with both target and proxy data

        Returns:
            Dictionary with estimated bridge equation parameters
        """
        df = quarterly_data.drop_nulls(subset=[target_series] + proxy_series)

        if df.height < 10:
            raise ValueError(f"Insufficient data for bridge equation: {df.height} observations")

        # Extract arrays
        y = df.select(target_series).to_numpy().flatten()

        # Build design matrix
        X = np.column_stack([
            np.ones(len(y)),  # intercept
            *[df.select(p).to_numpy().flatten() for p in proxy_series]
        ])

        # OLS estimation
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

            # Compute R-squared
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Compute standard errors
            n = len(y)
            k = X.shape[1]
            mse = ss_res / (n - k) if n > k else 0
            var_beta = mse * np.linalg.inv(X.T @ X) if np.linalg.det(X.T @ X) != 0 else np.zeros((k, k))
            se_beta = np.sqrt(np.diag(var_beta))

        except np.linalg.LinAlgError:
            # Fallback to simple correlation
            beta = np.array([np.mean(y)] + [0] * len(proxy_series))
            se_beta = np.array([np.std(y)] + [0] * len(proxy_series))
            r_squared = 0

        return {
            "intercept": beta[0],
            "coefficients": dict(zip(proxy_series, beta[1:])),
            "se_intercept": se_beta[0],
            "se_coefficients": dict(zip(proxy_series, se_beta[1:])),
            "r_squared": r_squared,
            "n_obs": len(y),
        }

    def nowcast_credit(
        self,
        target_var: str = "total_credit",
        proxy_vars: Optional[list[str]] = None,
    ) -> NowcastResult:
        """
        Nowcast current quarter credit using latest proxy data.

        Args:
            target_var: Variable to nowcast
            proxy_vars: Proxy variables to use (default: bank credit proxies)

        Returns:
            NowcastResult with point estimate and confidence interval
        """
        if proxy_vars is None:
            proxy_vars = ["TOTLL", "BUSLOANS"]  # Weekly bank credit

        # Fetch proxy data
        proxy_data = self.fetch_proxy_data()

        # For nowcasting, get current quarter average of proxy
        current_quarter = datetime.now().replace(
            month=((datetime.now().month - 1) // 3) * 3 + 1,
            day=1
        )

        current_proxy = proxy_data.filter(
            pl.col("date") >= current_quarter
        )

        if current_proxy.height == 0:
            raise ValueError("No current quarter proxy data available")

        # Aggregate current quarter proxy data
        proxy_values = {}
        for var in proxy_vars:
            if var in current_proxy.columns:
                proxy_values[var] = current_proxy.select(var).mean().item()

        # Get bridge equation (estimate if not cached)
        cache_key = f"{target_var}_{'_'.join(proxy_vars)}"
        if cache_key not in self._bridge_models:
            # Need quarterly data for bridge equation
            # This would need to be passed in or fetched separately
            # For now, return a placeholder result
            return NowcastResult(
                estimate=0.0,
                confidence_interval=(0.0, 0.0),
                last_observation_date=current_proxy["date"].max(),
                nowcast_date=datetime.now(),
                proxy_used=", ".join(proxy_vars),
                model_r_squared=0.0,
            )

        model = self._bridge_models[cache_key]

        # Compute nowcast
        nowcast = model["intercept"]
        for var, coef in model["coefficients"].items():
            if var in proxy_values:
                nowcast += coef * proxy_values[var]

        # Compute confidence interval (simplified)
        se = model["se_intercept"]  # Simplified; should include prediction interval
        ci = (nowcast - 1.96 * se, nowcast + 1.96 * se)

        return NowcastResult(
            estimate=nowcast,
            confidence_interval=ci,
            last_observation_date=current_proxy["date"].max(),
            nowcast_date=datetime.now(),
            proxy_used=", ".join(proxy_vars),
            model_r_squared=model["r_squared"],
        )

    def nowcast_bank_share(self) -> NowcastResult:
        """
        Nowcast the bank share of credit using weekly H.8 data.

        This is key for the Boyarchenko & Elias finding that lender
        composition determines crisis probability.

        Returns:
            NowcastResult for bank share of credit
        """
        proxy_data = self.fetch_proxy_data()

        # Use total bank credit as proxy for bank share
        # Actual implementation would need total credit estimate too
        bank_vars = ["TOTLL", "BUSLOANS", "CONSUMER", "REALLN"]
        available_vars = [v for v in bank_vars if v in proxy_data.columns]

        if not available_vars:
            raise ValueError("No bank credit proxy data available")

        # Get latest values
        latest = proxy_data.tail(1)
        bank_credit = sum(
            latest.select(v).item() for v in available_vars
            if latest.select(v).item() is not None
        )

        # This is a simplified version - full implementation would
        # also nowcast nonbank credit to compute share
        return NowcastResult(
            estimate=bank_credit,
            confidence_interval=(bank_credit * 0.95, bank_credit * 1.05),
            last_observation_date=latest["date"].item(),
            nowcast_date=datetime.now(),
            proxy_used=", ".join(available_vars),
            model_r_squared=0.0,  # Placeholder
        )

    def compute_credit_growth_nowcast(
        self,
        periods: int = 4,
    ) -> pl.DataFrame:
        """
        Nowcast credit growth using high-frequency proxies.

        Args:
            periods: Periods for growth calculation (4 = YoY for quarterly)

        Returns:
            DataFrame with nowcasted credit growth
        """
        proxy_data = self.fetch_proxy_data()

        # Aggregate to quarterly
        quarterly = self._aggregate_to_quarterly(proxy_data, method="last")

        # Compute growth rates
        credit_cols = ["TOTLL", "BUSLOANS", "CONSUMER", "REALLN"]
        available = [c for c in credit_cols if c in quarterly.columns]

        for col in available:
            quarterly = quarterly.with_columns(
                ((pl.col(col) / pl.col(col).shift(periods) - 1) * 100)
                .alias(f"{col}_growth_yoy")
            )

        # Aggregate bank credit growth
        growth_cols = [f"{c}_growth_yoy" for c in available if f"{c}_growth_yoy" in quarterly.columns]
        if growth_cols:
            quarterly = quarterly.with_columns(
                pl.mean_horizontal([pl.col(c) for c in growth_cols]).alias("bank_credit_growth_nowcast")
            )

        return quarterly


class FinancialConditionsMonitor:
    """
    Monitor financial conditions relevant to credit supply.

    Uses high-frequency financial conditions indices to assess
    the credit environment in real-time.
    """

    def __init__(self):
        self.fetcher = FREDDataFetcher()

    def fetch_conditions_data(self, lookback_days: int = 365) -> pl.DataFrame:
        """
        Fetch financial conditions indicators.

        Args:
            lookback_days: Days of history to fetch

        Returns:
            DataFrame with financial conditions data
        """
        from datetime import timedelta

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        conditions_series = ["NFCI", "STLFSI4", "BAMLH0A0HYM2"]

        return self.fetcher.fetch_multiple_series(
            conditions_series,
            start_date=start_date,
            end_date=end_date,
        )

    def assess_credit_environment(self) -> dict:
        """
        Assess current credit environment using financial conditions.

        Returns:
            Dictionary with credit environment assessment
        """
        data = self.fetch_conditions_data()

        if data.height == 0:
            return {"status": "unavailable", "message": "Could not fetch conditions data"}

        latest = data.tail(1)

        assessment = {
            "date": latest["date"].item() if "date" in latest.columns else None,
            "indicators": {},
        }

        # NFCI: negative = loose conditions, positive = tight
        if "NFCI" in latest.columns:
            nfci = latest["NFCI"].item()
            if nfci is not None:
                assessment["indicators"]["NFCI"] = {
                    "value": nfci,
                    "interpretation": "tight" if nfci > 0 else "loose",
                }

        # Credit spreads: higher = tighter conditions
        if "BAMLH0A0HYM2" in latest.columns:
            spread = latest["BAMLH0A0HYM2"].item()
            if spread is not None:
                assessment["indicators"]["HY_spread"] = {
                    "value": spread,
                    "interpretation": "tight" if spread > 400 else "normal" if spread > 300 else "loose",
                }

        # Overall assessment
        tight_count = sum(
            1 for ind in assessment["indicators"].values()
            if ind.get("interpretation") == "tight"
        )
        loose_count = sum(
            1 for ind in assessment["indicators"].values()
            if ind.get("interpretation") == "loose"
        )

        if tight_count > loose_count:
            assessment["overall"] = "tight"
        elif loose_count > tight_count:
            assessment["overall"] = "loose"
        else:
            assessment["overall"] = "neutral"

        return assessment


if __name__ == "__main__":
    # Test nowcasting
    nowcaster = CreditNowcaster()

    # Fetch proxy data
    proxy_df = nowcaster.fetch_proxy_data()
    print("Proxy data shape:", proxy_df.shape)
    print(proxy_df.tail())

    # Test credit growth nowcast
    growth_df = nowcaster.compute_credit_growth_nowcast()
    print("\nCredit growth nowcast:")
    print(growth_df.tail())

    # Test financial conditions
    monitor = FinancialConditionsMonitor()
    conditions = monitor.assess_credit_environment()
    print("\nFinancial conditions assessment:")
    print(conditions)
