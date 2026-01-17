"""
Shadow Nowcast Extension for PB-LLI V2

Implements the three-layer publication system:
1. Official layer (immutable until Z.1 release): through Q2'25
2. Shadow backfill layer: estimates for completed but unreleased quarters (Q3'25)
3. Progressive nowcast layer: real-time updates for current quarter (Q4'25)

Follows the methodology outlined in shadow_nowcast_extension.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Optional

import numpy as np
import polars as pl


@dataclass
class ShadowNowcastConfig:
    """Configuration for shadow nowcast extension."""

    # Weekly standardization window (260 weeks ≈ 5 years)
    weekly_std_window: int = 260

    # Robust z-score scale factor (1.4826 for MAD normalization)
    mad_scale: float = 1.4826

    # EWMA half-life for weekly factor smoothing (8 weeks)
    ewma_halflife: int = 8

    # Ridge regression penalty
    ridge_lambda: float = 5.0

    # Training winsorization percentiles
    winsorize_lower: float = 0.025
    winsorize_upper: float = 0.975

    # Uncertainty inflation factors
    kappa_x: float = 0.5  # Weekly factor uncertainty
    kappa_p: float = 1.0  # Disclosure pulse uncertainty

    # Disclosure pulse standardization window (8 quarters)
    disclosure_std_window: int = 8

    # Bank weights for disclosure pulse (V1: equal weights)
    bank_weights: dict[str, float] = field(default_factory=lambda: {
        "GS": 1.0,
        "MS": 1.0,
        "JPM": 1.0,
    })

    # Model specification (determines which features to include)
    # Options: "ar1_dealer" (default, best OOS), "ar1_dealer_cftc", "ar1_ar2_dealer", "full"
    model_spec: str = "ar1_dealer"

    # Whether to include AR(1) term (lagged intensity growth)
    include_ar1: bool = True

    # Whether to include AR(2) term
    include_ar2: bool = False

    # Whether to include CFTC weekly factor
    include_cftc: bool = True


@dataclass
class ShadowEstimate:
    """Container for a shadow/nowcast estimate with uncertainty."""

    quarter: str  # e.g., "2025Q3"
    estimate_type: str  # "official", "shadow", "nowcast"
    pb_intensity_g: float  # Growth rate estimate
    pb_intensity_level: float  # Level estimate
    uncertainty_std: float  # Standard deviation of estimate
    confidence_lower: float  # 95% CI lower bound
    confidence_upper: float  # 95% CI upper bound
    completeness_weekly: float  # Fraction of quarter weeks observed (omega_x)
    completeness_disclosure: float  # Fraction of banks reported (c_t)
    as_of_date: date  # Date of estimate
    components: dict[str, float] = field(default_factory=dict)  # Attribution


class ShadowNowcaster:
    """
    Shadow Nowcast Extension for PB-LLI.

    Extends the official quarterly indicator into:
    - Shadow backfill for completed but unreported quarters
    - Progressive nowcast for the current quarter
    """

    def __init__(self, config: Optional[ShadowNowcastConfig] = None):
        """Initialize the shadow nowcaster."""
        self.config = config or ShadowNowcastConfig()
        self._model_fitted = False
        self._model_coefficients: dict[str, float] = {}
        self._residual_std: float = 0.0

    def build_weekly_factor(
        self,
        weekly_cot: Optional[pl.DataFrame],
        weekly_pd: Optional[pl.DataFrame],
    ) -> pl.DataFrame:
        """
        Build weekly leverage appetite factor from CFTC and NY Fed data.

        x_w = (1/J) * sum(z_cot_j) + (1/K) * sum(z_pd_k)

        Then apply EWMA smoothing with half-life of 8 weeks.

        Returns:
            DataFrame with columns: date, x_raw, x_smooth
        """
        config = self.config

        # Handle None inputs
        if weekly_cot is None:
            weekly_cot = pl.DataFrame()
        if weekly_pd is None:
            weekly_pd = pl.DataFrame()

        # Aggregate CFTC COT data to weekly factor
        cot_factor = self._build_cot_factor(weekly_cot)

        # Aggregate NY Fed PD data to weekly factor
        pd_factor = self._build_pd_factor(weekly_pd)

        # Combine factors
        if cot_factor.height > 0 and pd_factor.height > 0:
            # Join on date (allow 3-day tolerance for report dates)
            weekly_factor = self._merge_weekly_factors(cot_factor, pd_factor)
        elif cot_factor.height > 0:
            weekly_factor = cot_factor.rename({"cot_factor": "x_raw"})
        elif pd_factor.height > 0:
            weekly_factor = pd_factor.rename({"pd_factor": "x_raw"})
        else:
            return pl.DataFrame({"date": [], "x_raw": [], "x_smooth": []})

        # Apply EWMA smoothing
        rho = 2 ** (-1 / config.ewma_halflife)  # ≈ 0.917 for halflife=8

        weekly_factor = weekly_factor.sort("date")

        # Calculate EWMA
        x_raw = weekly_factor["x_raw"].to_numpy()
        x_smooth = np.zeros_like(x_raw)
        x_smooth[0] = x_raw[0]

        for i in range(1, len(x_raw)):
            if np.isnan(x_raw[i]):
                x_smooth[i] = x_smooth[i-1]
            else:
                x_smooth[i] = (1 - rho) * x_raw[i] + rho * x_smooth[i-1]

        weekly_factor = weekly_factor.with_columns(
            pl.Series("x_smooth", x_smooth)
        )

        return weekly_factor

    def _build_cot_factor(self, weekly_cot: pl.DataFrame) -> pl.DataFrame:
        """Build CFTC COT weekly factor."""
        if weekly_cot.height == 0:
            return pl.DataFrame()

        config = self.config

        # Group by report date and compute mean z-score
        if "net_pct_oi_z_5y" in weekly_cot.columns:
            z_col = "net_pct_oi_z_5y"
        elif "leveraged_net_pct_oi" in weekly_cot.columns:
            # Need to compute z-score ourselves using robust method
            weekly_cot = self._add_robust_zscore(
                weekly_cot,
                "leveraged_net_pct_oi",
                "contract",
                config.weekly_std_window
            )
            z_col = "leveraged_net_pct_oi_z"
        else:
            return pl.DataFrame()

        cot_factor = weekly_cot.group_by("report_date").agg(
            pl.col(z_col).mean().alias("cot_factor")
        ).rename({"report_date": "date"})

        return cot_factor

    def _build_pd_factor(self, weekly_pd: pl.DataFrame) -> pl.DataFrame:
        """Build NY Fed PD weekly factor."""
        if weekly_pd.height == 0:
            return pl.DataFrame()

        config = self.config

        # Group by week and compute mean z-score
        if "value_z_3y" in weekly_pd.columns:
            z_col = "value_z_3y"
        elif "value" in weekly_pd.columns:
            # Need to compute z-score ourselves
            weekly_pd = self._add_robust_zscore(
                weekly_pd,
                "value",
                "series",
                156  # 3 years for PD data
            )
            z_col = "value_z"
        else:
            return pl.DataFrame()

        pd_factor = weekly_pd.group_by("week_ending").agg(
            pl.col(z_col).mean().alias("pd_factor")
        ).rename({"week_ending": "date"})

        return pd_factor

    def _add_robust_zscore(
        self,
        df: pl.DataFrame,
        value_col: str,
        group_col: str,
        window: int,
    ) -> pl.DataFrame:
        """Add robust z-score using median/MAD."""
        config = self.config

        # Sort by date within each group before rolling calculations
        date_col = "report_date" if "report_date" in df.columns else "date"
        df = df.sort([group_col, date_col])

        # Use a minimum window of 52 weeks (1 year) to ensure enough observations
        min_window = min(window, 52)

        # Calculate rolling median and std per group
        df = df.with_columns([
            pl.col(value_col)
            .rolling_median(window_size=min_window, min_periods=10)
            .over(group_col)
            .alias(f"{value_col}_median"),
            pl.col(value_col)
            .rolling_std(window_size=min_window, min_periods=10)
            .over(group_col)
            .alias(f"{value_col}_std"),
        ])

        # Robust z-score with fallback for edge cases
        # Replace zero std with a small value to avoid division by zero
        df = df.with_columns(
            pl.when(pl.col(f"{value_col}_std") > 0.0001)
            .then(
                (pl.col(value_col) - pl.col(f"{value_col}_median")) /
                (config.mad_scale * pl.col(f"{value_col}_std") * 0.6745)
            )
            .otherwise(pl.lit(0.0))
            .alias(f"{value_col}_z")
        )

        # Fill any remaining NaN values with 0
        df = df.with_columns(
            pl.col(f"{value_col}_z").fill_null(0.0)
        )

        return df

    def _merge_weekly_factors(
        self,
        cot_factor: pl.DataFrame,
        pd_factor: pl.DataFrame,
    ) -> pl.DataFrame:
        """Merge COT and PD factors with date tolerance."""
        # Convert dates to ensure same type
        cot_factor = cot_factor.with_columns(pl.col("date").cast(pl.Date))
        pd_factor = pd_factor.with_columns(pl.col("date").cast(pl.Date))

        # Join with tolerance using asof join (closest match within 3 days)
        merged = cot_factor.sort("date").join_asof(
            pd_factor.sort("date"),
            on="date",
            tolerance="3d",
            strategy="nearest",
        )

        # Average the two factors
        merged = merged.with_columns(
            ((pl.col("cot_factor").fill_null(0) +
              pl.col("pd_factor").fill_null(0)) / 2)
            .alias("x_raw")
        )

        return merged.select(["date", "x_raw"])

    def aggregate_weekly_to_quarter(
        self,
        weekly_factor: pl.DataFrame,
        quarter_year: int,
        quarter_num: int,
        as_of_date: Optional[date] = None,
    ) -> tuple[float, float]:
        """
        Aggregate weekly factor to quarterly mean.

        For shadow backfill: use full quarter
        For nowcast: use partial quarter through as_of_date

        Returns:
            (x_bar, omega_x) where omega_x is fraction of quarter observed
        """
        # Define quarter boundaries
        quarter_start = date(quarter_year, (quarter_num - 1) * 3 + 1, 1)
        if quarter_num == 4:
            quarter_end = date(quarter_year + 1, 1, 1) - timedelta(days=1)
        else:
            quarter_end = date(quarter_year, quarter_num * 3 + 1, 1) - timedelta(days=1)

        # If as_of_date provided, use it as cutoff
        if as_of_date is not None:
            effective_end = min(as_of_date, quarter_end)
        else:
            effective_end = quarter_end

        # Filter to quarter weeks
        quarter_data = weekly_factor.filter(
            (pl.col("date") >= quarter_start) &
            (pl.col("date") <= effective_end)
        )

        # Full quarter weeks for completeness calculation
        full_quarter_weeks = 13  # Standard quarter
        observed_weeks = quarter_data.height
        omega_x = observed_weeks / full_quarter_weeks if full_quarter_weeks > 0 else 0

        # Calculate mean of smoothed factor
        if quarter_data.height > 0 and "x_smooth" in quarter_data.columns:
            x_bar = quarter_data["x_smooth"].mean()
        else:
            x_bar = 0.0

        return float(x_bar) if x_bar is not None else 0.0, omega_x

    def fit_shadow_model(
        self,
        quarterly_anchor: pl.DataFrame,
        weekly_factor: pl.DataFrame,
        dealer_supply: Optional[pl.DataFrame] = None,
    ) -> dict[str, Any]:
        """
        Fit ridge regression model for shadow estimation.

        Primary model specification (best OOS performance):
            y_t = α + β₁·y_{t-1} + β₂·D_t + ε_t

        Where:
        - y_t = Δln(I_t) = pb_intensity_g (log growth)
        - y_{t-1} = lagged intensity growth (AR(1) term)
        - D_t = dealer_supply_g (dealer anchor)

        The AR(1) term captures mean-reversion behavior (negative correlation ~-0.30).
        Model achieves in-sample R² ≈ 0.20 and OOS R² ≈ 0.05.

        Optional features (configurable):
        - x̄_t = quarterly mean of weekly CFTC leverage appetite factor
        - y_{t-2} = AR(2) term

        Returns:
            Model fit statistics including R², coefficients, and diagnostics
        """
        config = self.config

        if quarterly_anchor.height < 10:
            print("Warning: Insufficient data for shadow model fitting")
            return {"success": False, "reason": "insufficient_data"}

        # Check if we have weekly data
        has_weekly_data = weekly_factor.height > 0

        # Prepare target variable: y_t = Δln(I_t)
        anchor_filtered = quarterly_anchor.filter(
            pl.col("pb_intensity_g").is_not_null()
        )
        y = anchor_filtered["pb_intensity_g"].to_numpy()
        n_obs = len(y)

        # Get dealer supply if available
        dealer_g = None
        if "dealer_supply_g" in anchor_filtered.columns:
            dealer_g = anchor_filtered["dealer_supply_g"].to_numpy()

        # Build AR(1) term: lagged intensity growth
        y_lag1 = np.roll(y, 1)
        y_lag1[0] = np.nan

        # Build AR(2) term if requested
        y_lag2 = None
        if config.include_ar2:
            y_lag2 = np.roll(y, 2)
            y_lag2[:2] = np.nan

        # Build CFTC factor if available and requested
        x_quarterly = None
        if has_weekly_data and config.include_cftc:
            quarters = anchor_filtered.select(["year", "quarter"]).to_dicts()
            x_quarterly = np.zeros(n_obs)
            for i, q in enumerate(quarters):
                x_bar, _ = self.aggregate_weekly_to_quarter(
                    weekly_factor,
                    q["year"],
                    q["quarter"],
                )
                x_quarterly[i] = x_bar

        # Winsorize target (2.5/97.5 percentiles per spec)
        lower = np.percentile(y, config.winsorize_lower * 100)
        upper = np.percentile(y, config.winsorize_upper * 100)
        y_winsorized = np.clip(y, lower, upper)

        # Build feature matrix based on model specification
        # Primary model: AR(1) + Dealer (best OOS performance)
        feature_list = []
        feature_names = []

        # AR(1) is always included (key for mean-reversion)
        if config.include_ar1:
            feature_list.append(y_lag1)
            feature_names.append("ar1")

        # AR(2) optional
        if config.include_ar2 and y_lag2 is not None:
            feature_list.append(y_lag2)
            feature_names.append("ar2")

        # Dealer supply
        if dealer_g is not None:
            feature_list.append(np.nan_to_num(dealer_g, nan=0.0))
            feature_names.append("dealer")

        # CFTC factor (optional - low OOS value but included for transparency)
        if x_quarterly is not None and config.include_cftc:
            feature_list.append(np.nan_to_num(x_quarterly, nan=0.0))
            feature_names.append("cftc")

        if len(feature_list) == 0:
            print("Warning: No features available for model")
            return {"success": False, "reason": "no_features"}

        X = np.column_stack(feature_list)

        # Remove rows with NaN (primarily from AR lags)
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_winsorized)
        X_valid = X[valid_mask]
        y_valid = y_winsorized[valid_mask]
        n_valid = len(y_valid)

        if n_valid < 10:
            print(f"Warning: Only {n_valid} valid observations after removing NaN")
            return {"success": False, "reason": "insufficient_valid_data"}

        # Standardize features (mean 0, sd 1 per spec)
        X_mean = np.mean(X_valid, axis=0)
        X_std = np.std(X_valid, axis=0)
        X_std[X_std == 0] = 1  # Avoid division by zero
        X_standardized = (X_valid - X_mean) / X_std

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(n_valid), X_standardized])

        # Ridge regression with λ = 5 (per spec)
        lambda_reg = config.ridge_lambda
        I_mat = np.eye(X_with_intercept.shape[1])
        I_mat[0, 0] = 0  # Don't regularize intercept

        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y_valid

        try:
            beta = np.linalg.solve(XtX + lambda_reg * I_mat, Xty)
        except np.linalg.LinAlgError:
            print("Warning: Ridge regression failed, using OLS")
            beta = np.linalg.lstsq(X_with_intercept, y_valid, rcond=None)[0]

        # Calculate residuals and standard error
        y_pred = X_with_intercept @ beta
        residuals = y_valid - y_pred
        residual_std = np.std(residuals)

        # Calculate R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Historical statistics (for uncertainty quantification)
        y_mean = float(np.mean(y))
        y_std = float(np.std(y))

        # AR(1) coefficient from raw correlation
        ar1_corr = float(np.corrcoef(y[:-1], y[1:])[0, 1]) if len(y) > 2 else 0.0

        # Dealer correlation
        dealer_y_corr = 0.0
        if dealer_g is not None:
            valid_d = ~np.isnan(dealer_g)
            if np.sum(valid_d) > 5:
                dealer_y_corr = float(np.corrcoef(y[valid_d], dealer_g[valid_d])[0, 1])

        # CFTC correlation
        cftc_y_corr = 0.0
        if x_quarterly is not None:
            valid_c = ~np.isnan(x_quarterly)
            if np.sum(valid_c) > 5:
                cftc_y_corr = float(np.corrcoef(y[valid_c], x_quarterly[valid_c])[0, 1])

        # Store model coefficients with feature names
        self._model_coefficients = {
            "alpha": float(beta[0]),
            "feature_names": feature_names,
            "betas": {name: float(beta[i+1]) for i, name in enumerate(feature_names)},
            "X_mean": X_mean.tolist(),
            "X_std": X_std.tolist(),
            "has_weekly_data": has_weekly_data,
            "has_ar1": "ar1" in feature_names,
            "has_ar2": "ar2" in feature_names,
            "has_cftc": "cftc" in feature_names,
            # Historical statistics
            "y_mean": y_mean,
            "y_std": y_std,
            "ar1_corr": ar1_corr,
            "dealer_y_corr": dealer_y_corr,
            "cftc_y_corr": cftc_y_corr,
        }
        self._residual_std = float(residual_std)
        self._model_fitted = True

        return {
            "success": True,
            "n_observations": n_valid,
            "r_squared": float(r_squared),
            "residual_std": float(residual_std),
            "y_mean": y_mean,
            "y_std": y_std,
            "ar1_corr": ar1_corr,
            "dealer_y_corr": dealer_y_corr,
            "cftc_y_corr": cftc_y_corr,
            "has_weekly_data": has_weekly_data,
            "model_spec": config.model_spec,
            "feature_names": feature_names,
            "coefficients": self._model_coefficients,
        }

    def estimate_shadow_quarter(
        self,
        quarter_year: int,
        quarter_num: int,
        weekly_factor: pl.DataFrame,
        last_official_intensity: float,
        dealer_supply_g: Optional[float] = None,
        disclosure_pulse: Optional[float] = None,
        completeness_disclosure: float = 0.0,
        as_of_date: Optional[date] = None,
        last_pb_intensity_g: Optional[float] = None,
        quarters_from_official: int = 1,
    ) -> ShadowEstimate:
        """
        Generate shadow/nowcast estimate for a quarter.

        Uses the fitted model with features:
        - AR(1): lagged intensity growth (primary driver via mean-reversion)
        - Dealer supply growth
        - CFTC weekly factor (optional)

        Args:
            quarter_year: Year of quarter
            quarter_num: Quarter number (1-4)
            weekly_factor: Weekly factor DataFrame
            last_official_intensity: Last official PB intensity level
            dealer_supply_g: Dealer supply growth (if available)
            disclosure_pulse: Bank disclosure pulse (if available)
            completeness_disclosure: Fraction of banks reported (c_t)
            as_of_date: As-of date for partial quarter (nowcast)
            last_pb_intensity_g: Last known pb_intensity_g for AR(1) model
            quarters_from_official: Number of quarters from last official data

        Returns:
            ShadowEstimate with point estimate and uncertainty
        """
        config = self.config

        if not self._model_fitted:
            raise ValueError("Model not fitted. Call fit_shadow_model first.")

        # Get model structure
        alpha = self._model_coefficients["alpha"]
        feature_names = self._model_coefficients.get("feature_names", [])
        betas = self._model_coefficients.get("betas", {})
        X_mean = np.array(self._model_coefficients["X_mean"])
        X_std = np.array(self._model_coefficients["X_std"])
        y_mean = self._model_coefficients.get("y_mean", 0.01)
        y_std = self._model_coefficients.get("y_std", 0.07)
        ar1_corr = self._model_coefficients.get("ar1_corr", -0.3)

        # Aggregate weekly factor for CFTC feature
        x_bar, omega_x = self.aggregate_weekly_to_quarter(
            weekly_factor, quarter_year, quarter_num, as_of_date
        )
        omega_x = min(omega_x, 1.0)

        # Build feature vector matching training order
        y_hat = alpha
        components = {"alpha": alpha}

        for i, name in enumerate(feature_names):
            beta = betas.get(name, 0.0)

            if name == "ar1":
                # AR(1) term: lagged intensity growth
                if last_pb_intensity_g is not None:
                    x_standardized = (last_pb_intensity_g - X_mean[i]) / X_std[i] if X_std[i] > 0 else 0
                    contribution = beta * x_standardized
                    y_hat += contribution
                    components["ar1"] = contribution

            elif name == "ar2":
                # AR(2) term: would need 2-lag data (skip for now)
                components["ar2"] = 0.0

            elif name == "dealer":
                # Dealer supply growth
                if dealer_supply_g is not None:
                    x_standardized = (dealer_supply_g - X_mean[i]) / X_std[i] if X_std[i] > 0 else 0
                    contribution = beta * x_standardized
                    y_hat += contribution
                    components["dealer"] = contribution

            elif name == "cftc":
                # CFTC weekly factor
                if omega_x > 0:
                    x_standardized = (x_bar - X_mean[i]) / X_std[i] if X_std[i] > 0 else 0
                    contribution = beta * x_standardized
                    y_hat += contribution
                    components["cftc"] = contribution

        # Add disclosure pulse contribution if available
        if disclosure_pulse is not None:
            pulse_contrib = 0.1 * disclosure_pulse
            y_hat += pulse_contrib
            components["disclosure"] = pulse_contrib

        # For multi-quarter forecasts, blend toward mean
        if quarters_from_official > 1:
            # Decay toward historical mean as we forecast further out
            decay = 0.7 ** (quarters_from_official - 1)
            y_hat = decay * y_hat + (1 - decay) * y_mean
            components["mean_reversion_adj"] = (1 - decay) * (y_mean - y_hat / decay if decay > 0 else 0)

        # Calculate uncertainty
        base_var = self._residual_std ** 2

        # Inflate uncertainty for distance from official and incomplete data
        omega_x_safe = max(omega_x, 0.1)
        inflation = 1.0
        inflation += 0.3 * (quarters_from_official - 1)  # Distance penalty
        if omega_x_safe < 1.0:
            inflation += config.kappa_x * (1 - omega_x_safe) / omega_x_safe
        inflation += config.kappa_p * (1 - completeness_disclosure)
        inflation += 0.2 * (quarters_from_official - 1)  # Additional uncertainty per quarter out

        if omega_x < 1.0:
            inflation += config.kappa_x * (1 - omega_x_safe) / omega_x_safe
        inflation += config.kappa_p * (1 - completeness_disclosure)

        uncertainty_var = base_var * inflation
        uncertainty_std = np.sqrt(uncertainty_var)

        # Ensure reasonable uncertainty bounds (at least y_std * 0.5)
        uncertainty_std = max(uncertainty_std, y_std * 0.5)

        # Convert growth to level
        pb_intensity_level = last_official_intensity * np.exp(y_hat)

        # Determine estimate type
        today = as_of_date or date.today()

        if omega_x >= 0.9 and completeness_disclosure >= 0.8:
            estimate_type = "shadow"  # Complete quarter with good data
        elif quarters_from_official == 1 and omega_x >= 0.9:
            estimate_type = "shadow"  # First quarter out with full weekly data
        else:
            estimate_type = "nowcast"  # Partial or further-out quarter

        # 95% confidence intervals
        z_95 = 1.96

        return ShadowEstimate(
            quarter=f"{quarter_year}Q{quarter_num}",
            estimate_type=estimate_type,
            pb_intensity_g=float(y_hat),
            pb_intensity_level=float(pb_intensity_level),
            uncertainty_std=float(uncertainty_std),
            confidence_lower=float(y_hat - z_95 * uncertainty_std),
            confidence_upper=float(y_hat + z_95 * uncertainty_std),
            completeness_weekly=float(omega_x),
            completeness_disclosure=float(completeness_disclosure),
            as_of_date=today,
            components=components,
        )

    def chain_intensity_levels(
        self,
        estimates: list[ShadowEstimate],
        last_official_level: float,
    ) -> list[ShadowEstimate]:
        """
        Chain shadow estimates forward using level reconstruction.

        I_t = I_{t-1} * exp(y_hat_t)
        """
        current_level = last_official_level

        for est in estimates:
            current_level = current_level * np.exp(est.pb_intensity_g)
            est.pb_intensity_level = float(current_level)

        return estimates


class BankDisclosureExtractor:
    """
    Extract prime-relevant metrics from bank earnings disclosures.

    V1 targets:
    - GS: Equities financing net revenues
    - MS: Equity net revenues
    - JPM: Equity Markets revenue
    """

    def __init__(self, config: Optional[ShadowNowcastConfig] = None):
        """Initialize the disclosure extractor."""
        self.config = config or ShadowNowcastConfig()
        self._historical_metrics: dict[str, list[dict]] = {
            "GS": [],
            "MS": [],
            "JPM": [],
        }

    def extract_gs_equities_financing(
        self,
        quarter_year: int,
        quarter_num: int,
    ) -> Optional[dict]:
        """
        Extract Goldman Sachs Equities Financing net revenues.

        Source: SEC Exhibit 99.2 (earnings presentation)

        Returns:
            Dict with metric value and metadata, or None if not available
        """
        # TODO: Implement actual EDGAR extraction
        # For V1, return placeholder indicating extraction needed
        return None

    def extract_ms_equity_revenues(
        self,
        quarter_year: int,
        quarter_num: int,
    ) -> Optional[dict]:
        """
        Extract Morgan Stanley Equity net revenues.

        Source: Earnings release PDF
        """
        # TODO: Implement actual extraction
        return None

    def extract_jpm_equity_markets(
        self,
        quarter_year: int,
        quarter_num: int,
    ) -> Optional[dict]:
        """
        Extract JPMorgan Equity Markets revenue.

        Source: Earnings release PDF
        """
        # TODO: Implement actual extraction
        return None

    def calculate_disclosure_pulse(
        self,
        quarter_year: int,
        quarter_num: int,
        available_disclosures: dict[str, float],
    ) -> tuple[float, float]:
        """
        Calculate coverage-weighted disclosure pulse.

        p_t = sum(w_i * s_i,t) / sum(w_i)

        where s_i,t is the standardized QoQ growth for bank i.

        Args:
            quarter_year: Year
            quarter_num: Quarter number
            available_disclosures: Dict of bank -> metric value

        Returns:
            (disclosure_pulse, completeness)
        """
        config = self.config

        if not available_disclosures:
            return 0.0, 0.0

        weighted_sum = 0.0
        total_weight = 0.0
        all_weight = sum(config.bank_weights.values())

        for bank, value in available_disclosures.items():
            if bank not in config.bank_weights:
                continue

            weight = config.bank_weights[bank]

            # Calculate standardized growth
            # Need historical values for standardization
            if bank in self._historical_metrics and len(self._historical_metrics[bank]) >= 2:
                history = self._historical_metrics[bank]
                prev_value = history[-1].get("value", value)

                # QoQ log growth
                if prev_value > 0 and value > 0:
                    growth = np.log(value / prev_value)
                else:
                    growth = 0.0

                # Standardize using trailing 8 quarters
                if len(history) >= config.disclosure_std_window:
                    recent = [h.get("growth", 0) for h in history[-config.disclosure_std_window:]]
                    mu = np.mean(recent)
                    sigma = np.std(recent) + 1e-6
                    s_it = (growth - mu) / sigma
                else:
                    s_it = growth / 0.1  # Assume 10% baseline std
            else:
                s_it = 0.0  # First observation, no growth available

            weighted_sum += weight * s_it
            total_weight += weight

            # Store for future standardization
            self._historical_metrics[bank].append({
                "quarter": f"{quarter_year}Q{quarter_num}",
                "value": value,
                "growth": growth if 'growth' in dir() else 0.0,
            })

        pulse = weighted_sum / total_weight if total_weight > 0 else 0.0
        completeness = total_weight / all_weight if all_weight > 0 else 0.0

        return pulse, completeness


def get_quarter_from_date(d: date) -> tuple[int, int]:
    """Get (year, quarter) from a date."""
    return d.year, (d.month - 1) // 3 + 1


def get_quarter_end_date(year: int, quarter: int) -> date:
    """Get the last day of a quarter."""
    if quarter == 4:
        return date(year, 12, 31)
    else:
        next_quarter_start = date(year, quarter * 3 + 1, 1)
        return next_quarter_start - timedelta(days=1)


def format_quarter_label(year: int, quarter: int) -> str:
    """Format quarter as Q1'25 style label."""
    return f"Q{quarter}'{str(year)[2:]}"
