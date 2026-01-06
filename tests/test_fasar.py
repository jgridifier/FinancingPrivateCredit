"""
Tests for the FASAR (Flex-Adjusted Syndicate Absorption Ratio) indicator.

Includes:
- Unit tests for each component
- Integration tests with mock data
- Historical scenario tests based on real market events

Run with: pytest tests/test_fasar.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from financing_private_credit.indicators.FASAR import (
    FASARIndicator,
    FASARSpec,
    CommitmentDeal,
    RigidityScorer,
    RigidityClassification,
    CLOVelocityCalculator,
)
from financing_private_credit.indicators.FASAR.rigidity import (
    extract_rigidity_evidence,
    compute_preliminary_rigidity,
    apply_binary_fallback,
)
from financing_private_credit.indicators.FASAR.nowcast import (
    FASARNowcaster,
    WarehouseStressIndicator,
)
from financing_private_credit.indicators.FASAR.forecast import (
    ScarTissueForecaster,
    ScarTissueSpec,
)


class TestRigidityScoring:
    """Test the rigidity scoring from 8-K text."""

    def test_sungard_detection(self):
        """Test detection of SunGard / Certain Funds language."""
        text = """
        The Commitment Letter provides for certain funds financing on a
        SunGard basis, meaning the lenders must fund regardless of market
        conditions on the closing date.
        """
        evidence = extract_rigidity_evidence(text)

        assert evidence.sungard_clause is True
        assert len(evidence.relevant_snippets) > 0

        score = compute_preliminary_rigidity(evidence)
        assert score >= 0.7, "SunGard clause should indicate high rigidity"

    def test_successful_syndication_condition(self):
        """Test detection of successful syndication condition (low rigidity)."""
        text = """
        The Bridge Facility is subject to successful syndication of the
        loans prior to funding. If syndication is not completed, the
        lenders may terminate their commitments.
        """
        evidence = extract_rigidity_evidence(text)

        assert evidence.successful_syndication_condition is True

        score = compute_preliminary_rigidity(evidence)
        assert score <= 0.3, "Syndication condition should indicate low rigidity"

    def test_market_flex_detection(self):
        """Test detection of market flex provisions."""
        text = """
        The commitment includes customary market flex provisions, allowing
        the lead arrangers to adjust pricing and terms based on market
        conditions at the time of syndication.
        """
        evidence = extract_rigidity_evidence(text)

        assert evidence.market_flex_mentioned is True

        score = compute_preliminary_rigidity(evidence)
        assert score < 0.7, "Market flex should reduce rigidity"

    def test_capped_flex_detection(self):
        """Test detection of capped flex (higher rigidity)."""
        text = """
        The commitment includes market flex provisions, however the flex
        is limited to 75 basis points on the interest margin.
        """
        evidence = extract_rigidity_evidence(text)

        assert evidence.flex_is_capped is True
        assert evidence.flex_cap_amount == "75bps"

    def test_mae_excludes_market(self):
        """Test MAE clause excluding market conditions (higher rigidity)."""
        text = """
        Material Adverse Effect shall exclude any changes resulting from
        general market conditions, economic conditions, or changes in
        interest rates.
        """
        evidence = extract_rigidity_evidence(text)

        assert evidence.mae_excludes_market is True

        score = compute_preliminary_rigidity(evidence)
        assert score >= 0.5, "MAE excluding market should increase rigidity"

    def test_binary_fallback_trapped(self):
        """Test binary fallback when bank is trapped."""
        from financing_private_credit.indicators.FASAR.rigidity import RigidityEvidence

        evidence = RigidityEvidence(
            sungard_clause=True,
            successful_syndication_condition=False,
        )

        score = apply_binary_fallback(evidence)
        assert score == 1.0, "Trapped bank should have rigidity 1.0"

    def test_binary_fallback_safe(self):
        """Test binary fallback when bank can exit."""
        from financing_private_credit.indicators.FASAR.rigidity import RigidityEvidence

        evidence = RigidityEvidence(
            sungard_clause=False,
            successful_syndication_condition=True,
        )

        score = apply_binary_fallback(evidence)
        assert score == 0.0, "Safe bank should have rigidity 0.0"

    def test_rigidity_scorer_without_llm(self):
        """Test rigidity scorer in regex-only mode."""
        scorer = RigidityScorer(use_llm=False)

        text = """
        Goldman Sachs has committed to provide $5 billion in bridge financing
        under SunGard terms with limited conditionality.
        """

        score, classification, evidence = scorer.score(text, "test_deal")

        assert score >= 0.7
        assert classification == RigidityClassification.TRAPPED
        assert evidence.sungard_clause is True


class TestCLOVelocity:
    """Test CLO velocity calculation."""

    def test_velocity_from_spreads(self):
        """Test velocity calculation from credit spreads."""
        calc = CLOVelocityCalculator(spread_baseline=300.0)

        # Normal spreads (300bps) should give velocity ~1.0
        spread_data = pl.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(30)],
            "spread": [300.0] * 30,
        })

        result = calc.calculate(spread_data=spread_data)

        assert 0.9 <= result.velocity <= 1.1, "Normal spreads should give velocity ~1.0"

    def test_velocity_from_wide_spreads(self):
        """Test that wide spreads reduce velocity."""
        calc = CLOVelocityCalculator(spread_baseline=300.0)

        # Wide spreads (500bps) should reduce velocity
        spread_data = pl.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(30)],
            "spread": [500.0] * 30,
        })

        result = calc.calculate(spread_data=spread_data)

        assert result.velocity < 0.8, "Wide spreads should reduce velocity"

    def test_velocity_from_tight_spreads(self):
        """Test that tight spreads increase velocity."""
        calc = CLOVelocityCalculator(spread_baseline=300.0)

        # Tight spreads (150bps) should increase velocity
        spread_data = pl.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(30)],
            "spread": [150.0] * 30,
        })

        result = calc.calculate(spread_data=spread_data)

        assert result.velocity > 1.2, "Tight spreads should increase velocity"

    def test_choke_signal_detection(self):
        """Test detection of JBBB vs HYG divergence."""
        calc = CLOVelocityCalculator()

        # JBBB drops 2%, HYG flat - should trigger choke
        today = datetime.now()
        etf_data = pl.DataFrame({
            "date": [today - timedelta(days=2), today],
            "ticker": ["JBBB", "JBBB"],
            "close": [100.0, 98.0],  # 2% drop
        }).vstack(pl.DataFrame({
            "date": [today - timedelta(days=2), today],
            "ticker": ["HYG", "HYG"],
            "close": [100.0, 100.5],  # Slight gain
        }))

        result = calc.calculate(etf_data=etf_data)

        assert result.choke_signal is True, "Should detect choke signal"


class TestFASARCalculation:
    """Test core FASAR calculation."""

    def test_fasar_basic_calculation(self):
        """Test basic FASAR calculation with mock deals."""
        indicator = FASARIndicator()

        # Create mock deals
        deals = [
            CommitmentDeal(
                deal_id="deal1",
                bank_ticker="GS",
                announcement_date=datetime.now(),
                commitment_amount=5000.0,  # $5B
                target_company="Target Corp",
                acquirer_company="Acquirer Inc",
                rigidity_score=0.8,  # High rigidity
                has_sungard_clause=True,
            ),
            CommitmentDeal(
                deal_id="deal2",
                bank_ticker="GS",
                announcement_date=datetime.now(),
                commitment_amount=3000.0,  # $3B
                target_company="Target 2",
                acquirer_company="Acquirer 2",
                rigidity_score=0.3,  # Low rigidity
                has_market_flex=True,
            ),
        ]

        # Calculate with normal velocity
        result = indicator.calculate_from_deals(deals, clo_velocity=1.0)

        assert result.height == 1  # One bank
        assert result["ticker"][0] == "GS"

        # Trapped volume = 5000*0.8 + 3000*0.3 = 4000 + 900 = 4900
        # FASAR = 4900 / 1.0 = 4900
        expected_trapped = 5000 * 0.8 + 3000 * 0.3
        actual_trapped = result["total_trapped_volume"][0]
        assert abs(actual_trapped - expected_trapped) < 0.01

    def test_fasar_multiple_banks(self):
        """Test FASAR calculation with multiple banks."""
        indicator = FASARIndicator()

        deals = [
            CommitmentDeal(
                deal_id="gs1",
                bank_ticker="GS",
                commitment_amount=5000.0,
                announcement_date=datetime.now(),
                target_company="T1",
                acquirer_company="A1",
                rigidity_score=0.9,
            ),
            CommitmentDeal(
                deal_id="jpm1",
                bank_ticker="JPM",
                commitment_amount=8000.0,
                announcement_date=datetime.now(),
                target_company="T2",
                acquirer_company="A2",
                rigidity_score=0.4,
            ),
        ]

        result = indicator.calculate_from_deals(deals, clo_velocity=1.0)

        assert result.height == 2
        assert "GS" in result["ticker"].to_list()
        assert "JPM" in result["ticker"].to_list()

    def test_fasar_risk_levels(self):
        """Test FASAR risk level classification."""
        indicator = FASARIndicator()
        spec = FASARSpec()
        indicator._spec = spec

        # High FASAR = HIGH_RISK
        emoji, status = indicator.get_warning_level(2.5)
        assert "TRAPPED" in status

        # Moderate FASAR = ELEVATED
        emoji, status = indicator.get_warning_level(1.7)
        assert "ELEVATED" in status

        # Normal FASAR
        emoji, status = indicator.get_warning_level(0.5)
        assert "NORMAL" in status

    def test_fasar_with_cet1_adjustment(self):
        """Test that CET1 adjustment dampens FASAR for well-capitalized banks."""
        indicator = FASARIndicator()
        indicator._spec = FASARSpec(apply_cet1_adjustment=True)

        deals_df = pl.DataFrame({
            "bank_ticker": ["GS", "DB"],
            "commitment_amount": [5000.0, 5000.0],
            "rigidity_score": [0.8, 0.8],
            "is_syndicated": [False, False],
        })

        # GS has high CET1, DB has low CET1
        cet1_df = pl.DataFrame({
            "ticker": ["GS", "DB"],
            "cet1_ratio": [0.15, 0.08],  # 15% vs 8%
        })

        result = indicator._calculate_fasar_by_bank(deals_df, 1.0, cet1_df)

        gs_fasar = result.filter(pl.col("ticker") == "GS")["effective_fasar"][0]
        db_fasar = result.filter(pl.col("ticker") == "DB")["effective_fasar"][0]

        # DB should have higher effective FASAR due to lower capital buffer
        assert db_fasar > gs_fasar


class TestFASARNowcast:
    """Test FASAR nowcasting."""

    def test_aaa_spread_widening_signal(self):
        """Test AAA spread widening detection."""
        nowcaster = FASARNowcaster()

        # Simulate 15bps widening in a week
        today = datetime.now()
        spread_data = pl.DataFrame({
            "date": [today - timedelta(days=7), today],
            "aaa_clo_spread": [100.0, 115.0],  # 15bps widening
        })

        indicator = WarehouseStressIndicator.calculate(spread_data, today)

        assert indicator is not None
        assert indicator.is_stressed is True
        assert indicator.spread_change_bps == 15.0
        assert indicator.stress_level == "elevated"

    def test_nowcast_velocity_adjustment(self):
        """Test that nowcast adjusts velocity appropriately."""
        nowcaster = FASARNowcaster()

        # Base FASAR data
        base_fasar = pl.DataFrame({
            "ticker": ["GS"],
            "effective_fasar": [1.5],
            "raw_fasar": [1.5],
        })

        # Critical spread widening (25bps)
        today = datetime.now()
        spread_data = pl.DataFrame({
            "date": [today - timedelta(days=7), today],
            "aaa_clo_spread": [100.0, 125.0],
        })

        result = nowcaster.nowcast(
            base_fasar=base_fasar,
            spread_data=spread_data,
        )

        # Nowcast FASAR should be higher due to reduced velocity
        assert result.data["nowcast_fasar"][0] > 1.5


class TestFASARForecast:
    """Test FASAR forecasting (Scar Tissue model)."""

    def test_scar_tissue_forecast(self):
        """Test M&A market share forecast."""
        forecaster = ScarTissueForecaster()

        # Bank with high FASAR should see market share decline
        fasar_data = pl.DataFrame({
            "ticker": ["GS", "JPM"],
            "effective_fasar": [2.5, 0.5],  # GS trapped, JPM fine
            "current_market_share": [0.08, 0.12],  # 8% and 12%
        })

        result = forecaster.predict(fasar_data, horizon=4)

        assert result.predictions.height == 2

        gs_forecast = result.predictions.filter(pl.col("ticker") == "GS")
        jpm_forecast = result.predictions.filter(pl.col("ticker") == "JPM")

        # GS should have larger negative change
        assert gs_forecast["change_pct"][0] < jpm_forecast["change_pct"][0]

    def test_cet1_dampening(self):
        """Test that CET1 dampens the scar tissue effect."""
        forecaster = ScarTissueForecaster(
            spec=ScarTissueSpec(apply_cet1_dampening=True)
        )

        # Same FASAR, different capital levels
        fasar_data = pl.DataFrame({
            "ticker": ["STRONG", "WEAK"],
            "effective_fasar": [2.0, 2.0],
            "current_market_share": [0.10, 0.10],
            "cet1_ratio": [0.16, 0.09],  # 16% vs 9%
        })

        result = forecaster.predict(fasar_data, horizon=4)

        strong = result.predictions.filter(pl.col("ticker") == "STRONG")
        weak = result.predictions.filter(pl.col("ticker") == "WEAK")

        # Well-capitalized bank should have less impact
        assert strong["cet1_dampening"][0] < weak["cet1_dampening"][0]


class TestHistoricalScenarios:
    """
    Test FASAR against historical scenarios.

    These tests use stylized data based on real market events.
    """

    def test_2008_hung_loan_crisis(self):
        """
        Simulate 2008-style hung loan scenario.

        Context: In 2008, banks had committed to LBO financing that couldn't
        be syndicated when credit markets froze. This is exactly what FASAR
        is designed to detect.
        """
        indicator = FASARIndicator()

        # Pre-crisis: Banks had large commitments with high rigidity
        deals = [
            CommitmentDeal(
                deal_id="lbo_2008_1",
                bank_ticker="Citi",
                announcement_date=datetime(2007, 10, 1),
                commitment_amount=10000.0,  # $10B
                target_company="TXU Energy",
                acquirer_company="KKR/TPG",
                rigidity_score=0.95,  # Very high - SunGard terms
                has_sungard_clause=True,
                has_limited_conditionality=True,
            ),
            CommitmentDeal(
                deal_id="lbo_2008_2",
                bank_ticker="Citi",
                announcement_date=datetime(2007, 11, 1),
                commitment_amount=7000.0,
                target_company="Clear Channel",
                acquirer_company="Bain/THL",
                rigidity_score=0.90,
                has_sungard_clause=True,
            ),
        ]

        # CLO market frozen - very low velocity
        result = indicator.calculate_from_deals(
            deals,
            clo_velocity=0.1,  # Market essentially frozen
        )

        fasar = result["effective_fasar"][0]
        risk_level = result["risk_level"][0]

        assert fasar > 2.0, "FASAR should be very high in crisis"
        assert risk_level == "HIGH_RISK"

        print(f"\n2008 Scenario - Citi FASAR: {fasar:.2f} ({risk_level})")

    def test_2015_energy_hung_loans(self):
        """
        Simulate 2015-2016 energy sector hung loans.

        Context: Oil price collapse led to hung loans for energy sector LBOs.
        """
        indicator = FASARIndicator()

        deals = [
            CommitmentDeal(
                deal_id="energy_2015",
                bank_ticker="WFC",
                announcement_date=datetime(2015, 6, 1),
                commitment_amount=3000.0,
                target_company="Energy Corp",
                acquirer_company="PE Firm",
                rigidity_score=0.9,  # Energy sector had very rigid terms
                has_market_flex=True,
                flex_is_capped=True,  # Limited flex
            ),
        ]

        # CLO market weak but not frozen (0.35 = significantly impaired)
        result = indicator.calculate_from_deals(
            deals,
            clo_velocity=0.35,
        )

        fasar = result["effective_fasar"][0]

        assert fasar > 1.0, "FASAR should be elevated"
        assert fasar < 5.0, "But not as extreme as 2008"

        print(f"\n2015 Energy Scenario - WFC FASAR: {fasar:.2f}")

    def test_2020_covid_shock(self):
        """
        Simulate March 2020 COVID market shock.

        Context: Brief but severe market dislocation, CLO market froze
        temporarily but recovered quickly with Fed intervention.
        """
        indicator = FASARIndicator()
        nowcaster = FASARNowcaster()

        # Base deals before shock
        deals = [
            CommitmentDeal(
                deal_id="pre_covid",
                bank_ticker="JPM",
                announcement_date=datetime(2020, 2, 15),
                commitment_amount=5000.0,
                target_company="Target Corp",
                acquirer_company="PE Buyer",
                rigidity_score=0.6,
                has_market_flex=True,
            ),
        ]

        # Pre-shock: normal velocity
        pre_shock = indicator.calculate_from_deals(deals, clo_velocity=1.0)
        pre_fasar = pre_shock["effective_fasar"][0]

        # During shock: AAA CLO spreads widened dramatically
        shock_spreads = pl.DataFrame({
            "date": [
                datetime(2020, 3, 1),
                datetime(2020, 3, 15),
            ],
            "aaa_clo_spread": [120.0, 200.0],  # 80bps widening
        })

        indicator_result = WarehouseStressIndicator.calculate(
            shock_spreads,
            datetime(2020, 3, 15)
        )

        assert indicator_result.is_stressed is True
        assert indicator_result.stress_level == "critical"

        print(f"\nCOVID Scenario:")
        print(f"  Pre-shock FASAR: {pre_fasar:.2f}")
        print(f"  Spread widening: {indicator_result.spread_change_bps:.0f}bps")

    def test_normal_market_conditions(self):
        """
        Test FASAR under normal market conditions.

        Banks should show low/normal FASAR when:
        - Deals have market flex
        - CLO market is healthy
        """
        indicator = FASARIndicator()

        deals = [
            CommitmentDeal(
                deal_id="normal_1",
                bank_ticker="GS",
                announcement_date=datetime.now(),
                commitment_amount=2000.0,
                target_company="Normal Target",
                acquirer_company="Normal Acquirer",
                rigidity_score=0.3,  # Bank has flex
                has_market_flex=True,
                successful_syndication_condition=True,
            ),
        ]

        result = indicator.calculate_from_deals(
            deals,
            clo_velocity=1.2,  # Healthy market
        )

        fasar = result["effective_fasar"][0]
        risk_level = result["risk_level"][0]

        assert fasar < 1.0, "FASAR should be low in normal conditions"
        assert risk_level in ["NORMAL", "WATCH"]

        print(f"\nNormal Market - GS FASAR: {fasar:.2f} ({risk_level})")


class TestIndicatorMetadata:
    """Test indicator metadata and registration."""

    def test_fasar_metadata(self):
        """Test FASAR indicator metadata."""
        indicator = FASARIndicator()
        metadata = indicator.get_metadata()

        assert metadata.short_name == "FASAR"
        assert metadata.update_frequency == "weekly"
        assert "SEC EDGAR" in metadata.data_sources[0]
        assert "Ivashina" in metadata.paper_reference

    def test_fasar_registration(self):
        """Test that FASAR is properly registered."""
        from financing_private_credit.indicators import get_indicator, list_indicators

        # Check registration
        indicators = list_indicators()
        assert "fasar" in indicators

        # Check retrieval
        indicator = get_indicator("fasar")
        assert isinstance(indicator, FASARIndicator)


def run_fasar_diagnostic():
    """Run diagnostic tests for FASAR indicator."""
    print("=" * 60)
    print("FASAR Indicator Diagnostic")
    print("=" * 60)

    # Test rigidity scoring
    print("\n1. Rigidity Scoring:")
    scorer = RigidityScorer(use_llm=False)

    test_texts = [
        ("SunGard terms", "The commitment provides certain funds on a SunGard basis."),
        ("Syndication condition", "Subject to successful syndication of the loans."),
        ("Market flex", "Includes customary market flex provisions."),
    ]

    for name, text in test_texts:
        score, classification, _ = scorer.score(text)
        print(f"   {name}: score={score:.2f}, class={classification.value}")

    # Test CLO velocity
    print("\n2. CLO Velocity:")
    calc = CLOVelocityCalculator()

    spread_tests = [150, 300, 500]
    for spread in spread_tests:
        data = pl.DataFrame({
            "date": [datetime.now()],
            "spread": [float(spread)],
        })
        result = calc.calculate(spread_data=data)
        print(f"   Spread {spread}bps: velocity={result.velocity:.2f}")

    # Test FASAR calculation
    print("\n3. FASAR Calculation:")
    indicator = FASARIndicator()

    deals = [
        CommitmentDeal(
            deal_id="test",
            bank_ticker="TEST",
            announcement_date=datetime.now(),
            commitment_amount=5000.0,
            target_company="Target",
            acquirer_company="Acquirer",
            rigidity_score=0.7,
        ),
    ]

    for velocity in [0.5, 1.0, 1.5]:
        result = indicator.calculate_from_deals(deals, clo_velocity=velocity)
        fasar = result["effective_fasar"][0]
        level = result["risk_level"][0]
        print(f"   Velocity {velocity}: FASAR={fasar:.2f} ({level})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_fasar_diagnostic()
