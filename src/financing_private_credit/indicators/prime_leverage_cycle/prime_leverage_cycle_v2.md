Prime Brokerage Leverage Lead Indicator (PB-LLI)

Purpose: A public-data, implementation-ready indicator suite designed to forecast normal-times prime brokerage / broker-dealer performance 1–2 quarters ahead, by combining (i) hedge fund balance-sheet leverage demand, (ii) dealer balance-sheet supply, and (iii) a higher-frequency nowcast layer that updates inside the quarter.

Design principle: Optimize for cycle and run-rate forecasting (balances and monetization) in normal market conditions; treat “stress” as a secondary overlay rather than the objective function.

⸻

1) Theoretical foundations (forecasting-centric)

1.1 Leverage cycle and intermediary balance sheets
	•	Procyclical leverage: leverage expands when collateral values rise and funding is ample; contracts when funding tightens. This creates predictable swings in intermediated balance-sheet activity (and therefore prime brokerage balances and revenue). (Adrian–Shin framework; “Liquidity and Leverage.”)
	•	Funding liquidity → balance-sheet capacity: the same client demand produces different prime brokerage outcomes depending on dealer constraints (internal VaR, balance-sheet cost, liquidity conditions). This motivates a two-sided model: HF demand + dealer supply.

1.2 Hedge fund leverage channels are structurally decomposable

A key edge (without paywalled data) is to explicitly separate hedge-fund leverage into:
	•	Prime broker secured borrowing (margin accounts) — closest mapping to equity prime financing.
	•	Repo liabilities — closer mapping to rates/FICC financing.

This is directly supported by the Fed’s hedge-fund sector accounts and the way official analyses frame hedge fund leverage channels.

1.3 Why this should lead prime brokerage / dealer performance in normal times

Prime brokerage earnings (at the segment level) are driven by a stable “normal-times” identity:

\text{PB Revenue} \approx \underbrace{\text{Client Financing Balances}}_{\text{quantity}} \times
\underbrace{\text{Net Financing Spread}}_{\text{price}} \;+\;
\underbrace{\text{Ancillary activity}}_{\text{prime services, execution, etc.}}

Your indicator’s main predictive power should come from forecasting quantity (balances) and, secondarily, price (funding tightness proxies). “Stress” mainly matters through nonlinear terms and shock triggers, which we keep as overlays rather than core.

⸻

2) Indicator architecture (what you publish each period)

You produce three objects:
	1.	Quarterly Anchor Index (structural truth): high signal, low frequency
	2.	Weekly Nowcast Factor (timeliness / edge): higher noise, high frequency
	3.	Composite Forecast Signal: the output used for 1–2 quarter projections

⸻

3) Core data (open, reproducible)

3.1 Quarterly hedge fund demand (Fed Z.1 / FRED series)
	•	HF prime-broker secured borrowing (margin accounts): BOGZ1FL624123035Q
	•	HF total financial assets: BOGZ1FL624090005Q
	•	HF total financial liabilities: BOGZ1FL624190005Q
	•	HF repo liabilities (optional but recommended for channel split): BOGZ1FL622151005Q

3.2 Quarterly dealer supply proxy (Fed Z.1 / FRED series)
	•	Security brokers & dealers: receivables due from customers (margin loans and other receivables): BOGZ1FL663067003Q
(Optional: dealer repo series if you want a broader balance-sheet constraint proxy.)

3.3 Weekly timeliness layer (open sources)
	•	CFTC “Leveraged Funds” positioning (weekly): used as a risk appetite / leverage appetite proxy
	•	NY Fed Primary Dealer Statistics / repo statistics (weekly): used as a financing conditions / dealer intermediation pulse

⸻

4) Definitions (the minimal set that works)

4.1 Hedge fund equity buffer (no AUM proxy needed)

\text{HF\_Equity}_t = \text{HF\_Assets}_t - \text{HF\_Liabilities}_t

4.2 Prime brokerage leverage intensity (core “quantity” driver)

\text{PB\_Intensity}_t = \frac{\text{HF\_PB\_Borrowing}_t}{\text{HF\_Equity}_t}

Interpretation (normal times):
	•	Rising PB_Intensity → expanding client financing balances → tailwind to PB revenue run-rate 1–2 quarters ahead.

4.3 Channel split (optional but high insight)

\text{Repo\_Intensity}_t = \frac{\text{HF\_RepoLiab}_t}{\text{HF\_Equity}_t}

This helps you map signals to likely segment sensitivity (equities prime vs rates financing).

4.4 Dealer supply / capacity proxy

Use the broker-dealer customer receivables series as a supply-side check:

\text{Dealer\_SupplyGrowth}_t = \Delta \ln(\text{BD\_CustReceivables}_t)

Interpretation:
	•	If PB_Intensity rises but Dealer_SupplyGrowth stalls, expect either balance growth to slow or monetization/spreads to rise (depending on the environment). This wedge is informative for “normal-times” forecasting.

⸻

5) The “edge” layer: weekly nowcast of the quarterly anchor

5.1 Construct a weekly signal set

Create standardized weekly features (z-scores or percentile ranks) such as:
	•	CFTC leveraged funds net positioning in key contracts (equity index futures, rates futures) and changes over 4–8 weeks.
	•	Primary dealer financing / repo-related weekly series (levels and changes).

5.2 Map weekly features to a latent “HF leverage factor”

Recommended method: state-space / Kalman filter (simple, robust, transparent).
	•	Measurement equation (weekly):
y_{w} = H \cdot x_w + \epsilon_w
where y_w are your weekly proxies (CFTC + NY Fed), and x_w is the latent leverage factor.
	•	Anchor equation (quarterly):
force the quarterly average of x_w to track the observed quarterly change in PB_Intensity (or its growth).

This gives you a weekly nowcast of PB_Intensity growth before the Z.1 print arrives, which is the practical “edge” for 1–2 quarter forecasting.

Deliverable:
	•	PB-Nowcast (weekly) = expected quarterly PB_Intensity growth for the current quarter (“in-quarter” estimate)

⸻

6) Composite forecast signal (normal-times optimized)

6.1 Forecast target

Pick one (or publish both):
	•	Macro PB activity: Broker-dealer customer receivables growth (BOGZ1FL663067003Q)
	•	Bank-level prime/equities financing proxies: extracted from 10-Qs (EDGAR XBRL / tables). (This is a later stage; start macro.)

6.2 Composite signal

A practical, stable composite:

\text{PB\_Lead}_t =
a\cdot \Delta \text{PB\_Intensity}_t^{\text{nowcast}}
+
b\cdot \Delta \text{PB\_Intensity}_{t-1}
+
c\cdot \text{Dealer\_SupplyGrowth}_t

Normal-times calibration:
	•	constrain coefficients a,b,c to be stable (ridge regression / Bayesian shrinkage)
	•	optimize on “business-cycle” periods, not crisis windows

Output you publish each week:
	•	Next-quarter (t+1) PB activity forecast and two-quarter (t+2) forecast
	•	confidence bands driven by historical nowcast error

⸻

7) Interpretation framework (what users should do with it)

7.1 “Run-rate” regimes (not stress regimes)

Define regimes using rolling percentiles of the composite PB_Lead:
	•	Accelerating (top tercile): improving prime balances/revenue momentum likely 1–2 quarters ahead
	•	Stable (middle tercile): neutral
	•	Decelerating (bottom tercile): slowing balances/revenue momentum likely 1–2 quarters ahead

This is intentionally not a stress classifier; it is a run-rate classifier.

7.2 Attribution dashboard (high value for PMs)

Publish decomposition each period:
	•	Contribution from HF demand (PB_Intensity nowcast)
	•	Contribution from dealer supply (BD receivables growth)
	•	Contribution from channel split (optional: repo vs PB)

This is what makes it “insightful” rather than a single opaque number.

⸻

8) Practical implementation guidelines

8.1 Data pipeline and timing
	•	Quarterly series: pull from FRED; align to quarter-end; enforce consistent vintage handling (avoid look-ahead).
	•	Weekly series: pull from CFTC and NY Fed; align to week-ending; create a “weekly within-quarter” calendar.

8.2 Transformations (keep it simple and robust)
	•	Use log changes for levels (growth rates).
	•	Winsorize extreme weekly moves (to prevent one-off reporting quirks dominating the nowcast).
	•	Standardize predictors in rolling windows (e.g., 5 years).

8.3 Model governance (avoid overfitting)
	•	Primary evaluation: out-of-sample rolling backtest with a fixed publication lag.
	•	Use stability constraints: prefer models that keep sign and magnitude stable across subsamples.
	•	Keep “stress” terms separate (see below).

⸻

9) Secondary stress overlay (kept separate by design)

Because you want normal-times forecasting as the primary focus, stress is best handled as an override flag, not baked into the main coefficient estimation.

Example overlay:
	•	Trigger if PB_Intensity drops sharply QoQ and weekly funding/positioning proxies deteriorate rapidly.
	•	When triggered: widen forecast intervals and add a “downside skew” note for broker performance.

This respects your design constraint: the core indicator remains a normal-times forecaster; stress is a secondary diagnostic.

⸻

10) What I recommend you build first (implementation order)
	1.	Quarterly PB_Intensity (HF PB borrowing / HF equity) and Dealer_SupplyGrowth
	2.	Composite quarterly PB_Lead and validate against macro BD receivables
	3.	Weekly nowcast layer (Kalman/state-space) feeding in-quarter PB_Intensity growth
	4.	Channel split (repo vs PB) for segment attribution
	5.	Bank-level EDGAR extraction to map the signal into named primes (GS/MS/JPM, etc.) once the macro link is validated


----
# Below is a concrete, engineer/quant-ready package: data dictionary, release calendar, and a reference implementation outline (model spec, backtest protocol, and publication-lag rules). Everything is based on open sources (FRED/Z.1, CFTC COT, NY Fed Primary Dealer Statistics, and ALFRED/FRED API for vintages).

⸻

1) Data dictionary

1.1 Quarterly anchors (Fed Z.1 via FRED; “ground truth”)

All series are Quarterly, End of Period, units Millions of Dollars, Not Seasonally Adjusted, and belong to the Z.1 Financial Accounts release.

Canonical name	FRED series	Description	Transform	Use
hf_pb_borrowing	BOGZ1FL624123035Q	Hedge Funds; Loans, Total Secured Borrowing Via Prime Brokerages (Margin Accounts); Liability, Level	level; Δln(level) optional	Core HF prime leverage demand  ￼
hf_assets	BOGZ1FL624090005Q	Hedge Funds; Total Financial Assets, Level	level	Equity buffer denominator  ￼
hf_liabilities	BOGZ1FL624190005Q	Hedge Funds; Total Financial Liabilities, Level	level	Equity buffer denominator  ￼
hf_repo_liab (optional)	BOGZ1FL622151005Q	Hedge Funds; Repo liabilities (Z.1)	level; Δln(level) optional	Channel split (repo/FICC financing)
bd_cust_recv	BOGZ1FL663067003Q	Security Brokers and Dealers; Receivables Due from Customers (Margin Loans and Other Receivables); Asset, Level	level; Δln(level)	Dealer supply proxy and/or macro target  ￼

Derived quarterly variables (canonical definitions)
	•	hf_equity = hf_assets - hf_liabilities
	•	pb_intensity = hf_pb_borrowing / hf_equity
	•	pb_intensity_g = Δln(pb_intensity) (QoQ)
	•	dealer_supply_g = Δln(bd_cust_recv) (QoQ)
	•	(optional) repo_intensity = hf_repo_liab / hf_equity, repo_intensity_g = Δln(repo_intensity)

Validation rules
	•	If hf_equity <= 0: hard fail (data issue).
	•	If any series has missing values: do not forward-fill across quarters; treat as missing and block model update.

⸻

1.2 Weekly signals (nowcast “edge” inputs)

A) CFTC Commitments of Traders (COT) — “Leveraged Funds”
	•	Publication: “generally” Fridays at 3:30pm ET, reflecting Tuesday data.  ￼
	•	You will ingest contract-level time series (e.g., equity index futures; major rates futures) and compute stable summary features.

Canonical weekly feature set (per contract or aggregated across a basket)
	•	net_pos (or net_pos_pct_oi if available)
	•	chg_4w = x - x.shift(4)
	•	chg_8w = x - x.shift(8)
	•	z_5y = zscore(x, 260); chg_8w_z = zscore(chg_8w, 260)

B) NY Fed Primary Dealer Statistics
	•	Publication: updated Thursdays ~4:15pm ET with the previous week’s statistics.  ￼
	•	Use a small set of consistently-defined financing/intermediation series (levels + 4/8-week changes + z-scores).

⸻

1.3 Targets (for validation and production forecasting)

Phase 1 (macro validation):
	•	Target: bd_cust_recv QoQ growth (dealer_supply_g) as a macro proxy for dealer margin/financing activity.  ￼

Phase 2 (bank-level mapping):
	•	Targets: prime/equities financing / securities-services segment metrics extracted from SEC EDGAR 10-Q/10-K (XBRL plus table extraction where necessary). (Open; engineering-heavy.)

⸻

2) Release calendar and publication-lag rules

2.1 Z.1 / Financial Accounts schedule
	•	Fed states the next Z.1 release date/time on the Z.1 release page (e.g., Thu, Mar 12, 2026 at 12:00 noon).  ￼
	•	Fed also notes Z.1 data are “typically” released in the second week of Mar/Jun/Sep/Dec.  ￼
	•	FRED provides a release calendar for Z.1 as well (useful for automation).  ￼

2.2 Operational “as-of” availability table

Dataset	Reference period	Typical publication	As-of gating rule
Z.1 quarterly series (FRED)	Quarter-end t	On Fed Z.1 release date/time (e.g., noon ET)  ￼	Quarter t is unavailable until the release timestamp
CFTC COT	Tuesday	Friday 3:30pm ET  ￼	Use only after Friday 3:30pm ET
NY Fed PD stats	Prior week	Thursday ~4:15pm ET  ￼	Use only after Thursday ~4:15pm ET

2.3 Revision/vintage handling (non-negotiable for credible backtests)

Z.1 data can be revised. Backtests must use ALFRED/FRED real-time parameters so that each simulated run sees only what was known then:
	•	Use realtime_start / realtime_end to define the real-time window.  ￼
	•	Use series_vintagedates to obtain vintage/revision dates.  ￼
	•	Use series/observations with realtime_start / realtime_end (or vintage_dates) to pull the correct vintage.  ￼

Rule: Every backtest prediction is produced by a function run(as_of_datetime) that (i) gates datasets by release timestamp and (ii) pulls quarterly series using ALFRED real-time settings consistent with as_of_datetime.

⸻

3) Reference implementation outline (minimal interpretation risk)

3.1 Repository structure (recommended)

pb_lli/
  config/
    series_fred.yml
    series_weekly.yml
    release_rules.yml
  src/
    ingest/
      fred_z1_vintages.py
      cftc_cot.py
      nyfed_pd.py
    features/
      quarterly_core.py
      weekly_features.py
      alignment.py
    models/
      bridge_nowcast.py
      composite_forecast.py
    backtest/
      asof_runner.py
      walkforward.py
      metrics.py
    publish/
      snapshot.py
      report.py
  tests/
    test_asof_gating.py
    test_vintage_pulls.py
    test_alignment.py

3.2 Ingest layer (contracts)

A) fred_z1_vintages.py
	•	Inputs: list of FRED IDs; as_of_date
	•	Outputs:
	•	df_quarterly_levels (quarter-end index)
	•	metadata including vintage date used for each series
	•	Implementation notes:
	•	Use FRED API with realtime_end=as_of_date to avoid look-ahead.  ￼
	•	Optionally cache series_vintagedates for each series.  ￼

B) cftc_cot.py
	•	Inputs: URLs or downloaded files; as_of_datetime
	•	Output: weekly time series indexed by report week
	•	Gating: exclude the most recent report if as_of_datetime is before Friday 3:30pm ET.  ￼

C) nyfed_pd.py
	•	Inputs: NY Fed data endpoint/download; as_of_datetime
	•	Output: weekly series indexed by week ending
	•	Gating: exclude current update if as_of_datetime is before Thursday ~4:15pm ET.  ￼

⸻

4) Model specification

4.1 Quarterly core construction (deterministic)

From quarterly levels, compute:
	•	hf_equity, pb_intensity, pb_intensity_g, dealer_supply_g
	•	(optional) repo_intensity, repo_intensity_g

No statistical fitting happens here. This is a strict accounting transformation.

4.2 Weekly nowcast model (V1: bridge regression; V2: state-space)

V1 (recommended start): Bridge regression
	1.	Create weekly feature matrix Xw (CFTC + NY Fed) with stable transforms (4/8-week changes, 5-year z-scores).
	2.	Aggregate to quarterly Xq using within-quarter mean (or last-value; pick one and standardize).
	3.	Fit quarterly:
pb\_intensity\_g(t) = \alpha + \beta^\top X_q(t) + \varepsilon_t
	4.	Nowcast current-quarter pb_intensity_g_nowcast using partial-quarter aggregation of observed weeks.

Regularization: ridge regression / Bayesian shrinkage to stabilize coefficients in normal times.

V2 (optional upgrade): single-factor state-space / Kalman where weekly proxies measure a latent “HF leverage appetite” state anchored to quarterly pb_intensity_g.

4.3 Composite forecast (normal-times forecaster)

Define a composite leading signal:
PB\_Lead(t) = a\cdot pb\_intensity\_g\_nowcast(t) + b\cdot pb\_intensity\_g(t-1) + c\cdot dealer\_supply\_g(t)

Forecast a target (phase 1 macro):
dealer\_supply\_g(t+1) = \gamma + \theta \cdot PB\_Lead(t) + u_t
and optionally t+2.

Normal-times focus: fit on a sample that (i) down-weights extreme periods or (ii) uses robust loss so crisis windows do not dominate parameter selection.

⸻

5) Backtest protocol (walk-forward, “as-of” realistic)

5.1 Walk-forward design
	•	Refit cadence: quarterly, immediately after each Z.1 release timestamp.
	•	Prediction cadence: weekly, whenever either (a) NY Fed PD update is available or (b) COT update is available.

5.2 “As-of” simulation (mandatory)

For each as_of_datetime:
	1.	Determine eligible datasets via release-gating rules (Section 2.2).
	2.	Pull quarterly Z.1 data using ALFRED real-time parameters consistent with as_of_date.  ￼
	3.	Construct features; produce nowcast and composite forecast.

5.3 Metrics (report separately for normal-times vs stress diagnostics)

Primary (normal-times):
	•	MAE/RMSE for t+1 and t+2
	•	Directional hit rate
	•	Parameter stability (sign consistency; coefficient drift)

Secondary (diagnostic only):
	•	Performance during 2008–2009 and 2020 windows (kept out of the optimization objective if you want “normal times” as the primary design).

⸻

6) Publication rules (what gets produced each run)

6.1 Output contract (machine-readable + human-readable)

Each weekly run emits:
	•	Latest available quarterly pb_intensity, pb_intensity_g, dealer_supply_g
	•	Weekly pb_intensity_g_nowcast for current quarter (with timestamp)
	•	PB_Lead and t+1, t+2 forecasts + confidence bands
	•	Attribution: contributions of (i) nowcast term, (ii) lagged intensity, (iii) dealer supply

6.2 Guardrails
	•	If a weekly source is delayed (holiday), do not impute; hold last nowcast and flag “inputs pending.”
	•	If quarterly vintage changes (revision), refit at next scheduled refit; archive the vintage used for reproducibility using ALFRED vintage controls.  ￼

⸻

7) “Engineer handoff” checklist

A build is complete when:
	1.	run(as_of_datetime) is deterministic and vintage-aware (ALFRED controls).
	2.	Release gating is unit-tested against documented times for COT and NY Fed PD.  ￼
	3.	Walk-forward backtest reproduces a full history without look-ahead (auditable snapshots).
	4.	Weekly nowcast updates correctly inside the quarter; quarterly refit happens only after Z.1 release time.  ￼


Below is a minimal, high-signal “V1 basket” that tends to work well for a weekly nowcast of hedge-fund financing appetite, plus a small set of NY Fed Primary Dealer series keys (mnemonics) that proxy dealer balance-sheet intermediation and frictions.

I’m deliberately optimizing for: (i) long history, (ii) liquidity/representativeness, (iii) stable definitions, and (iv) quick build.

⸻

A) CFTC COT (TFF) contract recommendations (Leveraged Funds)

Use the Traders in Financial Futures (TFF) schema (Leveraged Funds)

TFF is already segmented into Dealers / Asset Managers / Leveraged Funds / Other, and explicitly discusses consolidated equity index contracts and how consolidations can change over time.  ￼

V1 “core” contract basket (start here)

Equity risk (beta + risk-on/off)
	1.	CME S&P 500 (Consolidated)
	2.	CME Nasdaq-100 (Consolidated)
	3.	CME Russell 2000 (Consolidated)
	4.	CBOE VIX (if you want a single convexity/vol proxy; optional but useful)

Rates risk (duration + funding regime)
5) 2-Year U.S. Treasury (CBOT)
6) 5-Year U.S. Treasury (CBOT)
7) 10-Year U.S. Treasury (CBOT)
8) Ultra 10-Year or 30-Year U.S. Treasury (CBOT) (pick one; don’t overfit with both initially)
9) SOFR (STIR proxy; use in addition to belly duration)

FX risk (macro levered positioning proxy)
10) EUR/USD
11) JPY/USD
12) GBP/USD (optional: add AUD/USD as “risk FX” once V1 is stable)

Why this basket works in practice (for your use-case)
	•	It captures the dominant macro risk books that are typically financed and scaled (equity beta, rates DV01, FX).
	•	TFF’s consolidation guidance helps you avoid double counting mini/micro variants and reduces “methodology breaks” when CFTC changes consolidation rules.  ￼

What to compute (V1 features)

For each contract (or for the basket aggregates):
	•	net = long - short
	•	net_pct_oi = net / open_interest (preferred when available)
	•	chg_4w, chg_8w
	•	z_5y(net_pct_oi) and z_5y(chg_8w)
	•	Aggregate indices:
	•	equity_net_z = mean(z of ES,NQ,RTY)
	•	rates_net_dv01_z = mean(z of (TU/FV/TY/US + SOFR)) (or duration-scaled if you do DV01 mapping)
	•	fx_net_z = mean(z of EUR,JPY,GBP)

⸻

B) NY Fed Primary Dealer series (mnemonics) recommendations

Practical point: use the OFR Short-term Funding Monitor API for PD mnemonics

OFR’s documentation shows the NYPD dataset is exposed via mnemonics (and provides examples).  ￼
NY Fed confirms PD stats are weekly, updated Thursdays ~4:15pm ET, with a history back to 1998 (but split into time periods).  ￼

V1 “core” PD series list (small but potent)

These are the highest signal for “dealer balance sheet capacity + frictions” with minimal series sprawl.

1) Financing volumes (repo/reverse repo)
	•	NYPD-PD_RP_T_TOT-A — Repo backed by U.S. Treasuries (Total) (key dealer financing channel)
	•	NYPD-PD_RRP_T_TOT-A — Reverse repo backed by U.S. Treasuries (Total)
	•	NYPD-PD_RP_T_TIPS_TOT-A (or ETIPS variant depending on naming) — Repo backed by Treasury TIPS (Total)
The API docs explicitly show an example mnemonic in this family: nypd-pd_rp_t_etips_tot-a.  ￼

2) Settlement frictions (fails)
	•	NYPD-PD_AFtD_AG-A — Aggregate fails to deliver: Agency/GSE securities  ￼
	•	NYPD-PD_AFtR_AG-A — Aggregate fails to receive: Agency/GSE securities (pair with AFtD)
	•	NYPD-PD_AFtD_CORS-A — Aggregate fails to deliver: Corporate securities  ￼
	•	NYPD-PD_AFtR_CORS-A — Aggregate fails to receive: Corporate securities

3) Securities lending balance sheet (optional but often additive)
	•	NYPD-PD_SB_TOT-A — Securities borrowed (Total)
	•	NYPD-PD_SL_TOT-A — Securities lent (Total)

Recommended minimal V1 subset (if you want the absolute fastest build):
	•	NYPD-PD_RP_T_TOT-A
	•	NYPD-PD_RRP_T_TOT-A
	•	NYPD-PD_AFtD_AG-A
	•	NYPD-PD_AFtR_AG-A

These four usually give you most of the dealer “capacity + friction” signal without getting lost in collateral subtrees.

What to compute (V1 features)

For each mnemonic:
	•	level (or log(level) if strictly positive)
	•	chg_4w, chg_8w
	•	z_3y and/or z_5y
	•	Construct two composites:
	•	dealer_financing_z = mean(z(repo_T_total), z(reverse_repo_T_total))
	•	settlement_friction_z = mean(z(AFtD_AG), z(AFtR_AG), z(AFtD_CORS), z(AFtR_CORS))

⸻

C) Implementation note: programmatic “key discovery” (so you don’t hardcode wrong mnemonics)

Some NYPD series names/variants (e.g., TIPS vs eTIPS) can differ across time or documentation conventions. The clean approach is:

# OFR STFM: list all mnemonics for NYPD
https://data.financialresearch.gov/v1/metadata/mnemonics?dataset=nypd

# OFR HFM: search TFF series metadata by keyword (e.g., “S&P 500 Consolidated”, “Nasdaq”, “SOFR”)
https://data.financialresearch.gov/hf/v1/metadata/search?query=*S%26P*
https://data.financialresearch.gov/hf/v1/metadata/search?query=*SOFR*

The existence/usage of these endpoints is described in OFR’s API documentation.  ￼

⸻

