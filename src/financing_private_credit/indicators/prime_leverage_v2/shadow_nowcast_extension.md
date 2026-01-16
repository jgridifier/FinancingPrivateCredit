Below is a step-by-step iterative build that starts from your current “official-through-2025:Q2” method and extends it into a full shadow + progressive nowcast system through the most recent quarter (e.g., 2025:Q4 as of mid-Jan 2026), then snaps back to truth when Z.1 updates.

I’ve included (i) the full V1 equations, (ii) default hyperparameters, and (iii) a minimal disclosure schema for GS/MS/JPM that is implementable with EDGAR / issuer IR PDFs.

⸻

0) Baseline (what you already have)

Official core (quarterly, Z.1 via FRED):
	•	B_t: hedge fund prime brokerage borrowing (BOGZ1FL624123035Q)
	•	A_t: hedge fund assets (BOGZ1FL624090005Q)
	•	L_t: hedge fund liabilities (BOGZ1FL624190005Q)
	•	E_t = A_t - L_t: hedge fund equity
	•	I_t = B_t / E_t: PB intensity (HF prime borrowing per equity)
	•	Work in logs/growth for modeling:
	•	y_t \equiv \Delta \ln I_t  (QoQ)

This “official” track ends at 2025:Q2 because the HF sector series are PF/ADV-sourced and lag, while other sectors can be more current.

⸻

1) Extension architecture (what changes)

You will publish three layers each week:
	1.	Official layer (immutable until Z.1 release): y_t for quarters where Z.1 HF series exist (through 2025:Q2).
	2.	Shadow backfill layer for missing completed quarter(s): e.g., estimate 2025:Q3 even though HF Z.1 stops at Q2.
	3.	Progressive nowcast layer for the most recent quarter being reported (e.g., 2025:Q4): updates as:
	•	weekly proxy data arrives, and
	•	bank earnings disclosures arrive (GS/MS/JPM this week, others later).

When a new Z.1 release eventually prints the HF series, you:
	•	overwrite shadow quarters with official values, and
	•	record nowcast errors for calibration.

⸻

2) Inputs for the shadow/nowcast (open data only)

2.1 Weekly proxy factor (always available, noisy but timely)

Build a single weekly “HF leverage appetite / dealer intermediation” factor x_w from:
	•	CFTC TFF “Leveraged Funds” positioning (your contract basket)
	•	NY Fed Primary Dealer statistics (repo/reverse repo + fails)

(You already have the contract/mnemonic plan; we keep V1 to one factor for stability.)

2.2 Quarterly dealer-side anchor (when available)

Use a dealer-side quarterly series as an anchor proxy for financing supply, e.g.:
	•	D_t = BD_CustRecv growth (Security Brokers and Dealers; receivables due from customers) if available.

If dealer quarterly data is itself lagged relative to “current quarter,” it naturally drops out for the most recent nowcast and the model relies more on weekly factor + disclosures.

2.3 Earnings disclosures pulse (event-driven, higher information content)

Create a quarterly “disclosure pulse” p_t as banks report. This is the key to extending from “one quarter behind” to “most recent quarter”.

Examples of prime-relevant disclosures that are explicitly referenced in open materials:
	•	Goldman: Equities financing is explicitly disclosed, with commentary that record equities financing reflected higher net revenues “in prime and portfolio financing.”  ￼
	•	Morgan Stanley: Equity net revenues explicitly attribute strength to “financing revenues from higher client balances in prime brokerage.”  ￼
	•	JPMorgan: Equity Markets revenue explicitly cites strength “particularly in Prime.”  ￼

⸻

3) V1 equations (implementation-complete)

3.1 Weekly factor construction (single factor x_w)

Let z^{cot}_{j,w} be standardized features from CFTC (per contract j), and z^{pd}_{k,w} from NY Fed PD series (per mnemonic k).

Feature standardization (robust, rolling):
z_{w} = \frac{u_w - \text{median}(u_{w-260:w-1})}{1.4826 \cdot \text{MAD}(u_{w-260:w-1})}
	•	Use 260 weeks (~5y) window; MAD-based scale to reduce crisis outlier sensitivity.

Raw weekly factor (equal-weight V1):
x^{raw}_w = \frac{1}{J}\sum_{j=1}^{J} z^{cot}_{j,w} \;+\; \frac{1}{K}\sum_{k=1}^{K} z^{pd}_{k,w}

Smoothing (EWMA; half-life 8 weeks):
x_w = (1-\rho)\,x^{raw}_w + \rho\,x_{w-1}, \quad \rho = 2^{-1/8}\approx 0.917

Default V1 weekly basket sizes:
	•	CFTC: 8–12 contracts (your minimal basket)
	•	PD: 4 series (repo, reverse repo, fails to deliver/receive agencies)

This yields a stable x_w that updates weekly.

⸻

3.2 Quarter mapping of weekly factor

Define quarter t consists of weeks w \in \mathcal{Q}(t).

Full-quarter aggregation:
\bar{x}_t = \frac{1}{|\mathcal{Q}(t)|}\sum_{w\in\mathcal{Q}(t)} x_w

Partial-quarter aggregation as-of date \tau (for progressive nowcast):
\bar{x}_{t|\tau} = \frac{1}{|\mathcal{Q}(t,\tau)|}\sum_{w\in\mathcal{Q}(t,\tau)} x_w
where \mathcal{Q}(t,\tau) are the weeks in quarter t observed by \tau.

⸻

3.3 Bank disclosure pulse p_t (minimal and robust)

For bank i, extract one quarterly prime-relevant metric m_{i,t} (schema below).

Compute QoQ log change:
g_{i,t}=\Delta\ln(m_{i,t})

Standardize within bank using trailing 8 quarters (normal-times oriented):
s_{i,t}=\frac{g_{i,t}-\mu_{i,t}^{(8q)}}{\sigma_{i,t}^{(8q)}+\epsilon}
with \epsilon = 10^{-6}.

Coverage-weighted pulse:
p_t=\frac{\sum_{i\in\mathcal{R}(t)} w_i\, s_{i,t}}{\sum_{i\in\mathcal{R}(t)} w_i}
where \mathcal{R}(t) are banks that have reported quarter t.
V1 weights: w_i=1 (equal), unless you have a defensible prime share weighting.

Progressive completeness scalar (tightens uncertainty as more banks report):
c_t=\frac{\sum_{i\in\mathcal{R}(t)} w_i}{\sum_{i\in\mathcal{B}} w_i}\in[0,1]

⸻

3.4 Shadow backfill model for missing completed quarter(s)

Target: quarterly y_t=\Delta \ln I_t where I_t=B_t/E_t. Fit on quarters where Z.1 HF series exist.

Ridge regression (V1):
y_t = \alpha + \beta_1 \bar{x}_t + \beta_2 D_t + \beta_3 p_t + \varepsilon_t

Where:
	•	\bar{x}_t is full-quarter weekly factor
	•	D_t is quarterly dealer anchor (if available)
	•	p_t is disclosure pulse (often missing historically; include when available)
	•	If D_t or p_t is missing for older history, include missing indicators or simply set to 0 and restrict training windows.

Ridge objective (standardized regressors):
\min_{\alpha,\beta}\sum_t (y_t-\alpha-\beta^\top X_t)^2 + \lambda \|\beta\|^2

Default hyperparameters (V1):
	•	Standardize all regressors to mean 0 / sd 1 on training sample
	•	Ridge penalty: \lambda = 5 (good “shrink-to-stability” default in small-sample macro series)
	•	Training window: expanding from 2002 onward (or from first reliable HF Z.1 availability)
	•	Robustness: winsorize y_t at 2.5/97.5 percentiles in training only (normal-times emphasis)

This gives you a shadow estimate for 2025:Q3:
\hat{y}_{2025:Q3}=\hat{\alpha}+\hat{\beta}_1\bar{x}_{2025:Q3}+\hat{\beta}_2 D_{2025:Q3}+\hat{\beta}_3 p_{2025:Q3}

⸻

3.5 Progressive nowcast model for the most recent quarter (as banks report)

For quarter t in-progress or “just ended” but not in Z.1 HF series:

\hat{y}_{t|\tau}=\hat{\alpha}+\hat{\beta}_1\bar{x}_{t|\tau}+\hat{\beta}_2 D_{t|\tau}+\hat{\beta}_3 p_{t|\tau}

Where:
	•	\bar{x}_{t|\tau} updates weekly
	•	p_{t|\tau} updates each time a bank reports
	•	D_{t|\tau} is either the quarterly dealer anchor when it becomes available, or 0 until it is

Uncertainty (publishable band):
Let \hat{\sigma}_\varepsilon^2 be in-sample residual variance. Inflate for incomplete information:

\text{Var}(\hat{y}_{t|\tau}) \approx \hat{\sigma}_\varepsilon^2 \left(1 + \kappa_x \frac{1-\omega_x}{\omega_x} + \kappa_p (1-c_t)\right)

Defaults:
	•	\omega_x = \frac{|\mathcal{Q}(t,\tau)|}{|\mathcal{Q}(t)|} = fraction of quarter weeks observed
	•	\kappa_x = 0.5
	•	\kappa_p = 1.0

So early in the quarter (few weeks, few banks) your band is wider; it tightens as \omega_x\to1 and c_t\to1.

⸻

3.6 Converting \hat{y}_t back into levels (so the indicator is complete)

You maintain an official level I_{t_0} at last official quarter t_0=2025:Q2.

Then for shadow quarters:
\hat{I}_{t} = I_{t-1}\cdot \exp(\hat{y}_t)
chaining forward quarter by quarter. For Q4 nowcast, you chain from Q3 shadow.

If you also want B_t and E_t separately (not just intensity), you can run the same framework on:
	•	\Delta\ln B_t
	•	\Delta\ln E_t
and then reconstruct I_t=\frac{B_t}{E_t}. V1 can start with intensity only (simpler, often more stable).

⸻

4) Minimal disclosure schema (GS / MS / JPM) — V1

You want one numeric metric per bank that is (a) clearly linked to prime/financing, (b) consistently disclosed, and (c) extractable from PDF/HTML.

4.1 Goldman Sachs (GS) — Exhibit 99.2 presentation (SEC 8-K)

Source: SEC Exhibit 99.2 HTML for the earnings presentation.

Fields
	•	gs_equities_financing_net_rev_q (USD mm): “Equities financing” net revenues (quarter)
	•	The GS presentation includes a quarterly table where “Equities financing” is shown (e.g., 4Q25 value) and commentary ties record equities financing to “prime and portfolio financing.”  ￼

Extraction target
	•	Locate the table containing “Equities financing” and parse the 4Q column value.

4.2 Morgan Stanley (MS) — earnings release PDF

MS does not always publish “prime balances” as a standalone numeric series, but it does provide a consistent “Equity net revenues” line and explicitly attributes drivers to prime brokerage client balances.

Fields
	•	ms_equity_net_rev_q (USD mm): “Equity net revenues” (quarter)
	•	ms_prime_balance_flag (binary/text): whether the narrative attributes changes to “financing revenues from higher client balances in prime brokerage”
	•	The earnings release explicitly states this driver.  ￼

Extraction target
	•	Parse the “Equity net revenues” line item (table) and store the narrative driver flag from the bullet.

4.3 JPMorgan (JPM) — earnings release PDF

JPM’s CIB disclosure includes explicit “Equity Markets revenue” and ties it to Prime.

Fields
	•	jpm_equity_markets_rev_q (USD mm): “Equity Markets revenue”
	•	jpm_prime_driver_flag (binary/text): narrative contains “particularly in Prime”
	•	The release states equity markets revenue was driven by higher revenue across products, “particularly in Prime.”  ￼

Extraction target
	•	Parse the “Equity Markets revenue” line and store the prime-driver phrase flag.

⸻

5) Putting it all together: iterative build plan (from Q2’25 stop to full system)

Step 1 — Freeze the “official” pipe (through 2025:Q2)
	•	Compute I_t, y_t for all available quarters.
	•	Ensure reproducibility with vintage logic (ALFRED) in backtests.

Step 2 — Build the weekly factor x_w
	•	Ingest CFTC + NY Fed PD weekly series.
	•	Produce x_w each week using the equations above.
	•	Validate: x_w should “look” like a leverage appetite proxy (rises in risk-on, falls in deleveraging), but don’t tune to crises.

Step 3 — Train the quarterly shadow model (intensity growth)
	•	Construct \bar{x}_t historically
	•	Add D_t when available
	•	Fit ridge model for y_t with defaults (\lambda=5, standardized regressors, winsorized training residual control)
	•	Produce shadow Q3’25 estimate \hat{y}_{2025:Q3} and \hat{I}_{2025:Q3}

Step 4 — Add disclosure pulse (event-driven) and enable shadow Q4’25
	•	Implement the GS/MS/JPM extractors and compute p_{2025:Q4|\tau} as those banks report
	•	Compute \bar{x}_{2025:Q4|\tau} from weekly data as-of today
	•	Generate \hat{y}_{2025:Q4|\tau} and a confidence band that tightens as c_t increases

Step 5 — Publication + reconciliation workflow

Each weekly run produces:
	•	official_y through Q2’25
	•	shadow_y for Q3’25 (static unless you revise methodology)
	•	nowcast_y for Q4’25 (updates as weekly data and earnings arrive)
	•	completeness: c_t and \omega_x
When Z.1 eventually prints HF series for Q3/Q4:
	•	replace shadow/nowcast with official
	•	log nowcast error and update calibration (without overfitting to stress windows)

⸻

6) Default hyperparameters (single place to copy into config)
	•	Weekly standardization window: 260 weeks
	•	Robust z-score: median/MAD (scale 1.4826)
	•	EWMA half-life: 8 weeks (\rho \approx 0.917)
	•	Quarterly model:
	•	regressors standardized (mean 0, sd 1)
	•	ridge penalty \lambda = 5
	•	training winsorization on y_t: 2.5/97.5 pct
	•	Uncertainty inflation:
	•	\kappa_x = 0.5
	•	\kappa_p = 1.0
	•	Disclosure pulse standardization window: 8 quarters

⸻

Extraction: Concrete recommendation for your use-case (GS/MS/JPM V1)

V1 approach (strongly recommended)
	•	GS: deterministic table extraction from Exhibit 99.2 HTML (find “Equities financing” row; take the latest-quarter column).
	•	MS/JPM: extract the quantitative line item (e.g., “Equity net revenues”, “Equity Markets revenue”) via table parsing; extract the prime-driver flag via keyword match in narrative (regex is fine here because it’s a phrase check, not numeric extraction).

This yields a high-coverage, low-maintenance disclosure pulse.

Practical implementation guidance (decision logic)

Extractor decision tree for each filing/exhibit:
	1.	If HTML contains tables → try deterministic table parse.
	2.	If table parse fails but XBRL tags exist → use XBRL.
	3.	If PDF only → use deterministic PDF table extraction (rule-based).
	4.	If still fails → LLM to identify the right table/row/value, then verify deterministically.

Logging requirements (non-negotiable for production):
	•	store accession number + exhibit URL
	•	store table index + row label + column header used
	•	store raw extracted string and parsed numeric value
	•	store validation outcomes and any overrides