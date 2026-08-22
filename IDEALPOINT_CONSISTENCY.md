# Ideal point models: consistency & CI coverage program

Design agreed 2026-08-20 (interview). Companion scripts: `audit/experiment_ip_pilot.py`
(pilot + MC harness). Results land in `audit/results_ip_*.json`.

## Question

For `IdealPointNN` (1D Gaussian latent, votes-only 2PL IRT measurement), is the
amortized-ELBO **joint** estimator consistent for the regression functionals, and do
confidence/credible intervals have valid coverage — versus the standard **two-step**
estimator (unsupervised ideal points, then OLS)?

## Interview decisions

| Decision | Choice |
|---|---|
| Estimands | y-on-θ coefficient; prevalence (party) coefficient; θᵢ credible-interval coverage |
| Regime | fixed small J (votes/legislator), growing n — the κ>0 regime; plus a linear–Gaussian positive-control rung |
| CI approach | characterize sampling distributions across MC reps FIRST; build CI machinery only after |
| Scale | pilot (calibration, 1 rep, checkpoint trajectory) → full MC |
| Latent dim | 1D only |
| Outcome DGP | y = c·θ + ε only (no covariate in outcome) |
| Scale ridge | **learned Σ everywhere** (practitioner setting) |
| Modalities | votes only |

## DGP (per replication; estimands fixed across reps)

```
x_i ~ Bern(0.5)                         party (prevalence covariate)
θ_i = β0 + β1·x_i + N(0, σ_u²)          1D ideal point      (β0=0, β1=1, σ_u=1)
b_j ~ N(0,1),  d_j ~ N(0, 0.5)          discrimination / difficulty, j = 1..J
V_ij ~ Bern(σ(θ_i·b_j − d_j))           2PL votes           (J=25 primary)
y_i  = c·θ_i + N(0, σ_ε²)               outcome             (c=1, σ_ε=1)
```

Marginal sd(θ) = √(0.25·β1² + σ_u²) = √1.25 ≈ 1.118.

## Identification under learned Σ (the gauge problem)

Vote logits θ·b − d are invariant to θ → sθ + a with (b, d) absorbing it; with a
learned prior nothing pins location/scale/sign. **Raw coefficients are not
identified.** The study therefore evaluates identified functionals:

- `PSI  = c·sd(θ)` — outcome effect per 1 SD of θ. Truth ≈ 1.118.
- `B1STD = β1/σ_u` — party gap in residual-SD units. Truth = 1.
- `RF   = c·β1` — reduced form (y-effect of party via θ). Fully gauge-invariant
  (c→c/s, β1→sβ1). Truth = 1. **Primary cross-arm metric** — needs no alignment at all.

Conventions: sign fixed per rep by corr(posterior means, θ_true); the gauge for
θ-coverage is the affine map matching the model-implied marginal moments of θ to the
true marginal (β0+β1/2, √1.25). Model-implied sd(θ) for the joint arm =
√(Var̂ᵢ[μ̂(xᵢ)] + Σ̂); for the fixed-prior two-step arm the model marginal is N(0,1).

## Theory predictions (what the experiments test)

1. **Joint**: ELBO ≈ marginal likelihood (1D log-concave posterior → mean-field
   Gaussian nearly exact, variational gap should be tiny — smaller than GTM's 0.08
   nats/doc). Marginal MLE is consistent for fixed-dim functionals as n→∞ even at
   fixed J → expect bias → 0, SD ∝ n^(−1/2). Threats: finite-step SGD bias (the GTM
   U-shape — calibrate `num_steps` on matched sims, it is invisible to every loss),
   amortization gap, prior overfit at large step counts.
2. **Two-step subtlety (regression calibration)**: OLS of y on the *exact* posterior
   mean E[θ|V] is **per-unit unbiased** (E[y|θ̂]=c·θ̂ by iterated expectations when
   θ̂ is the exact posterior mean). The real two-step bias comes from (a) prior
   misspecification in the unsupervised step (fixed N(0,1) vs true covariate-shifted
   prior), (b) VI/amortization error in θ̂, (c) standardization: even the exact-PM
   standardized slope is attenuated by corr(θ, θ̂) ≈ √reliability. The ORACLE-PM arm
   (Gauss–Hermite exact posterior under true params) separates these channels.
3. **θᵢ credible intervals**: θᵢ is inconsistent at fixed J; the meaningful notion is
   average (Bayes) coverage. Exact posterior 90% intervals have ~90% average
   coverage (ORACLE-PM benchmark); mean-field VI typically underestimates posterior
   variance → undercoverage; expect worse in the θ tails (shrinkage). Joint-arm
   readout uses y=0 in the encoder (`labels_in_encoder`) → intervals target
   p(θ|V,y) but are read with y zeroed; reported as a separate caveated number.
4. **Two-step CIs**: naive OLS SEs ignore measurement error AND first-step bias →
   coverage of PSI should degrade toward 0 as n grows (bias fixed, SE shrinking).
   This is the κ>0 Battaglia et al. story with κ ∝ √n/J.

## Arms

| Arm | Fit | Readout |
|---|---|---|
| JOINT | `IdealPointNN`, `labels_in_encoder=True`, linear head, learned σ_ε², `update_prior=True` (mean_net on x + learned Cholesky Σ) | ĉ from head weight; β̂1 from mean_net; σ̂_u from Σ̂ → PSI, B1STD, RF |
| TWO-STEP | unsupervised, `update_prior=False` (fixed N(0,1) prior — the standard normalization) | OLS y~θ̂ (+ naive SEs), OLS θ̂~x |
| ORACLE-PM | none (Gauss–Hermite exact posterior under true DGP) | OLS y~E[θ|V]; exact credible-interval coverage |
| ORACLE | none | OLS y~θ_true (finite-n floor) |

## Phases

0. **Pilot** (`experiment_ip_pilot.py`, REPS=1): n=2000, J=25, checkpoint trajectory
   [2k, 4k, 8k, 16k, 24k]. Job: verify wiring/sign/gauge machinery, trace the
   U-shape, pick `num_steps`, time a rep.
1. **Full MC**: R=30 reps × n ∈ {1000, 4000, 16000}, J=25 fixed, calibrated steps.
   Read: E[bias] vs n (joint → 0? two-step flat?), SD·√n stable?, normality (QQ),
   coverage of naive two-step CIs, average θ-coverage per arm.
2. **J-sensitivity** (secondary): J=100 cell — two-step bias should shrink ≈4×
   (bias ∝ 1/J), a sharp internal validation.
3. **Linear–Gaussian rung** (positive control): same harness with
   `IP_MODALITY=gauss`. Measurement: w_ij = λ_j·θ_i + δ_j + N(0, σ_w²) with
   **σ_w² = 0.5 KNOWN** — pinned to the embedding decoder's implicit fixed
   variance (its Gaussian NLL is plain SSE ⇒ σ²=0.5, AUDIT_REPORT LOW-3), so the
   model is *correctly specified with known measurement variance* and no core
   loss code is touched. λ_j ~ N(0, 0.25²), J=25 features → reliability ≈ 0.8,
   matched to the votes design (same input dim, same measurement precision, only
   the channel changes). Exact posterior is closed form (Normal–Normal; posterior
   variance constant across i). Theory here is the strongest: posterior exactly
   Gaussian and in the mean-field family, E[θ|w,x] linear ⇒ amortization gap
   exactly closable ⇒ ELBO = exact MLE. If consistency/coverage fails HERE it's
   optimization, not the variational family. Follow-up (not needed for the rung):
   make the embedding decoder σ² learnable (closes LOW-3 properly).
4. **CI machinery for the joint arm**: decided after Phase 1's sampling
   distributions (candidates: sandwich on the structural block, bootstrap).

## RESULTS — RUN 1 (2026-08-20, 30 reps/cell × 3 n × 2 modalities, RANDOM-design MC)

Raw per-rep JSONs: `audit/hpc_results/`; merged: `audit/results_ip_mc_merged.json`
(regenerate with `python audit/merge_ip_mc.py audit/hpc_results/`).

**⚠ MC-design caveat (caught in review):** run 1 redrew the ITEM parameters
(discriminations/loadings) every replication, so its across-rep spread mixes
sampling noise with design-specific-bias variation — a "random-design /
across-studies" MC, not the classical fixed-DGP frame that coverage evaluation
requires (and that BCHS's width theorem assumes). Consequences for reading the
tables below: **bias/consistency conclusions and the two-step coverage collapse
are frame-robust** (they hold conditional on every design); **all SD-based
claims** (SD·√n table, width comparisons, the hybrid-CI test at 0.80–0.93) are
contaminated for the two-step arm and are restated by run 3, the fixed-design
rerun (`ITEM_SEED=777`, items held fixed across reps and n-cells; harness flag
`IP_ITEM_SEED`). One frame-specific observation worth keeping, correctly
labeled: under design redraws the two-step estimator's dispersion is dominated
by its design-varying attenuation and stops shrinking in n (visible in the
lin-Gauss cells), while the joint estimator — unbiased design-by-design — keeps
clean √n behavior in either frame.

### 1. The joint estimator is consistent (both measurement channels)

Joint @24k steps — bias (SD) across 30 reps; truth ψ=1.118, rf=1, β₁/σᵤ=1:

| cell | ψ bias (SD) | rf bias (SD) | β₁/σᵤ bias (SD) |
|---|---|---|---|
| votes n=1k | −.018 (.048) | −.017 (.073) | −.000 (.063) |
| votes n=4k | −.005 (.023) | −.014 (.045) | −.011 (.044) |
| votes n=16k | +.002 (.011) | +.002 (.023) | +.000 (.024) |
| gauss n=1k | +.004 (.038) | −.004 (.051) | −.007 (.064) |
| gauss n=4k | −.000 (.025) | −.009 (.038) | −.011 (.034) |
| gauss n=16k | −.003 (.011) | −.003 (.015) | −.001 (.014) |

Bias → 0 in n; SD·√n ≈ constant (√n rate). σ̂ᵤ ≈ 1.02, σ̂²ε ≈ 1.00 across reps.
**Steps note:** at 16k the prevalence functionals are still ~3–8% attenuated at
every n; 24k is converged. No overshoot anywhere (the GTM U-shape is absent —
"enough steps" suffices, no sweet-spot hunting).

### 2. Two-step is inconsistent and its naive CIs collapse (the κ>0 prediction)

Two-step bias is FLAT in n and equals the ORACLE-PM (perfect two-step) value:
ψ ≈ −0.12 (=1−√rel), rf ≈ −0.19 votes / −0.23 gauss (=1−rel), β₁/σᵤ ≈ −0.11/−0.15.
Naive 95% CI coverage across reps:

| cell | ψ CI covers | β₁/σᵤ CI covers |
|---|---|---|
| votes n=1k / 4k / 16k | 6.7% / 0% / 0% | 63% / 6.7% / 0% |
| gauss n=1k / 4k / 16k | 0% / 3.3% / 0% | 33% / 0% / 0% |

SE shrinks around the wrong value → coverage → 0. At n=16k a practitioner's CI
essentially NEVER contains the truth, for either measurement channel.

### 3. VI adds nothing to the error budget

- corr(θ̂, θ) = 0.899 at the √reliability ceiling (0.90) in every cell, both arms.
- Two-step VI ≈ exact-posterior two-step on every functional at every n.
- θᵢ credible intervals (two-step, q(θ|V)): coverage 0.89 at all n vs exact 0.90;
  interval widths 1.01× exact (sd_ratio_pm). Mean-field is calibrated in 1D —
  the factorization penalty is vacuous and the log-concave posterior is
  near-Gaussian (cf. VARIATIONAL_FAMILIES.md: expressiveness matters on the
  multimodality axis; there is none here).
- ORACLE-PM per-unit slope: unbiased at every n (regression-calibration theorem
  confirmed empirically: E[y|θ̂]=c·θ̂ when θ̂ is the exact posterior mean — the
  two-step bias lives in standardization + prior misspecification, NOT shrinkage
  per se).

### 4. Joint-arm θ-interval undercoverage is a readout artifact, not VI failure

Joint y-zeroed readout coverage ≈ 0.80–0.82 everywhere, but the diagnostics
pin it: interval widths are 0.95× the SUPERVISED posterior p(θ|V,y) (what
training targets) and 0.85× the y-free posterior p(θ|V) (what the y=0 readout
actually conditions on). Correctly sized for the wrong conditioning set +
miscentered without the true y ⇒ mechanical undercoverage. Fix if needed:
read out θ with true y in the encoder, or report q(θ|V) from an unsupervised
readout pass.

### Verdict on the two headline questions

- **Consistency: YES** for the joint amortized-ELBO estimator, on all identified
  functionals, in both the exact-theory (linear–Gaussian) case and the beyond-
  theory (logistic IRT) case — with √n-rate concentration.
- **CI validity:** no CI machinery exists yet, but the sampling distributions are
  unbiased with √n scaling, so M-estimator/sandwich intervals are the natural
  next step (Phase 4). Two-step naive CIs are catastrophically invalid, and
  VARIATIONAL credible intervals for θᵢ are valid (in the Bayes-average sense)
  when read from the right conditioning set.

## RESULTS — RUN 3 (2026-08-21, FIXED-design MC: items held fixed via ITEM_SEED=777; 30 reps × 3 n × 2 modalities; the definitive frame)

Raw: `audit/hpc_results_run3/`; merged: `audit/results_ip_mc_run3.json`.
Design 777 reliability ≈ 0.86. All run-1 headline conclusions replicate; the
SD-based statements below supersede run 1's.

### Joint (@24k): consistent, √n rate, both channels

ψ bias (SD): votes −.004(.042) / +.002(.020) / +.001(.011); gauss −.005(.045) /
+.000(.022) / −.000(.010) at n=1k/4k/16k. rf and β₁/σᵤ likewise → 0 (all ≤.016).
Fixed-design confirms: two-step and joint have ~EQUAL sampling SDs (e.g. .0102 vs
.0106 at votes n=16k) — the BCHS location-shift picture exactly: same ruler,
wrong place.

### Two-step: bias flat (ψ ≈ −.09, rf ≈ −.13/−.16), coverage collapses

Naive 95% CI coverage (ψ): votes .37/.00/.00, gauss .30/.00/.00.

### POST-FIT OLS (the BCHS-inspired recipe: joint fit → OLS of y on
### FITTED-param Y-FREE posterior means E_hat[θ|V], plain OLS SEs)

| cell | ψ bias | SD | E[SE] | 95% cover (naive/HC) |
|---|---|---|---|---|
| votes 1k/4k/16k | −.009/−.003/−.003 | .042/.020/.011 | .037/.019/.009 | .93/.93 · .93/.97 · .90/.90 |
| gauss 1k/4k/16k | −.004/−.000/+.000 | .045/.022/.010 | .037/.019/.009 | .90/.90 · .90/.90 · .93/.93 |

**Point estimate: unbiased at every n, as precise as the joint read itself; rf
≈ 1.00 everywhere.** Coverage ≈ .90–.93 vs nominal .95 — the shortfall is fully
accounted for: every regression-based SE is ≈0.85× the true sampling SD because
ψ = ĉ·σ̂θ carries σ̂θ estimation noise invisible to a regression SE (0.85 width
⇒ predicted coverage ≈ .905 ✓). Same story for the HYBRID CI (joint center ±
two-step naive SE): fixed-design coverage .933 at every votes n (run 1's .80 was
the design-mixing artifact). BCHS width test, fair frame: E[SE_2s]/SD(ψ̂_2s) =
.84–.93 — correct for the raw slope, ~15% short for the standardized functional.

**Practitioner recipe status:** joint fit + one OLS afterwards gives unbiased
estimates with ≈.90–.93 CIs from plain regression output. Oracle-PM
(exact-posterior benchmark): per-unit slope ≈ 1.000, exact θ-interval coverage
≈ .900, at every n, both channels.

### Normality of the joint sampling distribution (justifies Gaussian CIs)

All 6 cells pass Shapiro–Wilk (p = .19–.65), QQ correlations .973–.990, no
systematic skew, 30 reps each.

### The σ̂θ delta correction CLOSES coverage to nominal

Since ψ̂ = ĉ·σ̂θ, the variance missing from every regression SE is
`ψ̂²·(κ−1)/(4n)` with κ the kurtosis of the θ marginal. Verified on the stored
run-3 reps (κ from true DGP moments = 2.92): E[SE_corr] matches SD(ψ̂_pf) in
every cell and coverage moves .90–.93 → **.933–.967, pooled .956 over 180
intervals** ≈ nominal .95. The harness now computes the fully data-driven
version (`pf_se_psi_corr`, `pf_psi_cover_corr`): κ̂ from the FITTED marginal's
closed-form mixture moments (mean_net over observed x + Σ̂) —
`SE²corr = SE²OLS + ψ̂²(κ̂−1)/(4n)`. One line; no sandwich, no bootstrap.

**Final inference recipe (paper-ready):** (1) joint fit; (2) OLS of y on fitted
y-free posterior means Ê[θ|V]; (3) inflate the OLS SE by the closed-form σ̂θ
term. Unbiased + nominal coverage on 180 fixed-design reps across two
measurement channels.

### J=100 cell (votes, n=4000, 30 reps — out-of-sample test of everything)

- **Two-step bias tracks the reliability formula exactly**: realized rel = .941
  → predicted ψ attenuation √rel = .970, observed 1.080/1.118 = .966; predicted
  rf attenuation rel = .941, observed .944. (At J=25: rel .846 → √rel .920,
  observed .919.) Bias shrinks with J as theory says — but its naive CI still
  covers only .33 (SE shrinks too).
- **Joint**: ψ bias −.005, rf −.011 @24k — consistent as always. (@16k still
  attenuated on prevalence functionals — the 24k rule holds at J=100 too.)
- **The correction's out-of-sample moment**: naive post-fit coverage DROPS to
  .800 at J=100 — counterintuitively, MORE votes per person makes the naive CI
  worse, because the slope's own noise shrinks with reliability while the σ̂θ
  term does not, so the missing share grows. The data-driven corrected SE
  (.0208 vs true SD .020) restores coverage to **.967**. The correction works
  precisely where it is needed most: high-reliability designs.

## RESULTS — PRODUCTION SWEEP (2026-08-22, fixed design, 6,900 reps — the paper numbers)

27 packed SLURM workers on Bocconi HPC (`prod_worker.sbatch`), all `WORKER_DONE`,
zero failures. Cells: {ip,lg} × n∈{1k,4k,16k} × 1000 reps (J=25) + votes J∈{10,50,100}
× 300 reps at n=4000. Raw reps: `audit/hpc_results_prod/`; merged:
`audit/results_ip_mc_prod.json`; paper outputs: `tables/sim_{point,coverage,vi_appendix}.tex`,
`figures/sim_{distributions,bias,coverage,mechanism}.{pdf,png}`; section draft:
`papers/simulations_section.tex`. All run-3 conclusions reproduce at 1000-rep
precision (MC-SE on bias ≈ ±0.001, on coverage ≈ ±0.007):

- **Joint consistent, √n, Gaussian** — ψ bias ≤ .003 in all 9 cells (incl. J-grid);
  SD .043→.021→.011 (votes) exactly halving per 4× n; Shapiro p ≥ .105 and QQ-corr
  ≥ .996 in every cell at 1000 reps. b1_std and rf bias ≤ .007. lin-Gauss ≡ votes.
- **Two-step location shift at production precision** — ψ bias −.088/−.086/−.087
  (votes), −.092/−.091/−.089 (lg): flat to the third digit. rf bias −.13/−.16.
  SDs equal joint's in every cell (pure location shift, BCHS). Naive CI coverage
  .307→.004→**.000** (votes), .284→.002→.000 (lg).
- **Post-fit OLS + delta correction = the method** — bias ≤ .004 everywhere; naive
  coverage .881–.933; corrected coverage .940–.970, **pooled .955 over the 6,000
  core reps** (vs .95 nominal). Weakest cell ip_n16000 .940 (binomial SE .008, ~1.3
  SE below nominal — noise). rf from post-fit E ≈ .992–.998.
- **J-grid mechanism locked** — rel .697/.857/.897/.943 at J=10/25/50/100; two-step
  ψ ratio-to-truth .817/.922/.943/.970 vs √rel .835/.926/.947/.971; rf tracks rel.
  Two-step coverage at n=4000: .000/.004/.063/.483 — still far below nominal even
  at J=100. Naive post-fit coverage FALLS with J (.933/.909/.897/.880) while
  corrected stays nominal (.970/.960/.943/.950) — the "more votes, worse naive CI"
  gotcha at 300-rep precision.
- **Oracle-PM** per-unit ĉ ∈ [.999, 1.001] everywhere, per-unit CI cover .935–.959;
  VI at ceiling: corr .919–.922 vs √rel .923–.927; MF θ-cov .885–.889 vs exact .900–.902.

## Model config (mirrors GTM MC conventions)

`ae_type="vae"`, `w_prior=1` (true ELBO), `vi_type="mean_field"` (1D: full-rank ≡
mean-field; posterior unimodal log-concave so MoG unnecessary), encoder [64,64],
linear decoders **with bias** (`decoder_args={"vote_responses": {"bias": True}}` —
the per-bill difficulty d_j; default bias=False would misspecify the measurement
model), batch 256, `num_workers=0`, optimizer groups main lr 1e-3 / prior lr 1e-4
wd 0, `return_best_model=False`.

Note: `learn_prior_cov` is a no-op for Gaussian latents — `GaussianPrior` always
learns full Σ when `update_prior=True`; `update_prior=False` gives `FixedGaussianPrior`
N(0, I). This matches the chosen design exactly.
