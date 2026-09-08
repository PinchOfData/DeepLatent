# GTM fast iteration (2026-09-08): why the pilot showed no two-step bias and an off-target joint

Single draws, no Monte Carlo. Same fixed DGP as `audit/experiment_gtm_pilot.py`
(design seed 777, replication seed 9100, n = 10,000, K = 5 with a 6%-share topic,
V = 200, 25 words/document unless stated). Truth: beta = (2, 1, 0, -1, -2), sigma_y = 1.
Scripts and JSON live in `audit/fast/`.

## 1. The pilot's two-step is regression calibration, which is unbiased by construction

The pilot's two-step arm learns the *same* covariate prior and full covariance as the
joint arm and regresses y on the posterior mean of the shares E[theta | w, x]. For a
linear outcome y = theta'beta + eps with E[eps | w, x] = 0, OLS of y on the exact
posterior mean recovers beta exactly in population (E[theta_bar theta'] = E[theta_bar theta_bar']),
for every document length. This is the regression-calibration identity that the
ideal-point simulation section already uses to define the post-fit readout, and the
ideal-point paper section states that the exact-posterior-mean oracle is per-unit unbiased.
So no bias was to be expected from this arm. The bias analysed by Battaglia et al. is for
the *unshrunk* plug-in theta_hat = S(BB')^{-1}B(x_i/C_i) (their Assumption 3(iv) and
their Table 7 implementation); the ideal-point two-step used a *fixed N(0,1) prior without
covariates* plus a standardized estimand. The GTM pilot replaced both with a correctly
specified, consistently estimated prior, and the bias vanished with it.

Oracle two-step estimators on the pilot draw under the TRUE decoder and prior
(`gtm_oracle_truthonly.py`; OLS of y on the per-document share estimate; scale = beta_hat.beta/beta.beta):

| Readout (true parameters, n = 10,000)             | C = 10 | C = 25 | C = 100 |
|---|---:|---:|---:|
| OLS on true theta (floor)                          | 1.017 | 0.971 | 0.987 |
| E[theta \| w, x], true prior (regression calibration) | 1.012 | 0.948 | 0.977 |
| E[theta \| w], fixed N(0, I) prior, no covariates (ideal-point convention) | 1.237 | 1.123 | 1.079 |
| Per-document MLE plug-in (Battaglia-type, unshrunk) | 0.650 | 0.745 | 0.905 |

MAE at C = 25: floor 0.035, regression calibration 0.199, fixed prior 0.202, MLE plug-in 0.321.
The plug-in attenuation grows as documents shorten, exactly the kappa = sqrt(n) E[1/C]
mechanism; the fixed-prior readout *inflates* (it over-shrinks toward a common centre, so the
regressor has too little variance); exact regression calibration stays centred on 1 at every C.
HC1 standard errors of the calibration readout at C = 25: (0.10, 0.07, 0.25, 0.09, 0.08),
so its errors on this draw (largest on the 6% topic and on topic 1, whose sum stays near 2)
are sampling noise on a weakly identified contrast, not bias.

Readout variants under the *fitted* unsupervised pilot model (`gtm_readout_variants.py`), same draw:

| Readout under the fitted two-step model (C = 25) | Scale | MAE |
|---|---:|---:|
| Posterior mean of theta (256 encoder draws)        | 0.896 | 0.173 |
| Softmax of the encoder mean logit                  | 0.813 | 0.263 |
| Single posterior draw (`get_latent_factors` default `num_samples=1`) | 0.666 | 0.491 |

A user who calls the package's default readout gets the plug-in-type bias (a single draw is
an unshrunk noisy regressor). The proper posterior mean does not.

## 2. Likelihood at the truth: the words do not identify the rare topic

`gtm_lik_at_truth.py` evaluates log p(w | x) and log p(w, y | x) per document by importance
sampling with each model's own encoder as proposal (ESS median 290-650 of 1,024):

| Per-document log-likelihood (multinomial constant omitted) | train (10k) | held-out (2,048) |
|---|---:|---:|
| truth - fitted two-step, log p(w \| x)   | -0.013 (se 0.005) | +0.061 (se 0.010) |
| truth - fitted joint, log p(w \| x)      | -0.015 (se 0.005) | +0.057 (se 0.010) |
| truth - fitted joint, log p(w, y \| x)   | -0.014 (se 0.005) | +0.056 (se 0.010) |
| two-step - joint, log p(w \| x)          | -0.002 (se 0.001) | -0.004 (se 0.003) |

Both pilot fits sit at the truth's likelihood (slightly above it in sample, 0.06 nats/doc
below it out of sample) while their topic-3 decoder-profile correlation is only 0.33-0.39.
A decoder with the wrong rare topic costs at most 0.06 nats/doc: at 25 words and a 6% share
the words barely identify topic 3, so beta_3 (truth 0) and beta_1 (truth 2) are nearly
collinear for every estimator (their sum is about 2 in every fit; HC1 se of beta_3 is 0.25-0.39
even with exact posterior means). No estimator can be "very close" on beta_3 on one draw at C = 25.
This is also why the earlier overfitting audit saw nothing: the word likelihood is blind to the
outcome coefficients and to the rare topic.

## 3. The pilot's joint arm was under-trained in the outcome head

Pilot optimiser (lr 1e-3, prior lr 1e-4, batch 128, 64/64 encoder): joint scale 0.38, 0.73, 0.81,
0.86 at 2k/4k/8k/16k steps, still rising, while its word likelihood had converged (Section 2).
Same draw with the optimiser of the 30-rep Monte Carlo (lr 5e-3, prior lr 5e-4, batch 1024,
128-unit encoder; run A, `runA_mcopt.json`):

| Steps | Joint head beta (truth 2, 1, 0, -1, -2) | Scale | MAE | Two-step (learned prior) scale | MAE |
|---|---|---:|---:|---:|---:|
| 1k  | 2.07, 1.07, -0.87, -0.53, -1.75 | 0.924 | 0.347 | 0.894 | 0.209 |
| 2k  | 2.02, 1.08, -0.59, -0.66, -1.85 | 0.950 | 0.236 | 0.943 | 0.106 |
| 4k  | 1.91, 1.03, -0.11, -0.80, -2.03 | 0.970 | 0.093 | 1.017 | 0.072 |
| 8k  | 1.91, 0.94, 0.19, -0.87, -2.18  | 0.998 | 0.128 | 1.143 | 0.217 |
| 12k | 1.98, 0.93, 0.25, -0.89, -2.26  | 1.030 | 0.142 | | |
| 16k | 2.02, 0.91, 0.30, -0.91, -2.32  | 1.050 | 0.162 | | |
| 24k | 2.09, 0.89, 0.34, -0.93, -2.39  | 1.078 | 0.198 | | |

The joint reaches MAE 0.09 at 4k steps (floor 0.035; exact-posterior calibration 0.20) and then
drifts past the truth: the known U-shape, driven here by the unidentified rare-topic direction,
which the outcome likelihood can exploit. The learned-prior two-step passes through the truth
at 4k as well and drifts faster (scale 1.14 at 8k), consistent with the temperature degeneracy of a
learned Sigma at short documents. Neither arm shows a stable bias; both show training-time drift.

The drift is the learned-Sigma temperature degeneracy (short documents barely pin the absolute
logit scale). True contrast covariance trace 4.09; fitted Sigma trace by step:

| Steps | 1k | 2k | 4k | 8k | 12k | 16k | 24k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Joint Sigma trace     | 5.56 | 4.36 | 3.62 | 3.17 | 2.91 | 2.74 | 2.55 |
| Joint coefficient scale | 0.92 | 0.95 | 0.97 | 1.00 | 1.03 | 1.05 | 1.08 |
| Two-step Sigma trace  | 5.83 | 4.58 | 3.64 | 2.97 | 2.64 | 2.46 | 2.26 |
| Two-step coefficient scale | 0.89 | 0.94 | 1.02 | 1.14 | 1.21 | 1.27 | 1.33 |

Both arms cross the true covariance near 4k steps and keep deflating; an over-shrunk prior
over-shrinks the posterior means, which inflates the slope (same direction as the fixed N(0,I)
oracle). The outcome likelihood slows the drift in the joint arm but does not stop it; nothing
pins the scale in the unsupervised arm. Held-out word likelihood is flat along this direction
(Section 2), so no loss-based stopping rule sees it. The joint's post-fit calibration readout at
24k (scale 1.08, MAE 0.20) simply inherits the drifted prior.

Run B (`runB_pilotopt_64k.json`, pilot optimiser continued to 64k steps) walks the same path
about ten times more slowly: joint scale 0.81/0.87/0.91/0.96 and Sigma trace 5.5/4.5/4.0/3.6 at
8k/16k/32k/64k (two-step 0.83/0.88/0.90/0.94 and 6.1/4.9/4.3/3.8); topic-3 decoder correlation
rises 0.15 -> 0.84 (joint) and 0.15 -> 0.73 (two-step). Its final held-out IWAE equals run A's
to three decimals (joint -127.42, two-step -126.03) although run A's Sigma trace is 2.6 and its
coefficient scale 1.08 at that point: the likelihood is flat along the whole scale path.

## 4. The scale (temperature) of the linear-softmax GTM is not identified in practice

Ridge scan at the TRUE parameters (`gtm_ridge_scan.py`; 2,048 held-out documents; transform
eta -> eta/s, Sigma -> Sigma/s^2, D -> sD + per-word constant, beta -> s beta + intercept,
which leaves the model invariant to first order in theta - theta_bar):

| s (Sigma trace) | d log p(w\|x), C = 25 | d log p(w,y\|x), C = 25 | d log p(w\|x), C = 100 |
|---|---:|---:|---:|
| 0.85 (5.66) | -0.011 (se 0.004) | -0.011 | -0.039 (se 0.007) |
| 1.15 (3.09) | -0.016 (se 0.004) | -0.017 | -0.037 (se 0.005) |
| 1.30 (2.42) | -0.060 (se 0.007) | -0.062 | -0.130 (se 0.010) |
| 1.50 (1.82) | -0.166 (se 0.012) | -0.170 | -0.342 (se 0.017) |

The outcome likelihood contributes nothing to pinning the scale (beta rescales along the ridge), and
this linearised ridge still overstates the curvature: the *fitted* valley is flatter because the
decoder re-optimises. `gtm_drift_diag.py` scores the run A and run B final fits of the same data:

| Fit | Sigma trace | coef. scale | train IWAE log p(w\|x) | held-out IWAE | encoder gap |
|---|---:|---:|---:|---:|---:|
| joint A (24k, MC optimiser)     | 2.55 | 1.08 | -125.720 | -125.684 | 0.045 |
| joint B (64k, pilot optimiser)  | 3.64 | 0.96 | -125.722 | -125.684 | 0.041 |
| two-step A (24k, MC optimiser)  | 2.26 | 1.33 | -125.726 | -125.691 | 0.031 |
| two-step B (64k, pilot optimiser) | 3.80 | 0.94 | -125.726 | -125.690 | 0.031 |

Paired per-document differences A - B are within +-0.002 nats/doc (se 0.002) for IWAE and ELBO,
in sample and out of sample, for both arms (joint: also for log p(w, y | x)). Fits that differ by
20-25% in scale and by 0.1-0.4 in coefficient scale are indistinguishable by the exact likelihood
of 10,000 documents. Consequences:

- The topic-share scale, and with it the outcome-coefficient scale, is only weakly identified in
  this model class (softmax curvature is the only source), at 25 and at 100 words per document.
  The joint estimator inherits this: its coefficients cross the truth at a calibrated step budget
  (the established U-shape) and no likelihood-based criterion can locate that point.
- The learned-prior two-step (regression calibration) is unbiased only with the correct prior
  scale, which the unsupervised fit does not pin either; it drifts the same way, faster.
- The ideal-point simulations did not face this because their estimands were defined scale-free
  (standardised psi, reduced form c beta_1, beta_1/sigma_u) with a fixed N(0,1) prior gauge.
- An arithmetic-mixture (LDA-type) measurement model with anchor words pins the scale through the
  non-negativity of topic-word probabilities; the bias-free linear-softmax decoder has no such
  constraint (a per-word constant is absorbed because shares sum to one). This is a plausible
  reason the earlier anchor-word Monte Carlo was better behaved.

## 5. Large n, short documents: which two-step is biased

Truth-parameter readouts (`gtm_oracle_truthonly.py`) on one draw with n = 50,000 and C = 25:

| Readout (true parameters, n = 50,000, C = 25) | beta_hat | HC1 se | Scale | MAE |
|---|---|---|---:|---:|
| OLS on true theta (floor)                       | 2.03, 1.00, 0.00, -1.04, -1.98 | 0.03, 0.02, 0.06, 0.03, 0.03 | 1.007 | 0.020 |
| E[theta \| w, x], true prior (regression calibration) | 1.99, 1.01, 0.00, -1.04, -1.96 | 0.04, 0.03, 0.11, 0.04, 0.03 | 0.995 | 0.018 |
| E[theta \| w], fixed N(0, I) prior, no covariates | 2.44, 1.18, -0.19, -1.19, -2.24 | 0.04, 0.02, 0.08, 0.03, 0.03 | 1.174 | 0.247 |
| Per-document MLE plug-in (unshrunk)             | 1.45, 1.03, -0.06, -0.84, -1.58 | 0.03, 0.02, 0.04, 0.02, 0.02 | 0.792 | 0.245 |

At large n the regression-calibration two-step converges to the truth (its errors shrink with
the standard errors), while the fixed-prior and plug-in readouts sit at stable location shifts of
+17% and -21%, many standard errors away: the kappa > 0 regime of Battaglia et al. The "heavily
biased two-step" of the paper is therefore the plug-in/fixed-prior two-step, and the comparison
must be run against it (`--two-step-prior fixed`, or an explicit plug-in readout), not against a
two-step that estimates the correct prior and reads out exact posterior means.

Run C (`runC_w100_mcopt.json`, 100 words/document, MC optimiser): the rare topic is recovered
(topic-3 decoder correlation 0.88-0.95 from 2k steps on; share correlation 0.75) and both arms
pass through the truth near 2k steps (joint scale 0.97, MAE 0.09; learned-prior two-step 0.97,
MAE 0.05), then drift faster than at 25 words: Sigma trace 4.5/3.2/2.5/2.3 (joint) and
4.9/3.4/2.6/2.4 (two-step) at 2k/4k/8k/12k, coefficient scale 0.97/1.09/1.20/1.24 and
0.97/1.08/1.17/1.23. Longer documents fix topic identification, not the scale gauge.

## 6. Trained practitioner two-step (fixed N(0, I) prior, no covariates)

Run D (`runD_fixedprior_mcopt.json`, C = 25, MC optimiser, `--two-step-prior fixed`):

| Steps | Two-step encoder OLS beta (truth 2, 1, 0, -1, -2) | Scale | MAE |
|---|---|---:|---:|
| 4k | 1.60, 1.48, 0.63, -1.29, -2.42 | 1.081 | 0.444 |
| 8k | 1.59, 1.45, 0.70, -1.31, -2.43 | 1.081 | 0.460 |
| 8k, importance-weighted posterior mean | 1.58, 1.44, 0.72, -1.31, -2.42 | 1.074 | 0.458 |

Stable across checkpoints (the fixed prior pins the gauge, so nothing drifts) and badly biased:
the misspecified shrinkage distorts individual topic effects (topic 2 at 1.45, topic 3 at 0.70)
more than the global scale. On the same draw the joint arm reads 0.998-1.000 in scale and 0.13 MAE
at 8k. This is the comparison that reproduces the ideal-point pattern, provided the joint is read
at a calibrated budget or its gauge is pinned (Section 7).

Run E (`runE_w10_mcopt.json`, 10 words/document, MC optimiser): the 6% topic is not separable
(decoder correlation for topic 3 stays low in both arms) and its coefficient absorbs or sheds
effect from its neighbours: joint MAE 0.57/0.37/0.34 at 2k/4k/8k (scale 0.96/1.04/1.12) with
beta_3 at 1.43/0.93/0.65; learned-prior two-step MAE 0.55/0.68/0.77 with beta_3 at -1.4/-1.7/-1.9.
Global scales stay near 1 for both. At this document length the design's rare topic is simply
unidentified on one draw; the truth-parameter readouts (Section 1) show that even the exact
posterior mean has se 0.36 on beta_3 here, while the plug-in two-step is attenuated by 35%.
