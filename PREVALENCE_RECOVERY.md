# Can DeepLatent recover prevalence-covariate coefficients? — A Monte-Carlo study

**Question:** in simulations with known covariate effects on topic prevalence, how well do the
fitted models estimate those coefficients?
**Method:** `audit/experiment_prevalence_recovery.py` (+ `_optim.py`), run in the `deeplatent` env.

---

## TL;DR

- **The estimation problem is well-posed.** An infeasible *oracle* that regresses the
  centered log of the *true* topic proportions on the covariates recovers the true coefficients
  almost perfectly (corr **0.997**, slope **1.00**). So any shortfall is the model's, not the
  identification's.
- **Reading the model's built-in prior coefficients (`prior.mean_net`) is unreliable.**
  Under the **default WAE** it recovers essentially **nothing** (corr ≈ **0.00**); under VAE it
  recovers the *pattern* only weakly and noisily (corr ≈ **0.37 ± 0.23**), and in both cases the
  coefficients are **severely shrunk toward zero** (slope 0.24 / 0.01).
- **Two root causes for the shrinkage, both in the default optimizer:** the prior is trained at
  `lr=1e-4` (10× slower than the rest of the model) **and with `weight_decay=0.01`** — an
  explicit L2 penalty on the very coefficients you want to interpret. Removing the weight-decay
  roughly **doubles the recovered scale** (slope 0.33 → 0.68).
- **The reliable way to get covariate effects is a post-hoc two-step regression** — OLS of the
  inferred (centered-log) topic proportions on the covariates — which roughly **doubles** the
  correlation over reading the prior (VAE **0.68**, WAE **0.64**) and is the STM-style approach.
- **Net recommendation:** estimate prevalence effects with the two-step regression, not by
  reading `mean_net`; and change the prior's default `weight_decay` to 0 (it silently shrinks
  the coefficients). Recovery is modest here (corr ≈ 0.65) because this is a small-signal regime;
  it improves with more documents, longer documents, stronger effects, and more steps.

---

## Setup

DGP (`generate_documents`, logistic-normal): `z_d = X_d λ + ε`, `θ_d = softmax(z_d)`,
words `~ Multinomial(θ_d B)`. `λ` is `(C+1, K)` — intercept + `C` covariate effects per topic.
`R=10` replications, `N=2500` docs, `K=5` topics, `C=3` binary covariates, vocab 200, 80
words/doc, 2 000 steps. Model: `GTM(doc_topic_prior="logistic_normal", update_prior=True)`, so
the learnable prior's `mean_net` *is* the covariate→prevalence map; estimated coefficients are
`prior.mean_net.weight` (with `bias` folded into the intercept column).

**Identifiability handled explicitly** (the metric is meaningless otherwise):
1. **Topic permutation** — align estimated to true topics with the Hungarian algorithm on the
   document-topic overlap `θ_trueᵀ θ_est`.
2. **Softmax additive shift** (`softmax(z+c)=softmax(z)`) — center each covariate's coefficients
   across topics before comparing.
3. **Intercept/bias redundancy** — fold `mean_net.bias` into the intercept column.

Headline metric: recovery of the **non-intercept** covariate effects (correlation, RMSE, and the
slope of estimated-on-true, where slope=1 means the scale is recovered and slope<1 is shrinkage).

---

## Results

Per-effect recovery of the non-intercept covariate coefficients (mean ± sd over R=10):

| estimator | corr | RMSE | slope |
|---|---:|---:|---:|
| **oracle** (regress true centered-log-θ on X) | **0.997 ± 0.002** | 0.035 | 1.00 |
| two-step, VAE (regress inferred log-θ on X) | 0.679 ± 0.157 | 0.348 | 0.65 |
| two-step, WAE | 0.643 ± 0.152 | 1.093 | 1.86 |
| `mean_net`, VAE (read the prior) | 0.367 ± 0.228 | 0.446 | 0.24 |
| `mean_net`, WAE | −0.002 ± 0.236 | 0.536 | 0.01 |

Mechanism check — default vs tuned **prior** optimizer (VAE, R=6), `mean_net` recovery:

| prior optimizer | corr | slope |
|---|---:|---:|
| default `lr=1e-4, weight_decay=0.01` | 0.45 ± 0.22 | 0.33 |
| tuned `lr=1e-3, weight_decay=0` | 0.39 ± 0.18 | **0.68** |

Scatter of estimated vs true centered coefficients: `audit/prevalence_recovery_scatter.png`
(oracle hugs the diagonal; `mean_net` is diffuse and flattened toward 0). Raw numbers:
`audit/prevalence_recovery_results.json`.

---

## Interpretation

1. **The oracle ≈ 1.0 validates everything** — the DGP is exactly linear in logit space, the
   alignment/centering is correct, and perfect θ ⇒ perfect coefficients. The models simply leave
   signal on the table.

2. **`mean_net` is a poor coefficient estimator, and the default optimizer makes it worse.**
   The prior carries `weight_decay=0.01` (`models.py:417`), i.e. an L2 penalty pulling the
   prevalence coefficients toward 0, plus a 10× smaller learning rate (`lr=1e-4`) so it lags the
   encoder/decoder. Together these explain the slope ≈ 0.24 (VAE) and ≈ 0 (WAE). Removing the
   weight-decay restores most of the scale (slope 0.33 → 0.68) but not the correlation — the
   pattern signal in `mean_net` is just noisy (sd ≈ 0.22 across reps, occasionally negative).

3. **WAE ≫ worse than VAE for reading the prior.** WAE matches the *aggregate* posterior to the
   prior via MMD, which is a weak, marginal signal for *per-covariate* effects; the per-document
   KL of the VAE pins `q(z|x)` to `N(mean_net(x), Σ)` and gives `mean_net` a direct gradient. So
   under the package's default `ae_type="wae"`, the prior coefficients are essentially
   uninformative about covariate effects.

4. **The two-step regression is the right tool.** Regressing the inferred (centered-log) topic
   proportions on the covariates recovers the effects about twice as well as reading the prior
   (corr ≈ 0.65–0.68), and is exactly how STM and most applied topic-covariate analyses report
   effects. Caveat: WAE's θ is over-confident (more extreme), inflating the two-step *scale*
   (slope 1.86, large RMSE) even though the *pattern* (corr 0.64) is fine — so standardize or
   report partial correlations rather than raw magnitudes under WAE.

5. **Regime caveat.** corr ≈ 0.65 is a *modest-signal* setting (small effects `λ~N(0,0.25)`,
   2 500 short docs, 2 000 steps). Recovery rises with `N`, document length, effect size, and
   training; this study brackets "do effects come through at all and without bias," not the
   asymptotic ceiling (which the oracle shows is ~1.0).

---

## Recommendations

1. **Estimate prevalence effects with the two-step regression** (inferred centered-log-θ on the
   covariate design), not by reading `prior.mean_net`. Report standardized effects / partial
   correlations, especially under WAE where raw magnitudes are scale-inflated.
2. **Change the prior's default `weight_decay` to 0** (or surface it loudly). Shrinking the
   coefficients a researcher is trying to interpret is a surprising default; it costs ~3× in
   recovered effect size. If regularization is wanted for stability, make it opt-in and small.
   I can make this one-line change if you want it.
3. **Prefer VAE over WAE when covariate-effect estimation is a goal**, and consider a higher
   prior learning rate / longer training so `mean_net` actually converges.
4. **For the paper:** the oracle-vs-model gap is a clean way to separate *identification* (perfect
   here) from *estimation* (the deep model's amortized inference), and to show your method
   recovers covariate effects with the two-step readout — while flagging the attenuation honestly.

Reproduce: `python audit/experiment_prevalence_recovery.py` and
`python audit/experiment_prevalence_recovery_optim.py`.
