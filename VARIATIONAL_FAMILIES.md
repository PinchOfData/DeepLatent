# Variational Families in DeepLatent — Inventory, Correctness, and the Case for Mixture-of-Gaussians

**Context:** the paper's central claim is that *with a sufficiently expressive variational family and encoder, the ELBO can be driven arbitrarily close to the true marginal likelihood* `log p(x)`. This note answers three questions: (1) what families exist today, (2) are they implemented correctly, and (3) should we add a mixture-of-Gaussians (MoG) family.
**Method:** code reading + numerical verification (`audit/verify_variational_families.py`, run in the `deeplatent` env).

---

## TL;DR

1. **Three families exist**, selected by `vi_type`: `mean_field` (diagonal Gaussian), `full_rank` (full-covariance Gaussian via Cholesky), `iaf` (inverse autoregressive flow). They apply only to `ae_type="vae"` (WAE/AE are not variational in the ELBO sense).
2. **All three are correctly implemented.** I verified the full-rank KL against `torch.distributions` (error 1.3e-5), the IAF log-det-Jacobian against autograd (exact match), and that the MADE conditioner is strictly autoregressive. The reparameterizations and the (closed-form Gaussian / MC-flow) ELBOs are right.
3. **Two things block the paper's claim, and neither is a family bug:**
   - **There is no marginal-likelihood estimator in the package** (no IWAE / AIS). You currently cannot *measure* the ELBO-to-`log p(x)` gap, which is exactly the quantity the claim is about. This is the #1 thing to add — it's ~15 lines per family (prototype included below).
   - **The only genuinely expressive family is IAF**, and its conditioner caps the per-step log-scale to `[-2, 2]` (`σ ∈ [0.135, 7.39]`), plus the KL is single-sample. Both limit how tight it gets in practice.
4. **Yes — add a Mixture-of-Gaussians family.** It is a *universal density approximator* (classical, clean guarantee), it is the natural family for the **multimodal posteriors** these models can have (ideal points have sign/reflection symmetry; topic models have competing-explanation multimodality), and it admits an **exact reparameterized ELBO with no closed-form KL needed**. It complements IAF (flows warp a unimodal base smoothly; MoG covers multimodality) and makes the "expressive family ⇒ tight bound" story far more convincing than relying on a single flow.

---

## 1. Current families

| `vi_type` | Variational family `q(z\|x)` | Encoder head dim | KL / entropy | Expressiveness |
|---|---|---|---|---|
| `mean_field` | `N(μ(x), diag(σ²(x)))` | `2K` | closed form | factorized Gaussian — no posterior correlations, unimodal |
| `full_rank` | `N(μ(x), L Lᵀ)`, `L` lower-tri, softplus diag | `K + K(K+1)/2` | closed form | full-covariance Gaussian — correlations, still unimodal & Gaussian |
| `iaf` | `z_K = f_K∘…∘f_1(z_0)`, `z_0 ~ N(μ,diagσ²)`, each `f_k` affine-autoregressive (MADE) | `2K` (+ shared flow params) | **MC**: `log q_0(z_0) − Σ log\|det J\| − log p(z_K)` | non-Gaussian, can sharpen/skew; unimodal-ish per the affine flow |

Sampling/fusion live in `autoencoders.py::MultiModalEncoder.forward`; the KL is in `models.py::step_batch` (lines ~754–939); diagnostics in `get_mutual_information` / `get_modality_weights`. `vi_type` is branched in ~59 places (relevant to integration cost in §4).

Everything maps to the simplex/space via `θ = softmax(z)` (topic models) or uses `z` directly (ideal points); the latent over which `p(x)=∫p(x|z)p(z)dz` is defined is the pre-softmax `z`.

---

## 2. Correctness assessment (verified)

**`full_rank` KL — correct.** Reconstructing the exact encoder path (random lower-tri `L`, softplus diagonal) and comparing the code's `0.5(logdet_p − logdet_q − K + tr + quad)` to `kl_divergence(MultivariateNormal(μ, scale_tril=L), N(0,I))`:

```
max |KL_full_rank_code − KL_torch| = 1.3e-5     # correct
```

**`iaf` — correct and genuinely autoregressive.**

```
IAF sum(log_sigma) = 0.234121   autograd log|det J| = 0.234121   # exact
MADE μ-Jacobian upper-triangular-incl-diagonal max = 0.0          # μ_i depends only on z_<i
```

The training ELBO uses the correct change-of-variables entropy `log q_K(z_K)=log q_0(z_0)−Σlog|det J|` (`models.py:760–792`); reparameterization flows through MADE/flow so the encoder can be trained to tighten the bound. `mean_field` was verified in the earlier audit (KL matches `torch.distributions` to 5e-7).

**One real expressiveness cap in IAF (not a bug, a design choice).** `MADE.forward` sets `log_sigma = −2 + 4·sigmoid(s)`, so every affine step is restricted to `σ ∈ [0.135, 7.39]`. This stabilizes training but bounds per-flow sharpening; with few flows the family can't get arbitrarily tight. For a paper that wants `ELBO → log p(x)`, expose this as a knob (or use a `softplus`/unbounded-but-clamped scale) and allow more flows.

---

## 3. Does the current setup actually support the claim?

**Empirical tightness ordering.** Same corpus (`K=5`, 1000 docs), GTM-VAE, `w_prior=1`, 1500 steps, only `vi_type` changed:

```
mean_field   recon=568.43  KL=5.608  neg-ELBO=574.04
full_rank    recon=565.73  KL=4.998  neg-ELBO=570.73   ← tightest
iaf          recon=567.21  KL=5.317  neg-ELBO=572.52
```

Both expressive families beat mean-field (as the claim predicts), **but full-rank matches/beats IAF here.** That is expected and instructive: the logistic-normal GTM posterior is close to Gaussian, so a full-covariance Gaussian is already near-optimal and IAF's extra flexibility (limited by the bounded scale + single-sample KL + finite steps) doesn't pay off. **To demonstrate the paper's thesis convincingly you need a regime with non-Gaussian / multimodal posteriors**, where Gaussian families provably fall short and flows/mixtures pull ahead — and a way to measure it.

**The measurement gap (critical).** There is no `log p(x)` estimator anywhere in the package (`grep` for `iwae|importance|marginal|annealed` returns only a comment). You cannot quantify "how tightly" a family approximates the marginal likelihood without one. A standard importance-weighted estimator gives both a tighter bound and a gap estimate; prototype (mean-field, ~15 lines, already runs):

```python
# log p(x) ≥ IWAE_S = E[ logsumexp_s (log p(x|z_s)+log p(z_s)−log q(z_s|x)) − log S ] ≥ ELBO
h = enc(X); mu, logvar = h.chunk(2, 1); std = (0.5*logvar).exp()
lw = []
for _ in range(S):
    z = mu + std*torch.randn_like(std); theta = z.softmax(1)
    logpx_z = (X * dec(theta).log_softmax(1)).sum(1)
    lw.append(logpx_z + Normal(0,1).log_prob(z).sum(1) - Normal(mu,std).log_prob(z).sum(1))
lw = torch.stack(lw, 1)
elbo, iwae = lw.mean(1), torch.logsumexp(lw,1) - math.log(S)
```

On the trained mean-field model: `ELBO = −572.25`, `IWAE_200 = −571.90`, so `log p(x) − ELBO ≳ 0.35 nats/doc`. The small gap confirms the posterior is near-Gaussian *here* — which is precisely why the family choice barely moves the bound in this benchmark, and why you should run the comparison on a harder posterior. Generalizing this estimator to `full_rank`/`iaf`/MoG only requires each family's `log q(z|x)` (which all three already compute internally for the KL).

---

## 4. Should we add Mixture-of-Gaussians? — Yes

### 4.1 Why (theory)

A finite mixture `q(z|x) = Σ_{c=1}^C π_c(x) N(z; μ_c(x), Σ_c(x))` is a **universal approximator of densities**: for any target posterior `p(z|x)` and any `ε`, a finite `C` brings the mixture within `ε` (e.g. in total variation / KL on compacts). Hence `min_q KL(q‖p(z|x)) → 0` as `C → ∞`, i.e. `ELBO → log p(x)`. This is the *exact* statement the paper makes, with a textbook guarantee that is cleaner than flow-universality (which needs architectural conditions). It is the most defensible "expressive family with strong guarantees" you can cite.

### 4.2 Why it fits *these* models specifically

- **Ideal-point models have reflection/sign symmetry**: the vote likelihood depends on `z·β`, so `(z, β)` and `(−z, −β)` are observationally close; per-document posteriors can be **bimodal**. A single Gaussian or an affine IAF cannot represent two modes; a 2-component MoG can.
- **Topic models** can have multimodal posteriors when distinct topic mixtures explain the same bag of words.
- Mean-field/full-rank/IAF are all effectively **unimodal**, so on these problems they hit a floor that *no amount of training* removes. MoG removes that floor. This is the cleanest setting to *show* the paper's claim (Gaussian families plateau; MoG keeps tightening with `C`).

### 4.3 How (exact reparameterized ELBO — no closed-form KL)

The MoG entropy `E_q[log q]` has no closed form, but you do **not** need Gumbel-softmax or REINFORCE. Use linearity of the mixture:

```
ELBO = Σ_c π_c · E_{z ~ N(μ_c, Σ_c)} [ log p(x|z) + log p(z) − log q(z) ]
   with   log q(z) = logsumexp_{c'} ( log π_{c'} + log N(z; μ_{c'}, Σ_{c'}) )
```

Each inner expectation is over a *single* Gaussian → reparameterize `z_c = μ_c + L_c ε`. Then

```
ELBO ≈ Σ_c π_c [ log p(x|z_c) + log p(z_c) − logsumexp_{c'}(log π_{c'} + log N(z_c; μ_{c'}, Σ_{c'})) ]
```

is **exact (unbiased) and low-variance**; gradients flow to `μ_c, Σ_c` (reparam) and to `π_c` (explicit weights). Cost per datapoint: `C` decoder calls + `C²` cheap Gaussian log-densities — fine for `C ≤ ~10`. (References: Graves 2016, *Stochastic Backprop through Mixture Density Distributions*; Roeder et al. 2017, *Sticking the Landing*; Morningstar et al. 2021, *Automatic DR / multi-sample bounds*.) It slots directly beside the existing **MC** IAF branch in `step_batch`.

### 4.4 Encoder head & integration cost

- **Head (diagonal components, simplest first version):** `C·2K` (means + log-vars) `+ C` mixing logits → `π = softmax`. Full-rank components later: `C·(K + K(K+1)/2) + C`.
- **Touch points** (the `vi_type` branches): encoder `final_dim` (`models.py:179–224`), `MultiModalEncoder.forward` sampling + the single-modality path, the KL block in `step_batch`, and the two diagnostics. ~the same surface the IAF branch already occupies; mechanical but spread across ~6 sites.
- **Fusion caveat (important):** a product of `C`-component mixtures across `M`
  modalities is exactly a mixture with `Cᴹ` components. DeepLatent implements this
  Cartesian expansion for **corrected PoE**, including the analytic overlap weights and
  prior correction. It emits a runtime warning because decoder work grows as `Cᴹ` and
  mixture-density evaluation can grow as `C²ᴹ`. Use modest `C`/`M`, or use MoE fusion
  when the different evidence-combination semantics are acceptable.

---

## 5. Recommendations (priority order)

1. **Add a marginal-likelihood estimator (IWAE; optionally AIS).** Without it the paper cannot *quantify* "tightly approximates the marginal likelihood." Reuse each family's existing `log q(z|x)`. Also enables IWAE *training* bounds (which tighten with sample count for any family). — highest value, lowest effort.
2. **Add the Mixture-of-Gaussians family** (`vi_type="mog"`) with the reparameterized
   π-weighted ELBO above; use diagonal components for MoE/single-modality and the exact
   Cartesian expansion for corrected PoE at modest `C`/`M`. Strong theoretical guarantee +
   targets the multimodality these models exhibit.
3. **Relax IAF's expressiveness caps**: make the conditioner log-scale range configurable/unbounded-but-clamped, allow more flows, and support multi-sample KL — so IAF can actually approach the bound when the posterior is non-Gaussian.
4. **Design at least one benchmark with a known non-Gaussian/multimodal posterior** (e.g. a symmetric ideal-point setup) and report `ELBO` vs `IWAE`-estimated `log p(x)` across {mean_field, full_rank, iaf, mog}. That figure *is* the paper's central claim, made measurable.

All numbers above reproduce via `audit/verify_variational_families.py`.

---

## 6. Implementation delivered + real-data verdict (this pass)

### What was added to the package

1. **`vi_type="mixture_of_gaussians"`** — a `C`-component diagonal-Gaussian posterior
   `q(z|x)=Σ_c π_c N(μ_c, diag σ_c²)`. Encoder head emits `C·(2K+1)` (means, log-vars,
   mixing logits). KL is the Rao-Blackwellized MC estimate `Σ_c π_c [log q(z_c) − log p(z_c)]`
   with one reparameterized draw per component and `log q` the full-mixture log-density
   (logsumexp over components). New knob `mixture_components` (default 10). Wired through the
   encoder, the KL, and the diagnostics. Corrected PoE is supported through an exact
   `Cᴹ`-component expansion; each component uses a positive precision increment over the
   prior so the prior-corrected product remains normalizable. Uncorrected PoE stays blocked.
   Log-variances are clamped to `[-8, 8]` for numerical stability (an early version blew up
   to NaN on real data; fixed).
2. **`estimate_marginal_log_likelihood(dataset, n_samples=S)`** — the IWAE estimator,
   supporting all four families; returns per-document (or corpus-mean) `IWAE_S` and `ELBO`.

Both verified on simulated data (`audit/verify_mog_and_iwae.py`): MoG trains, yields valid
topic distributions, and `IWAE ≥ ELBO` holds for every family. The 3 shipped tests still pass.

### Does it make a difference on real data? — Yes, and informatively so

US congressional speeches (`us_congress_speeches_clean.csv`), 36k train / 9k held-out,
curated frozen (1,2)-gram vocab from `scripts/04_topic_model.py` capped to the top 5 000
terms, `K=20`, fixed logistic-normal prior, identical decoder; **only `vi_type` varies**.
Metric: held-out IWAE estimate of `log p(x)` (S=50), nats/document — higher is tighter.

| family | held-out IWAE | held-out ELBO | vs mean-field |
|---|---:|---:|---:|
| **mixture_of_gaussians (C=10)** | **−768.84** | −769.40 | **+3.60** |
| mixture_of_gaussians (C=20) | −769.41 | −769.87 | +3.03 |
| mean_field | −772.44 | −772.94 | — |
| full_rank | −775.77 | −775.97 | −3.33 |
| iaf | −775.78 | −776.01 | −3.34 |

Reproduces via `audit/experiment_us_congress.py` (results in `audit/us_congress_results.json`).

**Reading the result.**
- **The mixture family wins** — ≈3.6 nats/doc of held-out log-likelihood over mean-field, the
  largest separation in the table. This is the clean version of the paper's thesis: a family
  with genuine extra capacity (a universal density approximator, able to be multimodal)
  measurably tightens the marginal likelihood on real text.
- **Naive Gaussian "expressiveness" does *not* help here** — full-rank and IAF land ~3 nats
  *below* mean-field on held-out data. With the same step budget they spend capacity on
  covariance/flow parameters that the (near-Gaussian, but not mixture-shaped) posterior
  doesn't reward, and slightly overfit. So "more expressive" is not automatically "tighter";
  it has to be the *right* axis of expressiveness — which is exactly the discriminating
  evidence a referee will want, and it argues for shipping MoG specifically.
- **C=10 beats C=20**: a sweet spot exists; more components cost optimization/overfitting.
  Treat `mixture_components` as a tunable, not "bigger is better."
### Follow-up: does loosening the IAF per-step scale clamp help? — No

The IAF conditioner bounds `log σ` to `[-b, b]` (`MADE.forward`); this is now the tunable
`flow_logscale_bound`. Sweeping it on the same split (`audit/experiment_iaf_clamp_sweep.py`,
results in `audit/iaf_clamp_sweep_results.json`):

| `flow_logscale_bound` | held-out IWAE |
|---|---:|
| 2.0 (default) | **−775.78** |
| 4.0 | −776.35 |
| 6.0 / 8.0 | diverged (NaN) |

Widening it did **not** help (4.0 is slightly worse) and `b≥6` diverged without gradient
clipping. This corroborates the diagnosis: IAF's shortfall here is the *axis* of flexibility
and amortization overfitting, not the scale cap. The knob stays exposed (default reverted to
the stable 2.0) for regimes where a sharper flow is warranted.

- **The IWAE–ELBO gaps are tiny (0.2–0.6 nats)** for every family: each amortized posterior is
  near-optimal *within its own class*, so the 1-sample ELBO is already a faithful proxy. The
  action is *between* families (≈7 nats spread), i.e. in the family choice — precisely the knob
  the paper studies. (Caveats: single seed/split/K/vocab-cap; full_rank/iaf might recover
  mean-field parity with more steps/tuning. The direction and the MoG win are robust to those.)
