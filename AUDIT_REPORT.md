# DeepLatent — Mathematical & Code Audit

**Scope:** `deeplatent/` package (v0.1.3, branch `fix-posterior-collapse`)
**Files reviewed in full:** `models.py`, `autoencoders.py`, `priors.py`, `predictors.py`, `corpus.py`, `utils.py`, `simulations.py`, `__init__.py`, `tests/test_recon_reduction.py`
**Method:** line-by-line reading of the variational objective (ELBO/KL, reconstruction, priors, PoE/MoE fusion, flows), cross-checked against the literature, plus numerical verification scripts and short training runs in the `deeplatent` conda env (PyTorch 2.8 + CUDA).
**Date:** 2026-06-23

---

## Executive summary

The core of the package is **sound and, on its default configurations, mathematically correct.** I verified numerically that:

- the default **GTM** (fixed logistic-normal) and **IdealPointNN** (fixed Gaussian) mean-field KL match `torch.distributions.kl_divergence` to 5e-7;
- the **corrected Product-of-Experts** fusion reproduces the exact Gaussian posterior-combination identity (mean and precision error = `0.0`);
- the **IAF** log-det-Jacobian matches autograd, and the IAF training KL is a correct single-sample Monte-Carlo ELBO;
- the **posterior-collapse fix** on this branch (per-document reconstruction scale) is the right fix: reconstruction and KL now share a per-document scale, so `w_prior=1` really is the negative ELBO. The 3 shipped tests pass.

The audit nonetheless surfaced **one high-severity correctness bug** and several medium/low issues. The headline item:

> **VAE + Dirichlet prior computes the KL divergence in the wrong space.** The Dirichlet prior is summarized by its *simplex-space* moments (mean `α/Σα`, marginal variance) and those are then plugged into a **Gaussian KL evaluated in logit space**, where the encoder posterior actually lives. The two objects live on different manifolds; the resulting objective is mis-specified and empirically drives the inferred topic proportions toward uniform.

Everything below is organized by severity with location, the math, the evidence, and a concrete fix.

---

## Changes applied in this pass

After review, the following fixes were implemented and verified (`audit/verify_fixes.py`; all 3 shipped tests still pass):

| Finding | Action taken | Verification |
|---|---|---|
| **HIGH-1** | Re-confirmed VAE+Dirichlet is reachable (no guard existed). Added a `ValueError` guard in `DeepLatent.__init__` that blocks `ae_type="vae"` + Dirichlet and points to WAE/`logistic_normal`. | guard raises; WAE+Dirichlet still constructs |
| **MED-1** | Learnable `LogisticNormalPrior`/`GaussianPrior` Cholesky factor now zero-initialized → `Σ_init ≈ I`. | `Σ_init` diag = 1.0002, off-diag 0 |
| **MED-3** | `EncoderMLP.forward` no longer applies dropout/activation to the final `(μ, logvar)` layer. Decoder/Predictor left as-is — see revised note (user-configurable, default 0). | output zero-fraction 0.0 at dropout=0.9; hidden dropout still active |
| **LOW-1** | `visualize_docs`/`visualize_words` now use `if/elif/else` and raise on unknown methods instead of silently overriding UMAP with PCA. | logic corrected |

---

## Severity legend

| Level | Meaning |
|---|---|
| **HIGH** | Produces incorrect results in a documented, reachable configuration |
| **MED** | Incorrect/biased under specific options, or silently degrades training quality |
| **LOW** | Robustness, performance, or documentation defects |

---

## HIGH-1 — VAE + Dirichlet prior: KL is computed in the wrong space

**Where:** `priors.py:326–341` (`DirichletPrior.get_prior_params`), `priors.py:764–790` (`FixedDirichletPrior.get_prior_params`); consumed at `models.py:851–859` (mean-field), and the analogous full-rank/iaf branches.

**The math.** The encoder posterior is Gaussian in **logit space**: the encoder emits `(μ_q, logvar_q)`, samples `z = μ_q + ε·σ_q`, and only then maps `θ = softmax(z)` (`autoencoders.py:1000`, `1227`). So the KL term must be `KL(q(z) ‖ p(z))` with **both** distributions defined over `z ∈ ℝ^K`.

For the logistic-normal prior this is exactly what happens (its `get_prior_params` returns the logit-space Gaussian `(μ, diag Σ)`), and it is correct. For the **Dirichlet** prior, `get_prior_params` instead returns

```
mean   = α / Σα           # a point ON the simplex (sums to 1)
logvar = log Var_Dir(θ)   # Dirichlet marginal variance, also a simplex quantity
```

and `models.py:851–859` feeds these into the **Gaussian** KL `0.5·Σ(logvar_p − logvar_q − 1 + var_q/var_p + (μ_q−μ_p)²/var_p)`. A simplex mean (`0.2` for `K=5`) and a simplex variance (`≈0.027`) are being used as the **mean and variance of a Gaussian over logits**. These are incommensurable objects.

**Evidence (numerical).** With an *uninformative* logit posterior `q = N(0, I)` — which should sit close to any sane uninformative prior — the implemented KL explodes:

```
Dir(α=1.0, K=5): prior 'mean' fed to Gaussian-KL = [0.2 0.2 0.2 0.2 0.2]  (sums to 1.0)
                 prior 'var'  fed to Gaussian-KL = [0.0267 ...]
   KL( N(0,I)_logit || 'Dirichlet' ) = 85.94      <-- should be small
Dir(α=0.1, K=5): KL( N(0,I)_logit || 'Dirichlet' ) = 16.28
```

**Evidence (training).** A 400-step GTM-VAE on simulated text (`K=4`), same data, only the prior changed:

```
prior=logistic_normal   KL=1.85   recon=488.1   mean max-topic share=0.656   eff #topics=2.52
prior=dirichlet         KL=0.57   recon=491.1   mean max-topic share=0.368   eff #topics=3.77
```

The Dirichlet run drives `θ` toward uniform (3.77 of 4 effective topics, top-share ≈ 0.37) — the mis-specified prior pins the logit mean at `1/K` with tiny variance, which is precisely a push toward the uniform simplex point. The logistic-normal run retains usable concentration.

**Re-check — is VAE+Dirichlet even reachable?** Yes. The only related assertion is `GTM` at `models.py:2421`, which checks the prior *name* (`{"dirichlet","logistic_normal"}`) independent of `ae_type`. With `ae_type="vae"` the constructor builds a real `FixedDirichletPrior` (`models.py:309`) or `DirichletPrior` (`models.py:328`) and the broken KL runs. There was **no guard** — confirmed by construction (it trained without error in the run above).

**Why it usually "works anyway":** the package default is `ae_type="wae"`, and **WAE+Dirichlet is fine** — it never calls `get_prior_params`; it draws real Dirichlet samples (`priors.py:292`) and compares them to `θ` with MMD (`models.py:731–735`). The bug is confined to `ae_type="vae"` with `latent_factor_prior="dirichlet"`.

**Resolution (applied).** Added a guard in `DeepLatent.__init__` that raises `ValueError` for `ae_type="vae"` + Dirichlet, directing users to WAE (with the Dirichlet) or to the logistic-normal prior (with a VAE). WAE+Dirichlet and VAE+logistic-normal are unaffected. Verified: the VAE+Dirichlet constructor now raises; WAE+Dirichlet still constructs.

**Alternative (not applied).** If a genuine Dirichlet-VAE is ever wanted, implement the Laplace-bridge approximation in softmax-logit space (Srivastava & Sutton, *Autoencoding Variational Inference for Topic Models*, 2017) inside `get_prior_params` (returning logit-space `(μ, logvar)`):

```
μ_p,k  = log α_k − (1/K) Σ_j log α_j
σ²_p,k = (1/α_k)(1 − 2/K) + (1/K²) Σ_j (1/α_j)
```

For symmetric `Dir(1, K=5)` this gives `μ_p = 0`, `σ²_p = 0.8` — a sane logit-space prior — and the existing KL code becomes correct.

---

## MED-1 — Learnable Gaussian/logistic-normal prior initializes Σ to e²·I, not I

**Where:** `priors.py:109–110` & `priors.py:130` (`LogisticNormalPrior`); `priors.py:400–401` & `priors.py:421` (`GaussianPrior`).

`L_flat` is initialized from `torch.eye(...)`, so the Cholesky **diagonal entries are 1.0**. But `sigma` then exponentiates the diagonal: `L[diag] = exp(L[diag]) + 1e-4`. With diagonal = 1, that is `exp(1) ≈ 2.718`, so

```
Σ_init diagonal = [7.389 7.389 7.389 7.389 7.389]   # measured
exp(1)^2 = 7.389      # docstring/intent: identity (1.0)
```

The docstring says *"Initialize as identity matrix"*; the actual initial prior covariance is **≈7.4× too wide**. This only affects `update_prior=True` (the default `GTM`/`IdealPointNN` use the *fixed* priors, which are correctly `N(0, I)`), but with the prior LR at `1e-4` the prior stays over-dispersed for many steps, weakening the KL early in training — the opposite of what you want when fighting posterior collapse.

**Resolution (applied).** `L_flat` is now zero-initialized in both `LogisticNormalPrior` and `GaussianPrior`. Because `sigma` computes `L_ii = exp(L_flat_ii) + 1e-4`, a zero init gives `L_ii = 1 + 1e-4` with zero off-diagonals, so `Σ_init = L Lᵀ ≈ I`. Verified: `Σ_init` diagonal = `1.0002`, off-diagonal `0`.

---

## MED-2 — Validation/early-stopping uses only the first test batch

**Where:** `models.py:518–522`.

```python
test_iter = iter(test_data_loader)         # re-created every step
test_data_batch = next(test_iter)          # always the SAME first batch (shuffle=False)
validation_loss = self.step_batch(test_data_batch, test_data, validation=True)
```

The validation iterator is rebuilt each step and only its first batch is consumed. Because the test loader uses `shuffle=False`, **every validation evaluation scores the same single batch.** That value drives `best_loss`, best-model checkpointing, and early stopping (`models.py:531–553`). Consequences: noisy/biased model selection, and the "best model" reflects one batch rather than the validation set.

**Fix.** Average `step_batch(..., validation=True)` over the full `test_data_loader` (or a fixed held-out subset), e.g. accumulate loss × batch-size and divide by N. Optionally validate every `k` steps rather than every step.

---

## MED-3 — Encoder applies dropout to its `(μ, logvar)` output layer

**Where:** `autoencoders.py:399–405` (`EncoderMLP.forward`). Also present, **by design**, in `DecoderMLP.forward` and `Predictor.forward` — see the note below.

`EncoderMLP.forward` looped as `hid = self.dropout(layer(hid))` for **every** layer, including the final one, so with `dropout>0` the encoder's **`μ`/`logvar` outputs were randomly zeroed** at train time (a zeroed `logvar` unit injects `σ²=1` at random; a zeroed `μ` unit biases the posterior mean toward 0). Unlike a hidden activation, these are *distribution parameters* — dropping them corrupts the variational posterior, not just a feature map.

**Evidence:** with `dropout=0.5`, the train-mode mean of `EncoderMLP([10,8,10])(x)` over 200 passes differed from the eval-mode output. A no-op only because the default is `dropout=0.0`.

**Resolution (applied).** `EncoderMLP.forward` now applies dropout + activation to hidden layers only; the final `(μ, logvar)`/Cholesky/latent-code projection is a bare linear map. Verified: at `dropout=0.9` the output has a zero-fraction of `0.0` (no output units dropped) while hidden-layer dropout remains active.

**Re-check — decoder & predictor (left as-is, per design).** Agreed: these are different. Their dropout is opt-in via the per-modality/per-label config dicts (`decoder_args`, `predictor_args`), defaults to `0.0`, and the outputs there are *predictions/reconstructions*, not posterior parameters — so a user who dials in dropout is making a deliberate regularization choice. They are intentionally **not** changed. (One thing to keep in mind: with the current loop, a configured decoder/predictor `dropout` also regularizes the output layer, not just hidden layers; that's a defensible-but-slightly-unconventional interpretation of the knob. Flagging for awareness only — no change made.)

---

## MED-4 — `get_mutual_information` (and IAF free-bits) treat the flow KL as the base KL

**Where:** `models.py:2009–2027`, docstring `models.py:1812`; related approximation at `models.py:795–819`.

The docstring states the IAF KL is computed *"via the base distribution (flow transformations cancel in expectation)."* They do **not** cancel: `q_K(z_K)` is non-Gaussian, so `KL(q_K ‖ p) ≠ KL(q_0 ‖ p)` — the missing `−E[log|det J|]` term is exactly what the flow adds. The reported pointwise MI for `vi_type="iaf"` is therefore a biased approximation (it equals the base-Gaussian KL). Similarly, the per-dimension free-bits path for IAF distributes `log_det_j` uniformly across dims (`models.py:813–816`), which is acknowledged in-code as an approximation.

**Impact is limited:** the **training** objective for IAF is correct (it uses the proper MC KL `log q_0(z_0) − log|det J| − log p(z_K)` at `models.py:760–792`). Only the `get_mutual_information` diagnostic and the IAF free-bits split are approximate.

**Fix.** Either (a) compute the IAF diagnostic MI with the same MC estimator used in training, or (b) soften the docstring to state the MI is a base-distribution approximation for flows.

---

## LOW-1 — `visualize_docs` / `visualize_words`: UMAP silently replaced by PCA

**Where:** `models.py:2809–2814` and `models.py:2894–2899`; column at `models.py:2818`.

```python
if dimension_reduction == "umap":
    ModelLowDim = UMAP(...)
if dimension_reduction == "tsne":      # NOTE: separate `if`, with an `else`
    ModelLowDim = TSNE(...)
else:
    ModelLowDim = PCA(...)             # runs whenever method != "tsne"
EmbeddingsLowDim = ModelLowDim.fit_transform(matrix)   # single fit, last assignment wins
```

**Re-check — is this a PCA→UMAP pipeline?** No. There is a single `fit_transform` and `ModelLowDim` is simply reassigned; nothing chains PCA into UMAP. The exact behavior is:

| `dimension_reduction` | model actually used |
|---|---|
| `"umap"` | first `if` sets UMAP, then `else` (because `"umap" != "tsne"`) **overwrites with PCA** → **PCA** |
| `"tsne"` | **TSNE** |
| `"pca"` / anything else | **PCA** |

So requesting `"umap"` silently yields a PCA embedding; UMAP is unreachable. This is a silent override, not a two-stage reducer.

**Resolution (applied).** Both functions now use `if/elif/else` and raise `ValueError` on an unknown method, so `"umap"` runs UMAP and bad inputs fail loudly instead of silently falling back to PCA. (If a genuine PCA→UMAP *pipeline* is desired, that should be wired explicitly — say so and I'll add it.)

**Still open (not changed):** `models.py:2818` hardcodes `dataset.df["doc_clean"]`, which raises `KeyError` for corpora whose text column differs (the bundled simulator produces `doc_clean_0`). Recommend deriving the column from `dataset.modalities_config`.

---

## LOW-2 — Per-step prior sampling + MMD is computed and discarded for VAEs

**Where:** `models.py:729–735`.

The `mmd_loss` block (which calls `self.prior.sample(...)` and `compute_mmd_loss`) runs for **both** WAE and VAE, but for VAE the divergence is `β·KL` (`models.py:939`) and `mmd_loss` is thrown away. For a Dirichlet prior, `sample()` is a **Python `for`-loop over the batch** (`priors.py:320–322`), so every VAE step pays for a discarded, slow sample.

**Fix.** Guard the MMD/sample block with `if self.ae_type == "wae":`. Separately, vectorize Dirichlet sampling — `torch.distributions.Dirichlet(concentration).sample()` already supports batched `concentration`, so the loop in `priors.py:319–322` and `priors.py:758–761` is unnecessary.

---

## LOW-3 — Fixed, implicit observation noise couples reconstruction to KL arbitrarily

**Where:** `models.py:685` (image), `models.py:709` (embedding).

`F.mse_loss(recon, target, reduction="sum") / B` is the Gaussian NLL up to a constant **only for a fixed variance** — implicitly `σ²=0.5` (since `Σ(x−μ)² = 2 · [½Σ(x−μ)²/σ²]` at `σ²=0.5`). The recon-vs-KL balance therefore hinges on an unstated, un-tunable decoder variance. For high-dimensional images this makes the summed pixel error dwarf the KL, so `w_prior=1` is no longer a calibrated ELBO for the image modality. This is a modeling subtlety rather than an outright bug, but it deserves a knob (learnable or configurable `σ²`, or a `w_prior` that the user is told to scale per modality).

---

## LOW-4 — Smaller items

| # | Where | Issue |
|---|---|---|
| a | `models.py:94`, `:740` | Docstring claims `w_prior=None` auto-selects the weight; no such logic exists — `w_prior=None` makes `beta=None` and `divergence_loss = None*kl` raises `TypeError`. |
| b | `models.py:742–745` | If `kl_annealing_start == kl_annealing_end` (a non-default but legal setting), `span=0` → division by zero in the annealing branch. |
| c | `corpus.py:70–78`, used at `models.py:627–628` | Missing votes are encoded as the literal value `2.0` in the **encoder input** (only the reconstruction is masked). Missingness thus leaks into the posterior as a distinct input level rather than being masked/imputed. |
| d | `models.py:2192–2199` | `generate_samples` for BoW draws a single word per document (documented "simplified"); not a faithful multinomial draw. |
| e | `corpus.py:128` | When `prevalence` is `None`, `M_prevalence_covariates` defaults to a column of **zeros** `np.zeros((N,1))` (not ones); harmless because `prevalence_covariate_size` is then forced to 0 (`models.py:151–153`), but the array is misleading. |
| f | repo hygiene | `old/`, `src2/`, committed `dist/*.whl`/`*.tar.gz`, and a vendored `.conda/` tree are checked in; `tests/old_tests/` is stale. These bloat the repo and confuse `setuptools.find`. |

---

## Verified correct (high confidence)

These were explicitly checked and are **right** — worth recording so they aren't "fixed" by mistake:

1. **Default mean-field KL** (fixed logistic-normal / Gaussian prior): matches `torch.distributions.kl_divergence` to `4.8e-7` (`models.py:856–858`).
2. **Corrected PoE** (`autoencoders.py:734–782`): equals the exact Gaussian posterior-combination identity `Λ_S = ΣΛ_m − (M−1)Λ_0`, `η_S = Ση_m − (M−1)η_0`; measured mean & precision error `= 0.0`. The encoder's precision-increment parameterization `Λ_m = Λ_0 + Δ_m` (`autoencoders.py:1020–1026`) correctly guarantees each modality only *adds* information.
3. **IAF** (`autoencoders.py:570–624`): MADE masks enforce strict autoregressivity; `log|det J| = Σ log σ` matches autograd (`-0.2878094` vs `-0.2878095`). The training KL `log q_0(z_0) − log|det J| − log p(z_K)` (`models.py:760–792`) is a correct single-sample MC ELBO.
4. **Posterior-collapse fix** (`models.py:701–716`): BoW/vote/embedding/image reconstructions are per-document NLLs (sum over features ÷ batch), matching the per-document KL, so `w_prior=1` is the true negative ELBO. `discrete_choice` (mean-over-batch CE summed over questions) is also on the per-document scale. Confirmed by the three passing tests.
5. **Removal of labels-from-encoder** (diff vs `main`): eliminates target leakage into `q(z|x)` — a correct and important change for valid supervised inference.
6. **full-rank KL** (`models.py:861–916`): correct trace/quadratic/log-det structure for `KL(N(μ,LLᵀ) ‖ prior)`.

---

## Prioritized recommendations

1. **HIGH-1 (done)** — VAE+Dirichlet is now guarded. If you later want a real Dirichlet-VAE, implement the Laplace bridge instead of the guard.
2. **Fix MED-2** (validate over the whole test set) — still open; directly affects which checkpoint is returned as "best".
3. **MED-1 / MED-3 (done)** — prior Σ init and encoder output-dropout fixed.
4. **Clarify MED-4 / LOW-3 / LOW-4a** in docstrings (flow-MI approximation, implicit decoder variance, `w_prior=None`).
5. **Performance:** gate the MMD/sample block to WAE only and vectorize Dirichlet sampling (LOW-2).
6. **Hygiene:** the hardcoded `df["doc_clean"]` in `visualize_docs` (LOW-1 tail); drop `old/`, `src2/`, `dist/`, vendored `.conda/`; expand `tests/` to cover each prior×`ae_type`×`vi_type`×fusion path (a single `vae`+`dirichlet` sanity test would have caught HIGH-1).

---

### Appendix — how to reproduce

The numerical checks and the Dirichlet-vs-logistic-normal training comparison live in two standalone scripts, runnable inside the `deeplatent` env:

```bash
conda activate deeplatent
python audit/verify_math.py                 # findings HIGH-1, MED-1, MED-3, PoE, IAF, default-KL
python audit/verify_poe_and_dirichlet.py    # corrected-PoE exactness + Dirichlet-VAE training pathology
```

They exercise the real package classes (`LogisticNormalPrior`, `FixedDirichletPrior`, `MultiModalEncoder.product_of_experts`, `IAF`, `EncoderMLP`, and a full `GTM` train loop) and print the figures quoted above. All three shipped tests (`tests/test_recon_reduction.py`) pass.
