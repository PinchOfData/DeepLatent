# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepLatent is a Python package and an accompanying econometrics paper, **"Deep Latent Factor Models for the Social Sciences"** (Gauthier, Widmer, Ash — draft in `papers/paper2.txt`). The package estimates latent factor models (topic models `GTM`, ideal point models `IdealPointNN`) for structured and unstructured, possibly multimodal data via **amortized variational inference**: an extended VAE that maximizes a lower bound (ELBO) on the intractable marginal likelihood.

The design principle is **shallow decoders, deep encoders**. Decoders stay linear/interpretable (the measurement model of a classical latent factor model); the encoder is a deep neural network whose only job is inference. The paper's theory gives conditions under which the ELBO is tight so that maximizing it is equivalent to exact maximum likelihood: the total inference gap decomposes into a *variational gap* (closed by an expressive variational family — a mixture of Gaussians is a universal approximator of the posterior) and an *amortization gap* (closed by a deep encoder that universally approximates the map from data to optimal variational parameters). In the linear–Gaussian special case the exact posterior is in the family, and ELBO maximization inherits consistency and a √n CLT as an M-estimator.

### The headline empirical claim (joint estimation de-biases regression on latent factors)

Battaglia, Christensen, Hansen & Sacher (2025), `papers/reg_unstruct.pdf`, prove that the standard **two-step strategy** — estimate latent factors θ̂ᵢ from unstructured data, then regress Y on θ̂ᵢ — has a first-order asymptotic *bias* (location shift, not variance inflation) whenever measurement error is comparable to sampling error (their κ > 0; for topic models κ = lim √n·E[1/Cᵢ], with Cᵢ the document length). Their fix is **joint likelihood estimation** of the factor model and the regression, but they implement it with HMC/NumPyro, which does not scale.

DeepLatent's supervised mode **is** joint likelihood estimation — the outcome likelihood p(y|z) enters the ELBO alongside the modality likelihoods — and it scales via amortization and minibatch SGD. The research program in `audit/` demonstrates: (i) the bound is tight in practice (measured total gap ≈ 0.08 nats/doc, IWAE-verified), so the ELBO maximizer ≈ the marginal MLE; (ii) in Monte Carlos the joint estimator recovers topic→outcome coefficients unbiasedly (30-rep MC: E[c] = 1.00 at the calibrated step budget) while the two-step baseline is stably attenuated (E[c] ≈ 0.84); (iii) this delivers the scalable joint solution the reg_unstruct paper calls for.

## Commands

```bash
# Install from source (development). Required: the env expects an editable install
pip install -e .

# Run tests (NOTE: tests/* is in .gitignore — new test files are untracked by default)
pytest tests/

# Run a single test
pytest tests/test_recon_reduction.py::test_name

# Format / lint
black deeplatent/        # 88-char lines
flake8 deeplatent/
```

Environment notes (this machine — WSL2, small GPU):
- **Heavy experiments run on the Bocconi HPC — see `HPC.md`** (SSH aliases, SLURM
  partitions, the `deeplatent` conda env, job-array patterns, rsync workflow).
  Local WSL is for development and quick smoke tests only. Always use SLURM on the
  cluster — never run Python on the login node.
- 2 GB GPU: run at most one torch process at a time.
- DataLoader `num_workers=0` for Monte Carlo loops. `num_workers>0` combined with frequent iterator re-creation (small datasets → few batches/epoch → `train()` respawns the iterator constantly) exhausts WSL shared memory and crashes the VM. If workers are ever needed, use `persistent_workers=True`.

## Architecture

### Core Entry Points

- **`deeplatent/models.py`** — `DeepLatent` base class; `GTM` (topic models) and `IdealPointNN` (ideal point models) inherit from it. The constructor wires everything and calls `self.train()`. All training logic is in `DeepLatent.train()` / `step_batch()`.

- **`deeplatent/corpus.py`** — `Corpus` wraps DataFrames into PyTorch datasets with multimodal support. Modality dict keys use `"<modality>_<view>"` naming, parsed by `utils.parse_modality_view`.

### Model Components

- **`autoencoders.py`** — Encoder/decoder networks. `MultiModalEncoder` handles fusion of modalities and implements the variational families (incl. the mixture-of-Gaussians head and IAF flows). `ImageEncoder`/`ImageDecoder` are CNNs with FiLM conditioning. Decoders default to linear (`activation=None`, `bias=False` from models.py).

- **`priors.py`** — Learnable priors (`LogisticNormalPrior`, `DirichletPrior`, `GaussianPrior`) and fixed counterparts (`FixedLogisticNormalPrior`, ...). Priors implement `sample()`, `simulate()`, `get_prior_params()`. `LogisticNormalPrior(learn_cov=...)` controls whether Σ is learned or pinned at I; its `mean_net` is the prevalence-covariate → prior-mean map (the "structural" prevalence regression).

- **Contrast parameterization (v0.2.0, structural)**: logistic-normal latents are **(K−1)-dimensional contrast coordinates** η with θ = softmax(η @ V.T), V = `contrast_basis(K)` (orthonormal Helmert basis of the zero-sum subspace). Softmax is invariant along the all-ones logit direction, so the old K-dim latent carried K exactly-unidentified prior directions that drifted during training (measured gauge collapse). The distributional family on the simplex is *identical* — only the gauge is removed. Consequences: `model.n_latent = K−1` (vs `n_factors = K`); `get_latent_factors(to_simplex=False)` returns [N, K−1]; `get_prior_params`/`prior.sigma` are (K−1)-dim; **never hand-roll `F.softmax(z, dim=1)` on raw latents — use `model.latent_to_theta(z)`**; read prevalence coefficients with `model.get_prevalence_coefficients()` (lifted `V @ W`, centered, bias folded), not `prior.mean_net.weight`; Σ̂ topic-permutations require lifting first (`V @ Σ @ V.T`). Pre-0.2.0 logistic-normal checkpoints cannot be loaded (`load_model` raises). Gaussian (`IdealPointNN`) and Dirichlet latents are unaffected (no softmax gauge).

- **`predictors.py`** — `Predictor` and `MultiLabelPredictor` supervised heads. Regression labels carry a **learned observation noise** `noise_log_var` (per-label), making the prediction term a proper Gaussian log-likelihood log p(y|z) rather than bare MSE — required for the joint-estimation interpretation and for the coefficient scale to self-calibrate.

- **`simulations.py`** — synthetic DGPs (`generate_documents` with prevalence covariates, outcome labels, and `anchor_words` for topic separability; `generate_ideal_points`; multilingual docs).

### Data Flow

```
DataFrame → Corpus (multimodal preprocessing) → DataLoader → Model.train() → Checkpoints
```

### The loss (step_batch)

`loss = reconstruction + divergence + w_pred_loss * prediction`

- **Reconstruction**: per-document NLL (summed over tokens/features, divided by batch size). This per-document scaling is deliberate and load-bearing — see "Posterior collapse" below.
- **Divergence**: `vae` → β·KL(q‖prior) with closed-form/MC branches per `vi_type` (and KL annealing / free-bits options); `wae` → MMD·w_prior; `ae` → 0. `w_prior=1` with `ae_type="vae"` is the calibrated, true ELBO.
- **Prediction**: per-label Gaussian NLL with learned σ² (regression), BCE (binary), CE (multiclass). With this term active, training is joint likelihood estimation.

Diagnostics: `estimate_marginal_log_likelihood()` (IWAE — measures bound tightness), `get_mutual_information()`, `get_latent_factors(num_samples=..., return_std=...)` (posterior readout/refinement).

## Key Configuration Options

- **`ae_type`**: `"vae"` (variational — use this for likelihood-based work), `"wae"` (Wasserstein/MMD), `"ae"` (plain). VAE+Dirichlet prior is intentionally blocked (KL would be computed in the wrong space).
- **`vi_type`**: `"mean_field"`, `"full_rank"`, `"iaf"`, **`"mixture_of_gaussians"`** (universal variational family; `mixture_components`, default 10). MoG supports exact `corrected_poe` fusion through the `C^M` Cartesian component expansion (with a runtime explosion warning); uncorrected PoE is blocked.
- **`fusion`**: `"poe"`, `"corrected_poe"` (exact in the linear–Gaussian case: divides out duplicated prior factors), `"moe_gating"`, `"moe_average"`, `"moe_learned"`.
- **`labels_in_encoder`** (default False): concatenates outcomes y to encoder inputs so q(z|x,y) can target the supervised posterior p(z|x,y). True y at training, zeros at inference. This is the switch that makes supervised training genuinely joint.
- **`learn_prior_cov`** (default True): learn the logistic-normal prior Σ vs pin Σ=I. See "Temperature degeneracy" below before changing it.
- **`update_prior`**: learn the prior (incl. `mean_net` on prevalence covariates) vs keep it fixed. The two-step baseline convention is `update_prior=False` + fixed standard logistic-normal prior.
- **`w_prior`** (default 1): divergence weight. 1 = true ELBO (post posterior-collapse fix). `w_pred_loss` (default 1): weight on the outcome likelihood — keep at 1 and let learned σ² set the effective weight instead of hand-tuning.
- Optimizer uses separate param groups: main (lr 1e-3), prior (lr 1e-4, weight_decay 0 — wd shrinks prevalence coefficients), predictor. Override via `optim_args`.

### Modality Types Supported

- `bow` — bag-of-words (multinomial likelihood)
- `embedding` — pre-computed embeddings (Gaussian)
- `image` — raw images with lazy loading
- `vote` — voting records (missing values masked)
- `discrete_choice` — categorical responses

### Metadata Types

- `prevalence` — covariates shifting the prior over latent factors (x^p in the paper)
- `content` — covariates shifting the decoders (x^c)
- `labels` — outcome variables for supervised/joint estimation (y)
- `prediction` — additional predictors for labels (x^s)

## Established empirical findings (do not re-derive; see audit/ and root docs)

These were measured in this repo; scripts and JSON results live in `audit/`.

1. **Posterior collapse was a scale bug, not a modeling problem** (`experiments/POSTERIOR_COLLAPSE.md`): recon was per-token while KL was per-document → effective β≈document-length → collapse at w_prior=1. Fixed by per-document reconstruction (unconditional). Do not reintroduce per-token scaling.
2. **The bound is tight** (`audit/experiment_gap_decomp.py`, `experiment_supervised_gap.py`): total gap ≈ 0.08 nats/doc (~0.2% of |ELBO|); amortization share ~20%; IWAE plateaus; per-document refinement doesn't move coefficients. More encoder capacity / more MoG components do NOT help — there is no gap left to close.
3. **Temperature degeneracy of the logistic-normal** (`audit/experiment_scale_id.py`): with short documents the multinomial barely pins the absolute logit scale, so a learned prior Σ and the coefficients slide along a ridge (W→sW, Σ→s²Σ), producing a *single global scale* drift in recovered coefficients (corr with truth stays ≈0.999). Pinning Σ=I fixes prevalence-coefficient recovery but biases the outcome-coefficient scale — with realistic outcome noise (σ=1), learned Σ + learned σ² is the unbiased configuration.
4. **U-shaped coefficient bias in training time** (`audit/experiment_mc_calibrate.py`): under-trained → attenuated (c<1); over-trained → prior overfits and coefficients overshoot (c>1). The sweet spot (c≈1) is **invisible to every train/val loss** (the ELBO is ~blind to the coefficient scale) — no early-stopping criterion finds it. Fix `num_steps` at a value calibrated on a matched simulation and transfer it.
5. **Joint vs two-step, clean 30-rep Monte Carlo** (`audit/run_mc_30.sh` → `audit/results_mc_30.json`, N=10k, σ=1, anchor words): joint E[c]=1.000 at the calibrated 8k steps; two-step stably attenuated E[c]≈0.84 at every checkpoint (classic errors-in-variables). Residual caveat: topics with near-zero/adjacent effects can bleed into each other at small N (affects both estimators equally; anchor words mitigate).
6. **Prevalence coefficients** (`PREVALENCE_RECOVERY.md`): read them via two-step OLS on posterior factors or via the joint prior `mean_net` — but never with weight decay on the prior optimizer (shrinks them); wd=0 is now the default.
7. **Variational families** (`VARIATIONAL_FAMILIES.md`): on held-out US-Congress IWAE, MoG beats mean-field by ≈3.6 nats/doc while full-rank/IAF underperform — expressiveness must be on the multimodality axis, not the correlation axis.
8. **Code audit** (`AUDIT_REPORT.md`): core math verified correct. Still open: validation loss uses only the first test batch (MED-2); IAF MI diagnostic is approximate (MED-4); implicit fixed decoder σ²=0.5 for Gaussian modalities (LOW-3).

## Repository Layout

- `deeplatent/` — the installed package (only this is packaged).
- `papers/` — paper drafts (`paper2.txt` is current; `outline.txt` pitch; `slides.txt`) and the literature PDFs, incl. `reg_unstruct.pdf` (Battaglia et al. 2025 — the two-step-bias paper we answer).
- `audit/` — verification scripts + Monte Carlo experiments + results JSON. Each script has a docstring; results land in sibling `results_*.json`.
- `experiments/` — posterior-collapse experiments and writeup.
- `tests/` — pytest (note: gitignored; `git add -f` new tests).
- `figures/`, `tables/`, `logs/` — generated outputs. `notebooks/` — tutorials 01–07.
- `old/`, `src2/`, `dist/`, `.conda/` — legacy/build artifacts, slated for removal; do not build on them.

## Code Style

- `black` with 88-character lines; target Python 3.8+.
- Docstrings often encode empirical rationale (e.g., the prior learning-rate choice in `models.py`) — preserve them when editing.
