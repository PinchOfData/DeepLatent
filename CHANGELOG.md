# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-02

### Changed
- **Logistic-normal latents are now (K-1)-dimensional contrast coordinates** (STM/CTM convention), replacing the K-dimensional parameterization. Softmax is invariant to shifts along the all-ones logit direction, so the K-dim latent carried K parameter directions per configuration that the likelihood cannot identify (prior mean level, variance along the all-ones direction, and its cross-covariances); these gauge directions demonstrably drift during training and contaminate prior-covariance diagnostics. The latent is now `eta` in `R^{K-1}` with `theta = softmax(eta @ V.T)`, where `V = contrast_basis(K)` is a fixed orthonormal zero-sum (Helmert) basis. **The distributional family on the simplex is exactly unchanged** (`softmax(z) == softmax(V V^T z)` for every `z`; means and covariances map bijectively through `V`), and with orthonormal `V` a standard-normal prior in K dims corresponds exactly to a standard-normal prior in K-1 dims. ELBO/marginal-likelihood values remain directly comparable. Gaussian (`IdealPointNN`) and Dirichlet priors are untouched (no softmax gauge).
- `get_latent_factors(to_simplex=False)` now returns `[N, K-1]` contrast coordinates for logistic-normal models.
- `prior.get_prior_params` / `prior.sigma` return (K-1)-dimensional means/covariances for logistic-normal priors.
- `get_predictions(to_simplex=False)` now raises a clear `ValueError` for simplex priors (the predictor is trained on theta; feeding the raw latent was silently wrong before and is a shape mismatch now).

### Added
- `contrast_basis(n_topics)` in `deeplatent.utils` (exported): the fixed orthonormal zero-sum basis.
- `model.latent_to_theta(z)` / `encoder.latent_to_theta(z)`: the canonical latent-to-simplex map. Use this instead of hand-rolling `F.softmax(z, dim=1)` on raw latents.
- `model.get_prevalence_coefficients()` / `prior.get_prevalence_coefficients()`: centered per-topic prevalence coefficients in theta-logit space (`V @ mean_net.weight`, bias folded into the Intercept column) — the unique centered representative, directly comparable to centered ground-truth coefficients. Replaces hand-rolled `prior.mean_net.weight` reads.

### Migration Guide
- **Pre-0.2.0 logistic-normal checkpoints cannot be loaded** (`load_model` raises with an actionable message). Re-train with >=0.2.0, or check out the pre-0.2.0 commit / pin `deeplatent==0.1.3` to read old checkpoints. Gaussian/Dirichlet checkpoints are unaffected.
```python
# Old (pre-0.2.0): hand-rolled latent softmax and coefficient reads
theta = F.softmax(z, dim=1)                      # WRONG in >=0.2.0 (z is [N, K-1])
W = model.prior.mean_net.weight.cpu().numpy()    # WRONG shape in >=0.2.0 ([K-1, C])

# New (>=0.2.0)
theta = model.latent_to_theta(z)                 # [N, K-1] -> [N, K] simplex
W = model.get_prevalence_coefficients()          # [K, C], centered, bias folded
```

## [0.1.3] - 2025-01-23

### Added
- **Multi-label prediction support**: The `labels` parameter in `Corpus` now accepts a dictionary mapping label names to their configuration, enabling multiple prediction targets with different types (regression, binary, multiclass) in a single model.
- New `MultiLabelPredictor` class that creates independent MLP networks per label.
- Per-label predictor configuration via `predictor_args`, allowing different architectures and loss weights for each label.
- `get_predictions()` now returns a dictionary mapping label names to their predictions.
- Added `plot_training_loss()` method to visualize training and validation loss curves with optional smoothing. (Alex Pin)
- Improved `get_top_docs()` function for better document retrieval. (Alex Pin)

### Changed
- Removed `predictor_type` parameter from model initialization (now specified per-label in labels config).
- Updated notebook `04_gtm_with_metadata_simulation.ipynb` to use the new multi-label API.

### Migration Guide
```python
# Old API (v0.1.2)
corpus = Corpus(df, modalities=modalities, labels="~label-1")
model = GTM(corpus, predictor_type="regressor", predictor_args={"label": {"hidden_dims": []}})
predictions = model.get_predictions(corpus)  # Returns array

# New API (v0.1.3)
corpus = Corpus(df, modalities=modalities, labels={"label": {"column": "label", "type": "regression"}})
model = GTM(corpus, predictor_args={"label": {"hidden_dims": [], "loss_weight": 1.0}})
predictions = model.get_predictions(corpus)  # Returns {"label": array}
```

## [0.1.2] - 2025-11-27

- Corrected a BUG for vi_type == IAF.
- For votes, IdealPointNN() replaces NaN values with the value 2, so the encoder can learn what missing values mean for the posterior. The posterior is still only used to reconstruct observed voting patterns (not missing values which are masked). To reconstruct missing values, users should rely on discrete_choice instead. 
- Implemented print_topics as an argument if users want to see how topics evolve over training steps.
- Harmonized fusion strategies with normalizing flows. The flow is now always applied post-fusion when multiple modalities are present in the data.
- Added additional (Claude-generated) documentation for each method.

## [0.1.1] - 2025-11-20

- Added support for a new fusion strategy, "corrected_poe", that corrects some shortcomings of the naive PoE used in the computer science literature. Corrected PoE notably ensures that the encoder class can, in principle, contain the true posterior (this is not the case for a mixture of experts or a naive PoE).
- Added a method get_mutual_information() that answers the following question: "How much does each modality move us away from the prior?"

## [0.1.0] - 2025-10-30

- Original public release.