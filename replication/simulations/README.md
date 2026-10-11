# Simulations: likelihoods, multimodality, and covariates in the prior

Replication code for the paper's simulation figures. All three use a one-dimensional latent.

## Where things live

- **Code** (this folder): `code/` (numbered steps), `slurm/` (cluster jobs), `sync_to_hpc.sh`,
  `fetch_results.sh`.
- **Cluster** (`/home/Gauthier/simulations/`): the synced code, `pkg/deeplatent` (the copy of
  the package the jobs import; every job aborts unless it imports this copy and logs the md5 of
  each module), `results/`, `logs/`.
- **Laptop**: `results/` and `logs/` are fetched back with `fetch_results.sh`; `04_figures.py`
  draws the figures from `results/` into `figures/`, so their appearance can be changed without
  refitting anything.

## Figures

**Model (Figures 1 and 2).** `IdealPointNN` in its likelihood-based configuration: VAE with the
true ELBO (`w_prior = 1`), mean-field Gaussian posterior, fixed N(0, 1) prior, corrected
product-of-experts fusion, encoders with two hidden layers of 64 units, linear decoders with
intercepts, batch size 256, Adam at the package defaults, 30,000 steps (statistics also saved at
10,000 and 20,000). The N(0, 1) prior pins the latent's location and scale, so estimates are
aligned with the truth by sign only. Intervals are the posterior mean ± 1.96 posterior sd,
estimated from 2,000 posterior draws per unit; coverage is the share of units whose interval
contains the true value.

**Exact references.** For each unit, the exact posterior of θ given its data under the true item
parameters, computed on a grid (−7 to 7, step 0.01), and its best Gaussian approximation in
reverse KL (the target of a Gaussian variational family).

1. **One likelihood at a time** (`01_likelihoods.py`, `fig1_likelihoods`). N = 10,000 units,
   θ ~ N(0, 1), one model per likelihood, each design tuned to an exact posterior precision of
   about 11 (posterior sd ≈ 0.3):
   - continuous: 50 Gaussian features, w = b + λθ + e, noise variances U(0.5, 1.5)
     (the decoder learns them);
   - discrete choice: 25 survey questions with 2–5 options, multinomial logit;
   - counts: 100-token documents over a 1,000-word vocabulary, multinomial logit;
   - binary: 60 roll-call votes, two-parameter logistic model.
2. **Three modalities with different precisions** (`02_multimodal.py`, `fig2_multimodal`).
   N = 10,000 units, each observed through 60 votes, 12 survey questions and 15-token documents.
   The exact precision gains over the prior are about 10.6 (votes), 5.5 (survey) and 1.5 (text).
   One model is fitted on all three. Panels: the joint posterior and each modality's posterior on
   its own. Titles report the average posterior precision (1/sd²) against the exact one.
3. **Covariates in the prior** (`03_prior_mc.py`, `fig3_prior_mc`). Probabilistic PCA with
   covariates in the prior (a MIMIC model): z = g₀ + x'g + e, e ~ N(0, 1); 10 Gaussian
   features, w = b + λz + u. Covariates: one binary and two continuous; g = (−1, 0.75, −0.25).
   1,000 replications with N = 1,000 units each, new data each time, fixed parameters. Model:
   learned prior N(Gx, 1) (`learn_prior_cov=False` pins the residual variance, which identifies
   the scale), linear encoder and decoder, mean-field Gaussian posterior, true ELBO, prior
   learning rate 1e-3, 20,000 steps (10,000 and 15,000 also saved).
   - With a 1-D latent the exact posterior is Gaussian with a mean linear in (w, x). So the
     variational family and the linear encoder are exact, and the ELBO maximizer is the
     maximum-likelihood estimator.
   - Each replication also computes the exact MLE on the same sample. The gap between the
     estimate and the MLE measures optimization error; the gap between the MLE and the truth is
     sampling error. Both are in `figure_numbers.json`.
   - **Identification.** The slopes g are identified up to the latent's sign, which is aligned
     with the true loadings. The intercept g₀ is not identified, because the decoder intercepts
     absorb b + λg₀. So the figure shows the three slopes.

## Running

From the laptop (WSL, Bocconi VPN on):

```bash
bash replication/simulations/sync_to_hpc.sh       # code + package to the cluster, md5-checked
ssh bocconi-hpc2 "cd /home/Gauthier/simulations && sbatch slurm/01_likelihoods.sbatch \
    && sbatch slurm/02_multimodal.sbatch && sbatch slurm/03_prior_mc.sbatch"
bash replication/simulations/fetch_results.sh     # results/ and logs/ back to the laptop
cd replication/simulations/code && python 04_figures.py   # figures/ from results/
```

All jobs run on CPU (`medium_cpu`, 6 h limit). The Monte Carlo array has 10 tasks of 100
replications. It can be resumed: rerunning a task skips the replications already in its file.

## Outputs (`results/`)

- `fig1_likelihoods/<likelihood>.npz`: per unit, the true θ, the model's posterior mean
  (sign not aligned) and sd, and the exact posterior's mean, variance, and best-Gaussian
  mean and variance (`exact_mean`, `exact_var`, `exact_m`, `exact_s2`).
  `<likelihood>.json`: design, seeds, fit statistics at each checkpoint, provenance.
- `fig2_multimodal/<panel>.npz` (joint, vote, survey, bow): the same, per panel;
  `summary.json`.
- `fig3_prior_mc/task_<k>.jsonl`: one line per replication, with the slope estimates at each
  checkpoint (sign-aligned) and the exact MLE. `design.json` holds the true parameters.
- `figures/figure_numbers.json`: every number shown on the figures, plus diagnostics.
