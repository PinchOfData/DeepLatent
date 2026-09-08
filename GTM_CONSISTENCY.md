# GTM consistency pilot

Requested 2026-09-07. Harness: `audit/experiment_gtm_pilot.py`.
The first run is **one replication on the laptop**, before an HPC campaign.

## Specification

- Five topic shares, represented by four identifiable Helmert contrast logits.
- Three independent Bernoulli(0.5) prevalence covariates and an intercept.
- Logistic-normal prior with a linear covariate mean and a learned full covariance.
  The true contrast covariance has standard deviations 0.8, 0.9333, 1.0667, 1.2
  and correlations `0.45 ** abs(j-k)`. It is not an identity covariance.
- Ten diagonal Gaussian components in the variational mixture. The mixture can
  represent dependence; the **prior** covariance is a full Cholesky matrix.
- One linear, bias-free decoder of topic shares into 200 word logits, followed by
  softmax and a multinomial bag-of-words likelihood.
- A linear outcome head: `y = theta @ [2, 1, 0, -1, -2] + Normal(0, 1)`.
  The fitted outcome variance is learned, with unit likelihood weight.
- No decoder covariates, nonlinear decoder, KL downweighting, or weight decay.

The decoder, prevalence coefficients, covariance, and outcome coefficients are
fixed by design seed 777 independently of the replication seed. Document draws
vary with replication seed. The initial pilot uses seed 9100, 2,000 documents,
25 words per document, a 64/64 encoder, batches of 128, and checkpoints at
2,000/4,000/8,000/16,000 optimizer updates in each arm. Main/prior learning rates
are 0.001/0.0001. All checkpoints are retained; none is selected by its agreement
with the truth. Posterior diagnostics preserve the training random-number state.

## Why the old GTM harness is not reused unchanged

`generate_documents` draws words from `theta @ topic_word_probabilities`. Current
GTM training instead fits `softmax(theta @ decoder_logits)`. These are different
measurement models. This pilot generates from the latter so the fitted model is
correctly specified. This is a new experiment, not a reproduction of the July
topic-model Monte Carlo.

## Comparisons and identification

1. **Joint:** estimate decoder, prior, and outcome equation jointly; the encoder
   observes words, covariates, and the outcome during fitting.
2. **Two-step:** estimate an unsupervised GTM, then regress the outcome on posterior
   mean shares. By default it learns the same prevalence mean and full covariance
   as the joint arm. `--two-step-prior fixed` reproduces the older normalized-prior
   comparison if explicitly selected.
3. **Post-fit OLS:** regress the outcome on outcome-free posterior mean shares
   under each fitted measurement model and prior.
4. **Oracle:** OLS on the true shares from the same draw, showing sampling noise
   even when factors are known.

Topic labels are matched by decoder-profile similarity using Hungarian matching.
Outcome coefficients are centered across topics because shares sum to one and
the fitted outcome head has an intercept. Prevalence coefficients are centered
logit coefficients with the intercept folded into the prior mean. Prior covariance
is lifted through the contrast basis, permuted, and returned to contrast coordinates
before comparison. No arbitrary rescaling is used to improve recovery.

The correctly specified two-step comparison need not display the old attenuation:
regression on an exact outcome-free conditional mean can recover a linear outcome
coefficient. The comparison measures what the actual fitted procedures do.

## Posterior readout and inference

For supervised GTM, filling the encoder's outcome input with zero is conditioning
on zero; it does not marginalize the outcome. Outcome-free post-fit means therefore
use self-normalized importance sampling with target
`p(words | eta) p(eta | covariates)`. The proposal mixes the encoder distribution
(with its outcome input zeroed) with 15% fitted prior draws; its exact mixture
density appears in the importance weights. No outcome likelihood enters that
target. The initial readout uses 1,024 draws per document and reports ESS and
agreement between disjoint sample halves.

An independent 256-document draw checks encoder means against importance-weighted
means and nominal 90% topic-share posterior intervals under the appropriate
conditioning set. The defensive-proposal ELBO is labelled separately from the
encoder ELBO and must not be read as the trained encoder's variational gap.

OLS HC1 intervals are saved as **naive diagnostics**, omitting first-stage model
uncertainty. A single replication does not estimate sampling bias, consistency,
normality, or frequentist CI coverage. In particular, the scalar ideal-point
delta correction is not automatically valid for this multivariate GTM.

## Local execution and HPC

Both `HPC.md` here and `Desktop/alfred/instructions/hpc/workflows.md` were readable.
SSH to `bocconi-hpc2` was verified on 2026-09-07; the queue was empty at the check.
Use the newer repository-specific instructions for partitions and CPU limits.
All cluster Python must run through SLURM. Each worker must lift the inherited
CPU-time limit (`ulimit -t unlimited`), use one BLAS thread, and invoke the project
environment's Python by absolute path. No large GTM campaign has been submitted.

Local command:

```bash
env NUMBA_CACHE_DIR=/tmp/deeplatent_numba MPLCONFIGDIR=/tmp/deeplatent_mpl \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/gauthier/miniconda3/envs/deeplatent/bin/python -u \
  audit/experiment_gtm_pilot.py --out audit/results_gtm_pilot_20260907.json \
  > audit/gtm_pilot_20260907.log 2>&1
```

The JSON is written atomically after each checkpoint and includes all DGP arrays,
arguments, source revision, script hash, and elapsed times. Final fitted models
are saved next to the JSON as ignored `.ckpt` files for diagnosis on the same draw.

## Completed local pilot

Results: `audit/results_gtm_pilot_20260907.json`; readable report:
`audit/gtm_pilot_20260907.md`; figure: `figures/sim_gtm_pilot_20260907.{png,pdf}`.
The complete two-arm run took 416 seconds on one laptop CPU thread. A first
training-only attempt was repeated on the identical seeded data after correcting
a readout bookkeeping error; the joint checkpoint estimates reproduced exactly.

Outcome-coefficient mean absolute errors: oracle observed shares 0.090; joint
head 0.175; OLS after the joint fit 0.159; unsupervised encoder + OLS 0.371;
importance-corrected unsupervised posterior + OLS 0.362. Global effect-scale ratios
are 0.955/0.970 for joint head/post-fit and 0.997/0.995 for the unsupervised readouts.
Thus the two-step error on this draw concerns individual topic effects, rather
than the fixed global attenuation in the older simulations.

Recovery remains incomplete: topic 3 has mean share 0.062 and weak decoder-profile
correlation (joint 0.168; unsupervised 0.328). Prior-coefficient RMSE is 0.258/0.274;
relative covariance error is 0.301/0.627 for joint/unsupervised. The fitted outcome
variance is 1.021 against truth 1. These are single-draw errors, not estimates of
sampling bias or consistency.

Posterior importance sampling is stable (median ESS 839/860 of 1,024). On 256
held-out documents, the estimated IWAE-minus-encoder-ELBO gap is 0.058/0.059
nats/document. Diagnostics are saved in `audit/gtm_pilot_20260907_diagnostics.json`.
They evaluate `log p(words|eta) + log p(eta|x) [+ log p(y|theta)] - log q(eta)` on
1,024 encoder draws/document, comparing mean log weight to log-mean-exp weight.
The direct mixture density also matches `model._posterior_loglik` within 1e-6.

## Proposed next stage

Before a production Monte Carlo, use a small HPC calibration with the same fixed
DGP, three independent replications per cell, n in {2,000, 8,000, 16,000}, and
document lengths 25 and 100 (18 replications total, each fitting both arms).
Record 4k/8k/16k/32k checkpoints. Larger n tests sampling/optimization behavior;
longer documents test whether weak measurement drives topic and prior recovery.
Keep the rare topic in the fixed design rather than choosing new DGP parameters
after inspecting the results. The document-length choice has been raised with
the user. This calibration is proposed, not submitted.

`audit/gtm_mc.sbatch` implements one replication per array task, refuses to
overwrite results, and defaults to the same learned prior in both arms.
`audit/merge_gtm_mc.py` rejects duplicate seeds and incompatible designs/code,
reports point-estimate sampling summaries and naive HC1 coverage, and marks SD
as unavailable for a single replication. No joint coverage claim is made.

## n = 10,000 follow-up (2026-09-07)

At the user's request, reran one laptop replication with only `--n 10000` changed.
The encoder remains two hidden layers of 64 units each. DGP arrays, seeds, source
hashes, document length, prior/mixture specification, 16k-step budget, and readout
settings match the n=2,000 pilot. Results:
`audit/results_gtm_pilot_n10000_20260907.json`; report:
`audit/gtm_pilot_n10000_20260907.md`. The complete run took 437 seconds.

| Outcome estimator | MAE, n=2,000 | MAE, n=10,000 |
|---|---:|---:|
| Oracle true shares | 0.090 | 0.035 |
| Joint head | 0.175 | 0.274 |
| Joint post-fit OLS | 0.159 | 0.285 |
| Two-step encoder OLS | 0.371 | 0.181 |
| Two-step importance OLS | 0.362 | 0.206 |

The estimator ranking reverses on this draw: joint no longer has the lower
outcome-coefficient error. Prior recovery improves: joint prevalence RMSE
0.258 to 0.165 and relative covariance error 0.301 to 0.265; unsupervised values
0.274 to 0.178 and 0.627 to 0.532. The rare topic remains weakly recovered, although
the joint decoder-profile correlation improves from 0.168 to 0.394. The fitted
joint outcome variance is 1.015 (truth 1).

Estimated held-out IWAE-minus-ELBO gaps remain small: joint 0.054 and unsupervised
0.046 nats/document. These and density checks are recorded in
`audit/gtm_pilot_n10000_20260907_diagnostics.json`. The readout density matches the
package to within 1e-6. This is one draw per sample size, not a consistency or
coverage result. Holding optimizer updates fixed gives about five times fewer
passes over the training data at n=10,000. No HPC jobs were submitted.

## Generalization audit of the 10K training trajectory

After the user asked about overfitting, `audit/check_gtm_overfitting.py` replayed
the identical 10K run with likelihood evaluations every 1,000 updates. All eight
original arm/checkpoint coefficient vectors reproduced with maximum difference
**exactly zero**. The audit uses 2,048 fixed training documents and 2,048 independent
held-out documents, with 128 importance draws per document. It preserves training
RNG state and includes multinomial constants in likelihood scores.

Both arms' held-out negative log-likelihoods decreased at **every measured
checkpoint**, with the lowest observed value at 16K. Between 8K and 16K:

| Arm | Train NLL, 8K to 16K | Held-out NLL, 8K to 16K | Outcome-coefficient MAE, 8K to 16K |
|---|---|---|---|
| Joint | 71.0694 to 71.0254 | 71.1042 to 71.0700 | 0.3351 to 0.2744 |
| Two-step | 69.6247 to 69.5715 | 69.6526 to 69.6068 | 0.3290 to 0.1812 |

Thus there is no overall overfitting signal within the 1K checkpoint resolution.
Prevalence-coefficient errors do show late deterioration: joint RMSE 0.1605 to
0.1646; unsupervised 0.1297 to 0.1784 from 8K to 16K. These are different targets
from outcome-effect recovery and held-out likelihood. The original 16K endpoint
comparison was a fixed-budget comparison; parameter convergence was not established.

Raw audit: `audit/gtm_overfit_n10000_20260907.json`; paired per-document scores:
the sibling `.npz`; replayed model checkpoints: `audit/gtm_overfit_n10000_20260907/`.
The 10K report now includes the full checkpoint estimates and audit table; its
`figures/sim_gtm_pilot_n10000_20260907_overfit.{png,pdf}` plot shows the late
training and held-out likelihood trajectories. No extra Monte Carlo draw or HPC
job was added by this deterministic replay.
