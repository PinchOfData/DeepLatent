# Posterior collapse in the DeepLatent VAEs: diagnosis and fix

## TL;DR

The VAE posterior collapsed at `w_prior=1` (the value for which the objective is
supposed to be the ELBO and the theoretical guarantees hold) because the
**reconstruction term and the KL term were on different scales**: the
reconstruction was a *per-token / per-vote average* while the KL was a
*per-document average*. This silently over-weighted the KL by a factor of
roughly the document length `L` (number of tokens / votes per document), so
`w_prior=1` behaved like a β-VAE with β ≈ `L`, which collapses. The notebooks
worked around it with `w_prior ≈ 0.01 ≈ 1/L`, but then the objective is no
longer a valid lower bound on the marginal log-likelihood.

**Fix:** reconstruction is now **always** the per-document negative
log-likelihood (summed over the tokens/votes/features within a document,
averaged over the batch) — the same per-document scale as the KL. So `w_prior=1`
is the genuine (negative) ELBO, a valid lower bound, and the posterior does
**not** collapse. The previous per-token reduction was a bug and has been
removed (no option to re-enable it).

## The mismatch (before — removed)

In `DeepLatent.step_batch`:

```python
# reconstruction (BoW): divided by the number of tokens in the batch  ->  per-TOKEN
recon_loss = -torch.sum(target * log_probs) / torch.sum(target)
# KL: averaged over the batch                                          ->  per-DOCUMENT
kl = kl_raw.mean()
loss = recon_loss + w_prior * kl
```

The per-document ELBO is

```
ELBO_doc = sum_w n_w log p(w | theta)  -  KL_doc          (reconstruction SUMMED over tokens)
```

Dividing the reconstruction by `sum(target) ≈ B·L` but the KL only by `B`
down-weights the likelihood by `~L` relative to the KL. Equivalently, at
`w_prior=1` the KL is `~L`× too strong → collapse.

## The fix (now unconditional)

Reconstruction sums the per-document log-likelihood over tokens/votes/features
and averages over the batch — the same per-document scale as the KL:

```python
recon_loss = -torch.sum(target * log_probs) / target.shape[0]   # per-DOCUMENT (BoW)
loss = recon_loss + w_prior * kl                                # w_prior=1 == -ELBO
```

This applies to `bow`, `vote`, `embedding` and `image` reconstructions
(`discrete_choice` was already per-document). It holds for every `ae_type`; the
WAE/AE MMD/none penalties are unaffected in form (only the reconstruction scale
changed, and `w_prior` absorbs any retuning).

## Empirical confirmation

Simulated data, IAF posterior, fixed prior, single GPU. `raw_KL` is the true
per-document KL; `recovery` is topic-proportion correlation (GTM) or
ideal-point correlation (IdealPointNN). With the fix in place, `w_prior` is the
genuine ELBO weight, so the **collapse is reproduced by over-weighting the KL**
(`w_prior = L`, which matches the old per-token `w_prior=1`).

### GTM (logistic-normal, 6 topics, 6000 docs, 6000 steps)

| w_prior | what it is | raw_KL/doc | θ-var ratio | topic recovery |
|---|---|---|---|---|
| **1**   | **true ELBO (fix)**            | **6.70** | **0.97** | **0.973** ✅ |
| L≈200   | KL over-weighted (old per-token w=1) | ≈0 | ≈0.2 | ≈0.02 ❌ collapsed |
| 0.01    | old workaround (not a bound)   | 4.91 | 0.98 | 0.972 |

### IdealPointNN (1-D, text+votes, corrected PoE, 4000 units, 5000 steps)

| w_prior | what it is | raw_KL/doc | ideal-point recovery |
|---|---|---|---|
| **1**   | **true ELBO (fix)**            | **1.74** | **0.990** ✅ |
| large   | KL over-weighted (old per-token w=1) | ≈0 | ≈0.0 ❌ collapsed |
| 0.01    | old workaround (not a bound)   | 1.39 | 0.985 |

In both models, `w_prior=1` recovers the latent structure at least as well as
the hand-tuned `w_prior=0.01`, while being the correctly scaled ELBO. KL
annealing / free-bits were **not** needed.

## Reproduce

```bash
python experiments/posterior_collapse_gtm.py        --num_steps 6000 --figures
python experiments/posterior_collapse_idealpoint.py --num_steps 5000 --figures
pytest tests/test_recon_reduction.py -v
```
