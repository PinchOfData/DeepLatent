"""Likelihood at TRUE parameters vs fitted pilot checkpoints, with well-conditioned proposals.

Each target gets a proposal in its own coordinates:
  truth          : 50% two-step encoder rotated into true topic order + 50% true prior
  fitted two-step: 50% its own encoder + 50% its own prior
  fitted joint   : 50% its own encoder (y input = true y for p(w,y|x); zeros for the y-free
                   target) + 50% its own prior
Reports log p(w|x) and log p(w,y|x) per document (multinomial constant omitted), paired
differences truth - fitted with SE, and ESS. Also the outcome-free RC readout under each fit.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_gtm_pilot as H  # noqa: E402
from gtm_oracle_estimators import load_arm  # noqa: E402

torch.set_num_threads(8)
S, CH = 1024, 128


def mog_params(model, x, counts, y=None):
    parts = [counts, x]
    if model.labels_in_encoder:
        parts.append(torch.zeros((len(x), 1)) if y is None else y[:, None])
    raw = model.encoder.encoders["text_bow"](torch.cat(parts, dim=1))
    means, logvars, weights = model.encoder._mog_unpack(raw)
    return means, logvars.exp(), weights


def mixture(means, vars_, weights, R=None):
    if R is not None:  # z' = R z, |det R| = 1
        means = means @ R.T
        cov = R[None, None] * vars_[..., None, :] @ R.T[None, None]  # R diag(v) R^T
        comp = MultivariateNormal(means, covariance_matrix=cov)
    else:
        comp = MultivariateNormal(means, covariance_matrix=torch.diag_embed(vars_))
    return MixtureSameFamily(Categorical(probs=weights), comp)


def main():
    a = H.parser().parse_args(["--n", "10000", "--out", "/dev/null"])
    design = H.fixed_design(a)
    data = H.generate(a, design, a.n, a.seed)
    held = H.generate(a, design, 2048, a.seed + 200000)
    K, d = a.topics, a.topics - 1
    V = torch.tensor(design["basis"], dtype=torch.float32)
    D = torch.tensor(design["decoder"], dtype=torch.float32)
    prev = torch.tensor(design["prevalence"], dtype=torch.float32)
    Sig = torch.tensor(design["covariance"], dtype=torch.float32)
    beta = torch.tensor(design["beta"], dtype=torch.float32)
    truth = np.array(design["beta"])
    two = load_arm(a, data, False, "audit/results_gtm_pilot_n10000_20260907_two_step.ckpt")
    joint = load_arm(a, data, True, "audit/results_gtm_pilot_n10000_20260907_joint.ckpt")
    perm_two, _ = H.align_topics(two, design)
    perm_joint, _ = H.align_topics(joint, design)
    R_two = torch.tensor(design["basis"].T @ design["basis"][perm_two], dtype=torch.float32)  # fitted -> true coords

    def run(ds, tag, want_means):
        n = len(ds["y"])
        torch.manual_seed(7)
        lz = {k: [] for k in ["truth_w", "truth_wy", "two_w", "joint_w", "joint_wy"]}
        ess = {k: [] for k in lz}
        pm = {k: [] for k in ["two_w", "joint_w"]}
        with torch.no_grad():
            for lo in range(0, n, CH):
                sl = slice(lo, lo + CH)
                x = torch.tensor(ds["x"][sl], dtype=torch.float32)
                counts = torch.tensor(ds["counts"][sl])
                y = torch.tensor(ds["y"][sl], dtype=torch.float32)
                m = len(x)

                def importance(q, prior, logtarget, want_theta=None):
                    z = q.sample((S,))
                    z = torch.where((torch.rand((S, m)) < 0.5)[..., None], prior.sample((S,)), z)
                    logprop = torch.logaddexp(q.log_prob(z), prior.log_prob(z)) + math.log(0.5)
                    lw = logtarget(z) - logprop
                    w = lw.softmax(0)
                    out = [(lw.logsumexp(0) - math.log(S)).numpy(), (1 / w.square().sum(0)).numpy()]
                    if want_theta is not None:
                        out.append((w[..., None] * want_theta(z)).sum(0).numpy())
                    return out

                # truth
                mean_true = x @ prev.T @ V
                p_true = MultivariateNormal(mean_true, covariance_matrix=Sig)
                q_true = mixture(*mog_params(two, x, counts), R=R_two)

                def lt_truth_w(z):
                    th = torch.softmax(z @ V.T, -1)
                    return p_true.log_prob(z) + (counts[None] * (th @ D).log_softmax(-1)).sum(-1)

                def lt_truth_wy(z):
                    th = torch.softmax(z @ V.T, -1)
                    return lt_truth_w(z) - 0.5 * ((y[None] - th @ beta) ** 2 + math.log(2 * math.pi))

                r = importance(q_true, p_true, lt_truth_w); lz["truth_w"].append(r[0]); ess["truth_w"].append(r[1])
                r = importance(q_true, p_true, lt_truth_wy); lz["truth_wy"].append(r[0]); ess["truth_wy"].append(r[1])

                # fitted models
                for name, model, perm in [("two", two, perm_two), ("joint", joint, perm_joint)]:
                    mu, cov = model.prior.get_prior_params(x, return_full_cov=True)
                    prior = MultivariateNormal(mu, covariance_matrix=cov)

                    def theta_fn(z, model=model):
                        return model.latent_to_theta(z.reshape(-1, d)).reshape(S, m, K)

                    def lt_w(z, model=model, prior=prior):
                        th = theta_fn(z, model)
                        logits = model.decoders["text_bow"](th.reshape(-1, K)).reshape(S, m, -1)
                        return prior.log_prob(z) + (counts[None] * logits.log_softmax(-1)).sum(-1)

                    q0 = mixture(*mog_params(model, x, counts, None))
                    r = importance(q0, prior, lt_w, want_theta=lambda z, model=model: theta_fn(z, model))
                    lz[f"{name}_w"].append(r[0]); ess[f"{name}_w"].append(r[1]); pm[f"{name}_w"].append(r[2][:, perm])
                    if name == "joint":
                        def lt_wy(z, model=model):
                            th = theta_fn(z, model)
                            pred = model.predictor.predictors["y"](th.reshape(-1, K), None).reshape(S, m)
                            var = model.predictor.noise_log_var["y"].exp()
                            return lt_w(z) - 0.5 * ((y[None] - pred) ** 2 / var + var.log() + math.log(2 * math.pi))
                        qy = mixture(*mog_params(model, x, counts, y))
                        r = importance(qy, prior, lt_wy); lz["joint_wy"].append(r[0]); ess["joint_wy"].append(r[1])
        lz = {k: np.concatenate(v) for k, v in lz.items()}
        ess = {k: float(np.median(np.concatenate(v))) for k, v in ess.items()}
        pm = {k: np.vstack(v) for k, v in pm.items()} if want_means else {}
        print(f"\n=== {tag} ({n} docs): mean log-likelihood per doc (multinomial constant omitted); ESS median {ess}")
        for k, v in lz.items():
            print(f"  {k:9s} {v.mean():9.4f}")
        for lab, (p, q) in [("truth - two-step  log p(w|x)", ("truth_w", "two_w")),
                            ("truth - joint     log p(w|x)", ("truth_w", "joint_w")),
                            ("truth - joint     log p(w,y|x)", ("truth_wy", "joint_wy")),
                            ("two-step - joint  log p(w|x)", ("two_w", "joint_w"))]:
            diff = lz[p] - lz[q]
            print(f"  {lab}: {diff.mean():+.4f} (se {diff.std(ddof=1) / np.sqrt(len(diff)):.4f})")
        return pm

    pm = run(data, "train (all 10k)", True)
    for k, v in pm.items():
        o = H.ols(v, data["y"], truth)
        print(f"  RC readout under fitted {k}: beta={np.round(o['beta'], 3).tolist()} scale={o['scale_ratio']:.3f} MAE={o['mean_absolute_error']:.3f} HC1 se={np.round(o['naive_hc1_se'], 3).tolist()}")
    run(held, "held-out (2048)", False)


if __name__ == "__main__":
    main()
