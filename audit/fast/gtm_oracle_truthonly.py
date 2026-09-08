"""Truth-parameter two-step estimators for the pilot DGP at any (n, words); no training.

Two-stage importance sampling per document (prior proposal -> per-document Gaussian
fit -> defensive mixture proposal) so the exact posterior mean is accurate for short
and long documents alike. Estimators (OLS of y on the per-document share estimate):
  rc_true_prior     E[theta|w,x] under true prior (regression calibration; unbiased for linear y)
  fixed_N01_prior   E[theta|w] under fixed N(0,I) contrast prior, no covariates (IP-sim two-step)
  mle_plugin        per-document MLE on the simplex (unshrunk plug-in; Battaglia-type)
  oracle            OLS on true theta
Usage: python audit/fast/gtm_oracle_truthonly.py --n 10000 --words 100 [--seed 9100]
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_gtm_pilot as H  # noqa: E402

torch.set_num_threads(8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--words", type=int, default=25)
    p.add_argument("--seed", type=int, default=9100)
    p.add_argument("--draws", type=int, default=2048)
    p.add_argument("--prior-sd-scale", type=float, default=1.0)
    p.add_argument("--anchors", type=int, default=0)
    p.add_argument("--anchor-logit", type=float, default=4.0)
    p.add_argument("--out", default=None)
    o = p.parse_args()
    a = H.parser().parse_args(["--n", str(o.n), "--words", str(o.words), "--seed", str(o.seed), "--prior-sd-scale", str(o.prior_sd_scale), "--anchors", str(o.anchors), "--anchor-logit", str(o.anchor_logit), "--out", "/dev/null"])
    design = H.fixed_design(a)
    data = H.generate(a, design, a.n, a.seed)
    K, d, S, CH = a.topics, a.topics - 1, o.draws, 64
    V = torch.tensor(design["basis"], dtype=torch.float32)
    D = torch.tensor(design["decoder"], dtype=torch.float32)
    prev = torch.tensor(design["prevalence"], dtype=torch.float32)
    Sig = torch.tensor(design["covariance"], dtype=torch.float32)
    truth = np.array(design["beta"])
    fixed = MultivariateNormal(torch.zeros(d), covariance_matrix=torch.eye(d))

    def llw(z, counts):
        th = torch.softmax(z @ V.T, -1)
        return (counts[None] * (th @ D).log_softmax(-1)).sum(-1)

    def two_stage(x, counts, logprior):
        """Return posterior mean of theta and ESS for target exp(logprior(z) + llw)."""
        m = len(x)
        mean_true = x @ prev.T @ V
        p_true = MultivariateNormal(mean_true, covariance_matrix=Sig)
        # stage 1: broad proposal = true prior with inflated covariance
        p1 = MultivariateNormal(mean_true, covariance_matrix=2.0 * Sig)
        z = p1.sample((S,))
        lw = logprior(z) + llw(z, counts) - p1.log_prob(z)
        w = lw.softmax(0)
        mu = (w[..., None] * z).sum(0)
        diff = z - mu[None]
        cov = (w[..., None, None] * diff[..., :, None] * diff[..., None, :]).sum(0) + 0.02 * torch.eye(d)
        # stage 2: defensive mixture around the stage-1 fit
        q = MultivariateNormal(mu, covariance_matrix=1.5 * cov)
        z = q.sample((S,))
        z = torch.where((torch.rand((S, m)) < 0.3)[..., None], p1.sample((S,)), z)
        logprop = torch.logaddexp(q.log_prob(z) + math.log(0.7), p1.log_prob(z) + math.log(0.3))
        lw = logprior(z) + llw(z, counts) - logprop
        w = lw.softmax(0)
        th = torch.softmax(z @ V.T, -1)
        return (w[..., None] * th).sum(0).numpy(), (1 / w.square().sum(0)).numpy()

    torch.manual_seed(1)
    out = {"rc_true_prior": [], "fixed_N01_prior": []}
    ess = {k: [] for k in out}
    with torch.no_grad():
        for lo in range(0, a.n, CH):
            sl = slice(lo, lo + CH)
            x = torch.tensor(data["x"][sl], dtype=torch.float32)
            counts = torch.tensor(data["counts"][sl])
            p_true = MultivariateNormal(x @ prev.T @ V, covariance_matrix=Sig)
            pm, e = two_stage(x, counts, p_true.log_prob); out["rc_true_prior"].append(pm); ess["rc_true_prior"].append(e)
            pm, e = two_stage(x, counts, fixed.log_prob); out["fixed_N01_prior"].append(pm); ess["fixed_N01_prior"].append(e)
    # per-document MLE
    eta = torch.zeros((a.n, d), requires_grad=True)
    counts_all = torch.tensor(data["counts"])
    opt = torch.optim.Adam([eta], lr=0.1)
    for _ in range(500):
        opt.zero_grad()
        (-llw(eta[None], counts_all)[0].sum()).backward()
        opt.step()
        with torch.no_grad():
            eta.clamp_(-12, 12)
    est = {"oracle": H.ols(data["theta"], data["y"], truth)}
    for k, v in out.items():
        pm = np.vstack(v)
        est[k] = H.ols(pm, data["y"], truth)
        est[k]["share_rmse"] = float(np.sqrt(np.mean((pm - data["theta"]) ** 2)))
        est[k]["ess_median"] = float(np.median(np.concatenate(ess[k])))
        est[k]["ess_p05"] = float(np.quantile(np.concatenate(ess[k]), 0.05))
    th = torch.softmax(eta.detach() @ V.T, -1).numpy()
    est["mle_plugin"] = H.ols(th, data["y"], truth)
    est["mle_plugin"]["share_rmse"] = float(np.sqrt(np.mean((th - data["theta"]) ** 2)))
    print(f"\n=== n={a.n}, words={a.words}, seed={a.seed}: OLS of y on share estimate (truth {truth.tolist()}) ===")
    for k, v in est.items():
        ex = f" share_rmse={v['share_rmse']:.4f}" if "share_rmse" in v else ""
        ex += f" ESS med/p05={v['ess_median']:.0f}/{v['ess_p05']:.0f}" if "ess_median" in v else ""
        print(f"{k:17s} beta={np.round(v['beta'], 3).tolist()} se={np.round(v['naive_hc1_se'], 3).tolist()} scale={v['scale_ratio']:.3f} MAE={v['mean_absolute_error']:.3f}{ex}")
    if o.out:
        Path(o.out).write_text(json.dumps({"n": a.n, "words": a.words, "seed": a.seed, "estimators": est}, indent=2))


if __name__ == "__main__":
    main()
