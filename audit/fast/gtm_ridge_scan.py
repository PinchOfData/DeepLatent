"""How flat is the temperature ridge of the linear-softmax GTM at the true parameters?

Transform the TRUE parameters by a scale s:
    eta -> eta / s   (mu -> mu/s, Sigma -> Sigma/s^2)
    D   -> s D + 1 c',  c = (1 - s) theta_bar D    (per-word constant; absorbed because shares sum to 1)
    beta-> s beta,  intercept -> (1 - s) theta_bar beta
To first order in (theta - theta_bar) this leaves p(w | x) and p(y | theta) unchanged; only the
softmax curvature breaks it. Reports the held-out log p(w|x) and log p(w,y|x) per document
for each s relative to s = 1 (paired differences with SE), via two-stage importance sampling.
Usage: python audit/fast/gtm_ridge_scan.py --words 25 [--n-eval 2048]
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
import torch
from torch.distributions import MultivariateNormal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_gtm_pilot as H  # noqa: E402

torch.set_num_threads(8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--words", type=int, default=25)
    p.add_argument("--n-eval", type=int, default=2048)
    p.add_argument("--scales", type=float, nargs="+", default=[0.7, 0.85, 1.0, 1.15, 1.3, 1.5])
    p.add_argument("--draws", type=int, default=2048)
    p.add_argument("--prior-sd-scale", type=float, default=1.0)
    p.add_argument("--anchors", type=int, default=0)
    p.add_argument("--anchor-logit", type=float, default=4.0)
    o = p.parse_args()
    a = H.parser().parse_args(["--n", "10000", "--words", str(o.words), "--prior-sd-scale", str(o.prior_sd_scale), "--anchors", str(o.anchors), "--anchor-logit", str(o.anchor_logit), "--out", "/dev/null"])
    design = H.fixed_design(a)
    big = H.generate(a, design, a.n, a.seed)  # for theta_bar
    data = H.generate(a, design, o.n_eval, a.seed + 200000)
    K, d, S, CH = a.topics, a.topics - 1, o.draws, 64
    V = torch.tensor(design["basis"], dtype=torch.float32)
    D0 = torch.tensor(design["decoder"], dtype=torch.float32)
    prev = torch.tensor(design["prevalence"], dtype=torch.float32)
    Sig0 = torch.tensor(design["covariance"], dtype=torch.float32)
    beta0 = torch.tensor(design["beta"], dtype=torch.float32)
    theta_bar = torch.tensor(big["theta"].mean(0), dtype=torch.float32)

    def evaluate(s):
        D = s * D0 + ((1 - s) * (theta_bar @ D0))[None, :]
        beta = s * beta0
        b0 = float((1 - s) * (theta_bar @ beta0))
        Sig = Sig0 / s ** 2
        lz_w, lz_wy, ess = [], [], []
        torch.manual_seed(11)
        with torch.no_grad():
            for lo in range(0, o.n_eval, CH):
                sl = slice(lo, lo + CH)
                x = torch.tensor(data["x"][sl], dtype=torch.float32)
                counts = torch.tensor(data["counts"][sl])
                y = torch.tensor(data["y"][sl], dtype=torch.float32)
                m = len(x)
                mu = (x @ prev.T @ V) / s
                prior = MultivariateNormal(mu, covariance_matrix=Sig)

                def llw(z):
                    th = torch.softmax(z @ V.T, -1)
                    return (counts[None] * (th @ D).log_softmax(-1)).sum(-1)

                def lly(z):
                    th = torch.softmax(z @ V.T, -1)
                    return -0.5 * ((y[None] - th @ beta - b0) ** 2 + math.log(2 * math.pi))

                p1 = MultivariateNormal(mu, covariance_matrix=2.0 * Sig)
                z = p1.sample((S,))
                lw = prior.log_prob(z) + llw(z) - p1.log_prob(z)
                w = lw.softmax(0)
                m1 = (w[..., None] * z).sum(0)
                diff = z - m1[None]
                cov = (w[..., None, None] * diff[..., :, None] * diff[..., None, :]).sum(0) + 0.02 * torch.eye(d) / s ** 2
                q = MultivariateNormal(m1, covariance_matrix=1.5 * cov)
                z = q.sample((S,))
                z = torch.where((torch.rand((S, m)) < 0.3)[..., None], p1.sample((S,)), z)
                logprop = torch.logaddexp(q.log_prob(z) + math.log(0.7), p1.log_prob(z) + math.log(0.3))
                lw = prior.log_prob(z) + llw(z) - logprop
                lz_w.append((lw.logsumexp(0) - math.log(S)).numpy())
                lz_wy.append(((lw + lly(z)).logsumexp(0) - math.log(S)).numpy())
                ess.append((1 / lw.softmax(0).square().sum(0)).numpy())
        return np.concatenate(lz_w), np.concatenate(lz_wy), float(np.median(np.concatenate(ess)))

    th = big["theta"]
    print(f"DGP: anchors={a.anchors} logit={a.anchor_logit} sd-scale={a.prior_sd_scale}; mean shares {np.round(th.mean(0), 3).tolist()}; "
          f"median max share {np.median(th.max(1)):.2f}; frac max>0.9 {(th.max(1) > 0.9).mean():.2f}; Sigma trace {float(torch.trace(Sig0)):.2f}")
    ref_w, ref_wy, _ = evaluate(1.0)
    print(f"\n=== Ridge scan at TRUE parameters, words={o.words}, {o.n_eval} held-out docs; differences vs s=1 (nats/doc) ===")
    print(f"{'s':>5s} {'Sigma trace':>12s} {'d log p(w|x)':>14s} {'se':>7s} {'d log p(w,y|x)':>15s} {'se':>7s} {'ESS':>6s}")
    for s in o.scales:
        lw_, lwy_, e = evaluate(s)
        dw, dwy = lw_ - ref_w, lwy_ - ref_wy
        print(f"{s:5.2f} {float(torch.trace(Sig0)) / s ** 2:12.3f} {dw.mean():14.4f} {dw.std(ddof=1) / np.sqrt(len(dw)):7.4f} "
              f"{dwy.mean():15.4f} {dwy.std(ddof=1) / np.sqrt(len(dwy)):7.4f} {e:6.0f}")


if __name__ == "__main__":
    main()
