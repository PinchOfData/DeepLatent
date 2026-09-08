"""Why does training drift along the scale ridge? Compare two fits of the SAME data that sit at
different points of the valley (run A final: Sigma trace ~2.6; run B final: trace ~3.6):
train-set and held-out IWAE (encoder proposal), encoder ELBO, and the gap, per document.
If in-sample IWAE favours the drifted fit while held-out does not -> overfitting along the ridge.
If the ELBO gap is smaller at the drifted fit -> the variational objective pushes along the ridge.
"""
import math, sys
from pathlib import Path
import numpy as np
import torch
from torch.distributions import MultivariateNormal

sys.path.insert(0, "audit"); sys.path.insert(0, "audit/fast")
import experiment_gtm_pilot as H  # noqa: E402
from gtm_oracle_estimators import load_arm  # noqa: E402
from gtm_lik_at_truth import mog_params, mixture  # noqa: E402

torch.set_num_threads(8)
S, CH = 1024, 128


def score(model, ds, joint):
    n = len(ds["y"])
    K, d = model.n_topics, model.n_latent
    out = {"iwae_w": [], "elbo_w": [], "iwae_wy": [], "elbo_wy": []}
    torch.manual_seed(5)
    with torch.no_grad():
        for lo in range(0, n, CH):
            sl = slice(lo, lo + CH)
            x = torch.tensor(ds["x"][sl], dtype=torch.float32)
            counts = torch.tensor(ds["counts"][sl])
            y = torch.tensor(ds["y"][sl], dtype=torch.float32)
            m = len(x)
            mu, cov = model.prior.get_prior_params(x, return_full_cov=True)
            prior = MultivariateNormal(mu, covariance_matrix=cov)

            def lt_w(z):
                th = model.latent_to_theta(z.reshape(-1, d)).reshape(S, m, K)
                logits = model.decoders["text_bow"](th.reshape(-1, K)).reshape(S, m, -1)
                return prior.log_prob(z) + (counts[None] * logits.log_softmax(-1)).sum(-1), th

            for tag, yin in [("w", None)] + ([("wy", y)] if joint else []):
                q = mixture(*mog_params(model, x, counts, yin))
                z = q.sample((S,))
                lt, th = lt_w(z)
                if tag == "wy":
                    pred = model.predictor.predictors["y"](th.reshape(-1, K), None).reshape(S, m)
                    var = model.predictor.noise_log_var["y"].exp()
                    lt = lt - 0.5 * ((y[None] - pred) ** 2 / var + var.log() + math.log(2 * math.pi))
                lw = lt - q.log_prob(z)
                out[f"iwae_{tag}"].append((lw.logsumexp(0) - math.log(S)).numpy())
                out[f"elbo_{tag}"].append(lw.mean(0).numpy())
    return {k: np.concatenate(v) for k, v in out.items() if v}


def main():
    a = H.parser().parse_args(["--n", "10000", "--out", "/dev/null"])
    aA = H.parser().parse_args(["--n", "10000", "--hidden", "128", "--out", "/dev/null"])
    design = H.fixed_design(a)
    data = H.generate(a, design, a.n, a.seed)
    held = H.generate(a, design, 2048, a.seed + 200000)
    sub = {k: v[:4096] for k, v in data.items()}
    fits = {
        "joint A (24k, mc-opt)": load_arm(aA, data, True, "audit/fast/runA_mcopt_joint.ckpt"),
        "joint B (64k, pilot-opt)": load_arm(a, data, True, "audit/fast/runB_pilotopt_64k_joint.ckpt"),
        "two-step A (24k, mc-opt)": load_arm(aA, data, False, "audit/fast/runA_mcopt_two_step.ckpt"),
        "two-step B (64k, pilot-opt)": load_arm(a, data, False, "audit/fast/runB_pilotopt_64k_two_step.ckpt"),
    }
    res = {}
    for name, model in fits.items():
        tr = float(torch.trace(model.prior.sigma))
        res[name] = {"train": score(model, sub, "joint" in name), "held": score(model, held, "joint" in name), "trace": tr}
        r = res[name]
        line = f"{name:28s} Sigma trace={tr:5.2f}"
        for split in ["train", "held"]:
            for k in ["iwae_w", "elbo_w", "iwae_wy", "elbo_wy"]:
                if k in r[split]:
                    line += f" | {split} {k}={r[split][k].mean():9.4f}"
            line += f" gap_w={(r[split]['iwae_w'] - r[split]['elbo_w']).mean():.4f}"
        print(line, flush=True)
    for pair in [("joint A (24k, mc-opt)", "joint B (64k, pilot-opt)"), ("two-step A (24k, mc-opt)", "two-step B (64k, pilot-opt)")]:
        print(f"\n{pair[0]}  minus  {pair[1]}  (paired per-document differences, nats/doc)")
        for split in ["train", "held"]:
            for k in ["iwae_w", "elbo_w", "iwae_wy", "elbo_wy"]:
                if k in res[pair[0]][split]:
                    dd = res[pair[0]][split][k] - res[pair[1]][split][k]
                    print(f"  {split:5s} {k:8s} {dd.mean():+.4f} (se {dd.std(ddof=1) / np.sqrt(len(dd)):.4f})")


if __name__ == "__main__":
    main()
