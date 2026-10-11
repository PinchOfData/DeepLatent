"""Figure 3: Monte Carlo sampling distribution of the prior's covariate effects.

    python 03_prior_mc.py --task 0 --per-task 100       # replications 0-99

Model: probabilistic PCA with covariates in the prior (a MIMIC model),
    z_i = g0 + x_i' g + e_i,  e_i ~ N(0, 1);   w_ij = b_j + lam_j z_i + u_ij,  u_ij ~ N(0, s2_j),
with x_i = (binary, continuous, continuous), J = 10 Gaussian features and N = 1,000 units. Each
replication redraws (x, z, w) under fixed (g, lam, b, s2) and fits IdealPointNN with a learned
prior N(G x, 1) (learn_prior_cov=False pins the residual variance, which identifies the scale),
linear encoder and decoder, mean-field Gaussian posterior and the true ELBO. With a 1-D latent the
exact posterior is Gaussian with a mean linear in (w, x), so the variational family and the linear
encoder are exact and the ELBO maximizer is the maximum-likelihood estimator.

Identification: the slopes g are identified up to the latent's sign (aligned with the true
loadings); the intercept g0 is not, because the decoder intercepts absorb b_j + lam_j g0.

Each replication also computes the exact MLE of the same marginal likelihood (closed form,
L-BFGS from the truth), so  estimate - MLE  is optimization error and  MLE - truth  is sampling
error. One JSON line per replication in results/fig3_prior_mc/task_<task>.jsonl (resumable).
"""

import argparse
import json
import tempfile
import time

import numpy as np
import pandas as pd
import torch

import common as c

N, J = 1_000, 10
SLOPES = {"x1": -1.0, "x2": 0.75, "x3": -0.25}  # binary, continuous, continuous
G0 = 0.5
ITEM_SEED, DATA_SEED = 51, 52
PRIOR_SD2 = (1.0 + 1e-4) ** 2  # Sigma of GaussianPrior(learn_cov=False): (exp(0) + 1e-4)^2
KEY = "embedding_features"


def design():
    rng = np.random.default_rng(ITEM_SEED)
    return {
        "lam": rng.normal(0.0, 0.6, J),
        "b": rng.normal(0.0, 1.0, J),
        "s2": rng.uniform(0.5, 1.5, J),
    }


def simulate(p, rep):
    rng = np.random.default_rng([DATA_SEED, rep])
    x = pd.DataFrame(
        {"x1": rng.binomial(1, 0.5, N), "x2": rng.normal(size=N), "x3": rng.normal(size=N)}
    )
    z = G0 + x.to_numpy() @ np.array(list(SLOPES.values())) + rng.normal(size=N)
    w = p["b"] + np.outer(z, p["lam"]) + rng.normal(size=(N, J)) * np.sqrt(p["s2"])
    return x, z, w


def fit(x, w, seed, steps, prior_lr):
    from deeplatent import Corpus, IdealPointNN

    corpus = Corpus(
        x.assign(emb=0.0),
        prevalence="~ x1 + x2 + x3",
        modalities={
            "embedding": {
                "column": "emb",
                "views": {
                    "features": {"type": "embedding", "matrix": w.astype(np.float32)}
                },
            }
        },
    )
    optim = None
    if prior_lr is not None:
        optim = {
            "main": {"lr": 1e-3},
            "prior": {"lr": prior_lr, "weight_decay": 0.0},
            "predictor": {"lr": 1e-3},
        }
    model = IdealPointNN(
        train_data=corpus,
        n_ideal_points=1,
        ae_type="vae",
        vi_type="mean_field",
        update_prior=True,
        learn_prior_cov=False,
        fusion="corrected_poe",
        w_prior=1.0,
        encoder_args={KEY: {"hidden_dims": []}},
        decoder_args={KEY: {"hidden_dims": [], "bias": True}},
        batch_size=256,
        num_steps=0,
        num_workers=0,
        print_every_n_steps=10**9,
        return_best_model=False,
        ckpt_folder=tempfile.mkdtemp(),
        seed=seed,
        device="cpu",
        optim_args=optim,
    )
    cols = model.prevalence_colnames
    path = []
    for step in steps:
        c.train_to(model, corpus, step)
        lam_hat = model.decoders[KEY].decoder["dec_0"].weight.detach().numpy()[:, 0]
        W = model.prior.mean_net.weight.detach().numpy()[0]
        path.append({"step": step, "lam_hat": lam_hat, "g": [float(W[cols.index(k)]) for k in SLOPES]})
    c.cleanup(model)
    return path


def mle(x, w, p):
    """Exact MLE of the marginal likelihood w_i ~ N(b + lam (x_i'g), PRIOR_SD2 lam lam' + diag(s2)),
    by L-BFGS from the true parameters (the intercept g0 is absorbed into b)."""
    X = torch.tensor(x.to_numpy(), dtype=torch.float64)
    Wt = torch.tensor(w, dtype=torch.float64)
    true_b = p["b"] + p["lam"] * G0
    theta = {
        "b": torch.tensor(true_b, dtype=torch.float64),
        "lam": torch.tensor(p["lam"], dtype=torch.float64),
        "log_s2": torch.tensor(np.log(p["s2"]), dtype=torch.float64),
        "g": torch.tensor(list(SLOPES.values()), dtype=torch.float64),
    }
    for v in theta.values():
        v.requires_grad_(True)
    opt = torch.optim.LBFGS(
        list(theta.values()), max_iter=1000, tolerance_grad=1e-10,
        tolerance_change=1e-14, line_search_fn="strong_wolfe",
    )

    def nll():
        opt.zero_grad()
        cov = PRIOR_SD2 * torch.outer(theta["lam"], theta["lam"]) + torch.diag(
            theta["log_s2"].exp()
        )
        mean = theta["b"] + (X @ theta["g"])[:, None] * theta["lam"]
        loss = -torch.distributions.MultivariateNormal(mean, cov).log_prob(Wt).sum()
        loss.backward()
        return loss

    opt.step(nll)
    grad = max(float(v.grad.abs().max()) for v in theta.values())
    return theta["g"].detach().numpy().tolist(), grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--per-task", type=int, default=100)
    ap.add_argument("--steps", default="10000,20000")
    ap.add_argument("--prior-lr", type=float, default=None, help="default: package's 1e-4")
    ap.add_argument("--out", default=str(c.RESULTS / "fig3_prior_mc"))
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]
    torch.set_num_threads(1)

    out = c.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"task_{args.task:03d}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["rep"] for line in path.read_text().splitlines() if line}
    p = design()
    meta = out / "design.json"
    if not meta.exists():
        c.write_json(meta, {
            "N": N, "J": J, "slopes": SLOPES, "g0": G0, "prior_sd2": PRIOR_SD2,
            "item_seed": ITEM_SEED, "data_seed": DATA_SEED,
            "params": {k: v.tolist() for k, v in p.items()},
            "information": float((p["lam"] ** 2 / p["s2"]).sum()),
            "steps": steps, "prior_lr": args.prior_lr, "provenance": c.provenance(),
        })

    reps = range(args.task * args.per_task, (args.task + 1) * args.per_task)
    for rep in reps:
        if rep in done:
            continue
        t0 = time.time()
        x, _, w = simulate(p, rep)
        g_mle, grad = mle(x, w, p)
        fits = fit(x, w, seed=rep, steps=steps, prior_lr=args.prior_lr)
        for f in fits:  # align the latent's sign with the truth via the loadings
            sign = float(np.sign(f.pop("lam_hat") @ p["lam"]))
            f["sign"] = sign
            f["g"] = [sign * v for v in f["g"]]
        rec = {"rep": rep, "g_mle": g_mle, "mle_max_grad": grad, "path": fits,
               "seconds": time.time() - t0}
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"rep {rep}: g {np.round(fits[-1]['g'], 3)} | mle {np.round(g_mle, 3)} | "
              f"{rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
