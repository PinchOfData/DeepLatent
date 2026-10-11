"""Shared designs, data-generating processes, model settings and exact references for the
paper's simulation figures (see ../README.md).

Folder layout (on the HPC: /home/Gauthier/simulations, synced from replication/simulations):
    code/      this code        slurm/   job scripts
    results/   estimates written by the jobs, fetched back to the laptop with fetch_results.sh
SIMS_RESULTS overrides the results folder.

Every figure uses a one-dimensional latent theta. Figures 1-2 use the stock IdealPointNN in its
likelihood-based configuration: VAE with the true ELBO (w_prior = 1), mean-field Gaussian
posterior, fixed N(0, 1) prior (which pins the latent's location and scale, leaving only its sign),
corrected product-of-experts fusion, deep encoders and linear decoders with intercepts.

Exact references: for each unit and modality set, the exact posterior p(theta | data) under the
true item parameters and the N(0, 1) prior, computed on a fine grid, together with its
reverse-KL-optimal Gaussian (the best member of the model's variational family).
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logsumexp
from sklearn.feature_extraction.text import CountVectorizer

MODULE = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("SIMS_RESULTS", MODULE / "results"))

GRID = np.linspace(-7.0, 7.0, 1401)  # exact posteriors; theta ~ N(0, 1)

# Model settings for Figures 1-2.
HIDDEN = [64, 64]
BATCH = 256
READOUT_DRAWS = 2000  # posterior draws per unit for the reported mean and sd

# Item designs. Each likelihood is tuned to a comparable average posterior precision in
# Figure 1 and to clearly different precisions in Figure 2 (see results/*/summary.json).
FIG1_DESIGN = {
    "embedding": {"J": 50, "s_lam": 0.45},
    "survey": {"Q": 25, "s_a": 1.0},
    "bow": {"V": 1000, "C": 100, "s_a": 0.33},
    "vote": {"B": 60},
}
FIG2_DESIGN = {
    "vote": {"B": 60},
    "survey": {"Q": 12, "s_a": 1.0},
    "bow": {"V": 1000, "C": 15, "s_a": 0.33},
}


# ---------------------------------------------------------------------------------------
# Data-generating processes
# ---------------------------------------------------------------------------------------
def item_params(kind, cfg, seed):
    """Fixed item parameters of one modality (drawn once per design, not per replication)."""
    rng = np.random.default_rng(seed)
    if kind == "embedding":  # w_j = b_j + lam_j theta + e_j, e_j ~ N(0, s2_j)
        J = cfg["J"]
        return {
            "lam": rng.normal(0.0, cfg["s_lam"], J),
            "b": rng.normal(0.0, 1.0, J),
            "s2": rng.uniform(0.5, 1.5, J),
        }
    if kind == "survey":  # P(y_q = k) = softmax_k(a_qk theta + c_qk), K_q in {2,...,5}
        K = rng.integers(2, 6, cfg["Q"])
        return {
            "a": [rng.normal(0.0, cfg["s_a"], k) for k in K],
            "c": [rng.normal(0.0, 1.0, k) for k in K],
        }
    if kind == "bow":  # C tokens ~ Multinomial(softmax_v(a_v theta + c_v))
        V = cfg["V"]
        return {
            "a": rng.normal(0.0, cfg["s_a"], V),
            "c": rng.normal(0.0, 1.0, V),
            "C": cfg["C"],
        }
    if kind == "vote":  # P(y_b = 1) = logistic(beta_b theta - delta_b)
        B = cfg["B"]
        return {"beta": rng.normal(0.0, 1.0, B), "delta": rng.normal(0.0, 0.5, B)}
    raise ValueError(kind)


def simulate(kind, theta, p, rng):
    """Draw one modality's data for latent positions theta [N]."""
    N = len(theta)
    if kind == "embedding":
        return (
            p["b"] + np.outer(theta, p["lam"]) + rng.normal(size=(N, len(p["b"]))) * np.sqrt(p["s2"])
        )
    if kind == "survey":
        cols = []
        for a, c in zip(p["a"], p["c"]):
            L = np.outer(theta, a) + c
            P = np.exp(L - logsumexp(L, axis=1, keepdims=True))
            u = rng.uniform(size=(N, 1))
            cols.append((u > np.cumsum(P, axis=1)).sum(1))
        return np.stack(cols, axis=1)  # [N, Q] category indices
    if kind == "bow":
        L = np.outer(theta, p["a"]) + p["c"]
        P = np.exp(L - logsumexp(L, axis=1, keepdims=True))
        return rng.multinomial(p["C"], P)  # [N, V] counts
    if kind == "vote":
        return rng.binomial(1, expit(np.outer(theta, p["beta"]) - p["delta"])).astype(float)
    raise ValueError(kind)


MODALITY_KEY = {
    "embedding": "embedding_features",
    "survey": "survey_responses",
    "bow": "text_bow",
    "vote": "vote_responses",
}


def build_corpus(data):
    """Corpus with one modality per entry of data = {kind: simulated array}."""
    N = len(next(iter(data.values())))
    df = pd.DataFrame(index=range(N))
    mods = {}
    for kind, X in data.items():
        mod, view = MODALITY_KEY[kind].split("_")
        if kind == "embedding":
            df["emb"] = 0.0
            spec = {"type": "embedding", "matrix": X.astype(np.float32)}
            mods[mod] = {"column": "emb", "views": {view: spec}}
        elif kind == "survey":
            cols = [f"Q{q:02d}" for q in range(X.shape[1])]
            df[cols] = X
            mods[mod] = {"column": cols, "views": {view: {"type": "discrete_choice"}}}
        elif kind == "bow":
            vocab = [f"w{v:04d}" for v in range(X.shape[1])]
            df["text"] = [
                " ".join(np.repeat(vocab, row).tolist()) for row in X
            ]  # tokens of each document (order is irrelevant to a bag of words)
            vec = CountVectorizer(vocabulary=vocab)  # keeps the true column order
            mods[mod] = {"column": "text", "views": {view: {"type": "bow", "vectorizer": vec}}}
        elif kind == "vote":
            cols = [f"b{b:03d}" for b in range(X.shape[1])]
            df[cols] = X
            mods[mod] = {"column": cols, "views": {view: {"type": "vote"}}}
    from deeplatent import Corpus

    corpus = Corpus(df, modalities=mods)
    if "bow" in data:  # the text pipeline must reproduce the simulated counts exactly
        M = corpus.processed_modalities["text"]["bow"]["matrix"]
        M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
        assert np.array_equal(M, data["bow"]), "bag-of-words counts differ from the DGP"
    return corpus


# ---------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------
def build_model(corpus, kinds, seed, device="cpu"):
    from deeplatent import IdealPointNN

    keys = [MODALITY_KEY[k] for k in kinds]
    return IdealPointNN(
        train_data=corpus,
        n_ideal_points=1,
        ae_type="vae",
        vi_type="mean_field",
        update_prior=False,
        fusion="corrected_poe",
        w_prior=1.0,
        encoder_args={k: {"hidden_dims": HIDDEN} for k in keys},
        decoder_args={k: {"hidden_dims": [], "bias": True} for k in keys},
        batch_size=BATCH,
        num_steps=0,
        num_workers=0,
        print_every_n_steps=2000,
        return_best_model=False,
        ckpt_folder=tempfile.mkdtemp(),
        seed=seed,
        device=device,
    )


def train_to(model, corpus, step):
    model.num_steps = step
    model.train(corpus)


def readout(model, corpus, kind=None, seed=0):
    """Posterior mean and sd of theta per unit (from READOUT_DRAWS draws), fused over all
    modalities or from one modality's expert alone."""
    torch.manual_seed(seed)
    mean, sd = model.get_ideal_points(
        corpus,
        single_modality=MODALITY_KEY[kind] if kind else None,
        num_samples=READOUT_DRAWS,
        return_std=True,
    )
    return mean[:, 0], sd[:, 0]


def cleanup(model):
    shutil.rmtree(model.ckpt_folder, ignore_errors=True)


# ---------------------------------------------------------------------------------------
# Exact posteriors on the grid
# ---------------------------------------------------------------------------------------
def grid_terms(kind, X, p):
    """Log-likelihood of each unit on GRID and its first two theta-derivatives, [n, G] each."""
    G = GRID
    if kind == "embedding":
        A = ((X - p["b"]) * p["lam"] / p["s2"]).sum(1)  # [n]
        I = float((p["lam"] ** 2 / p["s2"]).sum())
        ll = A[:, None] * G[None, :] - 0.5 * I * G[None, :] ** 2
        return ll, A[:, None] - I * G[None, :], np.full_like(ll, -I)
    if kind == "survey":
        n = len(X)
        ll, g1, g2 = (np.zeros((n, len(G))) for _ in range(3))
        for q, (a, c) in enumerate(zip(p["a"], p["c"])):
            L = G[:, None] * a[None, :] + c[None, :]
            logp = L - logsumexp(L, axis=1, keepdims=True)  # [G, K]
            pw = np.exp(logp)
            Ea, Ea2 = pw @ a, pw @ a**2
            y = X[:, q].astype(int)
            ll += logp[:, y].T
            g1 += a[y][:, None] - Ea[None, :]
            g2 -= (Ea2 - Ea**2)[None, :]
        return ll, g1, g2
    if kind == "bow":
        L = G[:, None] * p["a"][None, :] + p["c"][None, :]
        logp = L - logsumexp(L, axis=1, keepdims=True)  # [G, V]
        pw = np.exp(logp)
        Ea, Ea2 = pw @ p["a"], pw @ p["a"] ** 2
        C = X.sum(1, keepdims=True)
        return X @ logp.T, (X @ p["a"])[:, None] - C * Ea[None, :], -C * (Ea2 - Ea**2)[None, :]
    if kind == "vote":
        Z = G[None, :, None] * p["beta"][None, None, :] - p["delta"][None, None, :]
        sig = expit(Z)
        Y = X[:, None, :]
        ll = -(np.logaddexp(0, -Z) * Y + np.logaddexp(0, Z) * (1 - Y)).sum(-1)
        g1 = ((Y - sig) * p["beta"]).sum(-1)
        g2 = -(sig * (1 - sig) * p["beta"] ** 2).sum(-1)
        return ll, g1, g2
    raise ValueError(kind)


def posterior_summaries(ll, g1, g2, iters=200):
    """Exact posterior mean/var (prior N(0, 1)) and the reverse-KL-optimal Gaussian N(m, s2),
    from the stationarity conditions E_q[h'] = 0 and 1/s2 = -E_q[h''] with h the log posterior
    (fixed-point iteration on the grid)."""
    h = ll - 0.5 * GRID[None, :] ** 2
    h1, h2 = g1 - GRID[None, :], g2 - 1.0
    w = np.exp(h - h.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    mean = w @ GRID
    var = w @ GRID**2 - mean**2
    m, s2 = mean.copy(), var.copy()
    for _ in range(iters):
        q = np.exp(-0.5 * (GRID[None, :] - m[:, None]) ** 2 / s2[:, None])
        q /= q.sum(1, keepdims=True)
        s2_new = -1.0 / (q * h2).sum(1)
        m_new = m + s2_new * (q * h1).sum(1)
        done = np.max(np.abs(m_new - m)) < 1e-10 and np.max(np.abs(s2_new - s2)) < 1e-12
        m, s2 = m_new, s2_new
        if done:
            break
    return {"mean": mean, "var": var, "m": m, "s2": s2}


def exact_reference(kinds, data, params, chunk=200):
    """Per-unit exact posterior summaries given the modalities in kinds (jointly)."""
    N = len(next(iter(data.values())))
    parts = []
    for lo in range(0, N, chunk):
        terms = [grid_terms(k, data[k][lo : lo + chunk], params[k]) for k in kinds]
        parts.append(posterior_summaries(*(sum(t[i] for t in terms) for i in range(3))))
    return {f: np.concatenate([p[f] for p in parts]) for f in parts[0]}


# ---------------------------------------------------------------------------------------
# Summaries and provenance
# ---------------------------------------------------------------------------------------
def panel_summary(theta, mean, sd, ref):
    """Fit statistics of one panel; the model's sign is aligned with the truth first."""
    sign = float(np.sign(np.corrcoef(mean, theta)[0, 1]))
    mean = sign * mean
    return {
        "sign": sign,
        "corr_truth": float(np.corrcoef(mean, theta)[0, 1]),
        "coverage95": float(np.mean(np.abs(theta - mean) <= 1.96 * sd)),
        "corr_exact_mean": float(np.corrcoef(mean, ref["mean"])[0, 1]),
        "rmse_exact_mean": float(np.sqrt(np.mean((mean - ref["mean"]) ** 2))),
        "precision_fitted": float(np.mean(1.0 / sd**2)),
        "precision_exact_rkl": float(np.mean(1.0 / ref["s2"])),
        "precision_exact_var": float(np.mean(1.0 / ref["var"])),
        "precision_unit_corr": float(np.corrcoef(1.0 / sd**2, 1.0 / ref["s2"])[0, 1]),
        "coverage95_exact": float(
            np.mean(np.abs(theta - ref["mean"]) <= 1.96 * np.sqrt(ref["var"]))
        ),
    }


def provenance():
    """Where deeplatent was imported from and the md5 of every module (logged in results)."""
    import deeplatent

    pkg = Path(deeplatent.__file__).resolve().parent
    return {
        "deeplatent": str(pkg),
        "torch": torch.__version__,
        "md5": {
            f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in sorted(pkg.glob("*.py"))
        },
    }


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
