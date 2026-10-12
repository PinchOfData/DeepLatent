"""Step 09: the exact speeches-only posterior of a saved step-05 model, against its speech encoder.

    python 09_speech_posterior_exact.py --tag votes_text_103_118_baselines_v2 --step 80000

For every legislator-congress with speeches: the model's own posterior of the position given the
speeches alone (and the prior covariates),
    p(theta | speeches) ∝ N(theta; prior mean, prior variance) x prod_w p_tw(theta)^c_w,
    p_tw(theta) = softmax_w(beta_w theta + a_tw),
with the saved word model (loadings beta, congress-t word baselines a_t incl. the common
intercept). log p(speeches | theta) = theta * sum_w c_w beta_w - n * logsumexp_w(beta_w theta +
a_tw) + const, and the logsumexp depends on the congress only, so the posterior is computed
exactly (up to the grid) on one fine grid of positions, in float64.

Compared with the speech encoder's answer, the speech expert as training forms it:
N(mu, 1/(prior precision + Delta)), mu and Delta = exp(-raw) from the encoder (eval mode), and
KL(encoder || exact), the speeches-only bound's gap for that unit. Means are signed like the
run's estimates. Writes results/<tag>/speech_posterior_<step>.csv and prints a summary by length.
"""

import argparse
import importlib
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import common as c

fit03 = importlib.import_module("03_fit_votes")
words08 = importlib.import_module("08_words_from_model")
TEXT = "speech_terms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--step", default="80000", help="checkpoint step, or warm_<N>")
    ap.add_argument("--grid-step", type=float, default=0.002)
    ap.add_argument("--margin", type=float, default=10.0, help="grid beyond the means' range")
    args = ap.parse_args()
    out = c.MODULE / "results" / args.tag
    record = json.loads((out / "summary.json").read_text())
    warm = args.step.startswith("warm_")
    step = int(args.step.removeprefix("warm_"))
    rec = next(r for r in record["checkpoints"]
               if r["step"] == step and ("phase" in r) == warm)
    sign = rec["sign"]
    ckpt = torch.load(rec["model"], map_location="cpu", weights_only=False)
    if TEXT not in ckpt["decoders"]:
        raise SystemExit("this model has no word model (--text encode-only): nothing to compute")

    lo, hi = map(int, record["args"]["congresses"].split("-"))
    units, rc, votes = fit03.load(range(lo, hi + 1))
    house = c.DATA / "house"
    S = sp.load_npz(house / "speech_counts.npz").tocsr()[units["unit"].to_numpy()]
    vocab = [w for w in (house / "vocab.txt").read_text().split("\n") if w]
    from deeplatent import Corpus
    from deeplatent.autoencoders import precision_increment

    corpus = Corpus(units[["bioguide", "trend", "congress"]],
                    prevalence="~ 0 + C(bioguide) + C(bioguide):trend",
                    content="~ 0 + C(congress)",
                    modalities={"speech": {"views": {"terms": {"type": "bow", "matrix": S,
                                                               "vocab": vocab}}}})
    assert list(corpus.prevalence_colnames) == list(ckpt["prevalence_colnames"]), \
        "prior covariates differ from the saved model's"
    assert list(corpus.content_colnames) == list(ckpt["content_colnames"]), \
        "content covariates differ from the saved model's"
    Z = torch.tensor(np.asarray(corpus.M_prevalence_covariates), dtype=torch.float32)
    X = np.asarray(corpus.M_content_covariates)

    # prior N(m_i, s2), as training uses it
    prior = ckpt["prior"].cpu()
    with torch.no_grad():
        m0 = prior.mean_net(Z)[:, 0].double().numpy()
        s2 = float(prior.sigma[0, 0])
    lam0 = 1.0 / (s2 + 1e-8)

    # word model: logits = beta * theta + a_t
    dec = ckpt["decoders"][TEXT]
    W = dec["decoder.dec_0.weight"].double().numpy()
    bias = dec["decoder.dec_0.bias"].double().numpy() if "decoder.dec_0.bias" in dec else 0.0
    beta = W[:, 0]
    congress = units["congress"].to_numpy()
    base = {t: W[:, 1:] @ X[np.flatnonzero(congress == t)[0]] + bias
            for t in sorted(set(congress))}

    # speech encoder: mean and raw log precision increment (eval mode: no dropout)
    enc = words08.speech_encoder(ckpt["encoder"])
    rows = np.flatnonzero(np.asarray(S.sum(1)).ravel() > 0)
    mu_e, raw_e = np.empty(len(rows)), np.empty(len(rows))
    with torch.no_grad():
        for b in range(0, len(rows), 512):
            r = rows[b:b + 512]
            x = torch.tensor(np.hstack([S[r].toarray().astype(np.float32), Z[r].numpy()]))
            o = enc(x)
            mu_e[b:b + 512], raw_e[b:b + 512] = o[:, 0].numpy(), o[:, 1].numpy()
    delta = precision_increment(torch.tensor(raw_e)).numpy()
    sd_e = 1.0 / np.sqrt(lam0 + delta)

    n = np.asarray(S[rows].sum(1)).ravel().astype(np.float64)
    s = S[rows] @ beta  # sum_w c_w beta_w
    lo_g = min(mu_e.min(), m0[rows].min()) - args.margin
    hi_g = max(mu_e.max(), m0[rows].max()) + args.margin
    grid = np.arange(lo_g, hi_g + args.grid_step, args.grid_step)
    lse = {}
    for t, a in base.items():  # logsumexp_w(beta_w theta + a_tw) on the grid, per congress
        bt, at = torch.tensor(beta), torch.tensor(a)
        lse[t] = np.concatenate([torch.logsumexp(torch.tensor(g)[:, None] * bt + at, 1).numpy()
                                 for g in np.array_split(grid, max(1, len(grid) // 500))])
    res = {k: np.empty(len(rows)) for k in ("exact_mean", "exact_sd", "kl", "edge_mass")}
    for t in base:
        k = np.flatnonzero(congress[rows] == t)
        for b in range(0, len(k), 256):
            kk = k[b:b + 256]
            lp = (-0.5 * lam0 * (grid[None] - m0[rows][kk, None]) ** 2
                  + s[kk, None] * grid[None] - n[kk, None] * lse[t][None])
            lp -= lp.max(1, keepdims=True)
            p = np.exp(lp)
            z = p.sum(1, keepdims=True) * args.grid_step
            logp = lp - np.log(z)  # normalized log density on the grid
            p = p / z
            mean = (p * grid).sum(1) * args.grid_step
            sd = np.sqrt((p * (grid[None] - mean[:, None]) ** 2).sum(1) * args.grid_step)
            # KL(N(mu_e, sd_e) || exact), by quadrature on the same grid
            logq = (-0.5 * ((grid[None] - mu_e[kk, None]) / sd_e[kk, None]) ** 2
                    - np.log(sd_e[kk, None] * np.sqrt(2 * np.pi)))
            q = np.exp(logq)
            res["kl"][kk] = (q * (logq - logp)).sum(1) * args.grid_step
            res["exact_mean"][kk], res["exact_sd"][kk] = mean, sd
            res["edge_mass"][kk] = (p[:, :50].sum(1) + p[:, -50:].sum(1)) * args.grid_step
    assert res["edge_mass"].max() < 1e-6, "posterior mass at the grid's edge: widen --margin"
    assert (sd_e / args.grid_step).min() > 10 and (res["exact_sd"] / args.grid_step).min() > 10, \
        "grid too coarse for the narrowest posterior: lower --grid-step"

    d = units.iloc[rows][["unit", "congress", "bioguide", "party_code"]].reset_index(drop=True)
    d = d.assign(n_terms=n, prior_mean=sign * m0[rows], exact_mean=sign * res["exact_mean"],
                 exact_sd=res["exact_sd"], enc_mean=sign * mu_e, enc_sd=sd_e, kl=res["kl"])
    est = pd.read_csv(out / f"estimates_{args.step}.csv")
    keep = [k for k in ("theta_votes", "theta_votes_sd", "theta_text", "theta_text_sd") if k in est]
    d = d.merge(est[["unit"] + keep], on="unit", how="left")
    d.to_csv(out / f"speech_posterior_{args.step}.csv", index=False)

    print(f"{args.tag}, {args.step}: {rec['model']}; {len(d):,} units with speeches; "
          f"prior variance {s2:.4f}; grid {len(grid):,} points of {args.grid_step}")
    print(f"encoder vs exact: means r {np.corrcoef(d.enc_mean, d.exact_mean)[0, 1]:.3f}, "
          f"rms difference {np.sqrt(((d.enc_mean - d.exact_mean) ** 2).mean()):.3f}; "
          f"sd ratio encoder/exact median {np.median(d.enc_sd / d.exact_sd):.2f}; "
          f"KL mean {d.kl.mean():.2f} nats, median {d.kl.median():.2f}")
    if "theta_text" in d:
        print(f"check: encoder mean vs the run's speeches-only readout r "
              f"{np.corrcoef(d.enc_mean, d.theta_text)[0, 1]:.4f}")
    d["bin"] = pd.qcut(d.n_terms, 10, labels=False)
    agg = {"words": ("n_terms", "median"), "exact_sd": ("exact_sd", "median"),
           "enc_sd": ("enc_sd", "median"), "kl_mean": ("kl", "mean")}
    t = d.groupby("bin").agg(**agg)

    def rms(x):
        return np.sqrt((x ** 2).groupby(d["bin"]).mean())

    t["rms_mean_diff"] = rms(d.enc_mean - d.exact_mean)
    if "theta_votes" in d:
        t["rms_exact_vs_votes"] = rms(d.exact_mean - d.theta_votes)
        t["rms_enc_vs_votes"] = rms(d.enc_mean - d.theta_votes)
        t["expected_exact_vs_votes"] = rms(np.sqrt(d.exact_sd ** 2 + d.theta_votes_sd ** 2))
    print("\nby length decile (medians of sd; rms of differences):")
    print(t.round(3).to_string())
    print(f"\nwrote {out / f'speech_posterior_{args.step}.csv'}")


if __name__ == "__main__":
    main()
