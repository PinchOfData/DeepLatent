"""Step 03 (stage 1): votes-only hierarchical ideal point model of the House.

    python 03_fit_votes.py --tag votes_103_118 --congresses 103-118 --steps 10000,25000,50000,100000

Model (IdealPointNN, one dimension), unit = legislator x congress:
    prior   theta_it ~ N(a_i + b_i * trend_it, 1)   (legislator intercept and linear trend, as
            DW-NOMINATE; members with fewer than 4 congresses in the sample get no trend;
            the variance is pinned at 1, which sets the scale)
    votes   P(yea_ij) = logistic(beta_j * theta_it + alpha_j) for the roll calls j of congress
            t; votes not cast are left out of the likelihood
Each congress's roll calls are their own vote modality (vote_c<t>: own encoder, own decoder),
and a unit has only its congress's modality (Corpus option "available"): roll calls of other
congresses are not part of the unit's data at all, unlike abstentions, which are missing votes
within its modality. Congresses are linked only through the legislator prior.
VAE with the true ELBO (w_prior = 1), mean-field Gaussian posterior, corrected PoE (a unit's
posterior is its one expert), encoders fed the congress's votes and the prior covariates (the
package default), linear decoders with intercepts; prior learning rate 1e-3; 256 units per step
(--batch-size 0: all units in every step). --align-every N: every N steps, a parameter-expansion
step moves each congress as a whole to where the prior wants it (see align_congresses; with
--align-stretch it also stretches each congress).

At each checkpoint: per-unit posterior mean and sd, and the prior mean a_i + b_i * trend_it, saved
with DW-NOMINATE (nominate_dim1: one score per legislator in Voteview's current release) and
Nokken-Poole (one score per legislator-congress); correlations overall and within party, after
aligning the sign with DW-NOMINATE. Outputs: results/<tag>/estimates_<step>.csv, summary.json;
the model at each checkpoint, model_<step>.ckpt (package save_model), in this run's own folder
/scratch/Gauthier/app2_house_ideal_points/checkpoints/<tag>/<run> (run_folder), recorded in
summary.json.
"""

import argparse
import contextlib
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import common as c

CKPT_ROOT = Path("/scratch/Gauthier/app2_house_ideal_points/checkpoints")


def load(congresses):
    """Units of the selected congresses and, per congress, the sparse vote matrix over its
    roll calls (rows: all units, +1 yea, -1 nay, 0 not cast or not a member of it)."""
    house = c.DATA / "house"
    units = pd.read_csv(house / "units.csv", dtype={"bioguide": str})
    rc = pd.read_csv(house / "rollcalls.csv")
    V = sp.load_npz(house / "votes.npz").tocsr()
    u = units[units["congress"].isin(congresses) & (units["n_votes"] > 0)].copy()
    r = rc[rc["congress"].isin(congresses)]
    Vu = V[u["unit"].to_numpy()]
    votes = {t: sp.csr_matrix(Vu[:, g["rollcall"].to_numpy()]) for t, g in r.groupby("congress")}
    n_cong = u.groupby("bioguide")["congress"].transform("nunique")
    centre = u.groupby("bioguide")["congress"].transform("mean")
    u["trend"] = np.where(n_cong >= 4, (u["congress"] - centre) / 5.0, 0.0)
    return u.reset_index(drop=True), r, votes


def key(t):
    return f"vote_c{t}"


def correlations(df, est):
    out = {}
    for ref in ("nominate_dim1", "nokken_poole_dim1"):
        ok = df[ref].notna()
        out[ref] = float(np.corrcoef(df.loc[ok, est], df.loc[ok, ref])[0, 1])
        for party, code in (("D", 100), ("R", 200)):
            m = ok & (df["party_code"] == code)
            out[f"{ref}_{party}"] = float(np.corrcoef(df.loc[m, est], df.loc[m, ref])[0, 1])
    return out


def start_values(model, corpus, units, votes):
    """Starting values that give every congress the same orientation (no constraint is imposed;
    every parameter is then estimated freely): each legislator's prior intercept starts at -1 for
    Democrats, +1 for Republicans, 0 otherwise, and each roll call's discrimination at +0.5 if
    Republicans voted yea more often than Democrats on it, -0.5 if less. No roll call spans two
    congresses, so otherwise each congress's latent axis starts with an arbitrary sign."""
    party = units.groupby("bioguide")["party_code"].last()
    side = party.map({100: -1.0, 200: 1.0}).fillna(0.0)
    cols = {name: j for j, name in enumerate(corpus.prevalence_colnames)}
    W = model.prior.mean_net.weight
    with torch.no_grad():
        W.zero_()
        model.prior.mean_net.bias.zero_()
        for leg, s in side.items():
            W[0, cols[f"C(bioguide)[{leg}]"]] = s
    def yea_rate(M, rows):
        S = M[rows]
        return np.asarray((S > 0).sum(0)).ravel() / np.maximum(1, np.asarray((S != 0).sum(0)).ravel())

    rep = (units["party_code"] == 200).to_numpy()
    dem = (units["party_code"] == 100).to_numpy()
    n_oriented = n_rollcalls = 0
    for t, M in votes.items():
        beta = 0.5 * np.sign(yea_rate(M, rep) - yea_rate(M, dem))
        lin = model.decoders[key(t)].decoder["dec_0"]
        with torch.no_grad():  # column 0: the ideal point; any content covariates start at 0
            lin.weight.zero_()
            lin.weight[:, 0] = torch.tensor(beta, dtype=lin.weight.dtype)
        n_oriented += int((beta != 0).sum())
        n_rollcalls += len(beta)
    print(f"start values: {int((side != 0).sum())} legislator intercepts at +-1, "
          f"{n_oriented} of {n_rollcalls} roll calls oriented by party", flush=True)


def start_truth(model, corpus, units, votes, rc, scale):
    """Diagnostic for simulated data (07_simulate_votes.py): start every parameter at the truth,
    on our scale theta = scale * truth. Prior intercept and trend of each legislator: their
    true positions' mean and linear slope over the career; roll-call decoders: logit =
    (beta_j / scale) * theta - beta_j * c_j, from the simulated cutpoints and discriminations."""
    sim = np.load(c.DATA / "house" / "sim_params.npz")
    u = units.assign(x=scale * sim["truth"][units["unit"].to_numpy()])
    cols = {name: j for j, name in enumerate(corpus.prevalence_colnames)}
    W = model.prior.mean_net.weight
    with torch.no_grad():
        W.zero_()
        model.prior.mean_net.bias.zero_()
        for leg, h in u.groupby("bioguide"):
            h = h[np.isfinite(h["x"])]
            if h.empty:
                continue
            W[0, cols[f"C(bioguide)[{leg}]"]] = float(h["x"].mean())
            name = f"C(bioguide)[{leg}]:trend"
            if name in cols and h["trend"].nunique() > 1:
                W[0, cols[name]] = float(np.polyfit(h["trend"], h["x"], 1)[0])
        for t, g in rc.groupby("congress"):
            j = g["rollcall"].to_numpy()
            lin = model.decoders[key(t)].decoder["dec_0"]
            beta = torch.tensor(sim["beta"][j], dtype=lin.weight.dtype)
            lin.weight.copy_((beta / scale).unsqueeze(1))
            lin.bias.copy_(-beta * torch.tensor(sim["cut"][j], dtype=lin.bias.dtype))
    print(f"start values: the truth (theta = {scale} x truth) for every legislator and roll call",
          flush=True)


def run_folder(tag):
    """This run's own folder for saved models, CKPT_ROOT/<tag>/<run>, <run> being the SLURM job id
    (job<id>) or, outside SLURM, the start time and process id. Created here and never reused,
    so runs never overwrite each other's models, even under the same tag."""
    job = os.environ.get("SLURM_JOB_ID")
    run = f"job{job}" if job else time.strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"
    folder = CKPT_ROOT / tag / run
    folder.mkdir(parents=True, exist_ok=False)
    return folder


@contextlib.contextmanager
def readout_rng(seed=0):
    """Seeded, reproducible draws for a readout between training segments, without touching
    training's random sequence: the global random state (CPU, and CUDA when present) is saved,
    seeded for the readout and restored after it. (A bare torch.manual_seed here made every
    training segment after a readout replay the same batches, modality draws and noise.)"""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        yield


def align_congresses(model, corpus, units, draws=50, stretch=False, modality_keys=None,
                     max_stretch=None, adam="rescale", content_shift=None):
    """Parameter-expansion step between training segments (the reduction step of PX-EM, Liu,
    Rubin & Wu 1998): move each congress as a whole to where the prior wants it.

    The votes of congress t cannot tell where congress t sits or how spread out it is: mapping
    all its positions x -> b * x + a, its roll calls' sharpness w -> w / b and intercepts
    c -> c - w * a / b leaves every vote probability unchanged. Only the prior links congresses,
    and gradient training barely moves a whole congress at once (its positions and ~1,000 roll
    calls would have to move in lockstep). So: choose (a_t, b_t) for every congress and the
    prior's coefficients to minimise the KL term, sum_i 0.5 [b^2 s_i^2 + (b mu_i + a - line_i)^2
    - 1 - log(b^2 s_i^2)] (posterior mean mu_i, sd s_i; prior variance pinned at 1, line_i the
    legislator's intercept + trend): given b, an OLS of b * mu on the prior's covariates plus
    congress dummies (the dummies give -a_t and are not part of the model); given a and the
    lines, b_t in closed form; alternated to convergence. stretch=False keeps b = 1 (shifts
    only: a single OLS). Then apply it: congress t's encoder mean output (last layer; a unit's
    posterior is its one congress's expert) -> b_t * mu + a_t, its raw log-variance + 2 log b_t
    (scales the precision increment Delta by 1 / b_t^2, so the posterior variance 1 / (1 +
    Delta) by about b_t^2 when the votes dominate the prior), its roll calls as above, and the
    prior's coefficients to the fit. The vote fit is unchanged, the KL improves by `gain`, and
    the model and its objective are unchanged. The overall location is kept (a common shift
    is unidentified). max_stretch caps each congress's stretch at one alignment to [1 /
    max_stretch, max_stretch] (a partial move; the shifts and prior are then refit given the
    capped stretches). A decoder with content covariates (step 05 --text shared-loadings feeds
    congress dummies to every decoder) keeps those columns: they do not multiply the position,
    so only its position column is edited. Adam's state of the edited parameters:
    adam="rescale" clears the momentum and rescales the running squared gradients with the
    parameters (a parameter multiplied by c has its gradient divided by c; otherwise a
    squeeze, b < 1, leaves stale, too-small second moments and Adam's next steps too large);
    adam="reset" deletes the state, so Adam restarts on those parameters (first steps of size
    about lr).

    modality_keys: congress -> its congress-specific modalities (default: its vote modality),
    each with a 1-D mean-field encoder and a linear decoder with intercepts, all edited alike
    (e.g. one text modality per congress, step 05: loadings / b_t, word intercepts - loading *
    a_t / b_t). With two experts, the fused posterior mean also involves the prior mean, which
    the refit moves, so it moves to b_t * mu + a_t only approximately (error (a_t - change in the
    unit's prior mean) / posterior precision); every decoder's predictions are unchanged.

    content_shift: {decoder key: {congress: column}} for a decoder shared by all congresses whose
    input has one column per congress (content covariates ~ 0 + C(congress), e.g. congress
    word intercepts in step 05 --text shared-loadings): congress t's column moves by -loading *
    a_t, so its predictions at the shifted positions are unchanged (shifts only). Its shared
    encoder is not edited, so a unit's position moves by a_t only through its congress-specific
    experts (approximately, by their share of its precision)."""
    from scipy.sparse.linalg import lsqr

    with readout_rng():
        mu, sd = model.get_ideal_points(corpus, num_samples=draws, return_std=True)
    mu, sd = mu[:, 0].astype(np.float64), sd[:, 0].astype(np.float64)
    L = sp.csr_matrix(np.asarray(corpus.M_prevalence_covariates, dtype=np.float64))
    P = L.shape[1]
    congresses = sorted(units["congress"].unique())
    cong = units["congress"].map({t: i for i, t in enumerate(congresses)}).to_numpy()
    rows = np.flatnonzero(cong > 0)
    C = sp.csr_matrix((np.ones(len(rows)), (rows, cong[rows] - 1)),
                      shape=(len(units), len(congresses) - 1))
    X = sp.hstack([L, C]).tocsr()
    fit = lambda A, y: lsqr(A, y, atol=1e-10, btol=1e-10, iter_lim=100000)[0]
    n = np.bincount(cong).astype(float)
    A = np.bincount(cong, weights=sd ** 2 + mu ** 2)
    b = np.ones(len(congresses))
    for _ in range(200 if stretch else 1):
        sol = fit(X, b[cong] * mu)
        coef, gamma = sol[:P], np.r_[0.0, sol[P:]]
        if not stretch:
            break
        B = np.bincount(cong, weights=mu * (-gamma[cong] - L @ coef))
        b_new = (-B + np.sqrt(B ** 2 + 4 * A * n)) / (2 * A)
        done = np.abs(b_new - b).max() < 1e-9
        b = b_new
        if done:
            break
    if stretch:
        if max_stretch:
            b = np.clip(b, 1 / max_stretch, max_stretch)
        sol = fit(X, b[cong] * mu)
        coef, gamma = sol[:P], np.r_[0.0, sol[P:]]
    k = mu.mean() - (b[cong] * mu - gamma[cong]).mean()  # keep the overall location
    a = -gamma + k
    new = b[cong] * mu + a[cong]
    line = L @ coef + k
    kl = lambda m, s2, ln: 0.5 * (s2 + (m - ln) ** 2 - np.log(s2)).sum()
    gain = kl(mu, sd ** 2, L @ fit(L, mu)) - kl(new, (b[cong] * sd) ** 2, line)
    intercept = np.array([":" not in name for name in corpus.prevalence_colnames])

    edited = []

    def rescale_sq(p, factor, index=None):
        state = model.optimizer.state.get(p, {})
        if "exp_avg_sq" in state:
            (state["exp_avg_sq"] if index is None else state["exp_avg_sq"][index]).mul_(factor)

    with torch.no_grad():
        for i, t in enumerate(congresses):
            for mk in (modality_keys or {}).get(t, [key(t)]):
                layers = model.encoder.encoders[mk].encoder
                last = layers[f"enc_{len(layers) - 1}"]  # mean_field 1-D: [mu, raw logvar]
                dec = model.decoders[mk].decoder["dec_0"]
                if stretch:
                    # only the position's column: other decoder inputs (content covariates,
                    # e.g. congress dummies) do not multiply the position and stay
                    last.weight[0] *= float(b[i])
                    last.bias[0] *= float(b[i])
                    last.bias[1] += 2 * float(np.log(b[i]))
                    dec.weight[:, 0] /= float(b[i])
                    if adam == "rescale":
                        rescale_sq(last.weight, 1 / float(b[i]) ** 2, 0)
                        rescale_sq(last.bias, 1 / float(b[i]) ** 2, 0)
                        rescale_sq(dec.weight, float(b[i]) ** 2, (slice(None), 0))
                    edited += [last.weight, dec.weight]
                last.bias[0] += float(a[i])
                dec.bias -= dec.weight[:, 0] * float(a[i])
                edited += [last.bias, dec.bias]
        if content_shift:
            assert not stretch, "content_shift supports shifts only"
            for dk, colmap in content_shift.items():
                Wd = model.decoders[dk].decoder["dec_0"].weight
                for i, t in enumerate(congresses):
                    Wd[:, colmap[t]] -= Wd[:, 0] * float(a[i])
                edited.append(Wd)
        W, bias = model.prior.mean_net.weight, model.prior.mean_net.bias
        W[0] = torch.tensor(coef + k * intercept, dtype=W.dtype, device=W.device)
        bias.zero_()
        edited += [W, bias]
    for p in edited:
        if adam == "reset":
            model.optimizer.state.pop(p, None)
            continue
        state = model.optimizer.state.get(p, {})
        if "exp_avg" in state:
            state["exp_avg"].zero_()
    t_c = np.array(congresses, dtype=float) - np.mean(congresses)
    return {"step": int(model.steps), "gain": float(gain),
            "shift_slope_per_congress": float(np.polyfit(t_c, a, 1)[0]),
            "shift_max_abs": float(np.abs(a).max()),
            "stretch_min": float(b.min()), "stretch_max": float(b.max()),
            "stretch_slope_per_congress": float(np.polyfit(t_c, b, 1)[0]),
            "shift": {str(t): round(float(v), 4) for t, v in zip(congresses, a)},
            "stretch": {str(t): round(float(v), 5) for t, v in zip(congresses, b)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--congresses", default="103-118")
    ap.add_argument("--steps", default="10000,25000,50000,100000")
    ap.add_argument("--hidden", default="256,64")
    ap.add_argument("--lr", type=float, default=1e-3, help="encoder and decoder learning rate")
    ap.add_argument("--prior-lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256, help="units per step; 0 = all units")
    ap.add_argument("--align-every", type=int, default=0,
                    help="every this many steps, move each congress to where the prior wants it "
                         "(align_congresses); 0 = never")
    ap.add_argument("--align-stretch", action="store_true",
                    help="the alignment also stretches each congress (default: shifts only)")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--print-every", type=int, default=1000)
    ap.add_argument("--start", choices=["party", "none", "truth"], default="party",
                    help="truth: simulated data only (diagnostic), see start_truth")
    ap.add_argument("--truth-scale", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4, help="persistent data-loading workers")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior", choices=["trend", "constant"], default="trend",
                    help="trend: legislator intercept + linear trend (default); constant: "
                         "intercept only (diagnostic)")
    args = ap.parse_args()
    lo, hi = map(int, args.congresses.split("-"))
    steps = [int(s) for s in args.steps.split(",")]
    out = c.MODULE / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from deeplatent import Corpus, IdealPointNN
    import deeplatent

    units, rc, votes = load(range(lo, hi + 1))
    keys = [key(t) for t in votes]
    views = {
        f"c{t}": {"type": "vote", "matrix": M, "available": (units["congress"] == t).to_numpy()}
        for t, M in votes.items()
    }
    corpus = Corpus(
        units[["bioguide", "trend"]],
        prevalence="~ 0 + C(bioguide)" + (" + C(bioguide):trend" if args.prior == "trend" else ""),
        modalities={"vote": {"views": views}},
    )
    n_votes = sum(M.nnz for M in votes.values())
    print(f"{len(units):,} units, {units['bioguide'].nunique():,} legislators, {len(rc):,} roll "
          f"calls in {len(votes)} congresses (one vote modality each), {n_votes:,} votes; "
          f"{corpus.M_prevalence_covariates.shape[1]:,} prior covariates "
          f"({time.time() - t0:.0f}s)", flush=True)

    ckpt = run_folder(args.tag)
    print(f"models saved in {ckpt}", flush=True)
    model = IdealPointNN(
        train_data=corpus,
        n_ideal_points=1,
        ae_type="vae",
        vi_type="mean_field",
        update_prior=True,
        learn_prior_cov=False,
        fusion="corrected_poe",
        w_prior=1.0,
        encoder_args={k: {"hidden_dims": [int(h) for h in args.hidden.split(",")]} for k in keys},
        decoder_args={k: {"hidden_dims": [], "bias": True} for k in keys},
        batch_size=args.batch_size or len(units),
        num_steps=0,
        num_workers=args.workers,
        print_every_n_steps=args.print_every,
        return_best_model=False,
        ckpt_folder=str(ckpt),
        seed=args.seed,
        device=args.device,
        optim_args={"main": {"lr": args.lr}, "prior": {"lr": args.prior_lr, "weight_decay": 0.0},
                    "predictor": {"lr": 1e-3}},
    )
    if args.start == "truth":
        start_truth(model, corpus, units, votes, rc, args.truth_scale)
    if args.start == "party":
        start_values(model, corpus, units, votes)
    pkg = Path(deeplatent.__file__).resolve().parent
    record = {
        "args": vars(args), "units": len(units), "legislators": int(units["bioguide"].nunique()),
        "rollcalls": len(rc), "congresses": len(votes), "votes": int(n_votes),
        "prior_covariates": int(corpus.M_prevalence_covariates.shape[1]),
        "deeplatent": str(pkg),
        "md5": {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in sorted(pkg.glob("*.py"))},
        "checkpoint_folder": str(ckpt), "checkpoints": [], "alignments": [],
    }
    Z = torch.tensor(corpus.M_prevalence_covariates, device=args.device)
    stops = set(steps)
    if args.align_every:
        stops |= set(range(args.align_every, max(steps) + 1, args.align_every))
    for step in sorted(stops):
        model.num_steps = step
        model.train(corpus)
        if args.align_every and step % args.align_every == 0:
            info = align_congresses(model, corpus, units, stretch=args.align_stretch)
            record["alignments"].append(info)
            print("ALIGN", json.dumps({k: v for k, v in info.items()
                                       if k not in ("shift", "stretch")}), flush=True)
        if step not in steps:
            continue
        with readout_rng():
            mean, sd = model.get_ideal_points(corpus, num_samples=args.draws, return_std=True)
        with torch.no_grad():
            prior_mean = model.prior.mean_net(Z).cpu().numpy()[:, 0]
        est = units[["unit", "congress", "bioguide", "party_code", "state_abbrev", "trend",
                     "n_votes", "nominate_dim1", "nokken_poole_dim1"]].copy()
        est["theta_mean"], est["theta_sd"], est["prior_mean"] = mean[:, 0], sd[:, 0], prior_mean
        sign = np.sign(np.corrcoef(est["theta_mean"], est["nominate_dim1"].fillna(0))[0, 1])
        for k in ("theta_mean", "prior_mean"):
            est[k] *= sign
        est.to_csv(out / f"estimates_{step}.csv", index=False)
        model.save_model(str(ckpt / f"model_{step}.ckpt"))
        rec = {"step": step, "minutes": (time.time() - t0) / 60, "sign": float(sign),
               "model": str(ckpt / f"model_{step}.ckpt"),
               "theta": correlations(est, "theta_mean"),
               "prior_mean": correlations(est, "prior_mean"),
               "theta_sd_median": float(est["theta_sd"].median())}
        record["checkpoints"].append(rec)
        (out / "summary.json").write_text(json.dumps(record, indent=1))
        print("CKPT", json.dumps(rec), flush=True)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
