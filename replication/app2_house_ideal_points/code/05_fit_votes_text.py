"""Step 05 (stage 2): votes and speeches. The hierarchical House model of step 03 with the
legislators' floor speeches as a second modality.

    python 05_fit_votes_text.py --tag votes_text_103_118 --congresses 103-118 --steps 5000,10000,20000,40000

Same units, prior, vote modalities (one per congress, vote_c<t>) and starting values as step 03,
plus the speeches: the unit's House floor speeches aggregated into term counts over the
32,668-term vocabulary (data/house/speech_counts.npz), with a multinomial likelihood and a linear
decoder with word intercepts, i.e. a multinomial Wordfish (log-odds of term w: alpha_w + beta_w *
theta). Default (--text per-congress): one text modality per congress, speech_c<t>, with its own
word loadings, intercepts and encoder, like the roll calls, so the text measures positions within
a congress and only the legislator prior links congresses (text no longer carries language
change across congresses, and the alignment step of step 03 stays exact for the decoders); the
alignment (--align-every, --align-stretch) moves each congress's vote and text encoders and
decoders together. --text shared-loadings: the pooled design (one text modality speech_terms,
one text encoder, one decoder with word loadings shared by all congresses) plus word intercepts
by congress (content covariates ~ 0 + C(congress), which enter every decoder: for a vote
decoder they only duplicate its roll calls' intercepts). A congress's shift is absorbed exactly
by its word intercepts, so the alignment moves each congress's vote encoder and roll calls and
its word intercepts; the shared text encoder cannot be moved by congress, so a unit with
speeches moves only through its vote expert (by about its share of the unit's precision) and
repeated alignments do the rest. A stretch is not absorbed (the shared loadings see each
congress's spread): --align-stretch is refused and the spreads are set by the text and the
prior through training (the warm start below may still stretch: the speeches are off then, and
only each decoder's position column is rescaled). --text pooled: the earlier design, one text modality speech_terms for
all congresses (shared loadings and intercepts). Units without speeches lack the text modality (Corpus option
"available"). Text encoders: deep MLP (default 64,64, dropout 0.2 on its hidden layers) fed the
raw term counts and the prior covariates; vote encoders as in step 03. Word intercepts start at
log frequencies, word loadings at 0 (--text-loadings-start). With two modalities per
unit, training drops a modality at random per unit, so each expert also learns to infer the
ideal point alone.

Warm start (--warm-votes-steps N): the model is first trained N steps on the votes alone (the
speeches switched off: every unit then fuses its vote expert only, and the text encoder and
decoder get no gradient), with the alignment of step 03 (--warm-align-every, --warm-align-stretch),
which is exact there; then the speeches are switched on and training continues from step N to
the last --steps (alignment then per --align-every; 0 = none). The congresses thus start where
the votes and the legislator prior put them, and any later movement comes from the text.
Writes results/<tag>/estimates_warm_<N>.csv (votes-only posterior at the end of the warm start).

At each checkpoint, per unit: the fused posterior (votes and text), the votes-only and the
text-only posteriors, and each modality's precision increment Delta_m over the prior (prior
precision 1, so the fused precision is 1 + Delta_votes + Delta_text), with the text share
Delta_text / (Delta_votes + Delta_text); correlations of each estimate with DW-NOMINATE and
Nokken-Poole. Outputs: results/<tag>/estimates_<step>.csv, summary.json; the model at each
checkpoint (model_<step>.ckpt, and model_warm_<N>.ckpt at the end of the warm start) in this run's
own folder, as in step 03.
"""

import argparse
import copy
import hashlib
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import common as c

fit03 = importlib.import_module("03_fit_votes")  # load, key, start_values, run_folder
TEXT = "speech_terms"


def correlations(df, est):
    """Correlations with DW-NOMINATE and Nokken-Poole, overall and within party, over the units
    where both are defined (the text-only estimate is missing for units without speeches)."""
    out = {}
    for ref in ("nominate_dim1", "nokken_poole_dim1"):
        ok = df[ref].notna() & df[est].notna()
        out[ref] = float(np.corrcoef(df.loc[ok, est], df.loc[ok, ref])[0, 1])
        for party, code in (("D", 100), ("R", 200)):
            m = ok & (df["party_code"] == code)
            out[f"{ref}_{party}"] = float(np.corrcoef(df.loc[m, est], df.loc[m, ref])[0, 1])
    return out


def text_start_values(model, text_mats, loadings="zero"):
    """Word intercepts start at the log frequencies of each text modality's counts (the Wordfish
    starting values). Word loadings start at 0 (default: word use does not depend on the ideal
    point until the votes have placed the legislators; random loadings make each text encoder
    falsely certain from the first steps, which blew the per-congress text runs up to NaN within
    1,000 steps) or at the package's random initialization ("random", the earlier runs)."""
    for k, S in text_mats.items():
        freq = np.asarray(S.sum(0)).ravel() + 0.5
        lin = model.decoders[k].decoder["dec_0"]
        with torch.no_grad():
            lin.bias.copy_(torch.tensor(np.log(freq / freq.sum()), dtype=lin.bias.dtype))
            if loadings == "zero":
                lin.weight.zero_()


def text_start_values_by_congress(model, S, units, corpus, loadings="zero"):
    """--text shared-loadings: each congress's word intercepts (the decoder's congress columns)
    start at that congress's log frequencies and the common intercept (bias) at 0; loadings (the
    first column) at 0 or the package's random initialization, as in text_start_values."""
    lin = model.decoders[TEXT].decoder["dec_0"]
    cols = {name: j for j, name in enumerate(corpus.content_colnames)}
    with torch.no_grad():
        lin.bias.zero_()
        for t in sorted(units["congress"].unique()):
            freq = np.asarray(S[(units["congress"] == t).to_numpy()].sum(0)).ravel() + 0.5
            lin.weight[:, 1 + cols[f"C(congress)[{t}]"]] = torch.tensor(
                np.log(freq / freq.sum()), dtype=lin.weight.dtype)
        if loadings == "zero":
            lin.weight[:, 0] = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--congresses", default="103-118")
    ap.add_argument("--steps", default="5000,10000,20000,40000")
    ap.add_argument("--hidden", default="256,64", help="vote encoders")
    ap.add_argument("--text-hidden", default="64,64", help="text encoder")
    ap.add_argument("--text-dropout", type=float, default=0.2, help="text encoder hidden layers")
    ap.add_argument("--text-loadings-start", choices=["zero", "random"], default="zero",
                    help="word loadings' starting values (random: the package's initialization)")
    ap.add_argument("--lr", type=float, default=1e-3, help="encoder and decoder learning rate")
    ap.add_argument("--prior-lr", type=float, default=1e-3)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--print-every", type=int, default=1000)
    ap.add_argument("--start", choices=["party", "none"], default="party")
    ap.add_argument("--text", choices=["per-congress", "shared-loadings", "pooled"],
                    default="per-congress",
                    help="one text modality per congress with its own loadings, with loadings "
                         "shared across congresses (own word intercepts), or one for all")
    ap.add_argument("--align-every", type=int, default=0,
                    help="every this many steps, move each congress to where the prior wants it "
                         "(03_fit_votes.align_congresses; per-congress or shared-loadings text); "
                         "0 = never")
    ap.add_argument("--align-stretch", action="store_true",
                    help="the alignment also stretches each congress (default: shifts only)")
    ap.add_argument("--warm-votes-steps", type=int, default=0,
                    help="first train this many steps on the votes alone (speeches off)")
    ap.add_argument("--warm-align-every", type=int, default=0,
                    help="alignment during the warm start (votes alone: exact); 0 = none")
    ap.add_argument("--warm-align-stretch", action="store_true",
                    help="the warm-start alignment also stretches each congress")
    ap.add_argument("--align-max-stretch", type=float, default=0,
                    help="cap each congress's stretch per alignment to [1/x, x]; 0 = no cap")
    ap.add_argument("--align-adam", choices=["rescale", "reset"], default="rescale",
                    help="Adam state of the edited parameters (see align_congresses)")
    ap.add_argument("--workers", type=int, default=4, help="persistent data-loading workers")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    lo, hi = map(int, args.congresses.split("-"))
    steps = [int(s) for s in args.steps.split(",")]
    out = c.MODULE / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from deeplatent import Corpus, IdealPointNN
    import deeplatent

    units, rc, votes = fit03.load(range(lo, hi + 1))
    house = c.DATA / "house"
    S = sp.load_npz(house / "speech_counts.npz").tocsr()[units["unit"].to_numpy()]
    vocab = [w for w in (house / "vocab.txt").read_text().split("\n") if w]
    assert len(vocab) == S.shape[1], (len(vocab), S.shape)
    units["n_terms"] = np.asarray(S.sum(1)).ravel()
    has_text = units["n_terms"].to_numpy() > 0

    keys = [fit03.key(t) for t in votes]
    vote_views = {
        f"c{t}": {"type": "vote", "matrix": M, "available": (units["congress"] == t).to_numpy()}
        for t, M in votes.items()
    }
    if args.text == "shared-loadings" and args.align_stretch:
        raise SystemExit("--align-stretch is not exact with shared word loadings (shifts only)")
    if args.text == "per-congress":
        # congress t's view keeps only its own units' rows (the others are absent placeholders)
        text_mats, text_views, text_keys = {}, {}, {}
        for t in votes:
            own = (units["congress"] == t).to_numpy()
            text_mats[f"speech_c{t}"] = (sp.diags(own.astype(S.dtype)) @ S).tocsr()
            text_views[f"c{t}"] = {"type": "bow", "matrix": text_mats[f"speech_c{t}"].tocsr(),
                                   "vocab": vocab, "available": own & has_text}
            text_keys[t] = f"speech_c{t}"
    else:
        if args.align_every and args.text == "pooled":
            raise SystemExit("--align-every needs --text per-congress or shared-loadings "
                             "(pooled text sees the frames)")
        text_mats, text_views, text_keys = {TEXT: S}, {"terms": {"type": "bow", "matrix": S,
                                                                 "vocab": vocab,
                                                                 "available": has_text}}, {}
    corpus = Corpus(
        units[["bioguide", "trend", "congress"]],
        prevalence="~ 0 + C(bioguide) + C(bioguide):trend",
        content="~ 0 + C(congress)" if args.text == "shared-loadings" else None,
        modalities={"vote": {"views": vote_views}, "speech": {"views": text_views}},
    )
    tkeys = list(text_mats)
    off = torch.zeros(len(units), dtype=torch.bool)
    # The same units seen through one kind of data: a unit then fuses those experts alone.
    votes_only = copy.copy(corpus)
    votes_only.availability = {**corpus.availability, **{k: off for k in tkeys}}
    text_only = copy.copy(corpus)
    text_only.availability = {**corpus.availability, **{k: off for k in keys}}
    n_votes = sum(M.nnz for M in votes.values())
    print(f"{len(units):,} units, {units['bioguide'].nunique():,} legislators, {len(rc):,} roll "
          f"calls in {len(votes)} congresses, {n_votes:,} votes; {int(has_text.sum()):,} units with "
          f"speeches, {int(units['n_terms'].sum()):,} terms over {len(vocab):,} words; "
          f"{corpus.M_prevalence_covariates.shape[1]:,} prior covariates ({time.time() - t0:.0f}s)",
          flush=True)

    ckpt = fit03.run_folder(args.tag)
    print(f"models saved in {ckpt}", flush=True)
    vote_hidden = [int(h) for h in args.hidden.split(",")]
    text_hidden = [int(h) for h in args.text_hidden.split(",")]
    model = IdealPointNN(
        train_data=corpus,
        n_ideal_points=1,
        ae_type="vae",
        vi_type="mean_field",
        update_prior=True,
        learn_prior_cov=False,
        fusion="corrected_poe",
        w_prior=1.0,
        encoder_args={**{k: {"hidden_dims": vote_hidden} for k in keys},
                      **{k: {"hidden_dims": text_hidden, "dropout": args.text_dropout}
                         for k in tkeys}},
        decoder_args={k: {"hidden_dims": [], "bias": True} for k in keys + tkeys},
        batch_size=256,
        num_steps=0,
        num_workers=args.workers,
        print_every_n_steps=args.print_every,
        return_best_model=False,
        ckpt_folder=str(ckpt),
        seed=0,
        device=args.device,
        optim_args={"main": {"lr": args.lr}, "prior": {"lr": args.prior_lr, "weight_decay": 0.0},
                    "predictor": {"lr": 1e-3}},
    )
    if args.start == "party":
        fit03.start_values(model, corpus, units, votes)
    if args.text == "shared-loadings":
        text_start_values_by_congress(model, S, units, corpus, args.text_loadings_start)
    else:
        text_start_values(model, text_mats, args.text_loadings_start)

    names = list(model.encoder.encoders.keys())
    vote_cols, text_cols = [names.index(k) for k in keys], [names.index(k) for k in tkeys]
    pkg = Path(deeplatent.__file__).resolve().parent
    record = {
        "args": vars(args), "units": len(units), "legislators": int(units["bioguide"].nunique()),
        "rollcalls": len(rc), "congresses": len(votes), "votes": int(n_votes),
        "units_with_text": int(has_text.sum()), "terms": int(units["n_terms"].sum()),
        "vocabulary": len(vocab),
        "prior_covariates": int(corpus.M_prevalence_covariates.shape[1]),
        "deeplatent": str(pkg),
        "md5": {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in sorted(pkg.glob("*.py"))},
        "checkpoint_folder": str(ckpt), "checkpoints": [], "alignments": [],
    }
    Z = torch.tensor(corpus.M_prevalence_covariates, device=args.device)
    by_congress = {t: [fit03.key(t), text_keys[t]] for t in votes} if text_keys else None
    content_shift = None
    if args.text == "shared-loadings":
        cols = {name: j for j, name in enumerate(corpus.content_colnames)}
        content_shift = {TEXT: {t: 1 + cols[f"C(congress)[{t}]"] for t in votes}}
    W = args.warm_votes_steps
    if W:
        warm = {W}
        if args.warm_align_every:
            warm |= set(range(args.warm_align_every, W + 1, args.warm_align_every))
        for step in sorted(warm):
            model.num_steps = step
            model.train(votes_only)
            if args.warm_align_every and step % args.warm_align_every == 0:
                info = fit03.align_congresses(model, votes_only, units,
                                              stretch=args.warm_align_stretch)
                record["alignments"].append({**info, "phase": "votes only"})
                print("ALIGN (votes only)", json.dumps({k: v for k, v in info.items()
                                                       if k not in ("shift", "stretch")}),
                      flush=True)
        with fit03.readout_rng():
            v_mean, v_sd = model.get_ideal_points(votes_only, num_samples=args.draws,
                                                  return_std=True)
        with torch.no_grad():
            prior_mean = model.prior.mean_net(Z).cpu().numpy()[:, 0]
        est = units[["unit", "congress", "bioguide", "party_code", "state_abbrev", "trend",
                     "n_votes", "n_terms", "nominate_dim1", "nokken_poole_dim1"]].copy()
        est["theta_mean"], est["theta_sd"], est["prior_mean"] = v_mean[:, 0], v_sd[:, 0], prior_mean
        sign = np.sign(np.corrcoef(est["theta_mean"], est["nominate_dim1"].fillna(0))[0, 1])
        for k in ("theta_mean", "prior_mean"):
            est[k] *= sign
        est.to_csv(out / f"estimates_warm_{W}.csv", index=False)
        model.save_model(str(ckpt / f"model_warm_{W}.ckpt"))
        rec = {"step": W, "phase": "votes only (warm start)", "minutes": (time.time() - t0) / 60,
               "sign": float(sign), "model": str(ckpt / f"model_warm_{W}.ckpt"), **{k: correlations(est, k) for k in ("theta_mean", "prior_mean")}}
        record["checkpoints"].append(rec)
        (out / "summary.json").write_text(json.dumps(record, indent=1))
        print("CKPT", json.dumps(rec), flush=True)
    stops = {s_ for s_ in steps if s_ > W}
    if args.align_every:
        stops |= set(range(args.align_every, max(steps) + 1, args.align_every)) - set(range(W + 1))
    for step in sorted(stops):
        model.num_steps = step
        model.train(corpus)
        if args.align_every and step % args.align_every == 0:
            info = fit03.align_congresses(model, corpus, units, stretch=args.align_stretch,
                                          modality_keys=by_congress,
                                          max_stretch=args.align_max_stretch or None,
                                          adam=args.align_adam, content_shift=content_shift)
            record["alignments"].append(info)
            print("ALIGN", json.dumps({k: v for k, v in info.items()
                                       if k not in ("shift", "stretch")}), flush=True)
        if step not in steps:
            continue
        with fit03.readout_rng():
            mean, sd = model.get_ideal_points(corpus, num_samples=args.draws, return_std=True)
            v_mean, v_sd = model.get_ideal_points(votes_only, num_samples=args.draws,
                                                  return_std=True)
            t_mean, t_sd = model.get_ideal_points(text_only, num_samples=args.draws,
                                                  return_std=True)
            inc = model.get_modality_weights(corpus, increments=True)
        t_mean[~has_text], t_sd[~has_text] = np.nan, np.nan  # no speeches: the prior
        with torch.no_grad():
            prior_mean = model.prior.mean_net(Z).cpu().numpy()[:, 0]
        est = units[["unit", "congress", "bioguide", "party_code", "state_abbrev", "trend",
                     "n_votes", "n_terms", "nominate_dim1", "nokken_poole_dim1"]].copy()
        est["theta_mean"], est["theta_sd"] = mean[:, 0], sd[:, 0]
        est["theta_votes"], est["theta_votes_sd"] = v_mean[:, 0], v_sd[:, 0]
        est["theta_text"], est["theta_text_sd"] = t_mean[:, 0], t_sd[:, 0]
        est["prior_mean"] = prior_mean
        est["precision_votes"] = np.nansum(inc[:, vote_cols], axis=1)  # each unit has one
        est["precision_text"] = np.where(has_text, np.nansum(inc[:, text_cols], axis=1), np.nan)
        est["text_share"] = est["precision_text"] / (est["precision_votes"] + est["precision_text"])
        sign = np.sign(np.corrcoef(est["theta_mean"], est["nominate_dim1"].fillna(0))[0, 1])
        for k in ("theta_mean", "theta_votes", "theta_text", "prior_mean"):
            est[k] *= sign
        est.to_csv(out / f"estimates_{step}.csv", index=False)
        model.save_model(str(ckpt / f"model_{step}.ckpt"))
        rec = {"step": step, "minutes": (time.time() - t0) / 60, "sign": float(sign),
               "model": str(ckpt / f"model_{step}.ckpt"),
               **{k: correlations(est, k)
                  for k in ("theta_mean", "theta_votes", "theta_text", "prior_mean")},
               "theta_sd_median": float(est["theta_sd"].median()),
               "precision_votes_median": float(est["precision_votes"].median()),
               "precision_text_median": float(est["precision_text"].median()),
               "text_share_median": float(est["text_share"].median())}
        record["checkpoints"].append(rec)
        (out / "summary.json").write_text(json.dumps(record, indent=1))
        print("CKPT", json.dumps(rec), flush=True)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
