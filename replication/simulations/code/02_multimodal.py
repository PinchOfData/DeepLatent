"""Figure 2: three modalities with different precisions, fitted jointly.

    python 02_multimodal.py [--n 10000]

Draws theta ~ N(0, 1) and votes, survey responses and text for each unit (FIG2_DESIGN), fits one
IdealPointNN on all three (corrected product of experts; each training batch keeps a random
subset of modalities, so every modality's expert is trained to be that modality's posterior on
its own), and reads out the joint posterior and each modality's posterior. Saves, per panel
(joint, vote, survey, bow), results/fig2_multimodal/<panel>.npz with the true theta, the model's
posterior mean and sd per unit, and the exact posterior summaries given the same data; summary
statistics, design and provenance go to summary.json.
"""

import argparse
import time

import numpy as np

import common as c

KINDS = ["vote", "survey", "bow"]
ITEM_SEED = {"vote": 31, "survey": 32, "bow": 33}
DATA_SEED = 41


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--ckpts", default="10000,20000,30000")
    ap.add_argument("--seed", type=int, default=0, help="model initialization seed")
    ap.add_argument("--out", default=str(c.RESULTS / "fig2_multimodal"))
    args = ap.parse_args()
    ckpts = [int(s) for s in args.ckpts.split(",")]
    t0 = time.time()

    params = {k: c.item_params(k, c.FIG2_DESIGN[k], ITEM_SEED[k]) for k in KINDS}
    rng = np.random.default_rng(DATA_SEED)
    theta = rng.normal(size=args.n)
    data = {k: c.simulate(k, theta, params[k], rng) for k in KINDS}
    corpus = c.build_corpus(data)
    panels = {"joint": None, **{k: k for k in KINDS}}
    refs = {
        p: c.exact_reference(KINDS if k is None else [k], data, params)
        for p, k in panels.items()
    }
    for p, r in refs.items():
        print(f"{p}: exact precision {np.mean(1 / r['s2']):.2f}", flush=True)

    model = c.build_model(corpus, KINDS, seed=args.seed)
    record = {
        "n": args.n,
        "design": c.FIG2_DESIGN,
        "item_seed": ITEM_SEED,
        "data_seed": DATA_SEED,
        "model_seed": args.seed,
        "readout_draws": c.READOUT_DRAWS,
        "provenance": c.provenance(),
        "checkpoints": [],
    }
    for step in ckpts:
        c.train_to(model, corpus, step)
        est = {p: c.readout(model, corpus, k) for p, k in panels.items()}
        stats = {p: c.panel_summary(theta, *est[p], refs[p]) for p in panels}
        record["checkpoints"].append(
            {"step": step, "minutes": (time.time() - t0) / 60, "panels": stats}
        )
        for p, s in stats.items():
            print(f"CKPT step {step} {p}: " + ", ".join(
                f"{k} {v:.3f}" for k, v in s.items()), flush=True)
    c.cleanup(model)

    out = c.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p, (mean, sd) in est.items():
        np.savez_compressed(
            out / f"{p}.npz",
            theta=theta,
            mean=mean,
            sd=sd,
            **{f"exact_{f}": v for f, v in refs[p].items()},
        )
    c.write_json(out / "summary.json", record)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
