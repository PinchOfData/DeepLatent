"""Figure 1: recovery of ideal points under each likelihood, one model per likelihood.

    python 01_likelihoods.py --likelihood {embedding,survey,bow,vote} [--n 10000]

Draws theta ~ N(0, 1) for N units and one modality from FIG1_DESIGN, fits IdealPointNN (see
common.py), and saves to results/fig1_likelihoods/<likelihood>.npz the true theta, the model's
posterior mean and sd per unit (sign not yet aligned; summary.json records the sign), and the
exact posterior summaries per unit. <likelihood>.json holds the design, the fit statistics at
each checkpoint and the provenance of the code.
"""

import argparse
import time

import numpy as np

import common as c

ITEM_SEED = {"embedding": 11, "survey": 12, "bow": 13, "vote": 14}
DATA_SEED = {"embedding": 21, "survey": 22, "bow": 23, "vote": 24}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--likelihood", required=True, choices=list(c.FIG1_DESIGN))
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--ckpts", default="10000,20000,30000")
    ap.add_argument("--seed", type=int, default=0, help="model initialization seed")
    ap.add_argument("--out", default=str(c.RESULTS / "fig1_likelihoods"))
    args = ap.parse_args()
    kind, ckpts = args.likelihood, [int(s) for s in args.ckpts.split(",")]
    t0 = time.time()

    params = c.item_params(kind, c.FIG1_DESIGN[kind], ITEM_SEED[kind])
    rng = np.random.default_rng(DATA_SEED[kind])
    theta = rng.normal(size=args.n)
    X = c.simulate(kind, theta, params, rng)
    corpus = c.build_corpus({kind: X})
    ref = c.exact_reference([kind], {kind: X}, {kind: params})
    print(f"{kind}: N={args.n}, exact precision {np.mean(1 / ref['s2']):.2f}", flush=True)

    model = c.build_model(corpus, [kind], seed=args.seed)
    record = {
        "likelihood": kind,
        "n": args.n,
        "design": c.FIG1_DESIGN[kind],
        "item_seed": ITEM_SEED[kind],
        "data_seed": DATA_SEED[kind],
        "model_seed": args.seed,
        "readout_draws": c.READOUT_DRAWS,
        "provenance": c.provenance(),
        "checkpoints": [],
    }
    for step in ckpts:
        c.train_to(model, corpus, step)
        mean, sd = c.readout(model, corpus)
        stats = c.panel_summary(theta, mean, sd, ref)
        record["checkpoints"].append({"step": step, "minutes": (time.time() - t0) / 60, **stats})
        print(f"CKPT {kind} step {step}: " + ", ".join(
            f"{k} {v:.3f}" for k, v in stats.items()), flush=True)
    c.cleanup(model)

    out = c.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{kind}.npz",
        theta=theta,
        mean=mean,
        sd=sd,
        **{f"exact_{f}": v for f, v in ref.items()},
    )
    c.write_json(out / f"{kind}.json", record)
    print(f"DONE {kind} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
