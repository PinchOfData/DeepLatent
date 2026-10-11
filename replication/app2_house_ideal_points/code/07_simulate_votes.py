"""Step 07 (diagnostic): semi-synthetic House votes with known ideal points, to test whether the
step-03 estimator produces a time drift of its own.

    python 07_simulate_votes.py --out ../data_sim --seed 0
    APP2_DATA_ROOT=../data_sim python 03_fit_votes.py --tag sim_votes_103_118 ...

Truth: each legislator-congress's ideal point is its Nokken-Poole score (DW-NOMINATE's location
over time; the DW-NOMINATE career score where Nokken-Poole is missing). Everything else is the real
data's structure: the same units, the same roll calls, and the same pattern of who voted on what
(abstentions and non-membership included). Each roll call j gets a cutpoint c_j (the position of a
randomly drawn member who voted on it, plus N(0, 0.05^2) noise) and a discrimination beta_j (random
sign, magnitude uniform on [5, 15] in Nokken-Poole units); a member votes yea with probability
logistic(beta_j * (x_it - c_j)). A fit that is free of artifacts recovers the truth up to one linear
map, so its chamber and party means by congress track Nokken-Poole's (no drift).

Writes <out>/house/{units.csv, rollcalls.csv, votes.npz} (units.csv as the real one, with n_votes
recounted and the truth kept in nokken_poole_dim1), <out>/house/sim_params.npz (truth per unit,
cutpoint and discrimination per roll call, indexed like votes.npz) and <out>/house/sim.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common as c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(c.MODULE / "data_sim"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    house = c.DATA / "house"
    units = pd.read_csv(house / "units.csv", dtype={"bioguide": str})
    rc = pd.read_csv(house / "rollcalls.csv")
    V = sp.load_npz(house / "votes.npz").tocsc()  # units x roll calls; nonzero = voted

    truth = units["nokken_poole_dim1"].fillna(units["nominate_dim1"]).to_numpy()
    keep_unit = np.isfinite(truth)
    # cutpoints: the position of a random member who voted on the roll call, plus noise
    counts = np.diff(V.indptr)
    pick = V.indptr[:-1] + np.floor(rng.uniform(size=V.shape[1]) * np.maximum(counts, 1)).astype(int)
    pick = np.minimum(pick, max(V.nnz - 1, 0))
    anchor = truth[V.indices[pick]]
    cut = np.where(counts > 0, np.nan_to_num(anchor) + rng.normal(0, 0.05, V.shape[1]), 0.0)
    beta = rng.choice([-1.0, 1.0], V.shape[1]) * rng.uniform(5, 15, V.shape[1])

    V = V.tocsr()
    rows = np.repeat(np.arange(V.shape[0]), np.diff(V.indptr))
    cols = V.indices
    ok = keep_unit[rows]
    rows, cols = rows[ok], cols[ok]
    p = 1 / (1 + np.exp(-beta[cols] * (truth[rows] - cut[cols])))
    y = np.where(rng.uniform(size=len(p)) < p, 1, -1).astype(np.int8)
    S = sp.csr_matrix((y, (rows, cols)), shape=V.shape)

    out = Path(args.out) / "house"
    out.mkdir(parents=True, exist_ok=True)
    sim_units = units.copy()
    sim_units["n_votes"] = np.diff(S.indptr)
    sim_units.to_csv(out / "units.csv", index=False)
    rc.to_csv(out / "rollcalls.csv", index=False)
    sp.save_npz(out / "votes.npz", S)
    np.savez(out / "sim_params.npz", truth=truth, cut=cut, beta=beta)
    info = {"seed": args.seed, "truth": "nokken_poole_dim1 (nominate_dim1 where missing)",
            "units_without_truth": int((~keep_unit).sum()), "votes": int(S.nnz),
            "yea_share": float((y > 0).mean()), "beta": "random sign x U(5, 15)",
            "cutpoint": "random voter's truth + N(0, 0.05^2)"}
    (out / "sim.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
