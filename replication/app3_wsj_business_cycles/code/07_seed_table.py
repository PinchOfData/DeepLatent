"""Step 07: average a summary of step 04 over seeds.

Runs that differ only by their seed (tags <name>_s<seed>) form one row: the mean and standard deviation over seeds of
every score (R^2 by period for the LASSO, ridge and own-head predictions; diversity, coherence, topics used) and the
number of seeds. Rows are ordered as LDA, the two-step head, then the joint runs by lam.

Writes results/<summary>_seeds.csv and prints the table (mean, with the sd in brackets).

Usage: python 07_seed_table.py --summary summary_final_change.csv
"""
import argparse

import numpy as np
import pandas as pd

import common

SCORES = ["lasso_train_r2", "lasso_val_r2", "lasso_test_r2", "ridge_train_r2", "ridge_val_r2", "ridge_test_r2",
          "head_train_r2", "head_val_r2", "head_test_r2", "diversity", "npmi", "topics_used"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()
    df = pd.read_csv(common.RESULTS / a.summary)
    df["run"] = df.tag.str.replace(r"_s\d+$", "", regex=True)
    cols = [c for c in SCORES if c in df]
    g = df.groupby("run", sort=False)
    out = g[cols].mean().add_suffix("_mean").join(g[cols].std().add_suffix("_sd")).join(g.size().rename("seeds"))
    out = out.join(g["lam"].first())
    kind = np.where(out.index.str.startswith("lda"), 0, np.where(out.index.str.startswith("twostephead"), 1, 2))
    out = out.assign(kind=kind).sort_values(["kind", "lam"]).drop(columns="kind")
    out.to_csv(common.RESULTS / a.summary.replace(".csv", "_seeds.csv"))

    show = pd.DataFrame({"seeds": out.seeds})
    for c in cols:
        m, s = out[f"{c}_mean"], out[f"{c}_sd"]
        nd = 0 if c == "topics_used" else 3 if c == "npmi" else 2
        show[c.replace("_r2", "")] = [("" if np.isnan(x) else f"{x:.{nd}f}" + ("" if np.isnan(y) else f" ({y:.{nd}f})"))
                                      for x, y in zip(m, s)]
    pd.set_option("display.width", 250)
    print(show.to_string())


if __name__ == "__main__":
    main()
