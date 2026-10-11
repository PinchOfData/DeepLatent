"""Check that 03_count_terms.count_matrix reproduces P&P build_counts() (R) exactly.

Sources the vendored Codes/R/02_party_classifier_gst.R with GST_R_SOURCE_ONLY=1, runs its
build_counts() on the same cleaned documents and compares the two count matrices entry by entry.

Usage: python check_counts_vs_R.py <parquet or csv with a doc_clean column> [n_docs]
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import common

spec = importlib.util.spec_from_file_location("count_terms", common.CODE / "03_count_terms.py")
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)

R_PROGRAM = r"""
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages(source(args[1]))
docs  <- readLines(args[2], encoding = "UTF-8", warn = FALSE)
vocab <- readLines(args[3], encoding = "UTF-8", warn = FALSE)
C <- build_counts(docs, vocab)$C
C <- as(C, "TsparseMatrix")
write.table(data.frame(i = C@i, j = C@j, x = C@x), args[4], sep = ",", row.names = FALSE)
"""


def main() -> None:
    path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, nrows=50 * n)
    docs = df["doc_clean"].dropna().astype(str).str.replace("\n", " ").sample(
        min(n, df["doc_clean"].notna().sum()), random_state=0).tolist()
    vocab = common.load_vocab()
    py = ct.count_matrix(docs, vocab)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "docs.txt").write_text("\n".join(docs) + "\n", encoding="utf-8")
        (tmp / "prog.R").write_text(R_PROGRAM)
        env = dict(os.environ, GST_R_SOURCE_ONLY="1", GST_R_NCORES="1")
        subprocess.run(["Rscript", str(tmp / "prog.R"), str(common.VENDOR / "R" / "02_party_classifier_gst.R"),
                        str(tmp / "docs.txt"), str(common.VOCAB_PATH), str(tmp / "C.csv")],
                       check=True, env=env)
        t = pd.read_csv(tmp / "C.csv")
    r = sp.csr_matrix((t["x"].to_numpy(), (t["i"].to_numpy(), t["j"].to_numpy())), shape=py.shape)
    diff = abs(py - r)
    print(f"{len(docs)} docs, {int(py.sum()):,} terms (python) vs {int(r.sum()):,} (R); "
          f"entries that differ: {diff.nnz}; max abs difference {diff.max() if diff.nnz else 0}")
    if diff.nnz:
        rows = np.unique(diff.nonzero()[0])[:3]
        for i in rows:
            print("doc", i, docs[i][:300])
        sys.exit(1)
    print("IDENTICAL")


if __name__ == "__main__":
    main()
