"""Topic-word tables (CSV and LaTeX), written by 05_fit_gtm.py after every fit and by 06_exhibits.py.

Three rankings of each topic's terms:
  - probable: the topic's word distribution from the model at average covariates (the average congress
    baseline, party at 0; GTM.get_topic_word_distribution, for either decoder);
  - frex: FREX of that distribution (harmonic mean of the within-topic ranks of probability and
    exclusivity, beta_kw / sum_k beta_kw, weight 0.5, as in STM);
  - observed: the data's word counts weighted by the speeches' posterior topic shares,
    sum_d theta_dk c_dw, normalized within topic, ranked by FREX (depends on the data, not on the
    decoder's weights for words a congress never uses).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd


def congress_columns(colnames: list) -> dict:
    """{congress: column} of the congress main effects, for either coding of the formula:
    "~ C(congress) + ..." (Intercept + C(congress)[T.t], the first congress being the reference) or
    "~ 0 + C(congress) + ..." (C(congress)[t] for every congress)."""
    cols = {}
    for j, n in enumerate(colnames):
        m = re.fullmatch(r"C\(congress\)\[(?:T\.)?(\d+)\]", n)
        if m:
            cols[int(m.group(1))] = j
    return cols


def congress_baselines(C: np.ndarray, colnames: list, congresses: list, background=0.0) -> np.ndarray:
    """len(congresses) x V baseline word logits of each congress (party at 0): background (the SAGE
    decoder's m; 0 for the linear decoder) + the congress's covariate row."""
    cols = congress_columns(colnames)
    icpt = C[colnames.index("Intercept")] if "Intercept" in colnames else np.zeros(C.shape[1])
    return np.stack([background + icpt + (C[cols[t]] if t in cols else 0.0) for t in congresses])


def average_content(colnames: list) -> np.ndarray:
    """Content covariates of the average congress with party at 0, for either coding of the formula:
    every congress dummy at 1/T ("~ 0 + C(congress) + ..."), or the intercept at 1 and each of the T - 1
    non-reference dummies at 1/T ("~ C(congress) + ...")."""
    cols = list(congress_columns(colnames).values())
    x = np.zeros(len(colnames), dtype=np.float32)
    if "Intercept" in colnames:
        x[colnames.index("Intercept")] = 1.0
        x[cols] = 1.0 / (len(cols) + 1)
    else:
        x[cols] = 1.0 / len(cols)
    return x


def model_distribution(tm) -> np.ndarray:
    """K x V word distributions of the fitted model's topics at average covariates."""
    return tm.get_topic_word_distribution(content_covariates=average_content(tm.content_colnames)).astype(np.float64)


def observed_distribution(theta: np.ndarray, X) -> np.ndarray:
    """K x V topic-weighted word counts of the data, normalized within topic."""
    T = np.asarray((X.T @ theta).T)
    return T / T.sum(axis=1, keepdims=True)


def frex(beta: np.ndarray, weight: float = 0.5) -> np.ndarray:
    # Words with probability 0 in every topic (unused in the sample, so exp underflows) get exclusivity 0.
    total = beta.sum(axis=0, keepdims=True)
    excl = np.divide(beta, total, out=np.zeros_like(beta), where=total > 0)

    def ecdf(x):
        return (np.argsort(np.argsort(x, axis=1), axis=1) + 1) / x.shape[1]

    return 1.0 / (weight / ecdf(excl) + (1 - weight) / ecdf(beta))


def write(out: Path, vocab: list, labels: list, shares: np.ndarray, beta: np.ndarray, theta: np.ndarray, X,
          n: int = 10) -> pd.DataFrame:
    """beta: K x V topic word distributions at average covariates (model_distribution)."""
    K = len(labels)
    obs = observed_distribution(theta, X)
    scores = {"probable": beta, "frex": frex(beta), "observed": frex(obs)}
    df = pd.DataFrame({"topic": range(K), "label": labels, "share": np.round(shares, 3)})
    for name, s in scores.items():
        df[name] = [", ".join(vocab[j] for j in np.argsort(-s[k])[:n]) for k in range(K)]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "topic_words.csv", index=False)
    for name in scores:
        lines = ["\\begin{tabular}{l|r|l}", "\\toprule",
                 "\\textbf{Topic} & \\textbf{Share} & \\textbf{Top Words} \\\\", "\\midrule"]
        for _, r in df.iterrows():
            lines.append(f"{r['label']} & {r['share']:.3f} & {r[name]} \\\\".replace("&amp;", "&").replace("_", "\\_"))
        lines += ["\\bottomrule", "\\end{tabular}"]
        (out / f"topic_words_{name}.tex").write_text("\n".join(lines) + "\n")
    return df
