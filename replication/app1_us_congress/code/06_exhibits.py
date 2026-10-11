"""Step 06: the Application-1 exhibits of the slides, from a fitted model in results/<tag>/.

Rebuilds the fitted GTM (a small corpus with every congress x party fixes the design columns, then
load_model) and writes results/<tag>/exhibits/:
 1. polarization_by_congress.csv, trends_in_speech_polarization.png/.pdf: the Gentzkow-Shapiro-Taddy
    index by congress under four counterfactuals, computed as in the simulation (polarization_mc of
    scripts/sim_polarization_decomposition.py). For each congress, scenario and Monte Carlo
    repetition, --n-docs topic-share vectors per party are drawn, and the decoder's word distribution
    is evaluated at party_signed = +1 (R) or -1 (D) when party acts on content, 0 otherwise. The topic
    shares (--topic-shares): "posterior" (default) resamples the congress's speeches' posterior-mean
    topic shares (theta.npy): Republican speeches for R and Democratic speeches for D when party acts
    on topic choice, half of each when it does not; "prior" draws them from the fitted prior at
    party_signed = +1, -1 or 0 (fits before 2026-10-08; the prior's party coefficients reproduce only
    about a third of the topic-share gaps between the parties' speeches, so the between-topic part was
    understated). q_P is the mean word distribution over the draws; the index is
    1/2 q_R.rho + 1/2 q_D.(1 - rho),
    rho = q_R / (q_R + q_D). Scenarios: party affects topic prevalence and content (total), prevalence
    only (between-topic), content only (within-topic), neither (placebo). Bands: 2.5-97.5% quantiles
    over --mc-reps repetitions (Monte Carlo noise only, not estimation uncertainty).
 2. topic_words.csv, topic_words_{probable,frex,observed}.tex: see topic_words.py.
 3. topics_tsne.png: t-SNE of the posterior topic shares of --tsne-docs speeches, colored by top topic.
 4. content_over_time.csv/.tex: distinctive terms of each congress's baseline word logits, net of their
    average over congresses (the .tex keeps the slide's years, after the average congress's top terms).
 5. word_embeddings.csv/.tex: most similar terms by cosine similarity of the terms' rows of the
    effective decoder matrix W2 W1 (topic and covariate loadings, each column centered across terms; for
    the SAGE decoder, the terms' topic, covariate and topic x party deviations).
    The rows of W2 alone are not used: W2 W1 = (W2 A)(A^-1 W1) for any invertible A, and without
    weight decay the directions of W2 that W1 does not use keep their random initial values.
 6. topic_deviations_for_republicans.csv/.png: Republican minus Democratic mean posterior topic share
    in each congress, averaged over congresses with equal weights; 95% intervals from the
    within-congress sampling variances of the two means.
 7. party_specific_language.csv/.tex: per topic, its baseline (FREX) terms and the terms with the largest
    average (over congresses) Republican-minus-Democratic difference in the topic's word distribution.
Topic labels come from inputs/topic_labels_<tag>.csv (topic,label) when it exists.

Usage: python 06_exhibits.py --tag pilot_k20 [--mc-reps 100] [--n-docs 1000]
"""
import argparse
import json
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from patsy import dmatrix

import common
import topic_words
from deeplatent import GTM, Corpus

DEC_KEY = "text_bow"
SCENARIOS = {  # name: (party in prevalence, party in content)
    "total": (True, True),
    "between_topic": (True, False),
    "within_topic": (False, True),
    "placebo": (False, False),
}
LABELS = {
    "total": "Total: party affects topics and content",
    "between_topic": "Between-topic: party affects topics only",
    "within_topic": "Within-topic: party affects content only",
    "placebo": "Placebo: party affects neither",
}
COLORS = {"total": "tab:purple", "between_topic": "tab:green", "within_topic": "tab:orange", "placebo": "tab:grey"}
SLIDE_YEARS = [1917, 1919, 1921, 1939, 1941, 1943, 1945, 1947, 1973, 1977, 1979, 1981, 1993, 2001, 2003,
               2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023]
QUERY_WORDS = ["taxation", "immigration", "war", "health", "wage", "election", "regulation", "fiscal",
               "debt", "deficit", "campaign", "insurance", "recession", "growth", "welfare", "crime"]


def start_year(c: int) -> int:
    return 1789 + 2 * (c - 1)


def load_counts(docs, terms):
    """The fit's count matrix, rebuilt from each speech's row in its terms file."""
    parts = []
    for (source, congress), g in docs.groupby(["source", "congress"], sort=False):
        parts.append((g.index.to_numpy(), sp.load_npz(terms / f"{common.stem(source, congress)}.npz")[g["term_row"].to_numpy()]))
    order = np.concatenate([i for i, _ in parts])
    return sp.vstack([m for _, m in parts]).tocsr()[np.argsort(order)]


def design_corpus(docs, vocab, terms, formula) -> Corpus:
    """Corpus over `docs` (rows of docs.csv.gz) with their term counts and the fit's design formula."""
    docs = docs.reset_index(drop=True).assign(row=lambda d: np.arange(len(d)))
    return Corpus(docs, modalities={"text": {"column": "row", "views": {"bow": {
        "type": "bow", "vectorizer": common.PrecomputedCounts(load_counts(docs, terms), vocab)}}}},
        prevalence=formula, content=formula)


def build_model(cfg, corpus, device, ckpt):
    """The fitted GTM: an untrained GTM with the fit's configuration, then load_model(ckpt)."""
    assert corpus.prevalence_colnames == cfg["prevalence_colnames"]
    assert corpus.content_colnames == cfg["content_colnames"]
    with tempfile.TemporaryDirectory() as tmp:
        tm = GTM(
            train_data=corpus, n_topics=cfg["n_topics"], doc_topic_prior="logistic_normal", ae_type="vae",
            vi_type="mixture_of_gaussians", mixture_components=cfg["mixture_components"],
            update_prior=True, learn_prior_cov=True,
            encoder_args={DEC_KEY: {"hidden_dims": cfg["encoder_hidden"], "activation": "relu", "bias": True,
                                    "dropout": cfg.get("dropout", 0.0)}},
            decoder_args={DEC_KEY: common.decoder_config(cfg, corpus.content_colnames)},
            w_prior=1.0, batch_size=cfg["batch_size"], num_steps=0, num_workers=0, return_best_model=False,
            ckpt_folder=tmp, seed=cfg["seed"], device=device,
        )
    load_checkpoint(tm, ckpt, device)
    return tm


def load_checkpoint(tm, ckpt, device) -> None:
    tm.load_model(str(ckpt))
    tm.device = device  # load_model restores the device attribute saved with the model
    tm.encoder.eval()
    tm.prior.eval()
    tm.decoders.eval()


def load_fit(res, device):
    cfg = json.load(open(res / "config.json"))
    docs = pd.read_csv(res / "docs.csv.gz", dtype={"speaker_session": str, "person_id": str})
    theta = np.load(res / "theta.npy")
    vocab = common.load_vocab(cfg["vocab"])
    # A few speeches of every congress x party reproduce the design columns of the fit.
    sub = docs.groupby(["congress", "party"], sort=True).head(2)
    tm = build_model(cfg, design_corpus(sub, vocab, common.terms_dir(cfg["vocab"]), cfg["formula"]), device,
                     res / "gtm.ckpt")
    return cfg, docs, theta, vocab, tm


def topic_labels(tag: str, K: int) -> list:
    path = common.INPUTS / f"topic_labels_{tag}.csv"
    if path.exists():
        lab = pd.read_csv(path).set_index("topic")["label"]
        return [lab.get(k, f"Topic {k}") for k in range(K)]
    return [f"Topic {k}" for k in range(K)]


def design_row(congress: int, party_signed: int, congresses: list, colnames: list, formula: str) -> np.ndarray:
    df = pd.DataFrame({"congress": pd.Categorical([congress], categories=congresses), "party_signed": [party_signed]})
    m = dmatrix(formula, df)
    assert list(m.design_info.column_names) == colnames
    return np.asarray(m, dtype=np.float32)[0]


def polarization(tm, congresses, colnames, formula, n_docs, reps, seed, chunk=10, theta=None, docs=None,
                 source="posterior") -> pd.DataFrame:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = tm.device
    decoder = tm.decoders[DEC_KEY]
    recs = []
    for t in congresses:
        rows = {ps: torch.as_tensor(design_row(t, ps, congresses, colnames, formula), device=dev)
                for ps in (-1, 0, 1)}
        if source == "posterior":
            in_t = docs["congress"].to_numpy() == t
            idx = {ps: np.flatnonzero(in_t & (docs["party"].to_numpy() == p)) for ps, p in ((1, "R"), (-1, "D"))}

        def shares(ps_prev, r):
            """n_docs * r topic-share vectors: r repetitions of n_docs speeches."""
            if source == "prior":
                return tm.prior.sample(N=n_docs * r, M_prevalence_covariates=rows[ps_prev].expand(n_docs * r, -1),
                                       to_simplex=True)
            if ps_prev:
                pick = rng.choice(idx[ps_prev], size=(r, n_docs))
            else:  # party does not act on topic choice: half Republican, half Democratic speeches
                h = n_docs // 2
                pick = np.concatenate([rng.choice(idx[1], size=(r, h)), rng.choice(idx[-1], size=(r, n_docs - h))], axis=1)
            return torch.as_tensor(theta[pick.ravel()], dtype=torch.float32, device=dev)

        def q_mean(ps_prev, ps_cont, r):
            """Mean word distribution of n_docs speeches' topic shares, for r repetitions at once."""
            with torch.no_grad():
                th = shares(ps_prev, r)
                x = torch.cat([th, rows[ps_cont].expand(n_docs * r, -1)], dim=1)
                return F.softmax(decoder(x), dim=1).view(r, n_docs, -1).mean(dim=1).double()

        for start in range(0, reps, chunk):
            r = min(chunk, reps - start)
            for name, (in_prev, in_cont) in SCENARIOS.items():
                qD = q_mean(-1 if in_prev else 0, -1 if in_cont else 0, r)
                qR = q_mean(1 if in_prev else 0, 1 if in_cont else 0, r)
                rho = qR / (qR + qD + 1e-12)
                pol = (0.5 * (qR * rho).sum(1) + 0.5 * (qD * (1 - rho)).sum(1)).cpu().numpy()
                recs += [{"congress": t, "scenario": name, "rep": start + i, "pol": float(p)} for i, p in enumerate(pol)]
        print(f"  polarization congress {t} done", flush=True)
    return pd.DataFrame(recs)


def plot_polarization(summ: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for name in SCENARIOS:
        d = summ[summ["scenario"] == name].sort_values("year")
        ax.plot(d["year"], d["mean"], color=COLORS[name], label=LABELS[name])
        ax.fill_between(d["year"], d["lo"], d["hi"], color=COLORS[name], alpha=0.2, lw=0)
    ax.set_xlabel("Congress (first year)")
    ax.set_ylabel("Speech polarization")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def latex(df: pd.DataFrame, path, colspec=None) -> None:
    cols = list(df.columns)
    lines = ["\\begin{tabular}{" + (colspec or "l" * len(cols)) + "}", "\\toprule",
             " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(v).replace("&", "\\&").replace("_", "\\_") for v in r.values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def top_terms(v: np.ndarray, vocab: list, n: int) -> str:
    return ", ".join(vocab[j] for j in np.argsort(-v)[:n])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mc-reps", type=int, default=100)
    ap.add_argument("--n-docs", type=int, default=1000)
    ap.add_argument("--topic-shares", choices=["posterior", "prior"], default="posterior",
                    help="polarization: speeches' posterior topic shares or draws from the fitted prior")
    ap.add_argument("--tsne-docs", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    res = common.RESULTS / a.tag
    out = res / "exhibits"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    cfg, docs, theta, vocab, tm = load_fit(res, device)
    K = cfg["n_topics"]
    labels = topic_labels(a.tag, K)
    congresses = sorted(docs["congress"].unique().tolist())
    colnames = cfg["content_colnames"]
    comp = {k: v.astype(np.float64) if isinstance(v, np.ndarray) else v
            for k, v in tm.get_word_logit_components().items()}
    B, C, m = comp["topics"], comp["covariates"], comp["background"]

    # 1. Polarization decomposition
    pol = polarization(tm, congresses, colnames, cfg["formula"], a.n_docs, a.mc_reps, a.seed, theta=theta, docs=docs,
                       source=a.topic_shares)
    pol.to_csv(out / "polarization_reps.csv.gz", index=False)
    summ = pol.groupby(["scenario", "congress"])["pol"].agg(
        mean="mean", lo=lambda x: x.quantile(0.025), hi=lambda x: x.quantile(0.975)).reset_index()
    summ["year"] = summ["congress"].map(start_year)
    summ.to_csv(out / "polarization_by_congress.csv", index=False)
    plot_polarization(summ, out / "trends_in_speech_polarization.png")

    # 2. Topics
    beta = topic_words.model_distribution(tm)
    Bfrex = topic_words.frex(beta)
    X = load_counts(docs, common.terms_dir(cfg["vocab"]))
    topic_words.write(out, vocab, labels, theta.mean(axis=0), beta, theta, X)

    # 3. t-SNE
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(theta), size=min(a.tsne_docs, len(theta)), replace=False)
    emb = TSNE(n_components=2, init="pca", random_state=a.seed, perplexity=30).fit_transform(theta[idx])
    top, mx = theta[idx].argmax(axis=1), theta[idx].max(axis=1)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(emb[:, 0], emb[:, 1], c=top, cmap="plasma", s=2 + 30 * mx ** 2, alpha=0.6, linewidths=0)
    ax.set_axis_off()
    fig.savefig(out / "topics_tsne.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. Content over time: each congress's baseline, net of their average
    E = topic_words.congress_baselines(C, colnames, congresses, background=m)
    D = E - E.mean(axis=0, keepdims=True)
    cot = pd.DataFrame({"congress": congresses, "year": [start_year(t) for t in congresses],
                        "top_terms": [top_terms(D[i], vocab, 6) for i in range(len(congresses))]})
    cot.to_csv(out / "content_over_time.csv", index=False)
    intercept = pd.DataFrame({"year": ["Average congress"], "top_terms": [top_terms(E.mean(axis=0), vocab, 6)]})
    sel = pd.concat([intercept, cot[cot["year"].isin(SLIDE_YEARS)][["year", "top_terms"]]], ignore_index=True)
    latex(sel.rename(columns={"year": "Year of Congress", "top_terms": "Top Words"}), out / "content_over_time.tex", "l|l")

    # 5. Word embeddings: each term's loadings on the topics and covariates (the rows of the linear decoder's
    # effective matrix; the SAGE decoder's deviations, incl. topic x party ones), centered across terms
    W = np.vstack([B, C] + ([comp["interactions"].reshape(-1, B.shape[1])] if "interactions" in comp else []))
    Emb = W.T - W.T.mean(axis=0, keepdims=True)
    Emb /= np.linalg.norm(Emb, axis=1, keepdims=True) + 1e-12
    col = {v: j for j, v in enumerate(vocab)}
    nb = {}
    for w in QUERY_WORDS:
        if w not in col:
            print(f"  embedding query '{w}' not in the vocabulary", flush=True)
            continue
        sim = Emb @ Emb[col[w]]
        sim[col[w]] = -np.inf
        nb[w] = [vocab[j] for j in np.argsort(-sim)[:8]]
    pd.DataFrame(nb).to_csv(out / "word_embeddings.csv", index=False)
    blocks = [list(nb)[i:i + 4] for i in range(0, len(nb), 4)]
    with open(out / "word_embeddings.tex", "w") as f:
        for b in blocks:
            latex(pd.DataFrame({w: nb[w] for w in b}), out / "_tmp.tex", "l|" * (len(b) - 1) + "l")
            f.write((out / "_tmp.tex").read_text() + "\n")
    (out / "_tmp.tex").unlink()

    # 6. Topic deviations for Republicans (posterior shares; equal weights across congresses)
    party = docs["party"].to_numpy()
    cong = docs["congress"].to_numpy()
    diffs, vars_ = [], []
    for t in congresses:
        r, d = theta[(cong == t) & (party == "R")], theta[(cong == t) & (party == "D")]
        diffs.append(r.mean(0) - d.mean(0))
        vars_.append(r.var(0, ddof=1) / len(r) + d.var(0, ddof=1) / len(d))
    ame = np.mean(diffs, axis=0)
    se = np.sqrt(np.sum(vars_, axis=0)) / len(congresses)
    dev = pd.DataFrame({"topic": range(K), "label": labels, "republican_minus_democrat": ame, "se": se})
    dev.to_csv(out / "topic_deviations_for_republicans.csv", index=False)
    dev = dev.sort_values("republican_minus_democrat")
    fig, ax = plt.subplots(figsize=(6, 0.35 * K + 1.5))
    ax.barh(np.arange(K), dev["republican_minus_democrat"], xerr=1.96 * dev["se"],
            color=np.where(dev["republican_minus_democrat"] > 0, "tab:red", "tab:blue"), alpha=0.75, capsize=3)
    ax.set_yticks(np.arange(K), dev["label"])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Republican minus Democratic topic share (average over congresses)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "topic_deviations_for_republicans.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 7. Party-specific language within topics: each topic's word distribution for Republicans minus
    # Democrats in every congress (decoder at the congress's design row), averaged over congresses
    acc_all = np.zeros((K, len(vocab)))
    for t in congresses:
        qR, qD = (tm.get_topic_word_distribution(content_covariates=design_row(t, ps, congresses, colnames,
                                                                                cfg["formula"])).astype(np.float64)
                  for ps in (1, -1))
        acc_all += qR - qD
    acc_all /= len(congresses)
    rows = []
    for k in range(K):
        acc = acc_all[k]
        rows += [{"topic": k, "label": labels[k], "type": "Baseline", "top_terms": top_terms(Bfrex[k], vocab, 6)},
                 {"topic": k, "label": labels[k], "type": "Republicans", "top_terms": top_terms(acc, vocab, 6)},
                 {"topic": k, "label": labels[k], "type": "Democrats", "top_terms": top_terms(-acc, vocab, 6)}]
    psl = pd.DataFrame(rows)
    psl.to_csv(out / "party_specific_language.csv", index=False)
    latex(psl[["label", "type", "top_terms"]].rename(columns={"label": "Topic Label", "type": "Type",
                                                                "top_terms": "Top Words"}),
          out / "party_specific_language.tex", "lll")
    print(f"exhibits written to {out}", flush=True)


if __name__ == "__main__":
    main()
