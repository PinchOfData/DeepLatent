"""Step 05: fit the GTM to the speeches (House and Senate pooled).

Model: the estimator validated in scripts/sim_polarization_decomposition.py (--time-effects period),
with the application's covariates. For speech d in congress t by a member of party p:
    theta_d = softmax(eta_d V'),  eta_d ~ N(Gamma' x_d, Sigma)       (logistic normal, learned Sigma)
    words_d ~ Multinomial(n_d, softmax(W2 W1 [theta_d, x_d]))                 (--decoder linear)
    words_d ~ Multinomial(n_d, sum_k theta_dk beta_k(x_d))                   (--decoder sage)
with x_d the design row of "~ 0 + C(congress) + C(congress):party_signed" (party_signed = +1 R, -1 D):
a separate baseline and a separate party effect for every congress, in the topic prevalence
(prior mean) and in the word content (decoder). (Fits before 2026-10-02 used "~ C(congress) + ...",
recorded in their config.json.) W2 W1 is a 256-wide hidden layer without activation
or bias. With K + p <= 256 columns in [theta, x] it restricts nothing (W2 W1 can be any V x (K + p)
matrix): it reparameterizes the simulation's linear decoder, and the rows of W2 serve as word
embeddings. --decoder sage (2026-10-08) is the STM's decoder instead: each word comes from one topic, whose
distribution beta_k(x) = softmax(m + kappa_k + x' kappa_cov [+ topic x party deviations]) is a SAGE model
(common.decoder_config: --interactions, --l1 or --jeffreys, --background). --init kmeans starts its topics at the word
frequencies of K clusters of the training speeches (spherical k-means on tf-idf rows of --init-docs speeches)
instead of near the background, as the STM initializes its topics from the data: from a random start the
mixture's first topics to specialize take most speeches and the others stay unused (topic collapse).
Estimation: VAE, loss = -ELBO (w_prior = 1), mixture-of-Gaussians posterior (10 components), learned
prior mean and covariance, main lr 1e-3 and prior lr 1e-4 without weight decay, optional KL annealing
(--kl-anneal). Early stopping with the package API: --val-frac of the speeches of every congress x
party are held out (test_data); every --validate-every steps the package scores the whole held-out set
(per-speech -ELBO) and keeps the best model; training stops after --patience evaluations without
improvement (and reloads the best model) or at --num-steps. If --num-steps is reached first, the
package returns the last model, not the best one: the log then shows both held-out losses.
--val-frac 0: no held-out set, the model after --num-steps is kept. --checkpoint-every N saves the model
every N steps to <common.CHECKPOINTS>/<tag>/ (package log_every_n_steps; on the HPC a scratch folder),
for 08_checkpoint_topics.py.
--from-checkpoint PATH: no training; the outputs below are written for that saved model instead (with
the same data options as its fit), e.g. for the checkpoint chosen after the review.
Documents (chosen here, not in the cleaning): the GST speeches (common.gst_sample: Republicans and
Democrats identified by name, voting members, no Extensions of Remarks) of the main build
(common.MAIN_SOURCES) with at least --min-words words in the raw text and at least --min-terms
terms of the vocabulary (--vocab), without duplicates (a speech with the same term counts and the same
speaker as one at most --duplicate-days days earlier, e.g. printed or parsed twice; -1 keeps them); --per-cell N draws N of them at random per congress x party
(stratified pilot), 0 keeps them all. --sample-vocab V applies the --min-terms and duplicate rules to the
counts of vocabulary V instead, so that a larger --vocab changes the model's words but not the speeches.

Outputs results/<tag>/: gtm.ckpt, config.json, trajectory.json (held-out loss at every evaluation up to
the kept model; the job log prints all of them), docs.csv.gz (speech metadata in model order, with
the held-out flag), theta.npy (posterior-mean topic shares, --posterior-samples draws), and
exhibits/topic_words.csv + topic_words_{probable,frex,observed}.tex (topic_words.py).

Usage: python 05_fit_gtm.py --tag full_k20 --n-topics 20 --per-cell 0 --kl-anneal 5000 30000
"""
import argparse
import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import common
import topic_words
from torch.utils.data import DataLoader

from deeplatent import GTM, Corpus

# One baseline per congress (no common intercept): with a common intercept the first congress is the
# reference, its baseline is the intercept (an average over all congresses early in training), and its
# uncentered party term absorbs the difference (43rd Congress: 74% Republican speeches).
FORMULA = "~ 0 + C(congress) + C(congress):party_signed"
DEC_KEY = "text_bow"


def duplicate_speeches(m, person, date, days: int) -> np.ndarray:
    """True for speeches that repeat an earlier speech: identical term counts (same terms, same counts),
    same person, at most `days` days after the previous copy. Rows of the CSR count matrix m."""
    m = m.tocsr()
    m.sort_indices()
    key = [hash((m.indices[m.indptr[i]:m.indptr[i + 1]].tobytes(), m.data[m.indptr[i]:m.indptr[i + 1]].tobytes()))
           for i in range(m.shape[0])]
    d = pd.DataFrame({"key": key, "person": person, "date": pd.to_datetime(date)}).sort_values(["key", "person", "date"])
    same = (d["key"] == d["key"].shift()) & (d["person"] == d["person"].shift())
    dup = same & ((d["date"] - d["date"].shift()).dt.days <= days)
    return dup.sort_index().to_numpy()


def load_documents(terms, min_words: int, min_terms: int, per_cell: int, seed: int, congresses=(43, 118),
                   duplicate_days: int = 7, sample_terms=None):
    """sample_terms: the counts (terms_dir) that the --min-terms and duplicate rules use; default terms."""
    rng = np.random.default_rng(seed)
    sample_terms = sample_terms or terms
    mats, metas = [], []
    for source, c in common.MAIN_SOURCES:
        if not congresses[0] <= c <= congresses[1]:
            continue
        m = sp.load_npz(terms / f"{common.stem(source, c)}.npz")
        meta = pd.read_csv(terms / source / f"{c:03d}_meta.csv.gz",
                           dtype={"speaker_session": str, "person_id": str})
        if sample_terms == terms:
            ms, n_terms = m, meta["n_terms"].to_numpy()
        else:
            ms = sp.load_npz(sample_terms / f"{common.stem(source, c)}.npz")
            ids = pd.read_csv(sample_terms / source / f"{c:03d}_meta.csv.gz", usecols=["speech_id", "n_terms"],
                              dtype={"speech_id": str})
            assert ids["speech_id"].equals(meta["speech_id"].astype(str)) and ms.shape[0] == m.shape[0]
            n_terms = ids["n_terms"].to_numpy()
        idx = np.flatnonzero(common.gst_sample(meta).to_numpy() & (meta["raw_word_count"].to_numpy() >= min_words)
                             & (n_terms >= min_terms))
        if duplicate_days >= 0:  # repeated speeches (printed twice, parsed twice): keep the first copy
            dup = duplicate_speeches(ms[idx], meta["person_id"].to_numpy()[idx], meta["date"].to_numpy()[idx],
                                     duplicate_days)
            if dup.any():
                print(f"  congress {c}: {dup.sum()} duplicate speeches dropped", flush=True)
            idx = idx[~dup]
        if per_cell:
            party = meta["party"].to_numpy()
            idx = np.sort(np.concatenate([
                rng.choice(cand, size=min(per_cell, len(cand)), replace=False)
                for cand in (idx[party[idx] == p] for p in ("D", "R")) if len(cand)
            ]))
        mats.append(m[idx])
        metas.append(meta.iloc[idx].assign(source=source, term_row=idx))
    X = sp.vstack(mats).tocsr()
    docs = pd.concat(metas, ignore_index=True)
    docs["party_signed"] = np.where(docs["party"] == "R", 1, -1)
    docs["row"] = np.arange(len(docs))
    return X, docs


def heldout_mask(docs, frac: float, seed: int, by: str = "speech") -> np.ndarray:
    """True for the held-out set: a random `frac` of the speeches of every congress x party (by="speech"), or all
    the speeches in that congress of a random `frac` of its speakers of each party (by="speaker": new speakers,
    e.g. to choose the L1 penalty by how well the party language generalizes beyond the speakers it was fit on)."""
    rng = np.random.default_rng(seed + 1)
    held = np.zeros(len(docs), dtype=bool)
    for idx in docs.groupby(["congress", "party"], sort=True).indices.values():
        if by == "speech":
            held[rng.choice(idx, size=int(round(frac * len(idx))), replace=False)] = True
        else:
            people = np.unique(docs["person_id"].to_numpy()[idx])
            out = rng.choice(people, size=int(round(frac * len(people))), replace=False)
            held[idx[np.isin(docs["person_id"].to_numpy()[idx], out)]] = True
    return held


def kmeans_topics(X, K: int, n_docs: int, seed: int):
    """K x V word counts (+1) of K clusters of n_docs random rows of X: k-means on the L2-normalized tf-idf
    rows (cosine geometry). Also returns the cluster sizes."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=min(n_docs, X.shape[0]), replace=False))
    S = X[idx].astype(np.float64)
    idf = np.log((1 + S.shape[0]) / (1 + np.asarray((S > 0).sum(axis=0)).ravel())) + 1.0
    Z = normalize(normalize(S, norm="l1") @ sp.diags(idf), norm="l2")
    labels = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=4096, n_init=3).fit_predict(Z)
    counts = np.vstack([np.asarray(S[labels == k].sum(axis=0)).ravel() for k in range(K)]) + 1.0
    return counts, np.bincount(labels, minlength=K)


def party_summaries(tm, K: int) -> dict:
    G = tm.get_prevalence_coefficients(fold_bias=False)  # K x p, lifted and centered
    names = tm.prevalence_colnames
    party_cols = [j for j, n in enumerate(names) if n.endswith(":party_signed")]
    C = tm.get_word_logit_components()["covariates"]
    cont_cols = [j for j, n in enumerate(tm.content_colnames) if n.endswith(":party_signed")]
    Cp = C[cont_cols] - C[cont_cols].mean(axis=1, keepdims=True)
    return {
        "prevalence_party_norm_mean": float(np.linalg.norm(G[:, party_cols], axis=0).mean()),
        "content_party_rms_mean": float(np.sqrt((Cp ** 2).mean(axis=1)).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-topics", type=int, default=20)
    ap.add_argument("--vocab", default=str(common.VOCAB_PATH))
    ap.add_argument("--min-words", type=int, default=0, help="words in the raw text")
    ap.add_argument("--min-terms", type=int, default=30, help="vocabulary terms")
    ap.add_argument("--sample-vocab", default=None, help="vocabulary of the --min-terms and duplicate rules")
    ap.add_argument("--per-cell", type=int, default=1500, help="speeches per congress x party; 0 = all")
    ap.add_argument("--duplicate-days", type=int, default=7,
                    help="drop a speech with the same term counts and speaker as one at most this many days "
                         "earlier; -1 keeps duplicates")
    ap.add_argument("--num-steps", type=int, default=400000, help="cap on training steps")
    ap.add_argument("--log-every", type=int, default=1000, help="steps between printed losses and topic words")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out share of every congress x party")
    ap.add_argument("--val-by", choices=["speech", "speaker"], default="speech",
                    help="hold out speeches, or speakers (all their speeches in the congress)")
    ap.add_argument("--validate-every", type=int, default=1000, help="steps between held-out evaluations")
    ap.add_argument("--patience", type=int, default=50, help="evaluations without improvement before stopping")
    ap.add_argument("--checkpoint-every", type=int, default=0, help="steps between saved checkpoints; 0 = none")
    ap.add_argument("--from-checkpoint", default=None,
                    help="skip training and use this saved model (absolute, or relative to the checkpoint folder)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--prior-lr", type=float, default=1e-4)
    ap.add_argument("--mixture-components", type=int, default=10)
    ap.add_argument("--encoder-hidden", type=int, nargs="+", default=[1024, 512, 256])
    ap.add_argument("--decoder", choices=["linear", "sage"], default="linear",
                    help="linear: topics add up in the word logits; sage: the STM's mixture of SAGE topics")
    ap.add_argument("--interactions", choices=["none", "party", "congress_party"], default="none",
                    help="sage: topic-specific party deviations (none, one for all congresses, one per congress)")
    ap.add_argument("--l1", type=float, default=0.0, help="sage: L1 penalty on the deviations (word counts)")
    ap.add_argument("--jeffreys", action="store_true",
                    help="sage: Normal-Jeffreys prior on every deviation (SAGE; estimated SCHOLAR-style)")
    ap.add_argument("--jeffreys-start", type=int, default=5000,
                    help="sage --jeffreys: training step of the first refresh of the precisions (1 before)")
    ap.add_argument("--jeffreys-every", type=int, default=100, help="sage --jeffreys: steps between refreshes")
    ap.add_argument("--background", choices=["fixed", "learned"], default="fixed",
                    help="sage: background log frequencies fixed at the corpus's or learned")
    ap.add_argument("--init", choices=["random", "kmeans"], default="random",
                    help="sage: initial topics near the background (random) or from k-means clusters of speeches")
    ap.add_argument("--init-docs", type=int, default=200000, help="sage --init kmeans: speeches clustered")
    ap.add_argument("--decoder-hidden", type=int, default=256, help="linear: width of the hidden layer")
    ap.add_argument("--dropout", type=float, default=0.1, help="encoder and decoder hidden layers")
    ap.add_argument("--kl-anneal", type=int, nargs=2, default=[-1, -1], metavar=("START", "END"),
                    help="KL weight 0 before START, linear to 1 at END (default: no annealing)")
    ap.add_argument("--posterior-samples", type=int, default=30)
    ap.add_argument("--congresses", default="43-118", help="first-last congress to include")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.from_checkpoint and not Path(a.from_checkpoint).is_absolute():
        a.from_checkpoint = str(common.CHECKPOINTS / a.from_checkpoint)
    out = common.RESULTS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    vocab = common.load_vocab(a.vocab)
    lo, _, hi = a.congresses.partition("-")
    X, docs = load_documents(common.terms_dir(a.vocab), a.min_words, a.min_terms, a.per_cell, a.seed,
                             (int(lo), int(hi or lo)), duplicate_days=a.duplicate_days,
                             sample_terms=common.terms_dir(a.sample_vocab) if a.sample_vocab else None)
    docs["heldout"] = heldout_mask(docs, a.val_frac, a.seed, a.val_by)
    print(f"{len(docs):,} speeches ({docs['heldout'].sum():,} held out), {X.nnz:,} nonzeros, congresses "
          f"{docs['congress'].min()}-{docs['congress'].max()}, by party {docs['party'].value_counts().to_dict()} "
          f"({time.time() - t0:.0f}s)", flush=True)
    counts = common.PrecomputedCounts(X, vocab)

    def make_corpus(d):
        return Corpus(d.reset_index(drop=True),
                      modalities={"text": {"column": "row", "views": {"bow": {"type": "bow", "vectorizer": counts}}}},
                      prevalence=FORMULA, content=FORMULA)

    train = make_corpus(docs[~docs["heldout"]])
    val = make_corpus(docs[docs["heldout"]]) if a.val_frac > 0 else None
    p = len(train.content_colnames)
    if a.decoder == "linear":
        assert a.n_topics + p <= a.decoder_hidden, "the hidden layer would restrict the decoder's rank"
    if val is not None:
        assert val.content_colnames == train.content_colnames
        assert val.prevalence_colnames == train.prevalence_colnames
    print(f"corpora built ({time.time() - t0:.0f}s)", flush=True)

    init_topics = None
    if a.decoder == "sage" and a.init == "kmeans" and not a.from_checkpoint:
        t_km = time.time()
        init_topics, sizes = kmeans_topics(X[np.flatnonzero(~docs["heldout"].to_numpy())], a.n_topics, a.init_docs,
                                           a.seed)
        lift = np.log(init_topics / init_topics.sum(1, keepdims=True)) - np.log(
            init_topics.sum(0) / init_topics.sum())
        pd.DataFrame({"cluster": range(a.n_topics), "speeches": sizes,
                      "top_terms": [", ".join(vocab[j] for j in np.argsort(-(lift[k] * (init_topics[k] > 50)))[:12])
                                    for k in range(a.n_topics)]}).to_csv(out / "init_topics.csv", index=False)
        print(f"k-means initial topics ({time.time() - t_km:.0f}s), cluster sizes {sizes.tolist()}", flush=True)

    ckpt_dir = common.CHECKPOINTS / a.tag
    if a.checkpoint_every:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="app1_ckpt_") as tmp:
        t_fit = time.time()
        tm = GTM(
            train_data=train,
            test_data=val,
            n_topics=a.n_topics,
            doc_topic_prior="logistic_normal",
            ae_type="vae",
            vi_type="mixture_of_gaussians",
            mixture_components=a.mixture_components,
            update_prior=True,
            learn_prior_cov=True,
            encoder_args={DEC_KEY: {"hidden_dims": a.encoder_hidden, "activation": "relu",
                                    "bias": True, "dropout": a.dropout}},
            decoder_args={DEC_KEY: common.decoder_config(vars(a), train.content_colnames, init_topics)},
            optim_args={"main": {"lr": a.lr, "weight_decay": 0.0},
                        "prior": {"lr": a.prior_lr, "weight_decay": 0.0}},
            w_prior=1.0,
            kl_annealing_start=a.kl_anneal[0],
            kl_annealing_end=a.kl_anneal[1],
            batch_size=a.batch_size,
            num_steps=0 if a.from_checkpoint else a.num_steps,
            num_workers=0,
            print_every_n_steps=a.log_every,  # training and held-out loss, split into reconstruction and KL
            print_topics=True,  # each topic's top words at every print
            validate_every_n_steps=a.validate_every,
            # Without held-out data the package would apply patience to single training batches.
            patience=a.patience if val is not None else float("inf"),
            log_every_n_steps=a.checkpoint_every or float("inf"),
            # The best model is tracked on the held-out loss only (without held-out data the package
            # would track single training batches and save a checkpoint at each improvement).
            return_best_model=val is not None,
            ckpt_folder=str(ckpt_dir) if a.checkpoint_every else tmp,
            seed=a.seed,
            device=torch.device(a.device),
        )
        fit_seconds = time.time() - t_fit
    if a.from_checkpoint:
        tm.load_model(str(a.from_checkpoint))
        tm.device = torch.device(a.device)  # load_model restores the device attribute saved with the model
        print(f"loaded {a.from_checkpoint} (step {tm.steps})", flush=True)
    tm.save_model(str(out / "gtm.ckpt"))

    # Training step of the kept model (a checkpoint's file name ends with its step).
    step = int(re.search(r"_(\d+)\.ckpt$", a.from_checkpoint).group(1)) if a.from_checkpoint else tm.steps
    fit = {"fit_seconds": fit_seconds, "train_steps_recorded": len(tm.train_losses), "model_step": step}
    if val is not None and tm.val_steps:
        loader = DataLoader(val, batch_size=a.batch_size, shuffle=False, num_workers=0)
        t_eval = time.time()
        returned = tm.evaluate(loader, val)
        fit.update({"eval_seconds": time.time() - t_eval, "kept_step": tm.val_steps[-1],
                    "best_heldout_loss": float(min(tm.val_losses)), "kept_model_heldout_loss": returned})
        print(f"held-out loss of the kept model (step {tm.val_steps[-1]}): {returned:.4f} at re-evaluation, "
              f"{tm.val_losses[-1]:.4f} when selected, best {min(tm.val_losses):.4f}; "
              f"one evaluation takes {fit['eval_seconds']:.1f}s", flush=True)
    with open(out / "trajectory.json", "w") as f:
        json.dump({"val_steps": tm.val_steps, "val_loss": tm.val_losses, "val_recon": tm.val_recon_losses,
                   "val_kl": tm.val_div_losses,
                   "train_loss_per_1000_steps": [float(np.mean(tm.train_losses[i:i + 1000]))
                                                 for i in range(0, len(tm.train_losses), 1000)]}, f)

    # Posterior-mean topic shares of every speech, training and held-out, in docs order.
    theta = np.empty((len(docs), a.n_topics), dtype=np.float32)
    held = docs["heldout"].to_numpy()
    for sel, corpus in ((~held, train), (held, val)):
        if sel.any():
            theta[sel] = np.asarray(tm.get_doc_topic_distribution(corpus, num_samples=a.posterior_samples))
    np.save(out / "theta.npy", theta)
    eig = np.linalg.eigvalsh(tm.prior.sigma.detach().cpu().numpy())
    fit.update({"sigma_eig_min": float(eig.min()), "sigma_eig_max": float(eig.max()),
                "median_max_share": float(np.median(theta.max(axis=1))),
                "median_entropy": float(np.median(-(theta * np.log(theta + 1e-12)).sum(axis=1))),
                **party_summaries(tm, a.n_topics)})
    print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in fit.items()), flush=True)
    topic_words.write(out / "exhibits", vocab, [f"Topic {k}" for k in range(a.n_topics)], theta.mean(axis=0),
                      topic_words.model_distribution(tm), theta, X)
    docs.drop(columns=["row"]).to_csv(out / "docs.csv.gz", index=False)
    with open(out / "config.json", "w") as f:
        json.dump({**vars(a), "formula": FORMULA, "n_docs": len(docs), "vocab_size": len(vocab),
                   "prevalence_colnames": tm.prevalence_colnames, "content_colnames": tm.content_colnames,
                   **fit, "seconds": time.time() - t0}, f, indent=1)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
