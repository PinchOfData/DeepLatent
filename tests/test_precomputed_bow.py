"""Bag-of-words views from precomputed sparse counts: a precomputed count matrix must give
the same items and the same model as the vectorizer path."""

import tempfile

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer

from deeplatent import GTM, Corpus

N = 60


def _data(seed=0):
    rng = np.random.default_rng(seed)
    words = np.array([f"w{k:02d}" for k in range(30)])
    docs = [" ".join(rng.choice(words[: 15 + (i % 2) * 15], rng.integers(5, 40))) for i in range(N)]
    return pd.DataFrame({"doc": docs, "x": rng.normal(size=N)})


def _corpora():
    df = _data()
    vec = CountVectorizer().fit(df["doc"])
    from_text = Corpus(
        df,
        modalities={"t": {"column": "doc", "views": {"bow": {"type": "bow", "vectorizer": vec}}}},
        prevalence="~ x",
    )
    counts = sp.csr_matrix(vec.transform(df["doc"]).astype(np.int32))
    vocab = list(vec.get_feature_names_out())
    precomputed = Corpus(
        df[["x"]],
        modalities={"t": {"views": {"bow": {"type": "bow", "matrix": counts, "vocab": vocab}}}},
        prevalence="~ x",
    )
    return from_text, precomputed


def _gtm(corpus, steps=30):
    return GTM(
        train_data=corpus,
        n_topics=3,
        ae_type="vae",
        vi_type="mean_field",
        doc_topic_prior="logistic_normal",
        update_prior=True,
        encoder_args={"t_bow": {"hidden_dims": [8]}},
        decoder_args={"t_bow": {"hidden_dims": [], "bias": True}},
        batch_size=16,
        num_steps=steps,
        num_workers=0,
        print_every_n_steps=10**9,
        return_best_model=False,
        ckpt_folder=tempfile.mkdtemp(),
        seed=0,
        device="cpu",
    )


def test_precomputed_counts_match_vectorizer():
    from_text, precomputed = _corpora()
    assert from_text.id2token == precomputed.id2token
    idx = [0, 5, 59]
    for i in idx:
        torch.testing.assert_close(
            from_text[i]["modalities"]["t"]["bow"], precomputed[i]["modalities"]["t"]["bow"]
        )
    torch.testing.assert_close(
        from_text.get_batch(idx)["modalities"]["t"]["bow"],
        precomputed.get_batch(idx)["modalities"]["t"]["bow"],
    )
    fits = [_gtm(c) for c in (from_text, precomputed)]
    assert fits[0].train_losses == fits[1].train_losses


def test_precomputed_counts_must_be_sparse_and_match_vocab():
    df = pd.DataFrame({"x": [0.0, 1.0]})
    with pytest.raises(TypeError, match="scipy.sparse"):
        Corpus(df, modalities={"t": {"views": {"bow": {"type": "bow", "matrix": np.ones((2, 3))}}}})
    with pytest.raises(ValueError, match="vocab"):
        Corpus(
            df,
            modalities={
                "t": {"views": {"bow": {"type": "bow", "matrix": sp.csr_matrix(np.ones((2, 3))), "vocab": ["a"]}}}
            },
        )
