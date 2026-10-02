"""Vote modality storage: dense columns (float32 tensors, row views) and precomputed sparse
matrices (+1 yea, -1 nay, 0 not observed) must give the same items and the same model."""

import tempfile

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from deeplatent import Corpus, IdealPointNN


def _votes(n=120, j=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X = (rng.uniform(size=(n, j)) < 1 / (1 + np.exp(-np.outer(x, rng.normal(0, 1.5, j))))).astype(
        float
    )
    X[rng.uniform(size=(n, j)) < 0.5] = np.nan
    return X


def _dense(X):
    cols = [f"r{k}" for k in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    return Corpus(df, modalities={"vote": {"column": cols, "views": {"rc": {"type": "vote"}}}})


def _sparse(X):
    S = sp.csr_matrix(np.where(np.isnan(X), 0, np.where(X > 0, 1, -1)).astype(np.int8))
    df = pd.DataFrame({"unit": np.arange(len(X))})
    return Corpus(df, modalities={"vote": {"views": {"rc": {"type": "vote", "matrix": S}}}})


def _fit(corpus, num_workers=0, steps=150):
    return IdealPointNN(
        train_data=corpus,
        n_ideal_points=1,
        ae_type="vae",
        vi_type="mean_field",
        update_prior=False,
        fusion="corrected_poe",
        w_prior=1.0,
        encoder_args={"vote_rc": {"hidden_dims": [16]}},
        decoder_args={"vote_rc": {"hidden_dims": [], "bias": True}},
        batch_size=32,
        num_steps=steps,
        num_workers=num_workers,
        print_every_n_steps=10**9,
        return_best_model=False,
        ckpt_folder=tempfile.mkdtemp(),
        seed=0,
        device="cpu",
    )


def test_dense_and_sparse_items_match():
    X = _votes()
    dense, sparse = _dense(X), _sparse(X)
    for i in (0, 7, len(X) - 1):
        a = dense[i]["modalities"]["vote"]["rc"]
        b = sparse[i]["modalities"]["vote"]["rc"]
        assert a["matrix"].dtype == b["matrix"].dtype == torch.float32
        torch.testing.assert_close(a["matrix"], b["matrix"])
        assert torch.equal(a["mask"], b["mask"])
        # coding: 1 yea, 0 nay, 2 and masked when missing
        expected = np.where(np.isnan(X[i]), 2.0, X[i])
        np.testing.assert_array_equal(a["matrix"].numpy(), expected)
        np.testing.assert_array_equal(a["mask"].numpy(), np.isnan(X[i]))


def test_dense_and_sparse_train_the_same_model():
    X = _votes()
    fits = [_fit(corpus) for corpus in (_dense(X), _sparse(X))]
    for p, q in zip(fits[0].decoders.parameters(), fits[1].decoders.parameters()):
        torch.testing.assert_close(p, q)


def test_sparse_votes_must_be_sparse():
    with pytest.raises(TypeError, match="scipy.sparse"):
        Corpus(
            pd.DataFrame({"u": [0, 1]}),
            modalities={"vote": {"views": {"rc": {"type": "vote", "matrix": np.ones((2, 3))}}}},
        )


def test_training_with_persistent_workers():
    model = _fit(_dense(_votes(n=64)), num_workers=2, steps=40)  # several epochs
    assert model.steps == 40
