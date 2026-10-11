"""Tests for the (K-1)-dim contrast parameterization of logistic-normal latents (v0.2.0).

Softmax is invariant to shifts along the all-ones logit direction, so a K-dim
logistic-normal latent carries K unidentified parameter directions. Since 0.2.0
the latent is eta in R^{K-1} with theta = softmax(eta @ V.T), V an orthonormal
zero-sum (Helmert) basis. These tests pin down: the basis properties, the exact
gauge identity, latent/theta shapes across vi_types, the structural no-op for
gaussian/dirichlet priors, distributional equivalence of the prior, the
coefficient-lifting helper, and the pre-0.2.0 checkpoint guard.
"""

import os
import tempfile

import numpy as np
import pytest
import torch
from sklearn.feature_extraction.text import CountVectorizer

from deeplatent import (
    Corpus,
    GTM,
    IdealPointNN,
    contrast_basis,
    generate_documents,
    generate_ideal_points,
)
from deeplatent.priors import FixedLogisticNormalPrior, LogisticNormalPrior

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 4


def _text_corpus(num_topics=K, num_docs=400, with_labels=False):
    label_kwargs = (
        dict(label_type="regression", label_coeffs=np.linspace(1, -1, num_topics))
        if with_labels
        else {}
    )
    _, df, *_ = generate_documents(
        num_docs=num_docs,
        num_topics=num_topics,
        vocab_size=60,
        num_covs=2,
        doc_topic_prior="logistic_normal",
        min_words=15,
        max_words=15,
        random_seed=0,
        **label_kwargs,
    )
    vec = CountVectorizer()
    vec.fit(df["doc_clean_0"])
    mods = {
        "text": {
            "column": "doc_clean_0",
            "views": {"bow": {"type": "bow", "vectorizer": vec}},
        }
    }
    labels = {"y": {"column": "label", "type": "regression"}} if with_labels else None
    return Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2", labels=labels)


def _gtm(corpus, tmp_path, **overrides):
    kwargs = dict(
        train_data=corpus,
        n_topics=K,
        ae_type="vae",
        vi_type="mean_field",
        doc_topic_prior="logistic_normal",
        update_prior=True,
        encoder_args={"text_bow": {"hidden_dims": [32]}},
        decoder_args={"text_bow": {"hidden_dims": []}},
        batch_size=64,
        num_steps=20,
        num_workers=0,
        print_every_n_steps=10**9,
        return_best_model=False,
        ckpt_folder=str(tmp_path),
        seed=0,
        device=DEVICE,
    )
    kwargs.update(overrides)
    return GTM(**kwargs)


def test_contrast_basis_properties():
    for k in (2, 3, 5, 20):
        V = contrast_basis(k)
        assert V.shape == (k, k - 1)
        assert torch.allclose(V.T @ V, torch.eye(k - 1), atol=1e-6)
        assert torch.allclose(V.sum(0), torch.zeros(k - 1), atol=1e-6)
        assert torch.equal(V, contrast_basis(k))  # deterministic
    with pytest.raises(ValueError):
        contrast_basis(1)


def test_softmax_gauge_identity():
    """softmax(z) == softmax(V V^T z): the exact-equivalence kernel of the change."""
    V = contrast_basis(K)
    z = torch.randn(256, K)
    theta_full = torch.softmax(z, dim=1)
    theta_contrast = torch.softmax((z @ V) @ V.T, dim=1)
    assert torch.allclose(theta_full, theta_contrast, atol=1e-5)


@pytest.mark.parametrize(
    "vi_type", ["mean_field", "full_rank", "iaf", "mixture_of_gaussians"]
)
def test_latent_and_theta_shapes(vi_type, tmp_path):
    corpus = _text_corpus()
    m = _gtm(corpus, tmp_path, vi_type=vi_type, mixture_components=3)
    assert m.contrast is True and m.n_latent == K - 1
    theta = m.get_doc_topic_distribution(corpus, num_samples=2)
    assert theta.shape == (400, K)
    assert np.abs(theta.sum(1) - 1).max() < 1e-4
    z = m.get_latent_factors(corpus, to_simplex=False)
    assert z.shape == (400, K - 1)
    lifted = m.latent_to_theta(torch.tensor(z, dtype=torch.float32, device=m.device))
    assert lifted.shape == (400, K)
    assert torch.allclose(lifted.sum(1), torch.ones(400, device=m.device), atol=1e-4)


def test_gaussian_and_dirichlet_are_noops(tmp_path):
    num_bills = 40
    _, dfi, *_ = generate_ideal_points(
        num_politicians=150,
        dim_ideal_points=1,
        num_bills=num_bills,
        num_survey_questions=2,
        doc_length=30,
        vocab_size=50,
        seed=0,
        progress_bar=False,
    )
    corp_ip = Corpus(
        dfi,
        modalities={
            "vote": {
                "column": [f"vote_{i+1}" for i in range(num_bills)],
                "views": {"responses": {"type": "vote"}},
            }
        },
    )
    m = IdealPointNN(
        ae_type="vae",
        vi_type="mean_field",
        n_ideal_points=2,
        train_data=corp_ip,
        encoder_args={"vote_responses": {"hidden_dims": [32]}},
        decoder_args={"vote_responses": {"hidden_dims": []}},
        batch_size=64,
        num_steps=10,
        num_workers=0,
        print_every_n_steps=10**9,
        return_best_model=False,
        ckpt_folder=str(tmp_path),
        seed=0,
        device=DEVICE,
    )
    assert m.contrast is False and m.n_latent == 2

    corpus = _text_corpus()
    md = _gtm(corpus, tmp_path, ae_type="wae", doc_topic_prior="dirichlet")
    assert md.contrast is False and md.n_latent == K


def test_prior_sample_matches_full_dim_reference():
    """Contrast prior samples match the K-dim N(0, I) logistic normal in distribution."""
    torch.manual_seed(0)
    prior = FixedLogisticNormalPrior(0, K)
    theta = prior.sample(20000, to_simplex=True).cpu().numpy()
    assert theta.shape == (20000, K)
    eta = prior.sample(2000, to_simplex=False)
    assert eta.shape == (2000, K - 1)

    rng = np.random.default_rng(0)  # reference: the pre-0.2.0 K-dim construction
    z = rng.standard_normal((20000, K))
    ref = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    assert np.abs(theta.mean(0) - ref.mean(0)).max() < 0.01
    assert np.abs(theta.std(0) - ref.std(0)).max() < 0.01


def test_get_prevalence_coefficients(tmp_path):
    corpus = _text_corpus()
    m = _gtm(corpus, tmp_path)
    W = m.get_prevalence_coefficients()
    assert W.shape == (K, 3)  # Intercept + 2 covariates
    assert np.abs(W.sum(0)).max() < 1e-5  # centered by construction

    W_prior, b_prior = m.prior.get_prevalence_coefficients()
    V = m.prior.V
    manual = (V @ m.prior.mean_net.weight).detach()
    assert torch.allclose(W_prior, manual, atol=1e-6)
    icol = m.prevalence_colnames.index("Intercept")
    manual_folded = manual.clone()
    manual_folded[:, icol] += b_prior
    assert np.allclose(W, manual_folded.cpu().numpy(), atol=1e-6)

    with pytest.raises(ValueError):
        LogisticNormalPrior(0, K).get_prevalence_coefficients()
    m_fixed = _gtm(corpus, tmp_path, update_prior=False)
    with pytest.raises(ValueError):
        m_fixed.get_prevalence_coefficients()


def test_predictions_guard_and_supervised_path(tmp_path):
    corpus = _text_corpus(with_labels=True)
    m = _gtm(
        corpus,
        tmp_path,
        vi_type="mixture_of_gaussians",
        mixture_components=3,
        labels_in_encoder=True,
        predictor_args={"y": {"hidden_dims": [], "loss_weight": 1.0}},
    )
    preds = m.get_predictions(corpus)
    assert preds["y"].shape == (400, 1)
    # head consumes theta (K dims), not the raw latent
    assert m.predictor.predictors["y"].neural_net["pred_0"].weight.shape[1] == K
    with pytest.raises(ValueError, match="to_simplex=False"):
        m.get_predictions(corpus, to_simplex=False)


def test_checkpoint_roundtrip_and_pre020_guard(tmp_path):
    corpus = _text_corpus()
    m = _gtm(corpus, tmp_path)
    ck = os.path.join(str(tmp_path), "m.ckpt")
    m.save_model(ck)

    m2 = _gtm(corpus, tmp_path)
    m2.load_model(ck)
    assert m2.get_doc_topic_distribution(corpus).shape == (400, K)
    m2.num_steps = m2.num_steps + 5  # one more training leg after reload
    m2.train(corpus)

    old = torch.load(ck, map_location="cpu", weights_only=False)
    old.pop("n_latent", None)  # pre-0.2.0 checkpoints have no n_latent key
    old["n_factors"] = K
    p_old = os.path.join(str(tmp_path), "old.ckpt")
    torch.save(old, p_old)
    with pytest.raises(ValueError, match="latent dimension mismatch"):
        m2.load_model(p_old)
