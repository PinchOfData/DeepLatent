"""Units that differ in which modalities they have (Corpus view option "available").

A unit's posterior is the corrected product of the experts it has (an absent expert is left
out, as if it were the prior; a unit with none keeps the prior), its likelihood counts only
the modalities it has, and the placeholder rows of absent modalities are never read. Also:
Corpus.get_batch (whole-batch fetching) returns exactly the default-collated items.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import default_collate

from deeplatent import Corpus, IdealPointNN
from deeplatent.models import _mixture_covariance

KEYS = ("a_w", "b_w")


def _assert_same(x, y):
    if isinstance(x, dict):
        assert x.keys() == y.keys()
        for k in x:
            _assert_same(x[k], y[k])
    else:
        assert x.dtype == y.dtype and x.shape == y.shape
        assert torch.equal(x, y) or torch.allclose(x, y, equal_nan=True, rtol=0, atol=0)


def _two_modality_corpus(
    n=64, seed=0, absent_fill=0.0, pattern=None, prevalence=True, declare=True
):
    """Embeddings a and b; by default units cycle through: both, only a, only b, none."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    pattern = np.arange(n) % 4 if pattern is None else pattern
    has = {"a": np.isin(pattern, (0, 1)), "b": np.isin(pattern, (0, 2))}
    mats = {}
    for k, load in (("a", 1.0), ("b", 0.6)):
        m = np.outer(theta, rng.normal(0, load, 4)) + rng.normal(0, 0.8, (n, 4))
        m[~has[k]] = absent_fill
        mats[k] = m.astype(np.float32)
    df = pd.DataFrame(
        {"x": rng.normal(size=n), "has_a": has["a"], "has_b": has["b"]}
    )
    mods = {
        k: {
            "column": "x",
            "views": {
                "w": {
                    "type": "embedding",
                    "matrix": mats[k],
                    **({"available": f"has_{k}"} if declare else {}),
                }
            },
        }
        for k in "ab"
    }
    return Corpus(df, modalities=mods, prevalence="~ x" if prevalence else None), pattern


def _model(corpus, keys=KEYS, vi_type="mean_field", steps=0, dim=1, **kw):
    return IdealPointNN(
        train_data=corpus,
        n_ideal_points=dim,
        ae_type="vae",
        vi_type=vi_type,
        fusion=kw.pop("fusion", "corrected_poe"),
        update_prior=True,
        mixture_components=2,
        encoder_args={k: {"hidden_dims": [8]} for k in keys},
        decoder_args={k: {"hidden_dims": [], "bias": True} for k in keys},
        batch_size=32,
        num_steps=steps,
        num_workers=0,
        print_every_n_steps=10**9,
        return_best_model=False,
        ckpt_folder=tempfile.mkdtemp(),
        seed=0,
        device=kw.pop("device", "cpu"),
        **kw,
    )


def _batch(model, corpus):
    batch = corpus.get_batch(np.arange(len(corpus)))
    inputs = model._encoder_inputs(batch, corpus)
    mask, rows = model._units_in_step(batch)
    return batch, inputs, mask, rows


# ----------------------------------------------------------------------------- Corpus


def test_get_batch_matches_collated_items():
    rng = np.random.default_rng(0)
    n = 30
    X = rng.choice([0.0, 1.0, np.nan], size=(n, 4))
    S = sp.csr_matrix(rng.choice([-1, 0, 1], size=(n, 5)).astype(np.int8))
    words = np.array(["aa", "bb", "cc", "dd"])
    df = pd.DataFrame(X, columns=[f"r{j}" for j in range(4)])
    df["doc"] = [" ".join(rng.choice(words, 6)) for _ in range(n)]
    df["q"] = rng.choice([0, 1, 2], n)
    df["x"] = rng.normal(size=n)
    df["y"] = rng.normal(size=n)
    df["has"] = rng.uniform(size=n) < 0.5
    corpus = Corpus(
        df,
        modalities={
            "text": {"column": "doc", "views": {"bow": {"type": "bow", "vectorizer": CountVectorizer()}}},
            "emb": {"column": "x", "views": {"w": {"type": "embedding", "matrix": rng.normal(size=(n, 3))}}},
            "dense": {"column": [f"r{j}" for j in range(4)], "views": {"v": {"type": "vote", "available": "has"}}},
            "sparse": {"views": {"v": {"type": "vote", "matrix": S}}},
            "survey": {"column": ["q"], "views": {"q": {"type": "discrete_choice"}}},
        },
        prevalence="~ x",
        content="~ y",
        labels={"y": {"column": "y", "type": "regression"}},
    )
    idx = [3, 0, 17, 29, 8]
    _assert_same(corpus.get_batch(idx), default_collate([corpus[i] for i in idx]))
    _assert_same(corpus[np.array(idx)], corpus.get_batch(idx))


def test_explicit_zero_votes_are_not_observed():
    S = sp.csr_matrix((np.array([1, 0, -1], dtype=np.int8), ([0, 0, 1], [0, 1, 2])), shape=(2, 3))
    assert S.nnz == 3  # the explicit 0 is stored
    corpus = Corpus(pd.DataFrame({"u": [0, 1]}), modalities={"v": {"views": {"rc": {"type": "vote", "matrix": S}}}})
    item = corpus[0]["modalities"]["v"]["rc"]
    np.testing.assert_array_equal(item["matrix"].numpy(), [1.0, 2.0, 2.0])
    np.testing.assert_array_equal(item["mask"].numpy(), [False, True, True])
    _assert_same(corpus.get_batch([0, 1]), default_collate([corpus[0], corpus[1]]))


def test_available_must_be_boolean():
    df = pd.DataFrame({"x": [0.0, 1.0], "flag": [1, 0]})
    view = {"type": "embedding", "matrix": np.zeros((2, 2))}
    with pytest.raises(TypeError, match="boolean"):
        Corpus(df, modalities={"e": {"views": {"w": {**view, "available": "flag"}}}})
    with pytest.raises(TypeError, match="length 2"):
        Corpus(df, modalities={"e": {"views": {"w": {**view, "available": np.ones(3, bool)}}}})


# ----------------------------------------------------------------------------- guard


@pytest.mark.parametrize(
    "kw", [{"fusion": "poe"}, {"fusion": "moe_average"}, {"vi_type": "iaf"}]
)
def test_availability_requires_corrected_poe(kw):
    corpus, _ = _two_modality_corpus()
    with pytest.raises(ValueError, match="corrected_poe"):
        _model(corpus, **kw)


# ----------------------------------------------------------------------------- fusion


@pytest.mark.parametrize("vi_type", ["mean_field", "full_rank", "mixture_of_gaussians"])
def test_fused_posterior_is_product_of_the_experts_a_unit_has(vi_type):
    corpus, pattern = _two_modality_corpus()
    dim = 1 if vi_type == "mean_field" else 2
    model = _model(corpus, vi_type=vi_type, dim=dim)
    model.encoder.eval()
    batch, inputs, mask, rows = _batch(model, corpus)
    prev = batch["M_prevalence_covariates"]

    def moments(info):
        if vi_type == "mean_field":
            return info[0], torch.diag_embed(info[1].exp())
        if vi_type == "full_rank":
            return info[0], info[1] @ info[1].transpose(-1, -2)
        _, means, trils, pi = info
        return (pi.unsqueeze(-1) * means).sum(1), _mixture_covariance(means, trils, pi)

    with torch.no_grad():
        fused = moments(model.encoder(inputs, prevalence_covariates=prev, unit_mask=mask, unit_rows=rows)[2][0])
        both = moments(model.encoder(inputs, prevalence_covariates=prev)[2][0])
        single = {
            k: moments(model.encoder(inputs, single_modality=k, prevalence_covariates=prev)[2][0])
            for k in KEYS
        }
        mu0, Sigma0 = model.prior.get_prior_params(prev, return_full_cov=True)
    if vi_type == "mean_field":  # mean-field corrected PoE uses the prior's diagonal
        Sigma0 = torch.diag_embed(torch.diagonal(Sigma0))
    prior = (mu0, Sigma0.expand(len(mu0), -1, -1))
    for p, ref in ((0, both), (1, single["a_w"]), (2, single["b_w"]), (3, prior)):
        m = torch.from_numpy(pattern == p)
        for got, want in zip(fused, ref):
            torch.testing.assert_close(got[m], want[m], rtol=1e-4, atol=1e-5)


def test_all_available_matches_undeclared_fusion():
    corpus, _ = _two_modality_corpus()
    model = _model(corpus)
    model.encoder.eval()
    batch, inputs, _, _ = _batch(model, corpus)
    prev = batch["M_prevalence_covariates"]
    ones = torch.ones(len(corpus), 2, dtype=torch.bool)
    with torch.no_grad():
        a = model.encoder(inputs, prevalence_covariates=prev, unit_mask=ones)[2][0]
        b = model.encoder(inputs, prevalence_covariates=prev)[2][0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


# ----------------------------------------------------------------------------- training


def test_absent_rows_are_never_read():
    """Placeholders of absent modalities (zeros vs NaN) change nothing, to the bit."""
    fits = [_model(_two_modality_corpus(absent_fill=f)[0], steps=25) for f in (0.0, np.nan)]
    assert np.isfinite(fits[1].train_losses).all()
    assert fits[0].train_losses == fits[1].train_losses
    for p, q in zip(fits[0].encoder.parameters(), fits[1].encoder.parameters()):
        assert torch.equal(p, q)
    for p, q in zip(fits[0].decoders.parameters(), fits[1].decoders.parameters()):
        assert torch.equal(p, q)


def test_dropout_keeps_one_of_each_units_own_modalities():
    corpus, pattern = _two_modality_corpus()
    model = _model(corpus)
    batch = corpus.get_batch(np.arange(len(corpus)))
    have = torch.from_numpy(np.stack([np.isin(pattern, (0, 1)), np.isin(pattern, (0, 2))], 1))
    for active in ([], ["a_w"], ["b_w"], list(KEYS)):
        torch.manual_seed(0)
        mask, rows = model._units_in_step(batch, active)
        assert not (mask & ~have).any()  # only modalities a unit has
        assert (mask.any(1) == have.any(1)).all()  # at least one, if it has any
        for m, k in enumerate(KEYS):
            used = torch.zeros(len(pattern), dtype=torch.bool)
            used[rows[k] if rows[k] is not None else slice(None)] = True
            assert torch.equal(used, mask[:, m])
        if active == list(KEYS):
            assert torch.equal(mask, have)


def test_training_loss_matches_per_unit_loglik():
    """The batch reconstruction loss (step_batch) is the per-unit log-likelihood over the
    modalities each unit has (_recon_loglik), averaged over the whole batch."""
    corpus, _ = _two_modality_corpus()
    model = _model(corpus, steps=10)
    batch = corpus.get_batch(np.arange(len(corpus)))
    torch.manual_seed(0)
    model.step_batch(corpus.get_batch(np.arange(len(corpus))), corpus, validation=True)
    recon = float(model.reconstruction_loss)
    inputs = model._encoder_inputs(batch, corpus)
    mask, rows = model._units_in_step(batch)
    torch.manual_seed(0)
    with torch.no_grad():
        _, z, _ = model.encoder(
            inputs, prevalence_covariates=batch["M_prevalence_covariates"], unit_mask=mask, unit_rows=rows
        )
        ll = model._recon_loglik(z, batch, corpus, None, rows)
    assert ll[3::4].abs().max() == 0  # units with no modality
    np.testing.assert_allclose(recon, float(-ll.mean()), rtol=1e-5)


def test_readouts_nan_where_absent():
    corpus, pattern = _two_modality_corpus()
    model = _model(corpus, steps=10)
    has = np.stack([np.isin(pattern, (0, 1)), np.isin(pattern, (0, 2))], 1)
    theta = model.get_ideal_points(corpus, num_samples=3)
    assert np.isfinite(theta).all()
    for m, k in enumerate(KEYS):
        single, sd = model.get_ideal_points(corpus, single_modality=k, num_samples=3, return_std=True)
        assert np.isnan(single[~has[:, m]]).all() and np.isfinite(single[has[:, m]]).all()
        assert np.isnan(sd[~has[:, m]]).all()
    w = model.get_modality_weights(corpus)
    assert (np.isnan(w) == ~has).all()
    np.testing.assert_allclose(np.nansum(w[has.any(1)], 1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(w[pattern == 1, 0], 1.0)
    mi = model.get_mutual_information(corpus)
    assert (np.isnan(mi) == ~has).all()
    iwae, elbo = model.estimate_marginal_log_likelihood(corpus, n_samples=4, reduce="none")
    assert np.isfinite(iwae).all()
    np.testing.assert_allclose(iwae[pattern == 3], 0.0, atol=1e-4)  # posterior = prior, no data


def test_trained_posterior_matches_exact_linear_gaussian():
    """Linear-Gaussian model, a quarter of the units with each pattern (both, only a, only b,
    none): after training, each unit's fused posterior matches the exact posterior given the
    modalities it has, under the fitted decoder (a missing or double-counted prior would
    move the variance by about 1/9)."""
    rng = np.random.default_rng(0)
    n = 1500
    theta = rng.normal(size=n)
    pattern = np.arange(n) % 4
    has = {"a": np.isin(pattern, (0, 1)), "b": np.isin(pattern, (0, 2))}
    mats = {
        "a": np.outer(theta, rng.normal(0, 1.0, 5)) + rng.normal(0, 0.8, (n, 5)),
        "b": np.outer(theta, rng.normal(0, 0.6, 4)) + rng.normal(0, 0.8, (n, 4)),
    }
    for k in "ab":
        mats[k][~has[k]] = np.nan
        mats[k] = mats[k].astype(np.float32)
    df = pd.DataFrame({"p": np.zeros(n), "has_a": has["a"], "has_b": has["b"]})
    mods = {
        k: {"column": "p", "views": {"w": {"type": "embedding", "matrix": mats[k], "available": f"has_{k}"}}}
        for k in "ab"
    }
    corpus = Corpus(df, modalities=mods)
    model = IdealPointNN(
        train_data=corpus, n_ideal_points=1, ae_type="vae", vi_type="mean_field",
        fusion="corrected_poe", update_prior=True, learn_prior_cov=False,
        encoder_args={k: {"hidden_dims": []} for k in KEYS},
        decoder_args={k: {"hidden_dims": [], "bias": True} for k in KEYS},
        batch_size=128, num_steps=1500, num_workers=0, print_every_n_steps=10**9,
        return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=0, device="cpu",
        optim_args={"main": {"lr": 1e-2}, "prior": {"lr": 1e-2, "weight_decay": 0.0}},
    )
    for lr in (1e-3, 1e-4):  # decay the step size so SGD noise does not mask a bias
        for g in model.optimizer.param_groups:
            g["lr"] = lr
        model.num_steps += 1500
        model.train(corpus)

    batch, inputs, mask, rows = _batch(model, corpus)
    with torch.no_grad():
        info = model.encoder(inputs, unit_mask=mask, unit_rows=rows)[2][0]
        mu_q, var_q = info[0][:, 0].numpy(), info[1][:, 0].exp().numpy()
        mu0, Sigma0 = model.prior.get_prior_params(None, return_full_cov=True)
        lam = np.full(n, 1.0 / float(Sigma0[0, 0]))
        eta = lam * float(mu0.reshape(-1)[0])
        for k in "ab":
            dec = model.decoders[f"{k}_w"]
            W = dec.decoder["dec_0"].weight[:, 0].numpy()
            b = dec.decoder["dec_0"].bias.numpy()
            s2 = dec.noise_log_var.exp().numpy()
            lam[has[k]] += (W**2 / s2).sum()
            eta[has[k]] += ((mats[k][has[k]] - b) * W / s2).sum(1)
    assert np.abs(mu_q - eta / lam).max() < 0.05
    ratio = var_q * lam
    assert ratio.min() > 0.92 and ratio.max() < 1.08
    np.testing.assert_allclose(ratio[pattern == 3], 1.0, rtol=1e-5)


# ----------------------------------------------------------------------------- votes


def test_votes_one_modality_per_congress():
    """Two congresses, each its own vote view (sparse, abstentions inside): a unit's
    likelihood is its own congress's observed votes only."""
    rng = np.random.default_rng(1)
    n, J = 60, (6, 5)
    cong = np.arange(n) % 2
    theta = rng.normal(size=n)
    mats = []
    for t, j in enumerate(J):
        p = 1 / (1 + np.exp(-np.outer(theta, rng.normal(0, 1.5, j))))
        V = np.where(rng.uniform(size=(n, j)) < p, 1, -1)
        V[rng.uniform(size=(n, j)) < 0.2] = 0  # abstentions
        V[cong != t] = 0  # roll calls of another congress
        mats.append(sp.csr_matrix(V.astype(np.int8)))
    df = pd.DataFrame({"x": rng.normal(size=n), "in0": cong == 0, "in1": cong == 1})
    views = {f"c{t}": {"type": "vote", "matrix": mats[t], "available": f"in{t}"} for t in range(2)}
    corpus = Corpus(df, modalities={"vote": {"views": views}}, prevalence="~ x")
    keys = ("vote_c0", "vote_c1")
    model = _model(corpus, keys=keys, steps=20)
    assert np.isfinite(model.train_losses).all()
    batch, inputs, mask, rows = _batch(model, corpus)
    z = torch.randn(n, 1)
    with torch.no_grad():
        ll = model._recon_loglik(z, batch, corpus, None, rows).numpy()
        manual = np.zeros(n)
        for t, k in enumerate(keys):
            logits = model.decoders[k](z).numpy()
            V = mats[t].toarray()
            obs = (V != 0) & (cong == t)[:, None]
            y = (V > 0).astype(float)
            bern = y * -np.logaddexp(0, -logits) + (1 - y) * -np.logaddexp(0, logits)
            manual += (bern * obs).sum(1)
    np.testing.assert_allclose(ll, manual, rtol=1e-5, atol=1e-5)

    # the training loss is the same per-unit log-likelihood, averaged over the batch
    torch.manual_seed(0)
    model.step_batch(corpus.get_batch(np.arange(n)), corpus, validation=True)
    torch.manual_seed(0)
    with torch.no_grad():
        _, z, _ = model.encoder(
            inputs, prevalence_covariates=batch["M_prevalence_covariates"], unit_mask=mask, unit_rows=rows
        )
        ll = model._recon_loglik(z, batch, corpus, None, rows)
    np.testing.assert_allclose(float(model.reconstruction_loss), float(-ll.mean()), rtol=1e-5)


# ----------------------------------------------------------------------------- review fixes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_training_and_readouts_on_gpu():
    corpus, pattern = _two_modality_corpus()
    model = _model(corpus, steps=5, device="cuda")
    assert np.isfinite(model.train_losses).all()
    iwae, _ = model.estimate_marginal_log_likelihood(corpus, n_samples=2, reduce="none")
    assert np.isfinite(iwae).all()
    assert np.isfinite(model.get_ideal_points(corpus)).all()
    has = np.stack([np.isin(pattern, (0, 1)), np.isin(pattern, (0, 2))], 1)
    assert (np.isnan(model.get_modality_weights(corpus)) == ~has).all()


@pytest.mark.parametrize("vi_type", ["mean_field", "full_rank", "mixture_of_gaussians"])
def test_readout_batch_without_a_modality(vi_type):
    """Units sorted so each readout batch lacks one modality, no prevalence covariates
    (the prior mean is a single row that must still match an empty batch)."""
    pattern = np.repeat([1, 2], 32)  # first batch only a, second only b
    corpus, _ = _two_modality_corpus(pattern=pattern, prevalence=False)
    model = _model(corpus, vi_type=vi_type, dim=1 if vi_type == "mean_field" else 2, steps=3)
    has = np.stack([pattern == 1, pattern == 2], 1)
    w = model.get_modality_weights(corpus)
    assert (np.isnan(w) == ~has).all()
    for m, k in enumerate(KEYS):
        single = model.get_ideal_points(corpus, single_modality=k)
        assert (np.isnan(single).any(1) == ~has[:, m]).all()
    if vi_type != "mixture_of_gaussians":
        assert (np.isnan(model.get_mutual_information(corpus)) == ~has).all()
    iwae, _ = model.estimate_marginal_log_likelihood(corpus, n_samples=2, reduce="none")
    assert np.isfinite(iwae).all()


def test_discrete_choice_placeholders_keep_rows_aligned():
    df = pd.DataFrame(
        {
            "q": [0, np.nan, 2, 1, np.nan, 2, 1, 0],
            "x": np.linspace(-1, 1, 8),
            "has": [True, False, True, True, False, True, True, True],
        }
    )
    view = {"type": "discrete_choice", "available": "has"}
    corpus = Corpus(df, modalities={"s": {"column": "q", "views": {"q": view}}})
    info = corpus.processed_modalities["s"]["q"]["q"]
    assert info["matrix"].shape[0] == len(df)
    for i in np.flatnonzero(df["has"]):
        assert info["columns"][info["matrix"][i].argmax()] == f"C(q)[{df['q'][i]}]"
    _assert_same(corpus.get_batch(range(8)), default_collate([corpus[i] for i in range(8)]))
    # a missing answer of a unit that has the modality is an error, not a dropped row
    with pytest.raises(Exception, match="missing"):
        Corpus(df.assign(has=True), modalities={"s": {"column": "q", "views": {"q": view}}})


def test_discrete_choice_training_loss_matches_per_unit_loglik():
    rng = np.random.default_rng(2)
    n = 48
    theta = rng.normal(size=n)
    has = np.arange(n) % 3 != 0
    q = np.where(theta > 0, rng.choice([1, 2], n), rng.choice([0, 1], n)).astype(float)
    q[~has] = np.nan
    emb = (np.outer(theta, rng.normal(size=3)) + rng.normal(0, 0.7, (n, 3))).astype(np.float32)
    df = pd.DataFrame({"q": q, "x": rng.normal(size=n), "has": has})
    corpus = Corpus(
        df,
        modalities={
            "s": {"column": "q", "views": {"q": {"type": "discrete_choice", "available": "has"}}},
            "e": {"column": "x", "views": {"w": {"type": "embedding", "matrix": emb}}},
        },
        prevalence="~ x",
    )
    model = _model(corpus, keys=("s_q", "e_w"), steps=5)
    batch = corpus.get_batch(np.arange(n))
    torch.manual_seed(0)
    model.step_batch(corpus.get_batch(np.arange(n)), corpus, validation=True)
    inputs = model._encoder_inputs(batch, corpus)
    mask, rows = model._units_in_step(batch)
    torch.manual_seed(0)
    with torch.no_grad():
        _, z, _ = model.encoder(
            inputs, prevalence_covariates=batch["M_prevalence_covariates"], unit_mask=mask, unit_rows=rows
        )
        ll = model._recon_loglik(z, batch, corpus, None, rows)
    np.testing.assert_allclose(float(model.reconstruction_loss), float(-ll.mean()), rtol=1e-5)


def test_lazy_image_placeholders_are_not_loaded():
    loaded = []

    def transform(paths):
        loaded.extend(paths)
        return [torch.full((1, 4, 4), float(paths[0][3:]))]

    df = pd.DataFrame({"img": [f"img{i}" for i in range(6)], "has": [True, False, True, False, True, True]})
    view = {"type": "image", "transform_fn": transform, "available": "has"}
    corpus = Corpus(df, modalities={"pic": {"column": "img", "views": {"v": view}}})
    batch = corpus.get_batch(range(6))
    _assert_same(batch, default_collate([corpus[i] for i in range(6)]))
    images = batch["modalities"]["pic"]["v"]
    assert images.shape == (6, 1, 4, 4)
    assert (images[1] == 0).all() and (images[2] == 2).all()
    assert not {"img1", "img3"} & set(loaded)


def test_unknown_available_key_warns():
    corpus, _ = _two_modality_corpus()
    corpus.availability["c_w"] = corpus.availability["a_w"]
    with pytest.warns(RuntimeWarning, match="no encoder or decoder"):
        _model(corpus)


def test_modality_weights_refuse_availability_without_corrected_poe():
    model = _model(_two_modality_corpus(declare=False)[0], fusion="poe")
    with pytest.raises(ValueError, match="corrected_poe"):
        model.get_modality_weights(_two_modality_corpus()[0])
