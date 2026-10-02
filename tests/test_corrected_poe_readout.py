"""Corrected-PoE (mean-field) readouts must match the experts used in training.

The raw encoder logvar of a corrected-PoE mean-field expert encodes -log(Δ_m), the
precision increment of modality m over the prior; training uses the unimodal
posterior q_m(z|x_m) = N(μ_m, 1/(Λ_0 + Δ_m)). These tests pin the two public
readouts to that definition: get_modality_weights (shares of Δ_m) and the
single-modality posterior (get_latent_factors / get_ideal_points).

Full rank: the expert is the unimodal posterior itself and Δ_m = Λ_m - Λ_0.
Mixtures: Δ_m = Cov(q_m)^(-1) - Λ_0 (moment-matched). In both, Λ_0 is recovered in
the tests from the fused posterior (Λ_fused = Λ_a + Λ_b - Λ_0), not from the prior,
so the readout is checked against what training actually adds.
"""

import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from deeplatent import Corpus, IdealPointNN

KEYS = ("a_w", "b_w")


def _model(
    update_prior,
    fusion="corrected_poe",
    n=64,
    j=5,
    seed=0,
    vi_type="mean_field",
    dim=1,
    **extra,
):
    """Untrained two-modality IdealPointNN (readouts are checked against the
    encoder itself, so training is not needed)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n, dim))
    mats = {
        k: (
            theta @ rng.normal(0.0, s, (dim, j)) + rng.normal(0.0, np.sqrt(0.5), (n, j))
        ).astype(np.float32)
        for k, s in zip(KEYS, (1.0, 0.1))
    }
    mods = {
        k.split("_")[0]: {
            "column": "p",
            "views": {"w": {"type": "embedding", "matrix": mats[k]}},
        }
        for k in KEYS
    }
    corpus = Corpus(pd.DataFrame({"p": np.zeros(n)}), modalities=mods)
    model = IdealPointNN(
        train_data=corpus,
        n_ideal_points=dim,
        ae_type="vae",
        vi_type=vi_type,
        update_prior=update_prior,
        fusion=fusion,
        w_prior=1.0,
        encoder_args={k: {"hidden_dims": [8]} for k in KEYS},
        decoder_args={k: {"hidden_dims": [], "bias": True} for k in KEYS},
        num_steps=0,
        batch_size=n,
        num_workers=0,
        print_every_n_steps=10**9,
        return_best_model=False,
        ckpt_folder=tempfile.mkdtemp(),
        seed=seed,
        device="cpu",
        **extra,
    )
    model.encoder.eval()
    return model, corpus, {k: torch.from_numpy(v) for k, v in mats.items()}


def _shift_raw_logvar(model, key, shift):
    """Move expert `key`'s raw logvar by `shift` (Δ_m scales by exp(-shift))."""
    last = list(model.encoder.encoders[key].encoder.values())[-1]
    with torch.no_grad():
        last.bias[model.n_latent :] += shift


def _raw_logvar(model, inputs, key):
    with torch.no_grad():
        return model.encoder.encoders[key](inputs[key]).chunk(2, dim=1)[1]


@pytest.mark.parametrize("update_prior", [False, True])
def test_modality_weights_are_precision_increment_shares(update_prior):
    model, corpus, inputs = _model(update_prior)
    # Strong expert a, weak expert b whose increment is BELOW the prior precision
    # (the case the double prior subtraction reported as exactly zero).
    _shift_raw_logvar(model, "a_w", -3.0)
    _shift_raw_logvar(model, "b_w", 3.0)
    delta = {k: torch.exp(-_raw_logvar(model, inputs, k)).squeeze(1) for k in KEYS}
    assert (delta["b_w"] < 1.0).all()

    w = model.get_modality_weights(corpus)
    expected_a = (delta["a_w"] / (delta["a_w"] + delta["b_w"])).numpy()
    np.testing.assert_allclose(w[:, 0], expected_a, rtol=1e-5)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-5)
    assert (w[:, 1] > 0).all()


@pytest.mark.parametrize("update_prior", [False, True])
def test_single_modality_posterior_matches_training_expert(update_prior):
    model, _, inputs = _model(update_prior)
    for key in KEYS:
        with torch.no_grad():
            _, _, single = model.encoder(inputs, single_modality=key)
            # Training path with only this modality active (as in step_batch's
            # random modality subsets): the fused posterior is the unimodal expert.
            _, _, trained = model.encoder(inputs, active_modalities=[key])
        (mu_s, lv_s), (mu_t, lv_t) = single[0], trained[0]
        torch.testing.assert_close(mu_s, mu_t)
        torch.testing.assert_close(lv_s, lv_t)
        if not update_prior:  # fixed N(0, 1) prior: Λ_0 = 1
            raw = _raw_logvar(model, inputs, key)
            torch.testing.assert_close(lv_s, -torch.log(1.0 + torch.exp(-raw)))


def test_single_modality_readout_std_is_unimodal_posterior_std():
    model, corpus, inputs = _model(update_prior=False)
    torch.manual_seed(0)
    for key in KEYS:
        _, sd = model.get_ideal_points(
            corpus, single_modality=key, num_samples=2000, return_std=True
        )
        raw = _raw_logvar(model, inputs, key)
        expected = (1.0 / torch.sqrt(1.0 + torch.exp(-raw))).numpy()
        np.testing.assert_allclose(sd.mean(), expected.mean(), rtol=0.03)


@pytest.mark.parametrize("update_prior", [False, True])
def test_mutual_information_uses_unimodal_posterior(update_prior):
    """MI_m = KL(q_m || prior) with q_m the unimodal posterior N(μ_m, 1/(Λ_0 + Δ_m)),
    not the raw logvar read as a variance (1/Δ_m), which makes a weak expert (Δ_m
    below the prior precision) look wider than the prior and hence informative."""
    model, corpus, inputs = _model(update_prior)
    _perturb_prior(model, update_prior)
    _shift_raw_logvar(model, "a_w", -3.0)
    _shift_raw_logvar(model, "b_w", 3.0)
    mi = model.get_mutual_information(corpus)
    with torch.no_grad():
        if update_prior:
            mu0, Sigma0 = model.prior.get_prior_params(None, return_full_cov=True)
            var0 = Sigma0[0, 0]
        else:
            mu0, logvar0 = model.prior.get_prior_params(None)
            var0 = logvar0.exp()

        def kl(mu, var):
            return 0.5 * (var / var0 + (mu - mu0) ** 2 / var0 - 1 - torch.log(var / var0))

        for m, key in enumerate(KEYS):
            mu, logvar = model.encoder(inputs, single_modality=key)[2][0]
            raw = _raw_logvar(model, inputs, key)
            np.testing.assert_allclose(mi[:, m], kl(mu, logvar.exp())[:, 0], rtol=1e-4)
            torch.testing.assert_close(logvar, model.encoder._corrected_unimodal_logvar(raw))
        # the raw reading overstates the weak expert b
        mu_b = model.encoder(inputs, single_modality="b_w")[2][0][0]
        assert (kl(mu_b, _raw_logvar(model, inputs, "b_w").exp())[:, 0].numpy() > mi[:, 1] + 0.5).all()


def test_uncorrected_poe_single_modality_uses_raw_precision():
    model, _, inputs = _model(update_prior=False, fusion="poe")
    for key in KEYS:
        with torch.no_grad():
            _, _, single = model.encoder(inputs, single_modality=key)
        torch.testing.assert_close(single[0][1], _raw_logvar(model, inputs, key))


def _last_layer(model, key):
    return list(model.encoder.encoders[key].encoder.values())[-1]


def _perturb_prior(model, update_prior):
    """Move a learned prior off N(0, I), so a wrong Λ_0 would show."""
    if update_prior:
        with torch.no_grad():
            n = model.prior.L_flat.numel()
            model.prior.L_flat.copy_(torch.tensor([0.4, 0.5, -0.3])[:n])


def _trace_shares(increments):
    tr = torch.stack([torch.diagonal(d, dim1=1, dim2=2).sum(1) for d in increments], 1)
    return (tr / tr.sum(1, keepdim=True)).numpy()


def _precision(scale_tril):
    return torch.linalg.inv(scale_tril @ scale_tril.transpose(-1, -2))


@pytest.mark.parametrize("update_prior", [False, True])
def test_full_rank_weights_are_precision_increment_shares(update_prior):
    dim = 2
    model, corpus, inputs = _model(update_prior, vi_type="full_rank", dim=dim)
    _perturb_prior(model, update_prior)
    diag = dim + torch.tensor([0, 2])  # L_00, L_11 after the means (tril order)
    ridge = 1e-6 * torch.eye(dim)  # product_of_experts_full_rank's regularization
    with torch.no_grad():
        _last_layer(model, "a_w").bias[diag] -= 0.5  # strong expert
        _last_layer(model, "b_w").bias[diag] += 0.3  # weak expert
        prec = {}
        for k in KEYS:
            L = model.encoder(inputs, single_modality=k)[2][0][1]
            prec[k] = torch.linalg.inv(L @ L.transpose(1, 2) + ridge)
        prec_fused = _precision(model.encoder(inputs)[2][0][1]) - ridge
    Lambda_0 = prec["a_w"] + prec["b_w"] - prec_fused
    w = model.get_modality_weights(corpus)
    expected = _trace_shares([prec[k] - Lambda_0 for k in KEYS])
    np.testing.assert_allclose(w, expected, rtol=1e-3, atol=1e-4)
    # The old readout, shares of total precision tr(Λ_m), differs here.
    assert not np.allclose(w, _trace_shares([prec[k] for k in KEYS]), atol=1e-2)


def _mog_model(update_prior, dim, C=3):
    """Strong mixture expert a, weak expert b, component means kept close."""
    model, corpus, inputs = _model(
        update_prior, vi_type="mixture_of_gaussians", dim=dim, mixture_components=C
    )
    _perturb_prior(model, update_prior)
    with torch.no_grad():
        for key, shift in (("a_w", -2.0), ("b_w", 1.0)):
            last = _last_layer(model, key)
            last.weight[: C * dim] *= 0.2
            last.bias[: C * dim] *= 0.2
            last.bias[C * dim : 2 * C * dim] += shift  # raw = -log Δ
    return model, corpus, inputs


@pytest.mark.parametrize("update_prior", [False, True])
def test_mog_weights_are_moment_matched_increment_shares(update_prior):
    model, corpus, inputs = _mog_model(update_prior, dim=2)
    with torch.no_grad():
        mix = {k: model.encoder(inputs, single_modality=k)[2][0][1:] for k in KEYS}
        trils_fused = model.encoder(inputs)[2][0][2]
    # Fused component (0, 0) has precision Λ_a0 + Λ_b0 - Λ_0.
    Lambda_0 = (
        _precision(mix["a_w"][1][:, 0])
        + _precision(mix["b_w"][1][:, 0])
        - _precision(trils_fused[:, 0])
    )

    def moment_precision(means, trils, pi):
        second = trils @ trils.transpose(-1, -2) + means.unsqueeze(
            -1
        ) @ means.unsqueeze(-2)
        mean = (pi.unsqueeze(-1) * means).sum(1)
        cov = (pi[..., None, None] * second).sum(1) - mean.unsqueeze(
            -1
        ) @ mean.unsqueeze(-2)
        return torch.linalg.inv(cov)

    w = model.get_modality_weights(corpus)
    expected = _trace_shares([moment_precision(*mix[k]) - Lambda_0 for k in KEYS])
    np.testing.assert_allclose(w, expected, rtol=1e-3, atol=1e-4)
    assert not np.allclose(w, 0.5, atol=1e-2)  # the old readout returned 1/M


def test_mog_weights_match_sampled_unimodal_variance():
    # 1-D, fixed N(0, 1) prior: Δ_m = 1 / Var(q_m) - 1, with Var(q_m) estimated
    # from the public single-modality sampler rather than the mixture formula.
    model, corpus, _ = _mog_model(update_prior=False, dim=1)
    w = model.get_modality_weights(corpus)
    torch.manual_seed(0)
    delta = np.stack(
        [
            1.0
            / model.get_ideal_points(
                corpus, single_modality=k, num_samples=4000, return_std=True
            )[1][:, 0]
            ** 2
            - 1.0
            for k in KEYS
        ],
        axis=1,
    )
    sampled = delta / delta.sum(1, keepdims=True)
    np.testing.assert_allclose(w.mean(0), sampled.mean(0), rtol=0.02)


def _wide_expert_b(model, vi_type):
    """Make expert b's unimodal posterior wider than the N(0, I) prior for every
    unit. Mixture: components sharper than the prior (Δ_c = e^2) but means at
    -2, 0, 2 with equal weights, so Var(q_b) = 1/(1 + e^2) + 8/3 > 1."""
    last = _last_layer(model, "b_w")
    with torch.no_grad():
        if vi_type == "full_rank":  # dim 2: L_00, L_11 follow the 2 means
            last.bias[2 + torch.tensor([0, 2])] += 3.0  # L_ii ~ 3, Λ_b ~ I/9
        else:  # dim 1, 3 components: [means | raw = -log Δ_c | pi logits]
            last.weight.zero_()
            last.bias.copy_(torch.tensor([-2.0, 0.0, 2.0, -2.0, -2.0, -2.0, 0, 0, 0]))


@pytest.mark.parametrize("vi_type", ["full_rank", "mixture_of_gaussians"])
def test_nonpositive_increment_warns_and_keeps_values(vi_type):
    extra = {"mixture_components": 3} if vi_type == "mixture_of_gaussians" else {}
    dim = 2 if vi_type == "full_rank" else 1
    model, corpus, _ = _model(False, vi_type=vi_type, dim=dim, **extra)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model.get_modality_weights(corpus)  # untouched experts: no warning
    _wide_expert_b(model, vi_type)
    with pytest.warns(RuntimeWarning, match="64 of 64 units have a non-positive"):
        w = model.get_modality_weights(corpus)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-4)
