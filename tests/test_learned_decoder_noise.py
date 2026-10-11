"""Learned per-feature Gaussian noise for embedding modalities.

The embedding likelihood is N(w_j | lambda_j' z + b_j, sigma^2_j) with sigma^2_j
learned (default) instead of pinned at 0.5 by a squared-error loss. A modality's
information about z is sum_j lambda_j^2 / sigma^2_j, so a pinned sigma^2 misstates
its contribution to posterior precision whenever the true noise is not 0.5.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from torch.distributions import Normal

from deeplatent import Corpus, IdealPointNN
from deeplatent.models import _gaussian_feature_nll

KEYS = ("a_w", "b_w")


def _data(n, sig2, j=10, seed=0):
    """Two embedding modalities from a 1D latent; modality m has noise sig2[m]
    (a scalar or one variance per feature)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    mats = {}
    for k, s2 in zip(KEYS, sig2):
        noise = rng.normal(size=(n, j)) * np.sqrt(np.broadcast_to(s2, (j,)))
        mats[k] = (theta[:, None] * rng.normal(0.0, 0.5, j) + noise).astype(np.float32)
    mods = {
        k.split("_")[0]: {
            "column": "p",
            "views": {"w": {"type": "embedding", "matrix": mats[k]}},
        }
        for k in KEYS
    }
    return Corpus(pd.DataFrame({"p": np.zeros(n)}), modalities=mods)


def _model(corpus, noise=None, num_steps=0, seed=0):
    dec = {"hidden_dims": [], "bias": True}
    if noise is not None:
        dec["noise"] = noise
    return IdealPointNN(
        train_data=corpus,
        n_ideal_points=1,
        ae_type="vae",
        vi_type="mean_field",
        update_prior=False,
        fusion="corrected_poe",
        w_prior=1.0,
        encoder_args={k: {"hidden_dims": [32]} for k in KEYS},
        decoder_args={k: dict(dec) for k in KEYS},
        num_steps=num_steps,
        batch_size=256,
        num_workers=0,
        print_every_n_steps=10**9,
        return_best_model=False,
        ckpt_folder=tempfile.mkdtemp(),
        seed=seed,
        device="cpu",
    )


def test_default_embedding_decoder_learns_per_feature_noise():
    corpus = _data(32, (0.5, 0.5))
    learned = _model(corpus)
    for k in KEYS:
        torch.testing.assert_close(
            learned.decoders[k].noise_log_var.detach(),
            torch.full((10,), float(np.log(0.5))),
        )
    fixed = _model(corpus, noise="fixed")
    assert all(not hasattr(fixed.decoders[k], "noise_log_var") for k in KEYS)
    with pytest.raises(ValueError, match="noise"):
        _model(corpus, noise="per_modality")


def test_nll_is_exact_gaussian_and_fixed_is_legacy_squared_error():
    torch.manual_seed(0)
    recon, target = torch.randn(7, 10), torch.randn(7, 10)
    dec = torch.nn.Module()
    dec.noise_log_var = torch.nn.Parameter(torch.randn(10))
    exact = -Normal(recon, torch.exp(0.5 * dec.noise_log_var)).log_prob(target).sum(1)
    torch.testing.assert_close(_gaussian_feature_nll(recon, target, dec), exact)
    legacy = ((recon - target) ** 2).sum(1)
    torch.testing.assert_close(
        _gaussian_feature_nll(recon, target, torch.nn.Module()), legacy
    )


def test_learned_noise_recovers_per_feature_variances():
    # Modality a: noise 0.2 on half its features, 2.0 on the other half;
    # modality b: noise 1.0 everywhere. Nothing is at the legacy 0.5.
    sig2 = (np.repeat([0.2, 2.0], 5), 1.0)
    # sigma^2 converges within ~6k steps at the default lr (12k: 0.196/1.99/0.98).
    model = _model(_data(2000, sig2, seed=1), num_steps=6000, seed=1)
    for k, s2 in zip(KEYS, sig2):
        est = model.decoders[k].noise_log_var.detach().exp().numpy()
        np.testing.assert_allclose(est, np.broadcast_to(s2, (10,)), rtol=0.15)


def test_checkpoint_without_noise_loads_as_legacy_sigma2(tmp_path):
    corpus = _data(32, (0.5, 0.5))
    old = _model(corpus, noise="fixed")
    path = str(tmp_path / "legacy.ckpt")
    old.save_model(path)
    new = _model(corpus)
    with torch.no_grad():
        for k in KEYS:
            new.decoders[k].noise_log_var.fill_(3.0)
    with pytest.warns(RuntimeWarning, match="predates learned Gaussian"):
        new.load_model(path)
    for k in KEYS:
        torch.testing.assert_close(
            new.decoders[k].noise_log_var.detach(),
            torch.full((10,), float(np.log(0.5))),
        )
