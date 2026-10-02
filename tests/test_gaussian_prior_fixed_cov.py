"""GaussianPrior(learn_cov=False): learn the covariate-dependent mean, pin Sigma = I.

With a learned decoder, (z, mean, Sigma) -> (a z, a mean, a^2 Sigma) leaves the
likelihood unchanged, so the prior's covariate effects are identified only once the
residual variance is fixed. IdealPointNN exposes this as learn_prior_cov=False.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from deeplatent import Corpus, IdealPointNN


def _model(learn_prior_cov, num_steps, n=256, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x1": rng.binomial(1, 0.5, n), "x2": rng.normal(size=n)})
    z = 0.5 - 1.0 * df.x1 + 0.75 * df.x2 + rng.normal(size=n)
    w = np.outer(z, rng.normal(0, 0.5, 6)) + rng.normal(0, 0.7, (n, 6))
    corpus = Corpus(
        df.assign(p=0.0),
        prevalence="~ x1 + x2",
        modalities={
            "a": {
                "column": "p",
                "views": {"w": {"type": "embedding", "matrix": w.astype(np.float32)}},
            }
        },
    )
    return IdealPointNN(
        train_data=corpus,
        n_ideal_points=1,
        ae_type="vae",
        vi_type="mean_field",
        update_prior=True,
        learn_prior_cov=learn_prior_cov,
        w_prior=1.0,
        encoder_args={"a_w": {"hidden_dims": []}},
        decoder_args={"a_w": {"hidden_dims": [], "bias": True}},
        num_steps=num_steps,
        batch_size=64,
        num_workers=0,
        print_every_n_steps=10**9,
        return_best_model=False,
        ckpt_folder=tempfile.mkdtemp(),
        seed=seed,
        device="cpu",
    )


def test_fixed_cov_learns_mean_only():
    untrained = _model(learn_prior_cov=False, num_steps=0)
    trained = _model(learn_prior_cov=False, num_steps=200)
    for m in (untrained, trained):
        assert "L_flat" not in dict(m.prior.named_parameters())
        assert "L_flat" in dict(m.prior.named_buffers())
        torch.testing.assert_close(
            m.prior.sigma.detach(), torch.full((1, 1), (1 + 1e-4) ** 2)
        )
    assert not torch.equal(
        untrained.prior.mean_net.weight, trained.prior.mean_net.weight
    )


@pytest.mark.parametrize("learn_prior_cov", [True, False])
def test_checkpoint_roundtrip_keeps_prior(tmp_path, learn_prior_cov):
    model = _model(learn_prior_cov=learn_prior_cov, num_steps=50)
    path = str(tmp_path / "m.ckpt")
    model.save_model(path)
    fresh = _model(learn_prior_cov=learn_prior_cov, num_steps=0, seed=1)
    fresh.load_model(path)
    for k, v in model.prior.state_dict().items():
        torch.testing.assert_close(fresh.prior.state_dict()[k], v)


def test_default_still_learns_cov():
    model = _model(learn_prior_cov=True, num_steps=200)
    assert "L_flat" in dict(model.prior.named_parameters())
    assert model.prior.L_flat.detach().abs().sum() > 0
