"""Exact corrected Product-of-Experts tests for Gaussian-mixture posteriors."""

from collections import OrderedDict
import warnings

import pytest
import torch
from torch import nn
from torch.distributions import MultivariateNormal

from deeplatent.autoencoders import MultiModalEncoder
from deeplatent.priors import FixedGaussianPrior, GaussianPrior


class _FreeMixtureOutput(nn.Module):
    """Encoder stub whose raw MoG output is directly differentiable in tests."""

    def __init__(self, output):
        super().__init__()
        self.output = nn.Parameter(output.clone())

    def forward(self, x):
        return self.output.expand(x.size(0), -1)


def _encoder(prior, components=3, modalities=2, dim=2):
    encoders = OrderedDict((f"modality_{m}", nn.Identity()) for m in range(modalities))
    return MultiModalEncoder(
        encoders,
        topic_dim=dim,
        prior=prior,
        ae_type="vae",
        poe="corrected",
        vi_type="mixture_of_gaussians",
        mixture_components=components,
    )


def _direct_corrected_log_density(points, mixtures, prior):
    """Unnormalized log prod_m q_m(z) / p0(z)^(M-1), independently."""
    mu_0, Sigma_0 = prior.get_prior_params(None, return_full_cov=True)
    mu_0 = mu_0.to(points)
    Sigma_0 = Sigma_0.to(points)
    Lambda_0 = torch.linalg.inv(Sigma_0)

    log_density = torch.zeros(points.shape[:2], dtype=points.dtype)
    for means, raw_increments, pi in mixtures:
        precision = Lambda_0.unsqueeze(0).unsqueeze(0) + torch.diag_embed(
            torch.exp(-raw_increments)
        )
        covariance = torch.linalg.inv(precision)
        component_dist = MultivariateNormal(means[0], covariance_matrix=covariance[0])
        component_log_prob = component_dist.log_prob(points[0].unsqueeze(1))
        log_density = log_density + torch.logsumexp(
            torch.log(pi[0]).unsqueeze(0) + component_log_prob, dim=1
        ).unsqueeze(0)

    prior_dist = MultivariateNormal(mu_0[0], covariance_matrix=Sigma_0)
    log_prior = prior_dist.log_prob(points[0]).unsqueeze(0)
    return log_density - (len(mixtures) - 1) * log_prior


def test_corrected_mog_poe_warns_and_has_cartesian_component_count():
    prior = FixedGaussianPrior(0, 2)
    with pytest.warns(RuntimeWarning, match=r"9 components \(3\^2\)"):
        encoder = _encoder(prior, components=3, modalities=2)

    mixtures = []
    for _ in range(2):
        means = torch.randn(4, 3, 2)
        raw = torch.randn(4, 3, 2)
        pi = torch.softmax(torch.randn(4, 3), dim=1)
        mixtures.append((means, raw, pi))

    tag, means, scale_trils, weights = encoder.product_of_experts_mog(mixtures)
    assert tag == "mog_full"
    assert means.shape == (4, 9, 2)
    assert scale_trils.shape == (4, 9, 2, 2)
    assert weights.shape == (4, 9)
    assert torch.allclose(weights.sum(1), torch.ones(4), atol=1e-6)
    assert torch.isfinite(means).all()
    assert torch.isfinite(scale_trils).all()
    assert torch.isfinite(weights).all()


def test_corrected_mog_poe_matches_density_ratio_with_correlated_prior():
    torch.manual_seed(7)
    prior = GaussianPrior(prevalence_covariate_size=0, n_dims=2)
    with torch.no_grad():
        # Nonzero off-diagonal Cholesky entry exercises the full-covariance path.
        prior.L_flat.copy_(torch.tensor([0.15, 0.4, -0.2]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        encoder = _encoder(prior, components=3, modalities=2)

    leaves = []
    mixtures = []
    for _ in range(2):
        means = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
        raw = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
        logits = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
        mixtures.append((means, raw, torch.softmax(logits, dim=1)))
        leaves.extend((means, raw, logits))

    _, fused_means, fused_scales, fused_weights = encoder.product_of_experts_mog(
        mixtures
    )
    points = torch.randn(1, 31, 2, dtype=torch.float64)
    fused_component_log_prob = encoder._mog_component_log_prob(
        points, fused_means, fused_scales
    )
    fused_log_density = torch.logsumexp(
        torch.log(fused_weights).unsqueeze(1) + fused_component_log_prob, dim=2
    )
    direct_log_density = _direct_corrected_log_density(points, mixtures, prior)

    # The direct expression is unnormalized, so the two log densities may differ
    # by one global constant but must have exactly the same shape in z.
    difference = fused_log_density - direct_log_density
    assert torch.allclose(
        difference, difference[:, :1].expand_as(difference), atol=5e-8, rtol=5e-8
    )

    loss = fused_log_density.square().mean()
    loss.backward()
    for leaf in leaves:
        assert leaf.grad is not None
        assert torch.isfinite(leaf.grad).all()


def test_single_modality_corrected_expansion_is_the_unimodal_mixture():
    torch.manual_seed(11)
    prior = FixedGaussianPrior(0, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        encoder = _encoder(prior, components=3, modalities=1)

    means = torch.randn(2, 3, 2)
    raw = torch.randn(2, 3, 2)
    pi = torch.softmax(torch.randn(2, 3), dim=1)
    _, fused_means, fused_scales, fused_pi = encoder.product_of_experts_mog(
        [(means, raw, pi)]
    )

    expected_covariance = torch.diag_embed(1.0 / (1.0 + torch.exp(-raw)))
    actual_covariance = fused_scales @ fused_scales.transpose(-1, -2)
    assert torch.allclose(fused_means, means, atol=1e-6)
    assert torch.allclose(fused_pi, pi, atol=1e-6)
    assert torch.allclose(actual_covariance, expected_covariance, atol=1e-6)


def test_multimodal_forward_returns_the_exact_expanded_mixture():
    torch.manual_seed(13)
    batch_size, input_dim, latent_dim, components = 5, 4, 2, 3
    output_dim = components * (2 * latent_dim + 1)
    encoders = OrderedDict(
        (name, nn.Linear(input_dim, output_dim)) for name in ("left", "right")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        encoder = MultiModalEncoder(
            encoders,
            topic_dim=latent_dim,
            prior=FixedGaussianPrior(0, latent_dim),
            ae_type="vae",
            poe="corrected",
            vi_type="mixture_of_gaussians",
            mixture_components=components,
        )

    theta, sample, info = encoder(
        {
            "left": torch.randn(batch_size, input_dim),
            "right": torch.randn(batch_size, input_dim),
        }
    )
    tag, means, scale_trils, weights = info[-1]
    assert tag == "mog_full"
    assert means.shape == (batch_size, components**2, latent_dim)
    assert scale_trils.shape == (
        batch_size,
        components**2,
        latent_dim,
        latent_dim,
    )
    assert weights.shape == (batch_size, components**2)
    assert sample.shape == (batch_size, latent_dim)
    assert theta.shape == (batch_size, latent_dim)
    assert torch.allclose(theta.sum(1), torch.ones(batch_size), atol=1e-6)


def test_explicit_weighted_likelihood_reaches_all_mixture_parameters():
    """Regression guard for the Rao-Blackwellized mixture-ELBO gradient."""
    torch.manual_seed(23)
    batch_size, latent_dim, components = 3, 2, 2
    output_dim = components * (2 * latent_dim + 1)
    raw_outputs = [
        torch.tensor([[-0.7, 0.2, 0.8, -0.3, -0.4, 0.5, 0.9, -0.6, 0.3, -0.2]]),
        torch.tensor([[0.4, -0.9, -0.1, 0.7, 0.2, -0.8, -0.5, 0.6, -0.4, 0.7]]),
    ]
    modality_encoders = OrderedDict(
        (name, _FreeMixtureOutput(raw))
        for name, raw in zip(("left", "right"), raw_outputs)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        encoder = MultiModalEncoder(
            modality_encoders,
            topic_dim=latent_dim,
            prior=FixedGaussianPrior(0, latent_dim),
            ae_type="vae",
            poe="corrected",
            vi_type="mixture_of_gaussians",
            mixture_components=components,
        )

    _, _, info = encoder(
        {
            "left": torch.zeros(batch_size, 1),
            "right": torch.zeros(batch_size, 1),
        }
    )
    _, _, _, weights = info[-1]

    # Treat these as fixed per-component reconstruction NLLs. The explicit sum is
    # exactly what training must optimize after Rao-Blackwellizing the component index.
    component_nll = torch.tensor(
        [[0.2, 1.1, 2.3, 0.7], [1.5, 0.4, 0.9, 2.0], [0.8, 1.7, 0.3, 1.2]]
    )
    weighted_loss = (weights * component_nll).sum() / batch_size
    loop_reference = (
        sum(
            weights[b, c] * component_nll[b, c]
            for b in range(batch_size)
            for c in range(components**2)
        )
        / batch_size
    )
    assert torch.allclose(weighted_loss, loop_reference, atol=1e-7)

    weighted_loss.backward()
    for modality_encoder in modality_encoders.values():
        gradient = modality_encoder.output.grad[0]
        assert gradient is not None and torch.isfinite(gradient).all()
        # Means and precision increments affect the analytic overlap weights; logits
        # affect their Cartesian products. All three blocks need score gradients.
        assert gradient[: components * latent_dim].abs().sum() > 1e-8
        assert (
            gradient[components * latent_dim : 2 * components * latent_dim].abs().sum()
            > 1e-8
        )
        assert gradient[-components:].abs().sum() > 1e-8


def test_uncorrected_mog_poe_remains_rejected():
    with pytest.raises(ValueError, match="exact fusion only"):
        MultiModalEncoder(
            OrderedDict((f"m{i}", nn.Identity()) for i in range(2)),
            topic_dim=2,
            prior=FixedGaussianPrior(0, 2),
            ae_type="vae",
            poe="uncorrected",
            vi_type="mixture_of_gaussians",
            mixture_components=2,
        )
