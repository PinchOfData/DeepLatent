#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.dirichlet import Dirichlet


class Prior(nn.Module):
    """
    Base template class for doc-topic priors.
    
    All priors are now learnable neural networks that can optionally condition on
    prevalence covariates. Subclasses implement specific prior distributions
    (Dirichlet, Logistic Normal, Gaussian) with either fixed or learnable parameters.
    
    Subclasses must implement:
        - sample(): Draw samples from the prior distribution
        - simulate(): Generate synthetic data for testing
        - get_prior_params(): Return distribution parameters for KL computation
    """

    def __init__(self):
        super().__init__()

    def sample(self, N, M_prevalence_covariates=None, to_simplex=False):
        """
        Sample from the prior distribution.
        
        Args:
            N: Number of samples to draw
            M_prevalence_covariates: Prevalence covariates [batch_size, covariate_dim].
                                    If None, uses global parameters (no covariates).
            to_simplex: Whether to map samples to the simplex via softmax.
                       Only applicable for certain prior types.
        
        Returns:
            Samples from the prior [N, latent_dim]
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def simulate(self, M_prevalence_covariates, **kwargs):
        """
        Simulate data to test the prior's behavior.
        
        Args:
            M_prevalence_covariates: Prevalence covariates for simulation
            **kwargs: Additional arguments specific to the prior type
            
        Returns:
            Simulated samples from the prior
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError


class LogisticNormalPrior(Prior):
    """
    Learnable Logistic Normal prior for topic modeling.

    This prior learns a K-dimensional multivariate Gaussian distribution, then
    maps samples to the (K-1)-simplex via softmax normalization. The mean can
    optionally depend on prevalence covariates via a linear transformation.
    
    The covariance matrix is parameterized via Cholesky decomposition to ensure
    positive definiteness: Σ = L L^T where L is lower triangular with positive
    diagonal elements.
    
    Mathematical formulation:
        z ~ N(μ(x), Σ)  where μ(x) = W*x (if covariates) or μ (global)
        θ = softmax(z)   maps to simplex
    
    Attributes:
        prevalence_covariate_size: Dimension of prevalence covariates
        n_topics: Number of topics (K)
        n_latent: Latent dimension (equals n_topics)
        mean_net: Linear layer for covariate-dependent mean (if covariates > 0)
        global_mean: Learnable global mean parameter (if no covariates)
        L_flat: Flattened lower triangular Cholesky factor
    """

    def __init__(
        self,
        prevalence_covariate_size,
        n_topics,
    ):
        super().__init__()
        self.prevalence_covariate_size = prevalence_covariate_size
        self.n_topics = n_topics
        self.n_latent = n_topics  # Use full K dimensions
        
        if prevalence_covariate_size > 0:
            # Learnable linear layer for mean prediction (K dimensions)
            self.mean_net = nn.Linear(prevalence_covariate_size, self.n_latent)
        else:
            # Learnable global mean parameters (K dimensions)
            self.global_mean = nn.Parameter(torch.zeros(self.n_latent))
            
        # Learnable Cholesky factor for covariance matrix (K x K).
        # `sigma` exponentiates the diagonal (L_ii = exp(L_flat_ii) + 1e-4), so a
        # zero-initialized L_flat yields L_ii = 1 + 1e-4 and off-diagonals 0, i.e.
        # Sigma = L L^T ~= I at initialization (the intended identity init).
        tril_count = self.n_latent * (self.n_latent + 1) // 2
        self.L_flat = nn.Parameter(torch.zeros(tril_count))

    @property
    def sigma(self):
        """
        Reconstruct covariance matrix from Cholesky factor.
        
        Computes Σ = L L^T where L is the lower triangular Cholesky factor.
        Diagonal elements of L are exponentiated to ensure positivity.
        
        Returns:
            Covariance matrix [K, K]
        """
        device = self.L_flat.device  # Get device from existing parameter
        L = torch.zeros(self.n_latent, self.n_latent, device=device)
        tril_indices = torch.tril_indices(self.n_latent, self.n_latent, device=device)
        L[tril_indices[0], tril_indices[1]] = self.L_flat
        
        # Ensure positive diagonal elements
        diag_idx = torch.arange(self.n_latent, device=device)
        L[diag_idx, diag_idx] = torch.exp(L[diag_idx, diag_idx]) + 1e-4
        
        return torch.mm(L, L.t())

    def get_parameters(self, M_prevalence_covariates):
        """
        Get mean and covariance for the current batch.
        
        If prevalence covariates are provided, computes batch-specific means
        via a learned linear transformation. Otherwise, uses a global mean
        parameter broadcasted to the batch size.
        
        Args:
            M_prevalence_covariates: Prevalence covariates [batch_size, covariate_dim]
                                    or None for global parameters
        
        Returns:
            means: Mean vectors [batch_size, K]
            covariance: Covariance matrix [K, K] (shared across batch)
        """
        if self.prevalence_covariate_size > 0:
            if not torch.is_tensor(M_prevalence_covariates):
                M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).float()
            means = self.mean_net(M_prevalence_covariates)
        else:
            batch_size = M_prevalence_covariates.shape[0] if M_prevalence_covariates is not None else 1
            means = self.global_mean.unsqueeze(0).expand(batch_size, -1)
            
        return means, self.sigma

    def sample(self, N, M_prevalence_covariates=None, to_simplex=True):
        """
        Sample from the Logistic Normal prior.
        
        Draws samples from the learned multivariate Gaussian and optionally
        applies softmax normalization to map to the probability simplex.
        
        Args:
            N: Number of samples to draw
            M_prevalence_covariates: Prevalence covariates [N, covariate_dim] or None
            to_simplex: If True, apply softmax to map samples to simplex.
                       If False, return raw Gaussian samples.
        
        Returns:
            Samples [N, K]. If to_simplex=True, samples lie on the simplex.
            
        Raises:
            ValueError: If batch size mismatch between N and M_prevalence_covariates
        """
        means, covariance = self.get_parameters(M_prevalence_covariates)
        
        if means.shape[0] != N:
            # Handle case where we need to sample N times but have different batch size
            if self.prevalence_covariate_size == 0:
                means = self.global_mean.unsqueeze(0).expand(N, -1)
            else:
                # This should not happen if called correctly
                raise ValueError(f"Batch size mismatch: got {means.shape[0]}, expected {N}")
        
        # Sample from multivariate normal with full covariance (K dimensions)
        dist = MultivariateNormal(means, covariance.unsqueeze(0).expand(means.shape[0], -1, -1))
        z_samples = dist.sample()  # Shape: [N, K]

        if to_simplex:
            # Normalize via softmax to get simplex
            z_samples = torch.softmax(z_samples, dim=1)  # Shape: [N, K]
            
        return z_samples.float()

    def get_prior_params(self, M_prevalence_covariates, return_full_cov=False):
        """
        Return the mean and covariance parameters of the metadata-informed prior.
        Returns K dimensional parameters.
        
        Args:
            M_prevalence_covariates: Prevalence covariates
            return_full_cov: If True, return full covariance matrix. If False, return diagonal log-variance.
        """
        means, covariance = self.get_parameters(M_prevalence_covariates)
        
        if return_full_cov:
            return means.float(), covariance.float()
        else:
            # For backward compatibility, return diagonal log-variance
            logvar = torch.log(torch.diag(covariance)).unsqueeze(0).expand_as(means)
            return means.float(), logvar.float()

    def simulate(self, M_prevalence_covariates, **kwargs):
        """
        Simulate data using current parameters.
        """
        return self.sample(M_prevalence_covariates.shape[0], M_prevalence_covariates, to_simplex=True)


class DirichletPrior(Prior):
    """
    Learnable Dirichlet prior for topic modeling.

    This prior learns K concentration parameters (α₁, ..., αₖ) that define
    a Dirichlet distribution over the topic simplex. The concentration parameters
    can optionally depend on prevalence covariates via a neural network.
    
    Mathematical formulation:
        α = exp(f(x)) where f is a learned function (if covariates)
        θ ~ Dirichlet(α)
    
    Attributes:
        prevalence_covariate_size: Dimension of prevalence covariates
        n_topics: Number of topics (K)
        n_latent: Latent dimension (equals n_topics)
        concentration_net: Neural network for covariate-dependent α (if covariates > 0)
        log_concentration: Learnable log-concentration parameters (if no covariates)
    """

    def __init__(
        self,
        prevalence_covariate_size,
        n_topics,
    ):
        super().__init__()
        self.prevalence_covariate_size = prevalence_covariate_size
        self.n_topics = n_topics
        self.n_latent = n_topics  # Use full K dimensions
        
        if prevalence_covariate_size > 0:
            # Learnable network to predict log-concentration parameters (K dimensions)
            self.concentration_net = nn.Sequential(
                nn.Linear(prevalence_covariate_size, self.n_latent),
                nn.Softplus()  # Ensures positive output
            )
        else:
            # Learnable global concentration parameters (log scale for stability, K dimensions)
            self.log_concentration = nn.Parameter(torch.zeros(self.n_latent))

    def get_concentration(self, M_prevalence_covariates):
        """
        Get Dirichlet concentration parameters for the current batch.
        
        If prevalence covariates are provided, computes batch-specific concentration
        parameters via a learned neural network with softplus activation (ensures
        positivity). Otherwise, uses global concentration parameters.
        
        Args:
            M_prevalence_covariates: Prevalence covariates [batch_size, covariate_dim]
                                    or None for global parameters
        
        Returns:
            Concentration parameters α [batch_size, K], all elements positive
        """
        if self.prevalence_covariate_size > 0:
            if not torch.is_tensor(M_prevalence_covariates):
                M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).float()
            concentration = self.concentration_net(M_prevalence_covariates)
        else:
            batch_size = M_prevalence_covariates.shape[0] if M_prevalence_covariates is not None else 1
            concentration = torch.exp(self.log_concentration).unsqueeze(0).expand(batch_size, -1)
            
        # Add small epsilon to avoid numerical issues
        concentration = concentration + 1e-6
        
        return concentration

    def sample(self, N, M_prevalence_covariates=None):
        """
        Sample from the Dirichlet prior.
        
        Draws samples from the Dirichlet distribution with learned concentration
        parameters. Samples automatically lie on the probability simplex.
        
        Args:
            N: Number of samples to draw
            M_prevalence_covariates: Prevalence covariates [N, covariate_dim] or None
        
        Returns:
            Samples on the simplex [N, K], where each row sums to 1
            
        Raises:
            ValueError: If batch size mismatch between N and M_prevalence_covariates
        """
        concentration = self.get_concentration(M_prevalence_covariates)
        
        if concentration.shape[0] != N:
            if self.prevalence_covariate_size == 0:
                concentration = torch.exp(self.log_concentration).unsqueeze(0).expand(N, -1) + 1e-6
            else:
                raise ValueError(f"Batch size mismatch: got {concentration.shape[0]}, expected {N}")
        
        # Sample from Dirichlet distribution
        device = concentration.device
        samples = torch.empty_like(concentration, device=device)
        for i in range(concentration.shape[0]):
            dist = Dirichlet(concentration[i])
            samples[i] = dist.sample()
            
        return samples.float()

    def get_prior_params(self, M_prevalence_covariates):
        """
        Return parameters for KL divergence computation.
        For Dirichlet, we return the concentration parameters as mean and logvar.
        """
        concentration = self.get_concentration(M_prevalence_covariates)
        # Return as mean and logvar for consistency
        # Dirichlet mean = alpha / sum(alpha)
        alpha_sum = concentration.sum(dim=1, keepdim=True)
        mean = concentration / alpha_sum
        
        # Dirichlet variance = alpha*(sum(alpha)-alpha) / (sum(alpha)^2 * (sum(alpha)+1))
        var = concentration * (alpha_sum - concentration) / (alpha_sum**2 * (alpha_sum + 1))
        logvar = torch.log(var + 1e-8)
        
        return mean.float(), logvar.float()

    def simulate(self, M_prevalence_covariates, **kwargs):
        """
        Simulate data using current learned parameters.
        
        Convenience method that samples from the prior with the current
        learned concentration parameters.
        
        Args:
            M_prevalence_covariates: Prevalence covariates [batch_size, covariate_dim]
            **kwargs: Additional arguments (unused, for API compatibility)
        
        Returns:
            Simulated samples on the simplex [batch_size, K]
        """
        return self.sample(M_prevalence_covariates.shape[0], M_prevalence_covariates)


class GaussianPrior(Prior):
    """
    Learnable Gaussian prior for ideal point models.

    This prior learns a multivariate Gaussian distribution over unconstrained
    latent variables (ideal points). The mean can optionally depend on prevalence
    covariates via a linear transformation.
    
    Unlike LogisticNormalPrior, samples are NOT mapped to the simplex - they
    remain as real-valued vectors in ℝⁿ. This is suitable for ideal point models
    where latent dimensions represent policy positions or other continuous factors.
    
    The covariance matrix is parameterized via Cholesky decomposition to ensure
    positive definiteness: Σ = L L^T where L is lower triangular.
    
    Mathematical formulation:
        z ~ N(μ(x), Σ)  where μ(x) = W*x (if covariates) or μ (global)
    
    Attributes:
        prevalence_covariate_size: Dimension of prevalence covariates
        n_dims: Dimension of latent space
        mean_net: Linear layer for covariate-dependent mean (if covariates > 0)
        global_mean: Learnable global mean parameter (if no covariates)
        L_flat: Flattened lower triangular Cholesky factor
    """

    def __init__(self, prevalence_covariate_size, n_dims):
        super().__init__()
        self.prevalence_covariate_size = prevalence_covariate_size
        self.n_dims = n_dims

        if self.prevalence_covariate_size > 0:
            # Learnable network for mean prediction
            self.mean_net = nn.Linear(prevalence_covariate_size, n_dims)
        else:
            # Learnable global mean
            self.global_mean = nn.Parameter(torch.zeros(n_dims))

        # Learnable Cholesky factor for covariance matrix.
        # `sigma` exponentiates the diagonal (L_ii = exp(L_flat_ii) + 1e-4), so a
        # zero-initialized L_flat yields L_ii = 1 + 1e-4 and off-diagonals 0, i.e.
        # Sigma = L L^T ~= I at initialization (the intended identity init).
        tril_count = n_dims * (n_dims + 1) // 2
        self.L_flat = nn.Parameter(torch.zeros(tril_count))

    @property
    def sigma(self):
        """
        Reconstruct covariance matrix from Cholesky factor.
        
        Computes Σ = L L^T where L is the lower triangular Cholesky factor.
        Diagonal elements of L are exponentiated to ensure positivity.
        
        Returns:
            Covariance matrix [n_dims, n_dims]
        """
        device = self.L_flat.device  # Get device from existing parameter
        L = torch.zeros(self.n_dims, self.n_dims, device=device)
        tril_indices = torch.tril_indices(self.n_dims, self.n_dims, device=device)
        L[tril_indices[0], tril_indices[1]] = self.L_flat
        
        # Ensure positive diagonal elements
        diag_idx = torch.arange(self.n_dims, device=device)
        L[diag_idx, diag_idx] = torch.exp(L[diag_idx, diag_idx]) + 1e-4
        
        return torch.mm(L, L.t())

    def get_parameters(self, M_prevalence_covariates):
        """
        Get mean and covariance for the current batch.
        
        If prevalence covariates are provided, computes batch-specific means
        via a learned linear transformation. Otherwise, uses a global mean
        parameter broadcasted to the batch size.
        
        Args:
            M_prevalence_covariates: Prevalence covariates [batch_size, covariate_dim]
                                    or None for global parameters
        
        Returns:
            means: Mean vectors [batch_size, n_dims]
            covariance: Covariance matrix [n_dims, n_dims] (shared across batch)
        """
        if self.prevalence_covariate_size > 0:
            if not torch.is_tensor(M_prevalence_covariates):
                M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).float()
            means = self.mean_net(M_prevalence_covariates)
        else:
            batch_size = M_prevalence_covariates.shape[0] if M_prevalence_covariates is not None else 1
            means = self.global_mean.unsqueeze(0).expand(batch_size, -1)

        return means, self.sigma

    def sample(self, N, M_prevalence_covariates=None, to_simplex=False):
        """
        Sample latent vectors from the Gaussian prior.
        
        Draws samples from the learned multivariate Gaussian distribution.
        Samples are unconstrained real-valued vectors (not on the simplex).
        
        Args:
            N: Number of samples to draw
            M_prevalence_covariates: Prevalence covariates [N, covariate_dim] or None
            to_simplex: Ignored for Gaussian prior (included for API compatibility)
        
        Returns:
            Unconstrained samples [N, n_dims] from ℝⁿ
            
        Raises:
            ValueError: If batch size mismatch between N and M_prevalence_covariates
        """
        means, covariance = self.get_parameters(M_prevalence_covariates)
        
        if means.shape[0] != N:
            if self.prevalence_covariate_size == 0:
                means = self.global_mean.unsqueeze(0).expand(N, -1)
            else:
                raise ValueError(f"Batch size mismatch: got {means.shape[0]}, expected {N}")

        # Sample from multivariate normal
        dist = MultivariateNormal(means, covariance.unsqueeze(0).expand(means.shape[0], -1, -1))
        z_samples = dist.sample()

        return z_samples.float()

    def get_prior_params(self, M_prevalence_covariates, return_full_cov=False):
        """
        Return the mean and covariance parameters for each document.
        
        Args:
            M_prevalence_covariates: Prevalence covariates
            return_full_cov: If True, return full covariance matrix. If False, return diagonal log-variance.
        """
        means, covariance = self.get_parameters(M_prevalence_covariates)
        
        if return_full_cov:
            return means.float(), covariance.float()
        else:
            # For backward compatibility, return diagonal log-variance
            logvar = torch.log(torch.diag(covariance)).unsqueeze(0).expand_as(means)
            return means.float(), logvar.float()

    def simulate(self, M_prevalence_covariates, **kwargs):
        """
        Generate synthetic latent variables from current learned parameters.
        
        Convenience method that samples from the prior with the current
        learned parameters.
        
        Args:
            M_prevalence_covariates: Prevalence covariates [batch_size, covariate_dim]
            **kwargs: Additional arguments (unused, for API compatibility)
        
        Returns:
            Simulated samples [batch_size, n_dims]
        """
        return self.sample(M_prevalence_covariates.shape[0], M_prevalence_covariates)


class FixedGaussianPrior(Prior):
    """
    Fixed Gaussian prior (standard normal) - not learnable.
    
    This prior uses a fixed standard multivariate normal distribution N(0, I)
    with zero mean and identity covariance. Parameters are not learned during
    training. Useful as a baseline or when you want a simple, non-informative prior.
    
    Mathematical formulation:
        z ~ N(0, I)
    
    Attributes:
        prevalence_covariate_size: Ignored (for API compatibility)
        n_dims: Dimension of latent space
        _device_tracker: Buffer to track device placement
    """

    def __init__(self, prevalence_covariate_size, n_dims):
        super().__init__()
        self.prevalence_covariate_size = prevalence_covariate_size
        self.n_dims = n_dims
        # Register a dummy parameter to track device
        self.register_buffer('_device_tracker', torch.tensor(0.0))

    @property
    def device(self):
        """Get the device (CPU/GPU) where the prior is located."""
        return self._device_tracker.device

    def sample(self, N, M_prevalence_covariates=None, to_simplex=False):
        """
        Sample from standard normal distribution.
        
        Draws samples from N(0, I). Covariates are ignored since this is
        a fixed, non-covariate-dependent prior.
        
        Args:
            N: Number of samples to draw
            M_prevalence_covariates: Ignored (for API compatibility)
            to_simplex: Ignored (for API compatibility)
        
        Returns:
            Samples from standard normal [N, n_dims]
        """
        return torch.randn(N, self.n_dims, device=self.device)

    def get_prior_params(self, M_prevalence_covariates, return_full_cov=False):
        """
        Return zero mean and unit variance.
        
        Args:
            M_prevalence_covariates: Prevalence covariates
            return_full_cov: If True, return full covariance matrix. If False, return diagonal log-variance.
        """
        batch_size = M_prevalence_covariates.shape[0] if M_prevalence_covariates is not None else 1
        means = torch.zeros(batch_size, self.n_dims, device=self.device)
        
        if return_full_cov:
            covariance = torch.eye(self.n_dims, device=self.device)
            return means, covariance
        else:
            logvar = torch.zeros_like(means)
            return means, logvar

    @property
    def sigma(self):
        """
        Return identity covariance matrix.
        
        Returns:
            Identity matrix I [n_dims, n_dims]
        """
        return torch.eye(self.n_dims, device=self.device)

    def simulate(self, M_prevalence_covariates, **kwargs):
        """
        Simulate data from standard normal distribution.
        
        Args:
            M_prevalence_covariates: Used only for batch size
            **kwargs: Additional arguments (unused)
        
        Returns:
            Simulated samples from N(0, I) [batch_size, n_dims]
        """
        return self.sample(M_prevalence_covariates.shape[0], M_prevalence_covariates)


class FixedLogisticNormalPrior(Prior):
    """
    Fixed Logistic Normal prior - not learnable.
    
    This prior uses a fixed logistic normal distribution: samples are drawn from
    a standard multivariate normal N(0, I) and then mapped to the simplex via
    softmax. Parameters are not learned during training.
    
    This is equivalent to a uniform Dirichlet prior over the simplex and serves
    as a non-informative baseline for topic models.
    
    Mathematical formulation:
        z ~ N(0, I)
        θ = softmax(z)
    
    Attributes:
        prevalence_covariate_size: Ignored (for API compatibility)
        n_topics: Number of topics (K)
        n_latent: Latent dimension (equals n_topics)
        _device_tracker: Buffer to track device placement
    """

    def __init__(self, prevalence_covariate_size, n_topics):
        super().__init__()
        self.prevalence_covariate_size = prevalence_covariate_size
        self.n_topics = n_topics
        self.n_latent = n_topics  # Use full K dimensions
        # Register a dummy parameter to track device
        self.register_buffer('_device_tracker', torch.tensor(0.0))

    @property
    def device(self):
        """Get the device (CPU/GPU) where the prior is located."""
        return self._device_tracker.device

    @property
    def sigma(self):
        """
        Return identity covariance matrix.
        
        Returns:
            Identity matrix I [K, K]
        """
        return torch.eye(self.n_latent, device=self.device)

    def sample(self, N, M_prevalence_covariates=None, to_simplex=True):
        """
        Sample from fixed logistic normal distribution.
        
        Draws samples from N(0, I) and optionally applies softmax to map to
        the simplex. Covariates are ignored since this is a fixed prior.
        
        Args:
            N: Number of samples to draw
            M_prevalence_covariates: Ignored (for API compatibility)
            to_simplex: If True, apply softmax to map samples to simplex.
                       If False, return raw Gaussian samples.
        
        Returns:
            Samples [N, K]. If to_simplex=True, samples lie on the simplex.
        """
        # Sample from standard multivariate normal (K dimensions)
        z_samples = torch.randn(N, self.n_latent, device=self.device)
        
        if to_simplex:
            # Normalize via softmax to get simplex
            z_samples = torch.softmax(z_samples, dim=1)  # Shape: [N, K]
            
        return z_samples.float()

    def get_prior_params(self, M_prevalence_covariates, return_full_cov=False):
        """
        Return zero mean and unit variance for K dimensions.
        
        Args:
            M_prevalence_covariates: Prevalence covariates
            return_full_cov: If True, return full covariance matrix. If False, return diagonal log-variance.
        """
        batch_size = M_prevalence_covariates.shape[0] if M_prevalence_covariates is not None else 1
        means = torch.zeros(batch_size, self.n_latent, device=self.device)
        
        if return_full_cov:
            covariance = torch.eye(self.n_latent, device=self.device)
            return means.float(), covariance.float()
        else:
            logvar = torch.zeros_like(means)
            return means.float(), logvar.float()

    def simulate(self, M_prevalence_covariates, **kwargs):
        """
        Simulate data using fixed standard normal parameters.
        
        Args:
            M_prevalence_covariates: Used only for batch size
            **kwargs: Additional arguments (unused)
        
        Returns:
            Simulated samples on the simplex [batch_size, K]
        """
        return self.sample(M_prevalence_covariates.shape[0], M_prevalence_covariates, to_simplex=True)


class FixedDirichletPrior(Prior):
    """
    Fixed Dirichlet prior with specified concentration parameter - not learnable.
    
    This prior uses a fixed symmetric Dirichlet distribution Dir(α, ..., α)
    where α is specified at initialization. Parameters are not learned during
    training.
    
    When α = 1, this gives a uniform distribution over the simplex. Values
    α < 1 encourage sparsity (few active topics), while α > 1 encourages
    more uniform topic distributions.
    
    Mathematical formulation:
        θ ~ Dir(α, α, ..., α)  where α is fixed
    
    Attributes:
        prevalence_covariate_size: Ignored (for API compatibility)
        n_topics: Number of topics (K)
        alpha: Fixed concentration parameter (default 1.0)
        _device_tracker: Buffer to track device placement
    """

    def __init__(self, prevalence_covariate_size, n_topics, alpha=1.0):
        super().__init__()
        self.prevalence_covariate_size = prevalence_covariate_size
        self.n_topics = n_topics
        self.alpha = alpha
        # Register a dummy parameter to track device
        self.register_buffer('_device_tracker', torch.tensor(0.0))

    @property
    def device(self):
        """Get the device (CPU/GPU) where the prior is located."""
        return self._device_tracker.device

    def sample(self, N, M_prevalence_covariates=None):
        """
        Sample from fixed symmetric Dirichlet distribution.
        
        Draws samples from Dir(α, ..., α) with the fixed concentration
        parameter. Samples automatically lie on the simplex. Covariates
        are ignored since this is a fixed prior.
        
        Args:
            N: Number of samples to draw
            M_prevalence_covariates: Ignored (for API compatibility)
        
        Returns:
            Samples on the simplex [N, K], where each row sums to 1
        """
        concentration = torch.full((N, self.n_topics), self.alpha, device=self.device)
        samples = torch.empty_like(concentration)
        for i in range(N):
            dist = Dirichlet(concentration[i])
            samples[i] = dist.sample()
        return samples.float()

    def get_prior_params(self, M_prevalence_covariates):
        """
        Return fixed Dirichlet parameters as mean and log-variance.
        
        Computes the mean and variance of the Dirichlet distribution for
        use in KL divergence calculations. Both are derived from the fixed
        concentration parameter α.
        
        Args:
            M_prevalence_covariates: Used only for batch size
        
        Returns:
            mean: Dirichlet mean [batch_size, K]
            logvar: Log-variance [batch_size, K]
        """
        batch_size = M_prevalence_covariates.shape[0] if M_prevalence_covariates is not None else 1
        concentration = torch.full((batch_size, self.n_topics), self.alpha, device=self.device)
        
        # Dirichlet mean = alpha / sum(alpha)
        alpha_sum = concentration.sum(dim=1, keepdim=True)
        mean = concentration / alpha_sum
        
        # Dirichlet variance = alpha*(sum(alpha)-alpha) / (sum(alpha)^2 * (sum(alpha)+1))
        var = concentration * (alpha_sum - concentration) / (alpha_sum**2 * (alpha_sum + 1))
        logvar = torch.log(var + 1e-8)
        
        return mean.float(), logvar.float()

    def simulate(self, M_prevalence_covariates, **kwargs):
        """
        Simulate data from fixed Dirichlet distribution.
        
        Args:
            M_prevalence_covariates: Used only for batch size
            **kwargs: Additional arguments (unused)
        
        Returns:
            Simulated samples on the simplex [batch_size, K]
        """
        return self.sample(M_prevalence_covariates.shape[0], M_prevalence_covariates)
