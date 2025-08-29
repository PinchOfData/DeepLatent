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
    All priors are now learnable neural networks.
    """

    def __init__(self):
        super().__init__()

    def sample(self, N, M_prevalence_covariates=None, to_simplex=False, epoch=None, initialization=False):
        """
        Sample from the prior.
        """
        raise NotImplementedError

    def simulate(self, M_prevalence_covariates, **kwargs):
        """
        Simulate data to test the prior.
        """
        raise NotImplementedError


class LogisticNormalPrior(Prior):
    """
    Learnable logistic Normal prior.

    We learn the mean and full covariance parameters for K dimensions that define a multivariate gaussian,
    then map samples to the simplex via softmax normalization.
    Uses Cholesky decomposition for positive definiteness.
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
            
        # Learnable Cholesky factor for covariance matrix (K x K)
        # Initialize as identity matrix (flattened lower triangular)
        self.L_flat = nn.Parameter(torch.eye(self.n_latent)[torch.tril_indices(self.n_latent, self.n_latent)[0], 
                                                             torch.tril_indices(self.n_latent, self.n_latent)[1]])

    @property
    def sigma(self):
        """Reconstruct covariance matrix from Cholesky factor (K x K)"""
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
        Get mean and covariance for the current batch (K dimensions).
        """
        if self.prevalence_covariate_size > 0:
            if not torch.is_tensor(M_prevalence_covariates):
                M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).float()
            means = self.mean_net(M_prevalence_covariates)
        else:
            batch_size = M_prevalence_covariates.shape[0] if M_prevalence_covariates is not None else 1
            means = self.global_mean.unsqueeze(0).expand(batch_size, -1)
            
        return means, self.sigma

    def sample(self, N, M_prevalence_covariates=None, to_simplex=True, epoch=None, initialization=False):
        """
        Sample from the prior and optionally map to simplex via softmax.
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
    Learnable Dirichlet prior.

    We learn all K concentration parameters directly.
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
        Get Dirichlet concentration parameters for the current batch (K dimensions).
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

    def sample(self, N, M_prevalence_covariates=None, epoch=None, initialization=False):
        """
        Sample from the Dirichlet prior.
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
        Simulate data using current parameters.
        """
        return self.sample(M_prevalence_covariates.shape[0], M_prevalence_covariates)


class GaussianPrior(Prior):
    """
    Learnable Gaussian prior over latent variables.

    We learn the mean and full covariance matrix via neural networks.
    Uses Cholesky decomposition for positive definiteness.
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

        # Learnable Cholesky factor for covariance matrix
        # Initialize as identity matrix (flattened lower triangular)
        self.L_flat = nn.Parameter(torch.eye(n_dims)[torch.tril_indices(n_dims, n_dims)[0], 
                                                      torch.tril_indices(n_dims, n_dims)[1]])

    @property
    def sigma(self):
        """Reconstruct covariance matrix from Cholesky factor"""
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
        """
        if self.prevalence_covariate_size > 0:
            if not torch.is_tensor(M_prevalence_covariates):
                M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).float()
            means = self.mean_net(M_prevalence_covariates)
        else:
            batch_size = M_prevalence_covariates.shape[0] if M_prevalence_covariates is not None else 1
            means = self.global_mean.unsqueeze(0).expand(batch_size, -1)

        return means, self.sigma

    def sample(self, N, M_prevalence_covariates=None, to_simplex=False, epoch=None, initialization=False):
        """
        Sample latent vectors from the Gaussian prior.
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
        Generate synthetic latent variables from current parameters.
        """
        return self.sample(M_prevalence_covariates.shape[0], M_prevalence_covariates)


class FixedGaussianPrior(Prior):
    """
    Fixed Gaussian prior (standard normal) - not learnable.
    """

    def __init__(self, prevalence_covariate_size, n_dims):
        super().__init__()
        self.prevalence_covariate_size = prevalence_covariate_size
        self.n_dims = n_dims
        # Register a dummy parameter to track device
        self.register_buffer('_device_tracker', torch.tensor(0.0))

    @property
    def device(self):
        return self._device_tracker.device

    def sample(self, N, M_prevalence_covariates=None, to_simplex=False, epoch=None, initialization=False):
        """
        Sample from standard normal distribution.
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
        """Return identity covariance matrix"""
        return torch.eye(self.n_dims, device=self.device)

    def simulate(self, M_prevalence_covariates, **kwargs):
        return self.sample(M_prevalence_covariates.shape[0], M_prevalence_covariates)


class FixedLogisticNormalPrior(Prior):
    """
    Fixed Logistic Normal prior - not learnable.
    Uses fixed zero mean and identity covariance matrix for K dimensions.
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
        return self._device_tracker.device

    @property
    def sigma(self):
        """Return identity covariance matrix (K x K)"""
        return torch.eye(self.n_latent, device=self.device)

    def sample(self, N, M_prevalence_covariates=None, to_simplex=True, epoch=None, initialization=False):
        """
        Sample from fixed logistic normal distribution and optionally map to simplex.
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
        Simulate data using fixed parameters.
        """
        return self.sample(M_prevalence_covariates.shape[0], M_prevalence_covariates, to_simplex=True)


class FixedDirichletPrior(Prior):
    """
    Fixed Dirichlet prior with specified concentration parameter - not learnable.
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
        return self._device_tracker.device

    def sample(self, N, M_prevalence_covariates=None, epoch=None, initialization=False):
        """
        Sample from fixed Dirichlet distribution.
        """
        concentration = torch.full((N, self.n_topics), self.alpha, device=self.device)
        samples = torch.empty_like(concentration)
        for i in range(N):
            dist = Dirichlet(concentration[i])
            samples[i] = dist.sample()
        return samples.float()

    def get_prior_params(self, M_prevalence_covariates):
        """
        Return fixed Dirichlet parameters.
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
        return self.sample(M_prevalence_covariates.shape[0], M_prevalence_covariates)
