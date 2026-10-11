"""The joint topic-prediction model of application 3, self-contained (no deeplatent import).

For record i of month t (the K-1 contrast coordinates remove the softmax's unidentified all-ones direction):

    eta_i ~ N(mu, Sigma)                            learned prior, Sigma = L L'
    theta_i = softmax(B eta_i)                      B: K x (K-1) orthonormal Helmert basis of the zero-sum subspace
    w_i | theta_i ~ Multinomial(n_i, sum_k theta_ik beta_k),  beta_k = softmax(m + kappa_k)

a mixture of topics as in LDA and SAGE/STM: m is the log background word frequency (fixed) and kappa_k topic k's
deviation from it. The month's prediction is linear in its topic attention,

    yhat_t = b + c' (thetabar_t - mbar) / s,     thetabar_t = average over the month's records of E_q[theta_i]

with mbar and s the mean and standard deviation of the attention over the training months with a target, computed
from the current encoder and held constant in the gradient (as the two-step regressions standardize their inputs).
c_k is the change in the standardized outcome per standard deviation of topic k's attention. Monthly attention moves
by fractions of a percentage point, so on the raw shares the coefficients are huge and any common drift of the
attention while the encoder trains moves every prediction (setup runs of 2026-10-10: in-sample R^2 swinging between
-5 and 0.98 at a correlation of 0.97-0.99); standardizing on the training months removes that drift. The shares sum
to one, so the standardized inputs satisfy s'z = 0 and c is identified up to a multiple of s, which leaves the
predictions unchanged: the coefficients are reported with that direction projected out. The variational
posterior is q(eta_i | w_i) = N(m(w_i), diag s^2(w_i)), an MLP of the record's raw term counts.

Training objective, per training record (N records in T training months):

    mean_i [ -ELBO_i ]  +  lam * (T_y / N) * mean_t [ (y_t - yhat_t)^2 / 2 ]

with y standardized on the training months and T_y the training months with a target. lam = 0 is the topic model
alone; lam = 1 adds the outcome as one Gaussian observation per month with the standardized variance (the honest
joint likelihood, in which ~5,000 records per month swamp one number); larger lam is the researcher's trade-off
between topics that describe the text and topics that predict the outcome.

Minibatches hold n random records of each of M random training months (or of every training month). The record term
is reweighted by N_t / n so that it estimates the corpus average. The month term needs the month's attention, an
average over all its records: 03_fit_joint.py keeps every training record's theta in a cache (refreshed for the batch
records at every step and for all records at every evaluation), takes the month's attention from it, and lets the
gradient flow through the batch records (value of the cache, gradient of the batch mean: unbiased for the gradient
of the month average, up to the cache's staleness). With the batch mean alone (--month-attention batch) the
amplified record-level variation makes the head's gradient so noisy that its level drifts (setup runs of 2026-10-10:
in-sample R^2 jumping between 0.6 and 0.97 at a correlation of 0.99); that estimator subtracts the unbiased estimate
(1 - n / N_t) s_t^2 / n of the subsampling noise, s_t^2 the batch variance of c' theta_i within the month.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def contrast_basis(K: int) -> np.ndarray:
    """K x (K-1) orthonormal basis of {x : sum(x) = 0} (Helmert)."""
    B = np.zeros((K, K - 1))
    for j in range(1, K):
        B[:j, j - 1] = 1.0
        B[j, j - 1] = -j
        B[:, j - 1] /= np.sqrt(j * (j + 1))
    return B


class GPUCSR:
    """A CSR count matrix held on the device; rows(idx) returns the dense rows."""

    def __init__(self, X, device):
        self.indptr = torch.as_tensor(X.indptr, dtype=torch.int64, device=device)
        self.indices = torch.as_tensor(X.indices, dtype=torch.int64, device=device)
        self.data = torch.as_tensor(X.data, dtype=torch.float32, device=device)
        self.n_rows, self.n_cols = X.shape

    def rows(self, idx: torch.Tensor) -> torch.Tensor:
        starts = self.indptr[idx]
        lens = self.indptr[idx + 1] - starts
        ends = torch.cumsum(lens, 0)
        total = int(ends[-1]) if len(ends) else 0
        row_of = torch.repeat_interleave(torch.arange(len(idx), device=idx.device), lens)
        pos = torch.repeat_interleave(starts - (ends - lens), lens) + torch.arange(total, device=idx.device)
        out = torch.zeros(len(idx), self.n_cols, device=idx.device)
        out[row_of, self.indices[pos]] = self.data[pos]
        return out


class JointTopicModel(nn.Module):
    def __init__(self, V: int, K: int, hidden, dropout: float, log_background: torch.Tensor,
                 init_kappa: torch.Tensor):
        super().__init__()
        self.K = K
        self.register_buffer("B", torch.tensor(contrast_basis(K), dtype=torch.float32))
        layers, d = [], V
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.encoder = nn.Sequential(*layers, nn.Linear(d, 2 * (K - 1)))
        self.prior_mu = nn.Parameter(torch.zeros(K - 1))
        self.prior_L_raw = nn.Parameter(torch.zeros(K - 1, K - 1))  # strict lower part + log diagonal
        self.register_buffer("m", log_background.float())
        self.kappa = nn.Parameter(init_kappa.float().clone())
        self.head = nn.Parameter(torch.zeros(K))
        self.head_bias = nn.Parameter(torch.zeros(()))

    def prior_params(self):
        L = torch.tril(self.prior_L_raw, -1) + torch.diag(torch.exp(torch.diagonal(self.prior_L_raw)))
        return self.prior_mu, L

    def encode(self, x: torch.Tensor):
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        # The band the package uses for its mixture posteriors (std in ~[0.018, 55]): unbounded, a log-variance can
        # overflow exp() and make the KL infinite (it did at step 941 of a real-data run without dropout).
        return mu, logvar.clamp(-8.0, 8.0)

    def to_theta(self, eta: torch.Tensor) -> torch.Tensor:
        return F.softmax(eta @ self.B.T, dim=-1)

    def beta(self) -> torch.Tensor:
        return F.softmax(self.m + self.kappa, dim=-1)  # K x V

    def kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL( N(mu, diag exp(logvar)) || N(prior_mu, L L') ) per record."""
        pm, L = self.prior_params()
        k = L.shape[0]
        Linv = torch.linalg.solve_triangular(L, torch.eye(k, device=L.device), upper=False)
        prec_diag = (Linv ** 2).sum(0)  # diag of Sigma^{-1} = Linv' Linv
        tr = (torch.exp(logvar) * prec_diag).sum(-1)
        maha = (((mu - pm) @ Linv.T) ** 2).sum(-1)
        logdet = 2 * torch.log(torch.diagonal(L)).sum()
        return 0.5 * (tr + maha - k + logdet - logvar.sum(-1))

    @staticmethod
    def nll(x: torch.Tensor, theta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """-log p(w_i | theta_i) per record (up to the multinomial coefficient)."""
        return -(x * torch.log((theta @ beta).clamp_min(1e-30))).sum(-1)

    def predict(self, theta: torch.Tensor, mbar: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """b + c'(theta - mbar)/s: applied to a month's mean attention it is the month's prediction; applied to one
        record's theta it is that record's contribution, whose mean over the month's records is the former."""
        return self.head_bias + ((theta - mbar) / s) @ self.head

    def sample_theta(self, mu, logvar, n_samples: int = 1) -> torch.Tensor:
        """Mean of n_samples draws of theta under q (n_samples = 1: one reparameterized draw)."""
        std = torch.exp(0.5 * logvar)
        draws = [self.to_theta(mu + std * torch.randn_like(std)) for _ in range(n_samples)]
        return draws[0] if n_samples == 1 else torch.stack(draws).mean(0)
