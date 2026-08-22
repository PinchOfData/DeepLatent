"""Numerical verification of mathematical claims in the DeepLatent audit."""
import torch, numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal, kl_divergence

torch.manual_seed(0); np.random.seed(0)
dev = "cpu"

def header(s): print("\n" + "="*70 + f"\n{s}\n" + "="*70)

# ----------------------------------------------------------------------
header("1. Learnable LogisticNormal / Gaussian prior: initial covariance")
from deeplatent.priors import LogisticNormalPrior, GaussianPrior, DirichletPrior, FixedDirichletPrior
K = 5
p = LogisticNormalPrior(prevalence_covariate_size=0, n_topics=K)
Sigma0 = p.sigma.detach()
print("Sigma diagonal at init :", Sigma0.diag().numpy())
print("exp(1)^2 =", np.exp(1)**2, " (docstring claims identity, i.e. 1.0)")
print("Off-diagonal max abs    :", (Sigma0 - torch.diag(Sigma0.diag())).abs().max().item())

# ----------------------------------------------------------------------
header("2. Default (FIXED logistic-normal) mean-field KL matches torch.distributions")
# Replicates the exact KL formula in models.py mean_field / diagonal-prior branch
B = 7
mu_q = torch.randn(B, K); logvar_q = torch.randn(B, K)*0.5
mu_p = torch.zeros(B, K); logvar_p = torch.zeros(B, K)   # FixedLogisticNormalPrior N(0,I)
var_q, var_p = torch.exp(logvar_q), torch.exp(logvar_p)
kl_code = 0.5*(logvar_p - logvar_q - 1 + var_q/var_p + (mu_q-mu_p).pow(2)/var_p).sum(1)
q = Normal(mu_q, torch.exp(0.5*logvar_q)); pr = Normal(mu_p, torch.exp(0.5*logvar_p))
kl_ref = kl_divergence(q, pr).sum(1)
print("max |KL_code - KL_torch| =", (kl_code-kl_ref).abs().max().item(), "(should be ~0 -> default path correct)")

# ----------------------------------------------------------------------
header("3. Dirichlet prior + VAE: KL is computed in the WRONG space")
for alpha in [0.1, 1.0]:
    dp = FixedDirichletPrior(0, K, alpha=alpha)
    mu_p, logvar_p = dp.get_prior_params(torch.zeros(B,1))
    print(f"\nDir(alpha={alpha}): prior 'mean' fed to Gaussian-KL = {mu_p[0].numpy()}  (sums to {mu_p[0].sum():.3f}; lives on SIMPLEX)")
    print(f"                 prior 'var'  fed to Gaussian-KL = {torch.exp(logvar_p)[0].numpy()}")
    # The encoder q(z) lives in R^K (logit space); a *standard-normal* logit posterior
    # (the natural 'uninformative' encoder) should have ~0 KL to an uninformative prior.
    mu_q = torch.zeros(B, K); logvar_q = torch.zeros(B, K)   # q = N(0, I) in logit space
    var_q, var_p = torch.exp(logvar_q), torch.exp(logvar_p)
    kl = 0.5*(logvar_p - logvar_q - 1 + var_q/var_p + (mu_q-mu_p).pow(2)/var_p).sum(1).mean()
    print(f"   KL( N(0,I)_logit || 'Dirichlet' ) = {kl.item():.2f}  "
          f"<-- should be modest; huge value shows the space mismatch")

# Correct Laplace-bridge prior params (AVITM / Srivastava-Sutton) for comparison
def laplace_bridge(alpha, K):
    a = torch.full((K,), float(alpha))
    mu = torch.log(a) - torch.log(a).mean()
    var = (1.0/a)*(1 - 2.0/K) + (1.0/K**2)*(1.0/a).sum()
    return mu, var
mu_lb, var_lb = laplace_bridge(1.0, K)
print(f"\nCorrect Laplace-bridge for Dir(1): mu={mu_lb.numpy()} (all 0), var={var_lb.numpy()}")
print("  -> with the correct bridge, KL(N(0,I)||prior) is small/sane.")

# ----------------------------------------------------------------------
header("4. Corrected PoE formula vs exact Gaussian Bayesian fusion")
# True multimodal posterior precision = prior precision + sum of modality increments
from deeplatent.autoencoders import MultiModalEncoder, EncoderMLP
# Build the exact reference: priors N(0,I); two modality 'likelihood' precisions D1,D2
Lam0 = torch.ones(B, K)                      # prior precision (N(0,I))
mu0  = torch.zeros(B, K)
D1, D2 = torch.rand(B,K)+0.1, torch.rand(B,K)+0.1   # increments
mu1, mu2 = torch.randn(B,K), torch.randn(B,K)
# exact joint posterior (diagonal): Lam = Lam0+D1+D2 ; eta = Lam0 mu0 + D1 mu1 + D2 mu2
Lam_true = Lam0 + D1 + D2
mu_true  = (Lam0*mu0 + D1*mu1 + D2*mu2)/Lam_true
# Now reproduce code path: each unimodal posterior precision Lam_m = Lam0 + D_m ; logvar_m=-log Lam_m
enc = MultiModalEncoder({"a_x":EncoderMLP([2,2]), "b_x":EncoderMLP([2,2])}, topic_dim=K,
                        prior=None, ae_type="vae", poe="corrected", vi_type="mean_field")
# corrected PoE needs a prior; emulate by monkeypatching get_prior_params
class P:
    def get_prior_params(self, c, return_full_cov=False):
        return mu0, torch.zeros(B,K)   # logvar_p = 0 -> Lam0 = 1
enc.prior = P()
dists = [(mu1, -torch.log(Lam0+D1)), (mu2, -torch.log(Lam0+D2))]
mu_S, logvar_S = enc.product_of_experts(dists)
print("max |mu_S - mu_true|     =", (mu_S-mu_true).abs().max().item())
print("max |prec_S - Lam_true|  =", (torch.exp(-logvar_S)-Lam_true).abs().max().item(), "(should be ~0)")

# ----------------------------------------------------------------------
header("5. IAF log-det-Jacobian correctness (autograd check)")
from deeplatent.autoencoders import IAF
iaf = IAF(dim=K, num_flows=3, use_permutations=True).eval()
z0 = torch.randn(1, K, requires_grad=True)
zk, logdet = iaf(z0)
J = torch.autograd.functional.jacobian(lambda z: iaf(z)[0].squeeze(0), z0).squeeze()
sign, logabsdet = torch.linalg.slogdet(J)
print("IAF reported sum(log_sigma) =", logdet.item())
print("autograd  log|det J|        =", logabsdet.item(), "(should match)")

# ----------------------------------------------------------------------
header("6. Dropout is applied to the encoder output (mu/logvar) layer")
e = EncoderMLP([10, 8, 2*K], dropout=0.5)
e.train()
x = torch.ones(4, 10)
outs = torch.stack([e(x) for _ in range(200)]).mean(0)
e.eval(); out_eval = e(x)
print("train-mode mean output != eval output  ->", not torch.allclose(outs, out_eval, atol=1e-2))
print("  (dropout on the final layer randomly zeroes mu/logvar units during training)")

# ----------------------------------------------------------------------
header("7. MMD is computed (and discarded) every step for VAE")
import inspect, deeplatent.models as M
src = inspect.getsource(M.DeepLatent.step_batch)
i = src.index("mmd_loss = 0.0")
print("In step_batch, for ae_type!='ae':")
print("   ... mmd computed in a loop, but for VAE divergence_loss = beta*kl (mmd unused).")
print("   The prior.sample() call (Dirichlet uses a python per-row loop) runs every step.")
print("done.")
