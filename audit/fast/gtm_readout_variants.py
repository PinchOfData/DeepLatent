"""Two-step readout variants under the converged unsupervised pilot fit (n=10k, 25 words)."""
import sys, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "audit"); sys.path.insert(0, "audit/fast")
import experiment_gtm_pilot as H
from gtm_oracle_estimators import load_arm
torch.set_num_threads(8)
a = H.parser().parse_args(["--n", "10000", "--out", "/dev/null"])
design = H.fixed_design(a); data = H.generate(a, design, a.n, a.seed); truth = np.array(design["beta"])
two = load_arm(a, data, False, "audit/results_gtm_pilot_n10000_20260907_two_step.ckpt")
perm, _ = H.align_topics(two, design)
torch.manual_seed(3)
single, mean_logit, pmean = [], [], []
with torch.no_grad():
    for lo in range(0, a.n, 256):
        sl = slice(lo, lo + 256)
        x = torch.tensor(data["x"][sl], dtype=torch.float32); counts = torch.tensor(data["counts"][sl])
        q = H.encoder_distribution(two, x, counts)
        z = q.sample((256,))
        th = two.latent_to_theta(z.reshape(-1, two.n_latent)).reshape(256, len(x), a.topics)
        single.append(th[0].numpy())                                  # get_latent_factors(num_samples=1)
        pmean.append(th.mean(0).numpy())                              # posterior mean of theta (256 draws)
        mean_logit.append(two.latent_to_theta(q.mean).numpy())        # softmax of posterior-mean logit
for name, v in [("posterior mean theta (256 draws)", pmean), ("single posterior draw (package default num_samples=1)", single), ("softmax of encoder mean logit", mean_logit)]:
    v = np.vstack(v)[:, perm]
    o = H.ols(v, data["y"], truth)
    print(f"{name:55s} beta={np.round(o['beta'], 3).tolist()} scale={o['scale_ratio']:.3f} MAE={o['mean_absolute_error']:.3f}")
