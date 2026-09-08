"""Print per-checkpoint outcome coefficients for the fast-iteration GTM runs."""
import json, sys
from pathlib import Path
import numpy as np

paths = sys.argv[1:] or sorted(Path("audit/fast").glob("run*.json"))
for p in paths:
    r = json.load(open(p))
    c = r["config"]
    truth = np.array(r["design"]["beta"])
    print(f"\n##### {Path(p).name}: n={c['n']} words={c['words']} hidden={c['hidden']} lr={c['lr']} prior_lr={c['prior_lr']} "
          f"batch={c['batch']} two_step_prior={c['two_step_prior']} status={r['status']}")
    o = r["oracle_true_shares"]
    print(f"oracle true theta      beta={np.round(o['beta'], 3).tolist()} scale={o['scale_ratio']:.3f} MAE={o['mean_absolute_error']:.3f}")
    for arm, rec in r["arms"].items():
        for ck in rec["checkpoints"]:
            eff = ck.get("head", ck.get("ols_encoder"))
            print(f"{arm:9s} @{ck['step']:6d}  beta={np.round(eff['beta'], 3).tolist()} scale={eff['scale_ratio']:.3f} "
                  f"MAE={eff['mean_absolute_error']:.3f} dec_corr={np.round(ck['decoder_profile_correlations'], 2).tolist()} "
                  f"prevRMSE={ck.get('prevalence_rmse', float('nan')):.3f} covErr={ck.get('covariance_relative_frobenius_error', float('nan')):.3f}"
                  + (f" sig2y={ck['outcome_noise_variance']:.3f}" if "outcome_noise_variance" in ck else ""))
        for ck in rec["checkpoints"]:
            if "ols_plugin_mle" in ck:
                m = ck["ols_plugin_mle"]
                print(f"{arm:9s} @{ck['step']:6d} plug-in MLE beta={np.round(m['beta'], 3).tolist()} scale={m['scale_ratio']:.3f} MAE={m['mean_absolute_error']:.3f} vertex={m['fraction_at_vertex']:.2f}")
        if "ols_outcome_free_posterior" in rec:
            pf = rec["ols_outcome_free_posterior"]
            print(f"{arm:9s} RC post-fit (final) beta={np.round(pf['beta'], 3).tolist()} scale={pf['scale_ratio']:.3f} "
                  f"MAE={pf['mean_absolute_error']:.3f} se={np.round(pf['naive_hc1_se'], 3).tolist()} "
                  f"ESS med={pf['posterior_diagnostics']['ess_median']:.0f} share_corr={np.round(pf['topic_share_correlations'], 2).tolist()}")
        if "heldout_posterior" in rec:
            hp = rec["heldout_posterior"]["importance_diagnostics"]
            print(f"{arm:9s} held-out: IWAE={hp['log_evidence_estimate']:.3f} defensive-ELBO={hp['defensive_proposal_elbo']:.3f}")
