"""Oracle two-step estimators and likelihood-at-truth on the n=10k pilot draw.

No training. Everything is computed under the TRUE decoder/prior of the pilot DGP
(and, for the likelihood comparison, under the fitted pilot checkpoints).

Estimators of the outcome coefficients (OLS of y on a per-document share estimate):
  rc_true_prior      E[theta | w, x] under the true prior p(eta|x)   (regression calibration)
  rc_marginal_prior  E[theta | w]    under the true marginal prior p(eta) (no covariates)
  fixed_N01_prior    E[theta | w]    under a fixed N(0, I) contrast prior (IP-sim convention)
  mle_plugin         per-document MLE of theta on the simplex (unshrunk plug-in, Battaglia-style)
  fitted_*           E[theta | w, x] under the fitted two-step / joint (y-free) parameters

Likelihood comparison: log p(w|x) and log p(w,y|x) under truth vs fitted parameters,
same documents, same importance proposal, paired differences with SE.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_gtm_pilot as H  # noqa: E402
from deeplatent import GTM  # noqa: E402

torch.set_num_threads(8)
dev = torch.device("cpu")
OUT = Path("audit/fast/oracle_n10000.json")
S = 1024          # importance draws per document
CH = 128          # documents per chunk
TRAIN_SUB = 2048  # training docs used for the likelihood comparison
HELD = 2048


def load_arm(a, data, joint, ckpt):
    corpus = H.make_corpus(a, data, joint)
    model = GTM(train_data=corpus, n_topics=a.topics, ae_type="vae",
                vi_type="mixture_of_gaussians", mixture_components=a.components,
                doc_topic_prior="logistic_normal", update_prior=True, learn_prior_cov=True,
                labels_in_encoder=joint,
                predictor_args={"y": {"hidden_dims": [], "loss_weight": 1.0}} if joint else {},
                encoder_args={"text_bow": {"hidden_dims": a.hidden, "dropout": 0.0}},
                decoder_args={"text_bow": {"hidden_dims": [], "dropout": 0.0}},
                w_prior=1.0, w_pred_loss=1.0, batch_size=a.batch, num_steps=0, num_workers=0,
                optim_args={"main": {"lr": a.lr, "weight_decay": 0.0},
                            "prior": {"lr": a.prior_lr, "weight_decay": 0.0}},
                return_best_model=False, ckpt_folder="/tmp/deeplatent_load", seed=a.seed, device=dev)
    model.load_model(ckpt)
    for m in [model.encoder, model.decoders, model.prior] + ([model.predictor] if model.predictor is not None else []):
        m.eval()
    return model


def main():
    a = H.parser().parse_args(["--n", "10000", "--out", "/dev/null"])
    design = H.fixed_design(a)
    data = H.generate(a, design, a.n, a.seed)
    held = H.generate(a, design, HELD, a.seed + 200000)
    K, d = a.topics, a.topics - 1
    V = torch.tensor(design["basis"], dtype=torch.float32)          # (K, K-1)
    D = torch.tensor(design["decoder"], dtype=torch.float32)        # (K, V)
    prev = torch.tensor(design["prevalence"], dtype=torch.float32)  # (K, C+1)
    Sig = torch.tensor(design["covariance"], dtype=torch.float32)   # (K-1, K-1)
    beta = torch.tensor(design["beta"], dtype=torch.float32)
    truth = np.array(design["beta"])

    two = load_arm(a, data, False, "audit/results_gtm_pilot_n10000_20260907_two_step.ckpt")
    joint = load_arm(a, data, True, "audit/results_gtm_pilot_n10000_20260907_joint.ckpt")
    perm_two, _ = H.align_topics(two, design)
    perm_joint, _ = H.align_topics(joint, design)
    print("perm two", perm_two, "perm joint", perm_joint, flush=True)

    # all 8 covariate cells for the marginal prior
    cells = torch.tensor([[1] + [(c >> j) & 1 for j in range(a.covariates)] for c in range(2 ** a.covariates)], dtype=torch.float32)
    cell_means = cells @ prev.T @ V  # (8, K-1)
    mvn_cells = torch.distributions.MultivariateNormal(cell_means, covariance_matrix=Sig)
    mvn_fixed = torch.distributions.MultivariateNormal(torch.zeros(d), covariance_matrix=torch.eye(d))

    def theta_of(z):
        return torch.softmax(z @ V.T, dim=-1)

    def loglik_words(theta, counts):  # theta (..., K), counts (n, V) broadcast on leading dims
        logits = theta @ D
        return (counts * logits.log_softmax(-1)).sum(-1)

    def run(dataset, tag, want_means):
        n = len(dataset["y"])
        torch.manual_seed(123)
        acc = {k: [] for k in ["lz_truth_w", "lz_truth_wy", "lz_two_w", "lz_joint_w", "lz_joint_wy",
                               "ess_truth_w", "ess_two_w", "ess_joint_wy", "ess_fixed", "ess_marg"]}
        means = {k: [] for k in ["rc_true_prior", "rc_marginal_prior", "fixed_N01_prior", "fitted_two_step", "fitted_joint_yfree"]}
        with torch.no_grad():
            for lo in range(0, n, CH):
                sl = slice(lo, lo + CH)
                x = torch.tensor(dataset["x"][sl], dtype=torch.float32)
                counts = torch.tensor(dataset["counts"][sl])
                y = torch.tensor(dataset["y"][sl], dtype=torch.float32)
                m = len(x)
                mean_true = x @ prev.T @ V
                p_true = torch.distributions.MultivariateNormal(mean_true, covariance_matrix=Sig)
                q = H.encoder_distribution(two, x, counts)  # y-free fitted encoder as proposal
                z = q.sample((S,))
                from_prior = torch.rand((S, m)) < 0.5
                z = torch.where(from_prior[..., None], p_true.sample((S,)), z)
                logprop = torch.logaddexp(q.log_prob(z), p_true.log_prob(z)) + math.log(0.5)
                theta = theta_of(z)  # (S, m, K)
                llw = loglik_words(theta, counts[None])  # (S, m)
                lly = -0.5 * ((y[None] - theta @ beta) ** 2 + math.log(2 * math.pi))
                lp_true = p_true.log_prob(z)
                # --- targets under truth
                lw_tw = lp_true + llw - logprop
                lw_twy = lw_tw + lly
                # marginal prior (mixture over 8 equiprobable covariate cells)
                lp_marg = torch.logsumexp(mvn_cells.log_prob(z[..., None, :]), dim=-1) - math.log(len(cells))
                lw_marg = lp_marg + llw - logprop
                lw_fixed = mvn_fixed.log_prob(z) + llw - logprop
                # --- targets under fitted two-step
                mu2, cov2 = two.prior.get_prior_params(x, return_full_cov=True)
                p2 = torch.distributions.MultivariateNormal(mu2, covariance_matrix=cov2)
                th2 = two.latent_to_theta(z.reshape(-1, d)).reshape(S, m, K)
                logits2 = two.decoders["text_bow"](th2.reshape(-1, K)).reshape(S, m, -1)
                lw_2w = p2.log_prob(z) + (counts[None] * logits2.log_softmax(-1)).sum(-1) - logprop
                # --- targets under fitted joint
                muj, covj = joint.prior.get_prior_params(x, return_full_cov=True)
                pj = torch.distributions.MultivariateNormal(muj, covariance_matrix=covj)
                thj = joint.latent_to_theta(z.reshape(-1, d)).reshape(S, m, K)
                logitsj = joint.decoders["text_bow"](thj.reshape(-1, K)).reshape(S, m, -1)
                lw_jw = pj.log_prob(z) + (counts[None] * logitsj.log_softmax(-1)).sum(-1) - logprop
                pred = joint.predictor.predictors["y"](thj.reshape(-1, K), None).reshape(S, m)
                var = joint.predictor.noise_log_var["y"].exp()
                lw_jwy = lw_jw - 0.5 * ((y[None] - pred) ** 2 / var + var.log() + math.log(2 * math.pi))

                def lz(lw):
                    return (lw.logsumexp(0) - math.log(S)).numpy()

                def ess(lw):
                    w = lw.softmax(0)
                    return (1 / w.square().sum(0)).numpy()

                def pmean(lw, th):
                    return (lw.softmax(0)[..., None] * th).sum(0).numpy()

                acc["lz_truth_w"].append(lz(lw_tw)); acc["lz_truth_wy"].append(lz(lw_twy))
                acc["lz_two_w"].append(lz(lw_2w)); acc["lz_joint_w"].append(lz(lw_jw)); acc["lz_joint_wy"].append(lz(lw_jwy))
                acc["ess_truth_w"].append(ess(lw_tw)); acc["ess_two_w"].append(ess(lw_2w)); acc["ess_joint_wy"].append(ess(lw_jwy))
                acc["ess_fixed"].append(ess(lw_fixed)); acc["ess_marg"].append(ess(lw_marg))
                if want_means:
                    means["rc_true_prior"].append(pmean(lw_tw, theta))
                    means["rc_marginal_prior"].append(pmean(lw_marg, theta))
                    means["fixed_N01_prior"].append(pmean(lw_fixed, theta))
                    means["fitted_two_step"].append(pmean(lw_2w, th2)[:, perm_two])
                    means["fitted_joint_yfree"].append(pmean(lw_jw, thj)[:, perm_joint])
                if lo % (CH * 16) == 0:
                    print(f"  {tag} {lo}/{n}", flush=True)
        acc = {k: np.concatenate(v) for k, v in acc.items()}
        means = {k: np.vstack(v) for k, v in means.items()} if want_means else {}
        return acc, means

    result = {"design_note": "pilot DGP n=10000 seed 9100; oracle estimators under TRUE parameters", "importance_draws": S}

    # ---- outcome-coefficient estimators on the full training draw
    t0 = time.time()
    acc, means = run(data, "train", True)
    print(f"train pass done in {time.time() - t0:.0f}s", flush=True)
    est = {"oracle_true_theta": H.ols(data["theta"], data["y"], truth)}
    for k, pm in means.items():
        est[k] = H.ols(pm, data["y"], truth)
        est[k]["share_rmse"] = float(np.sqrt(np.mean((pm - data["theta"]) ** 2)))
        est[k]["share_corr"] = [float(np.corrcoef(pm[:, j], data["theta"][:, j])[0, 1]) for j in range(K)]
    # per-document MLE (unshrunk plug-in) under the TRUE decoder
    eta = torch.zeros((a.n, d), requires_grad=True)
    counts_all = torch.tensor(data["counts"])
    opt = torch.optim.Adam([eta], lr=0.1)
    for it in range(400):
        opt.zero_grad()
        loss = -loglik_words(theta_of(eta), counts_all).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            eta.clamp_(-12, 12)
    th_mle = theta_of(eta.detach()).numpy()
    est["mle_plugin"] = H.ols(th_mle, data["y"], truth)
    est["mle_plugin"]["share_rmse"] = float(np.sqrt(np.mean((th_mle - data["theta"]) ** 2)))
    est["mle_plugin"]["share_corr"] = [float(np.corrcoef(th_mle[:, j], data["theta"][:, j])[0, 1]) for j in range(K)]
    est["mle_plugin"]["fraction_at_boundary"] = float((th_mle.max(1) > 0.99).mean())
    result["estimators"] = est
    result["train_ess"] = {k: {"median": float(np.median(v)), "p05": float(np.quantile(v, 0.05))} for k, v in acc.items() if k.startswith("ess")}
    print("\n=== Outcome coefficients, OLS of y on share estimate (truth 2,1,0,-1,-2) ===")
    for k, v in est.items():
        extra = f" share_rmse={v['share_rmse']:.4f}" if "share_rmse" in v else ""
        print(f"{k:22s} beta={np.round(v['beta'], 3).tolist()} scale={v['scale_ratio']:.3f} MAE={v['mean_absolute_error']:.3f}{extra}")
    print("ESS (train):", {k: round(v["median"]) for k, v in result["train_ess"].items()})

    # ---- likelihood at truth vs fitted, on a fixed training subset and on held-out docs
    sub = {k: v[:TRAIN_SUB] for k, v in data.items()}
    lik = {}
    for tag, ds in [("train_subset", sub), ("held_out", held)]:
        acc, _ = run(ds, tag, False)
        row = {}
        for name, key in [("log p(w|x): truth", "lz_truth_w"), ("log p(w|x): fitted two-step", "lz_two_w"),
                          ("log p(w|x): fitted joint", "lz_joint_w"), ("log p(w,y|x): truth", "lz_truth_wy"),
                          ("log p(w,y|x): fitted joint", "lz_joint_wy")]:
            row[name] = float(acc[key].mean())
        for name, (num, den) in [("truth - fitted two-step, log p(w|x)", ("lz_truth_w", "lz_two_w")),
                                 ("truth - fitted joint, log p(w|x)", ("lz_truth_w", "lz_joint_w")),
                                 ("truth - fitted joint, log p(w,y|x)", ("lz_truth_wy", "lz_joint_wy"))]:
            diff = acc[num] - acc[den]
            row[name] = {"mean": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff)))}
        row["ess_median"] = {k: float(np.median(v)) for k, v in acc.items() if k.startswith("ess")}
        lik[tag] = row
        print(f"\n=== Likelihood per document, {tag} ({len(ds['y'])} docs; multinomial constant omitted) ===")
        for k, v in row.items():
            print(f"  {k}: {v}")
    result["likelihood"] = lik
    OUT.write_text(json.dumps(result, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
