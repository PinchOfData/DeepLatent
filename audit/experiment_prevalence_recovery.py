"""Monte-Carlo recovery of PREVALENCE-COVARIATE coefficients.

DGP (deeplatent.generate_documents, logistic-normal): z_d = X_d @ lambda + eps,
theta_d = softmax(z_d), words ~ Multinomial(theta_d @ B). lambda is (C+1, K) — the
prevalence coefficients we try to recover.

Estimators compared (per replication, same data):
  * mean_net  — the model's own prior coefficients (prior.mean_net), update_prior=True
  * two_step  — OLS of centered-log(theta_hat) on X (post-hoc, STM-style)
  * oracle    — OLS of centered-log(TRUE theta) on X  (infeasible ceiling)
for ae_type in {vae, wae}.

Identifiability handled: (1) topic permutation via Hungarian on theta; (2) softmax
additive-shift -> center each covariate's coefficients across topics; (3) intercept/bias
redundancy -> fold mean_net.bias into the intercept column. Headline metric is recovery
of the NON-intercept covariate effects.
"""
import json, time, numpy as np, scipy.sparse, tempfile, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

R, N, K, C, VOCAB, WORDS = 10, 2500, 5, 3, 200, 80
NUM_STEPS, BATCH = 2000, 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def align(true_theta, est_theta):
    score = true_theta.T @ est_theta
    r, c = linear_sum_assignment(-score)
    return {int(i): int(j) for i, j in zip(r, c)}

def center_rows_over_topics(M):  # M [K, n_cov]: remove per-covariate additive shift
    return M - M.mean(axis=0, keepdims=True)

def ols(X, Y):  # [n_cov,K]
    return np.linalg.lstsq(X, Y, rcond=None)[0]

def cov_metrics(est_KC, true_KC):
    """est,true [K, n_cov] centered; assess non-intercept cols (1:) recovery."""
    e = center_rows_over_topics(est_KC)[:, 1:].ravel()
    t = center_rows_over_topics(true_KC)[:, 1:].ravel()
    corr = float(np.corrcoef(t, e)[0, 1])
    rmse = float(np.sqrt(np.mean((t - e) ** 2)))
    slope = float(np.polyfit(t, e, 1)[0])
    return corr, rmse, slope, (t, e)

rows, scatter = [], {"vae": [[], []], "wae": [[], []], "oracle": [[], []]}
for r in range(R):
    seed = 100 + r
    dft, df, tw, lam, _ = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB,
        num_covs=C, doc_topic_prior="logistic_normal", min_words=WORDS, max_words=WORDS,
        random_seed=seed)
    true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
    lamT = lam.T.astype(np.float64)  # [K, C+1]
    vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
    corpus = Corpus(df, modalities={"text": {"column": "doc_clean_0",
                    "views": {"bow": {"type": "bow", "vectorizer": vec}}}},
                    prevalence="~ cov_1 + cov_2 + cov_3")
    m = corpus.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(m):
        corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)
    X = corpus.M_prevalence_covariates.astype(np.float64)  # [N, C+1] = [Intercept, cov_1..3]

    # ---- oracle: regress centered-log(true theta) on X ----
    def centered_logit(theta):
        lg = np.log(np.clip(theta, 1e-8, 1.0))
        return lg - lg.mean(axis=1, keepdims=True)
    B_or = ols(X, centered_logit(true_theta)).T  # [K, C+1]
    c_or, rm_or, sl_or, (t_or, e_or) = cov_metrics(B_or, lamT)
    scatter["oracle"][0] += list(t_or); scatter["oracle"][1] += list(e_or)

    for ae in ["vae", "wae"]:
        t0 = time.time()
        kw = dict(w_prior=1.0) if ae == "vae" else {}
        model = GTM(train_data=corpus, n_topics=K, ae_type=ae, vi_type="mean_field",
                    doc_topic_prior="logistic_normal", update_prior=True,
                    encoder_args={"text_bow": {"hidden_dims": [128], "dropout": 0.0}},
                    decoder_args={"text_bow": {"hidden_dims": []}},
                    batch_size=BATCH, num_steps=NUM_STEPS, num_workers=0,
                    print_every_n_steps=10**9, return_best_model=False,
                    ckpt_folder=tempfile.mkdtemp(), seed=seed, device=device, **kw)
        theta_hat = model.get_doc_topic_distribution(corpus, num_samples=20).astype(np.float64)
        mp = align(true_theta, theta_hat)                       # true_t -> est_t
        theta_al = np.stack([theta_hat[:, mp[t]] for t in range(K)], axis=1)

        # estimator 1: mean_net coefficients (fold bias into intercept col)
        W = model.get_prevalence_coefficients().astype(np.float64)  # [K, C+1] lifted+centered (v0.2.0)
        W_al = np.stack([W[mp[t]] for t in range(K)])           # [K, C+1] true order
        c_mn, rm_mn, sl_mn, (tm, em) = cov_metrics(W_al, lamT)

        # estimator 2: two-step OLS on inferred theta
        B_ts = ols(X, centered_logit(theta_al)).T               # [K, C+1]
        c_ts, rm_ts, sl_ts, _ = cov_metrics(B_ts, lamT)

        if r < 6:  # collect scatter from a few reps
            scatter[ae][0] += list(tm); scatter[ae][1] += list(em)
        rows.append({"rep": r, "ae_type": ae,
                     "meannet_corr": c_mn, "meannet_rmse": rm_mn, "meannet_slope": sl_mn,
                     "twostep_corr": c_ts, "twostep_rmse": rm_ts, "twostep_slope": sl_ts,
                     "oracle_corr": c_or, "oracle_rmse": rm_or, "oracle_slope": sl_or,
                     "secs": time.time() - t0})
        print(f"rep {r} {ae}: mean_net corr={c_mn:.3f} rmse={rm_mn:.3f} slope={sl_mn:.2f} | "
              f"two-step corr={c_ts:.3f} | oracle corr={c_or:.3f}", flush=True)

# ---- aggregate ----
import statistics as st
def agg(key, ae=None):
    vals = [x[key] for x in rows if (ae is None or x["ae_type"] == ae)]
    return float(np.mean(vals)), float(np.std(vals))
print("\n=== MC recovery of prevalence-covariate effects "
      f"(R={R}, N={N}, K={K}, C={C}, V={VOCAB}) ===", flush=True)
print(f"{'estimator':<22}{'corr (mean±sd)':<22}{'RMSE':<16}{'slope':<10}", flush=True)
summary = {}
for label, key, ae in [("mean_net (vae)", "meannet", "vae"), ("two_step (vae)", "twostep", "vae"),
                       ("mean_net (wae)", "meannet", "wae"), ("two_step (wae)", "twostep", "wae"),
                       ("oracle (ceiling)", "oracle", None)]:
    cm, cs = agg(key + "_corr", ae); rm, rs = agg(key + "_rmse", ae); sm, ss = agg(key + "_slope", ae)
    print(f"{label:<22}{cm:.3f} ± {cs:.3f}      {rm:.3f} ± {rs:.3f}   {sm:.2f}", flush=True)
    summary[label] = {"corr_mean": cm, "corr_sd": cs, "rmse_mean": rm, "slope_mean": sm}

json.dump({"rows": rows, "summary": summary}, open("audit/prevalence_recovery_results.json", "w"), indent=2)

# scatter
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, key, ttl in zip(axes, ["oracle", "vae", "wae"],
                        ["Oracle (true theta)", "mean_net, VAE", "mean_net, WAE"]):
    t, e = np.array(scatter[key][0]), np.array(scatter[key][1])
    ax.scatter(t, e, s=10, alpha=0.5)
    lim = [min(t.min(), e.min()), max(t.max(), e.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_title(f"{ttl}\ncorr={np.corrcoef(t,e)[0,1]:.3f}")
    ax.set_xlabel("true coefficient (centered)"); ax.set_ylabel("estimated")
fig.suptitle("Prevalence-covariate coefficient recovery (non-intercept effects)")
fig.tight_layout()
fig.savefig("audit/prevalence_recovery_scatter.png", dpi=150, bbox_inches="tight")
print("\nsaved -> audit/prevalence_recovery_results.json, audit/prevalence_recovery_scatter.png", flush=True)
