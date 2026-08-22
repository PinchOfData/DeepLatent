"""Core prediction (STM-style): JOINT estimation (covariates in the prior) should beat
the TWO-STEP estimator on SHORT documents, because the joint model uses covariates while
inferring topics, whereas the two-step infers topics WITHOUT covariates (stage 1) and only
then regresses theta on covariates (stage 2) -- and with few words per doc, stage-1 theta
is poorly identified.

Per (length, rep): fit a JOINT model (prevalence prior, update_prior=True) and a NAIVE
model (no covariates anywhere). Estimators of the covariate effects:
  joint_meannet  = prior.mean_net of the joint model            (the joint estimate)
  joint_twostep  = OLS of centered-log(theta) on X, joint model (covariate-aware theta)
  naive_twostep  = OLS of centered-log(theta) on X, NAIVE model (covariate-naive theta) <- the classic two-step
  oracle         = OLS on TRUE theta (ceiling)
VAE + mixture_of_gaussians, new default optimizer (prior lr=1e-3, wd=0), trained to convergence.
"""
import json, time, numpy as np, scipy.sparse, tempfile, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

LENGTHS = [10, 20, 40, 80, 160]
R, N, K, C, VOCAB, STEPS = 2, 2000, 5, 3, 200, 6000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def cmetrics(estKC, trueKC):
    cr = lambda M: M - M.mean(0, keepdims=True)
    t = cr(trueKC)[:, 1:].ravel(); e = cr(estKC)[:, 1:].ravel()
    return float(np.corrcoef(t, e)[0, 1]), float(np.polyfit(t, e, 1)[0])
def clogit(theta):
    lg = np.log(np.clip(theta, 1e-8, 1.0)); return lg - lg.mean(1, keepdims=True)
def ols(X, Y): return np.linalg.lstsq(X, Y, rcond=None)[0]

def fit_mog(corpus, prevalence, seed):
    return GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
               mixture_components=10, doc_topic_prior="logistic_normal",
               update_prior=prevalence, w_prior=1.0,
               encoder_args={"text_bow": {"hidden_dims": [128]}},
               decoder_args={"text_bow": {"hidden_dims": []}},
               batch_size=256, num_steps=STEPS, num_workers=0, print_every_n_steps=10**9,
               return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=seed, device=device)

def theta_two_step(model, corpus, X, true_theta, lamT):
    th = model.get_doc_topic_distribution(corpus, num_samples=20).astype(np.float64)
    mp = align(true_theta, th)
    th_al = np.stack([th[:, mp[t]] for t in range(K)], axis=1)
    corr, _ = cmetrics(ols(X, clogit(th_al)).T, lamT)
    overall = float(np.corrcoef(true_theta.ravel(), th_al.ravel())[0, 1])
    return corr, overall, mp

records = []
for L in LENGTHS:
    for rep in range(R):
        seed = 200 + rep
        dft, df, tw, lam, _ = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C,
            doc_topic_prior="logistic_normal", min_words=L, max_words=L, random_seed=seed)
        true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
        lamT = lam.T.astype(np.float64)
        vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
        mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}
        joint_corpus = Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3")
        naive_corpus = Corpus(df, modalities=mods)
        for cp in (joint_corpus, naive_corpus):
            m = cp.processed_modalities["text"]["bow"]["matrix"]
            if scipy.sparse.issparse(m):
                cp.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)
        X = joint_corpus.M_prevalence_covariates.astype(np.float64)
        c_or, _ = cmetrics(ols(X, clogit(true_theta)).T, lamT)
        t0 = time.time()

        jm = fit_mog(joint_corpus, prevalence=True, seed=seed)      # joint (covariate prior)
        c_jts, theta_corr, mp = theta_two_step(jm, joint_corpus, X, true_theta, lamT)
        W = jm.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
        W[:, 0] += jm.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
        c_jmn, s_jmn = cmetrics(np.stack([W[mp[t]] for t in range(K)]), lamT)

        nm = fit_mog(naive_corpus, prevalence=False, seed=seed)     # naive (no covariates)
        c_nts, theta_corr_n, _ = theta_two_step(nm, naive_corpus, X, true_theta, lamT)

        rec = {"L": L, "rep": rep, "joint_meannet": c_jmn, "joint_twostep": c_jts,
               "naive_twostep": c_nts, "oracle": c_or, "theta_corr": theta_corr,
               "gap_vs_naive": c_jmn - c_nts}
        records.append(rec)
        print(f"L={L:>3} rep{rep}: joint_meannet={c_jmn:+.3f} joint_2step={c_jts:+.3f} "
              f"NAIVE_2step={c_nts:+.3f} | gap(joint-naive)={c_jmn-c_nts:+.3f} | oracle={c_or:.3f} "
              f"theta(joint)={theta_corr:.2f} ({time.time()-t0:.0f}s)", flush=True)

print("\n=== JOINT vs TWO-STEP (covariate-naive stage 1) across document length ===")
print(f"{'words':>6} {'joint_meannet':>16} {'naive_twostep':>16} {'gap':>14} {'oracle':>8}")
summary = []
for L in LENGTHS:
    rs = [r for r in records if r["L"] == L]
    def ms(k): v = [r[k] for r in rs]; return float(np.mean(v)), float(np.std(v))
    jm_, jms = ms("joint_meannet"); nt_, nts = ms("naive_twostep"); gp, gps = ms("gap_vs_naive"); orc, _ = ms("oracle")
    jts_, _ = ms("joint_twostep")
    print(f"{L:>6} {jm_:>7.3f}±{jms:.3f}  {nt_:>7.3f}±{nts:.3f}  {gp:>+7.3f}±{gps:.3f}  {orc:>7.3f}", flush=True)
    summary.append({"L": L, "joint_meannet": jm_, "joint_meannet_sd": jms, "joint_twostep": jts_,
                    "naive_twostep": nt_, "naive_twostep_sd": nts, "gap": gp, "gap_sd": gps, "oracle": orc})
json.dump({"records": records, "summary": summary}, open("audit/short_docs_results.json", "w"), indent=2)

Ls = [s["L"] for s in summary]
fig, ax = plt.subplots(figsize=(7.5, 5))
for key, lab, mk in [("joint_meannet", "joint (mean_net)", "o-"),
                     ("joint_twostep", "joint, two-step on its theta", "^-"),
                     ("naive_twostep", "two-step (covariate-naive topics)", "s--"),
                     ("oracle", "oracle (ceiling)", "k:")]:
    y = [s[key] for s in summary]; sd = [s.get(key + "_sd", 0) for s in summary]
    ax.errorbar(Ls, y, yerr=sd, fmt=mk, capsize=3, label=lab)
ax.set_xscale("log"); ax.set_xticks(Ls); ax.set_xticklabels(Ls)
ax.set_xlabel("words per document"); ax.set_ylabel("covariate-coefficient recovery (corr)")
ax.set_title("Joint estimation vs two-step across document length")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("audit/short_docs_recovery.png", dpi=150)
print("\nsaved -> audit/short_docs_results.json, audit/short_docs_recovery.png", flush=True)
