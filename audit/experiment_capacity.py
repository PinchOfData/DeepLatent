"""Does richer inference (wider encoder + more MoG components) drive the JOINT
prevalence-coefficient bias toward 0? Tests the variational-gap hypothesis: if the
residual ~0.024 bias of single100k is the amortization/approximation gap (not an
irreducible floor), then more encoder capacity + a richer posterior should shrink it.

Joint estimator ONLY (covariates in the prior, update_prior=True). No two-step, no
full_rank, no IAF -- to avoid wasting time. Identification = Hungarian topic-alignment
+ per-covariate centering, same as single100k. The coefficient is read directly from
prior.mean_net (the jointly-learned logit-space prevalence regression), so it is the
approximate joint-MLE coefficient -- no plug-in/clogit step.

Data is FIXED and identical to single100k (rng(7) lambda, seed=1000, N=100000,
10 words/doc) so bias here is directly comparable to that run's [256,256] baseline
(bias-min 0.024 @ ~55-60k).

Configs (encoder [512,512] for both):
  MoG-20 : 100000 steps
  MoG-80 : 200000 steps  (more components -> more params -> trained LONGER to converge;
                          the components run's 80k budget under-converged MoG-80)
eval/print every 5000 steps. Resumable: skips (comp, step) pairs already in the JSON."""
import json, os, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, L = 100000, 5, 3, 200, 10
HIDDEN = [512, 512]
CONFIGS = [(20, 100000), (80, 200000)]       # (mixture_components, num_steps)
EVAL_EVERY = 5000
SEED = 1000                                  # == single100k / mc20 rep 0
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0},
         "prior": {"lr": 1e-4, "weight_decay": 0.0}}
OUT = "audit/results_capacity.json"
BASELINE = "[256,256] MoG-20: bias-min 0.024 @ ~55-60k (single100k)"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIXED true coefficients lambda (shape [C+1, K]); identical to single100k / mc20.
lambda_fixed = (np.random.default_rng(7).standard_normal((C + 1, K)) * 0.5)
Lc = (lambda_fixed.T - lambda_fixed.T.mean(0, keepdims=True))   # identified true coeffs [K, C+1]
true_c = Lc[:, 1:]                                              # [K, C]

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def center_rows(M): return M - M.mean(0, keepdims=True)

def fit(corpus, comp, num_steps):
    return GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
               mixture_components=comp, doc_topic_prior="logistic_normal",
               update_prior=True, w_prior=1.0,
               encoder_args={"text_bow": {"hidden_dims": HIDDEN}},
               decoder_args={"text_bow": {"hidden_dims": []}},
               batch_size=256, num_steps=num_steps, num_workers=4, print_every_n_steps=10**9,
               optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(),
               seed=SEED, device=device)

def evaluate(model, corpus, true_theta, step, t0):
    th = model.get_doc_topic_distribution(corpus, num_samples=10).astype(np.float64)
    mp = align(true_theta, th)
    th_al = np.stack([th[:, mp[t]] for t in range(K)], axis=1)
    theta_corr = float(np.corrcoef(true_theta.ravel(), th_al.ravel())[0, 1])
    W = model.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
    W[:, 0] += model.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
    Wc = center_rows(np.stack([W[mp[t]] for t in range(K)]))
    bias = (Wc - Lc)[:, 1:]; mab = float(np.abs(bias).mean()); mx = float(np.abs(bias).max())
    recon = float(np.mean(model.train_recon_losses[-100:])); kl = float(np.mean(model.train_div_losses[-100:]))
    print(f"    [{step:>6}] recon={recon:6.2f} KL={kl:5.2f} | theta_corr={theta_corr:.3f} | "
          f"mean|bias|={mab:.3f} (max={mx:.3f})  ({time.time()-t0:.0f}s)", flush=True)
    return {"step": step, "theta_corr": theta_corr, "mean_abs_bias": mab,
            "max_abs_bias": mx, "coef": Wc[:, 1:].tolist()}

# ---------------- data (single rep, shared across configs) ----------------
print(f"encoder {HIDDEN} | {L} words/doc, N={N}, seed={SEED} | configs {CONFIGS}", flush=True)
print(f"baseline -> {BASELINE}\n", flush=True)
t_data = time.time()
dft, df, tw, lam, _ = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C,
    doc_topic_prior="logistic_normal", min_words=L, max_words=L, lambda_=lambda_fixed, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
corpus = Corpus(df, modalities={"text": {"column": "doc_clean_0",
                "views": {"bow": {"type": "bow", "vectorizer": vec}}}}, prevalence="~ cov_1 + cov_2 + cov_3")
m = corpus.processed_modalities["text"]["bow"]["matrix"]
if scipy.sparse.issparse(m):
    corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)
print(f"data ready ({time.time()-t_data:.0f}s)", flush=True)

results = json.load(open(OUT)) if os.path.exists(OUT) else {}

# ---------------- joint fits per config ----------------
for comp, steps in CONFIGS:
    key = str(comp)
    traj = results.get(key, [])
    done = {r["step"] for r in traj}
    checkpoints = list(range(EVAL_EVERY, steps + 1, EVAL_EVERY))
    if done >= set(checkpoints):
        print(f"=== MoG-{comp} [{','.join(map(str,HIDDEN))}] already complete, skipping ===", flush=True)
        continue
    print(f"=== MoG-{comp}, encoder {HIDDEN} | {steps} steps (eval every {EVAL_EVERY}) ===", flush=True)
    t0 = time.time()
    model = fit(corpus, comp, checkpoints[0])
    for i, step in enumerate(checkpoints):
        if i > 0:
            model.num_steps = step; model.train(corpus)
        if step in done:
            continue
        traj.append(evaluate(model, corpus, true_theta, step, t0))
        results[key] = traj
        json.dump(results, open(OUT, "w"), indent=2)
    del model; torch.cuda.empty_cache()

# ---------------- report ----------------
print(f"\n=== JOINT coefficient recovery vs inference richness (N={N}, 10 words/doc) ===")
print(f"reference {BASELINE}")
print(f"{'config':>22} {'best step':>9} {'min mean|bias|':>14} {'theta_corr@min':>15} {'final mean|bias|':>16}")
for comp, steps in CONFIGS:
    traj = results.get(str(comp), [])
    if not traj:
        continue
    best = min(traj, key=lambda r: r["mean_abs_bias"])
    final = max(traj, key=lambda r: r["step"])
    print(f"{'[512,512] MoG-'+str(comp):>22} {best['step']:>9} {best['mean_abs_bias']:>14.3f} "
          f"{best['theta_corr']:>15.3f} {final['mean_abs_bias']:>16.3f}", flush=True)
print("\nsaved -> " + OUT, flush=True)
