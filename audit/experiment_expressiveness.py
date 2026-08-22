"""Test 2: is the ~0.10 coefficient-bias floor (10 words/doc, N=10000) the ELBO/amortization
gap? Climb a ladder of variational expressiveness (encoder width/depth + MoG components),
everything else fixed, and see whether mean|bias| drops below the baseline floor and
theta_corr rises above ~0.77.

Baseline (from experiment_joint_long.py, encoder [128], MoG-10, same data/seed):
    step      theta_corr  mean|bias|
    20000     0.713       0.222
    40000     0.761       0.121
    60000     0.774       0.101   <- floor

If a bigger encoder / richer mixture pushes mean|bias| well below ~0.10 and theta_corr
above ~0.77, the floor is the variational gap. If nothing moves, the floor is statistical
(finite N) -> points to the N-scaling test instead."""
import json, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, SEED, L = 10000, 5, 3, 200, 100, 10
CHECKPOINTS = [20000, 40000, 60000]
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0},
         "prior": {"lr": 1e-4, "weight_decay": 0.0}}
CONFIGS = [
    {"name": "enc256x2_mog20", "hidden": [256, 256], "comp": 20},
    {"name": "enc512x2_mog40", "hidden": [512, 512], "comp": 40},
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def clogit(theta):
    lg = np.log(np.clip(theta, 1e-8, 1.0)); return lg - lg.mean(1, keepdims=True)
def center_rows(M): return M - M.mean(0, keepdims=True)

dft, df, tw, lam, _ = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C,
    doc_topic_prior="logistic_normal", min_words=L, max_words=L, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
Lc = center_rows(lam.T.astype(np.float64))
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
corpus = Corpus(df, modalities={"text": {"column": "doc_clean_0",
                "views": {"bow": {"type": "bow", "vectorizer": vec}}}}, prevalence="~ cov_1 + cov_2 + cov_3")
m = corpus.processed_modalities["text"]["bow"]["matrix"]
if scipy.sparse.issparse(m):
    corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)

def evaluate(model, step, t0):
    th = model.get_doc_topic_distribution(corpus, num_samples=20).astype(np.float64)
    mp = align(true_theta, th)
    th_al = np.stack([th[:, mp[t]] for t in range(K)], axis=1)
    overall = float(np.corrcoef(true_theta.ravel(), th_al.ravel())[0, 1])
    W = model.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
    W[:, 0] += model.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
    Wc = center_rows(np.stack([W[mp[t]] for t in range(K)]))
    bias = (Wc - Lc)[:, 1:]
    mab = float(np.abs(bias).mean())
    recon = np.mean(model.train_recon_losses[-100:]); kl = np.mean(model.train_div_losses[-100:])
    print(f"    [{step:>6}] recon={recon:6.2f} KL={kl:5.2f} | theta_corr={overall:.3f} | "
          f"mean|bias|={mab:.3f} (max={np.abs(bias).max():.3f})  ({time.time()-t0:.0f}s)", flush=True)
    return {"step": step, "theta_corr": overall, "mean_abs_bias": mab, "max_abs_bias": float(np.abs(bias).max())}

out = {}
for cfg in CONFIGS:
    print(f"\n=== {cfg['name']}: encoder {cfg['hidden']}, MoG-{cfg['comp']} | L={L}, N={N} ===", flush=True)
    t0 = time.time(); traj = []
    model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
                mixture_components=cfg["comp"], doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
                encoder_args={"text_bow": {"hidden_dims": cfg["hidden"]}},
                decoder_args={"text_bow": {"hidden_dims": []}},
                batch_size=256, num_steps=CHECKPOINTS[0], num_workers=0, print_every_n_steps=10**9,
                optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
    traj.append(evaluate(model, CHECKPOINTS[0], t0))
    for cp in CHECKPOINTS[1:]:
        model.num_steps = cp; model.train(corpus)
        traj.append(evaluate(model, cp, t0))
        out[cfg["name"]] = traj
        json.dump(out, open("audit/results_expressiveness.json", "w"), indent=2)

print("\n=== expressiveness ladder vs baseline (10 words/doc, at 60k steps) ===")
print(f"{'config':>22} {'theta_corr':>11} {'mean|bias|':>11} {'max|bias|':>10}")
print(f"{'baseline enc128_mog10':>22} {0.774:>11.3f} {0.101:>11.3f} {0.388:>10.3f}")
for cfg in CONFIGS:
    r = out[cfg["name"]][-1]
    print(f"{cfg['name']:>22} {r['theta_corr']:>11.3f} {r['mean_abs_bias']:>11.3f} {r['max_abs_bias']:>10.3f}", flush=True)
print("\nsaved -> audit/results_expressiveness.json", flush=True)
