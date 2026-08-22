"""Does scaling MoG components (tightening the variational bound) remove the residual
~0.05 coefficient-bias plateau, instead of IWAE? Hold encoder FIXED at [256,256] and the
data FIXED (N=30000, 10 words/doc), scale mixture_components 20 -> 80, train to convergence.

Reference (from experiment_nscaling.py, same data/seed, MoG-20 @ 60k): mean|bias|=0.049.
If mean|bias| drops below ~0.05 as components grow, a richer q tightens the bound and
reduces the variational bias -- the components lever works (no IWAE needed).
encoder [256,256], prior lr=1e-4 wd=0.0, update_prior=True."""
import json, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, SEED, L = 30000, 5, 3, 200, 100, 10
CHECKPOINTS = [40000, 60000, 80000]
HIDDEN = [256, 256]
COMPONENTS = [20, 80]
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0},
         "prior": {"lr": 1e-4, "weight_decay": 0.0}}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
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
print(f"N={N}, {L} words/doc, K={K} | encoder {HIDDEN} fixed | scaling MoG components {COMPONENTS}\n", flush=True)

def evaluate(model, step, t0):
    th = model.get_doc_topic_distribution(corpus, num_samples=10).astype(np.float64)
    mp = align(true_theta, th)
    th_al = np.stack([th[:, mp[t]] for t in range(K)], axis=1)
    overall = float(np.corrcoef(true_theta.ravel(), th_al.ravel())[0, 1])
    W = model.prior.mean_net.weight.detach().cpu().numpy().astype(np.float64).copy()
    W[:, 0] += model.prior.mean_net.bias.detach().cpu().numpy().astype(np.float64)
    Wc = center_rows(np.stack([W[mp[t]] for t in range(K)]))
    bias = (Wc - Lc)[:, 1:]; mab = float(np.abs(bias).mean())
    recon = np.mean(model.train_recon_losses[-100:]); kl = np.mean(model.train_div_losses[-100:])
    print(f"    [{step:>6}] recon={recon:6.2f} KL={kl:5.2f} | theta_corr={overall:.3f} | "
          f"mean|bias|={mab:.3f} (max={np.abs(bias).max():.3f})  ({time.time()-t0:.0f}s)", flush=True)
    return {"step": step, "theta_corr": overall, "mean_abs_bias": mab, "max_abs_bias": float(np.abs(bias).max())}

out = {}
for comp in COMPONENTS:
    print(f"=== MoG-{comp}, encoder {HIDDEN} | N={N}, {L} words/doc ===", flush=True)
    t0 = time.time(); traj = []
    model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
                mixture_components=comp, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
                encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
                batch_size=256, num_steps=CHECKPOINTS[0], num_workers=0, print_every_n_steps=10**9,
                optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
    traj.append(evaluate(model, CHECKPOINTS[0], t0))
    for cp in CHECKPOINTS[1:]:
        model.num_steps = cp; model.train(corpus)
        traj.append(evaluate(model, cp, t0))
        out[str(comp)] = traj
        json.dump(out, open("audit/results_components.json", "w"), indent=2)
    del model; torch.cuda.empty_cache()

print("\n=== coefficient bias vs MoG components (N=30000, 10 words/doc, converged) ===")
print(f"{'components':>11} {'theta_corr':>11} {'mean|bias|':>11} {'max|bias|':>10}")
print(f"{'(ref) 20@60k':>11} {0.796:>11.3f} {0.049:>11.3f} {0.143:>10.3f}")
for comp in COMPONENTS:
    r = out[str(comp)][-1]
    print(f"{comp:>11} {r['theta_corr']:>11.3f} {r['mean_abs_bias']:>11.3f} {r['max_abs_bias']:>10.3f}", flush=True)
print("\nsaved -> audit/results_components.json", flush=True)
