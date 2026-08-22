"""Does the TRAINING loss let patience early-stopping land on the coefficient sweet spot (c=1)?

At sigma=1.0 the joint coef is U-shaped in training time: c=0.82(4k)->1.01(8k)->1.09(10k)->1.28(16k);
the sweet spot (c~1, min |bias|) is a SHARP peak at ~8k steps. We want an AUTOMATIC stop there.
Per the design: NO validation set -- patience early-stops on the TRAINING loss directly (models.py:589).
So we test: does the training ELBO stop improving (plateau) at ~8k, so native patience halts at the
sweet spot? We log, vs steps:
  - training ELBO loss (per step)         -> simulate native patience rule (improve if < best-1e-3)
  - held-out PREDICTIVE MSE at checkpoints -> inference theta on a 1k held-out split (encoder y=0),
                                             apply learned head, MSE vs true y. GROUND-TRUTH marker of
                                             the sweet spot (min MSE <=> c~1); not used for stopping.
  - c (coef slope) at checkpoints          -> the bias itself.
Config: locked [128] MoG-10, N=10k (9k train / 1k held-out), sigma=1.0, lr=5e-3, batch=1024, learned Sigma.
"""
import json, os, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, L = 10000, 5, 3, 200, 10
SIGMA, COMP, SEED = 1.0, 10, 1000
HIDDEN, LR, BATCH = [128], 5e-3, 1024
CKPTS = [4000, 6000, 8000, 10000, 12000, 16000]
N_TEST = 1000
PATIENCES = [200, 500, 1000, 2000, 4000]
OPTIM = {"main": {"lr": LR, "weight_decay": 0.0}, "prior": {"lr": 5e-4, "weight_decay": 0.0}}
PRED_ARGS = {"y": {"hidden_dims": [], "loss_weight": 1.0}}
OUT = "audit/results_mc_loss.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda_fixed = np.random.default_rng(7).standard_normal((C + 1, K)) * 0.5
label_coeffs = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
true_b = label_coeffs - label_coeffs.mean()

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def center(v): return v - v.mean()

# --- one DGP draw at sigma=1.0, split train / held-out ---
dft, df, tw, lam, lc = generate_documents(
    num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C, doc_topic_prior="logistic_normal",
    min_words=L, max_words=L, lambda_=lambda_fixed,
    label_type="regression", label_coeffs=label_coeffs, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
mu_y = true_theta @ label_coeffs
y = mu_y + np.random.default_rng(SEED + 999).normal(0, SIGMA, N)
df = df.copy(); df["label"] = y

perm_idx = np.random.default_rng(123).permutation(N)
te, tr = perm_idx[:N_TEST], perm_idx[N_TEST:]
df_tr = df.iloc[tr].reset_index(drop=True); df_te = df.iloc[te].reset_index(drop=True)
tt_tr = true_theta[tr]; y_te = y[te].astype(np.float64)

vec = CountVectorizer(); vec.fit(df_tr["doc_clean_0"])
mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}
def mk(dfx):
    c = Corpus(dfx, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3",
               labels={"y": {"column": "label", "type": "regression"}})
    mm = c.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(mm):
        c.processed_modalities["text"]["bow"]["matrix"] = np.asarray(mm.todense(), np.float32)
    return c
corpus_tr, corpus_te = mk(df_tr), mk(df_te)

def head_wb(model):
    lin = model.predictor.predictors["y"].neural_net["pred_0"]
    return (lin.weight.detach().cpu().numpy().astype(np.float64)[0],
            float(lin.bias.detach().cpu().numpy()[0]))

def measure(model):
    th_tr = model.get_doc_topic_distribution(corpus_tr, num_samples=10).astype(np.float64)
    perm = [align(tt_tr, th_tr)[t] for t in range(K)]
    W, b0 = head_wb(model)
    bh = center(np.array([W[perm[t]] for t in range(K)]))
    c = float((bh @ true_b) / (true_b @ true_b)); mab = float(np.abs(bh - true_b).mean())
    th_te = model.get_doc_topic_distribution(corpus_te, num_samples=10).astype(np.float64)
    yhat = th_te @ W + b0
    return c, mab, float(np.mean((y_te - yhat) ** 2))

print(f"loss-vs-bias | N={N} ({N-N_TEST} tr/{N_TEST} held-out) sigma={SIGMA} [128] MoG-{COMP} "
      f"lr={LR} batch={BATCH} | NO val set: patience stops on TRAIN loss")
print("question: does train-loss patience halt at the c~1 sweet spot (~8k)?\n", flush=True)

model = GTM(train_data=corpus_tr, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
            mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
            learn_prior_cov=True, labels_in_encoder=True, predictor_args=PRED_ARGS,
            encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
            batch_size=BATCH, num_steps=CKPTS[0], num_workers=4, print_every_n_steps=10**9,
            optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)

def sm(arr, w=200):
    a = np.array(arr[-w:], np.float64); return float(a.mean()) if len(a) else float("nan")

t0 = time.time(); rows = []
for i, cp in enumerate(CKPTS):
    if i > 0:
        model.num_steps = cp; model.train(corpus_tr)
    c, mab, mse = measure(model)
    rows.append({"step": cp, "c": c, "mab": mab, "test_mse": mse, "train_loss": sm(model.train_losses)})
    print(f"  [{cp:>6}] c={c:.3f} mab={mab:.3f} | heldout_MSE={mse:.4f} | "
          f"train_loss(sm)={sm(model.train_losses):.4f}  ({time.time()-t0:.0f}s)", flush=True)

# --- simulate native patience early-stop on the raw per-step TRAIN loss (rule: improve if < best-1e-3) ---
def patience_stop(losses, patience, tol=1e-3):
    best, counter = np.inf, 0
    for s, l in enumerate(losses):
        if l < best - tol:
            best, counter = l, 0
        else:
            counter += 1
            if counter >= patience:
                return s + 1
    return len(losses)
def c_at(step):  # linear interp of c across checkpoints
    xs = [r["step"] for r in rows]; cs = [r["c"] for r in rows]
    return float(np.interp(step, xs, cs))

mse_min = min(rows, key=lambda r: r["test_mse"])
print(f"\nground truth: held-out MSE min @ {mse_min['step']} (c={mse_min['c']:.3f}, mab={mse_min['mab']:.3f}) "
      f"-- the sweet spot.")
print("simulated patience early-stop on TRAINING loss (tol=1e-3):")
pat = {}
for p in PATIENCES:
    s = patience_stop(model.train_losses, p)
    pat[p] = {"stop_step": s, "c_at_stop": c_at(s)}
    print(f"  patience={p:>5}: stop @ step {s:>6}  ->  c~{c_at(s):.3f}", flush=True)

json.dump({"config": {"N": N, "sigma": SIGMA, "lr": LR, "batch": BATCH}, "true_b": true_b.tolist(),
           "rows": rows, "mse_min_step": mse_min["step"], "patience": pat,
           "train_losses": model.train_losses}, open(OUT, "w"))
print(f"\n=> if patience stop-steps cluster near {mse_min['step']} (c~1), train-loss patience WORKS; "
      f"if they run long (c>1), it does NOT and we need the held-out MSE.")
print(f"saved -> {OUT}", flush=True)
