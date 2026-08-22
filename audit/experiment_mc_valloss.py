"""Does the held-out VALIDATION loss (or its recon/KL parts) bottom at the c=1 sweet spot?

Train-loss patience was fragile (total loss ~flat while c overshoots 1.0->1.30). The overshoot is the PRIOR
overfitting, so the hope is that held-out RECON or KL (divergence) turns UP at the onset (~8k) even though
the tiny outcome term can't see it. We compute the package's validation loss DECOMPOSED (recon / KL / pred /
total) at checkpoints via one held-out forward pass (model.step_batch(test_batch, validation=True)) -- fast,
training runs at full speed (num_workers=4, val only at checkpoints). If a held-out component U-shapes at
~8k, the package's per-step val-loss patience would stop at the sweet spot.
NOTE (package quirk): validation appends true test labels to the encoder (models.py:683,714) -> the held-out
OUTCOME term can peek; recon/KL still reflect prior overfitting.
Config: locked [128] MoG-10, N=10k (9k train / 1k held-out), sigma=1.0, lr=5e-3, batch=1024, learned Sigma.
"""
import json, os, time, numpy as np, scipy.sparse, tempfile, torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, L = 10000, 5, 3, 200, 10
SIGMA, COMP, SEED = 1.0, 10, 1000
HIDDEN, LR, BATCH = [128], 5e-3, 1024
CKPTS = [3000, 5000, 7000, 8000, 9000, 11000, 14000, 18000]
N_TEST = 1000
OPTIM = {"main": {"lr": LR, "weight_decay": 0.0}, "prior": {"lr": 5e-4, "weight_decay": 0.0}}
PRED_ARGS = {"y": {"hidden_dims": [], "loss_weight": 1.0}}
OUT = "audit/results_mc_valloss.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda_fixed = np.random.default_rng(7).standard_normal((C + 1, K)) * 0.5
label_coeffs = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
true_b = label_coeffs - label_coeffs.mean()

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def center(v): return v - v.mean()

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
test_batch = next(iter(DataLoader(corpus_te, batch_size=N_TEST, shuffle=False, num_workers=0)))

def head_wb(model):
    lin = model.predictor.predictors["y"].neural_net["pred_0"]
    return (lin.weight.detach().cpu().numpy().astype(np.float64)[0],
            float(lin.bias.detach().cpu().numpy()[0]))
def _f(x): return x.item() if isinstance(x, torch.Tensor) else float(x)
def val_decomp(model):
    model.step_batch(test_batch, corpus_te, validation=True)      # held-out forward, sets loss components
    return (_f(model.loss), _f(model.reconstruction_loss), _f(model.divergence_loss), _f(model.prediction_loss))
def measure(model):
    th_tr = model.get_doc_topic_distribution(corpus_tr, num_samples=10).astype(np.float64)
    perm = [align(tt_tr, th_tr)[t] for t in range(K)]
    W, b0 = head_wb(model)
    bh = center(np.array([W[perm[t]] for t in range(K)]))
    c = float((bh @ true_b) / (true_b @ true_b)); mab = float(np.abs(bh - true_b).mean())
    th_te = model.get_doc_topic_distribution(corpus_te, num_samples=10).astype(np.float64)
    mse = float(np.mean((y_te - (th_te @ W + b0)) ** 2))
    return c, mab, mse

print(f"VAL-LOSS decomposition vs c | N={N} ({N-N_TEST} tr/{N_TEST} held-out) sigma={SIGMA} [128] MoG-{COMP} "
      f"lr={LR} batch={BATCH}")
print("does held-out total / recon / KL bottom at the c~1 sweet spot (~8k)?\n", flush=True)

model = GTM(train_data=corpus_tr, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
            mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
            learn_prior_cov=True, labels_in_encoder=True, predictor_args=PRED_ARGS,
            encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
            batch_size=BATCH, num_steps=CKPTS[0], num_workers=4, print_every_n_steps=10**9,
            optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)

t0 = time.time(); rows = []
for i, cp in enumerate(CKPTS):
    if i > 0:
        model.num_steps = cp; model.train(corpus_tr)
    c, mab, mse = measure(model)
    vtot, vrec, vkl, vpred = val_decomp(model)
    rows.append({"step": cp, "c": c, "mab": mab, "test_mse": mse,
                 "val_total": vtot, "val_recon": vrec, "val_KL": vkl, "val_pred": vpred})
    print(f"  [{cp:>6}] c={c:.3f} mab={mab:.3f} mse={mse:.4f} | val_tot={vtot:.4f} "
          f"recon={vrec:.4f} KL={vkl:.4f} pred={vpred:.4f}  ({time.time()-t0:.0f}s)", flush=True)

json.dump({"config": {"N": N, "sigma": SIGMA, "lr": LR, "batch": BATCH}, "true_b": true_b.tolist(),
           "rows": rows}, open(OUT, "w"), indent=2)

ss = min(rows, key=lambda r: r["mab"])  # the actual sweet spot (min |bias|)
print(f"\nsweet spot (min |bias|): step {ss['step']} (c={ss['c']:.3f}, mab={ss['mab']:.3f})")
for key in ["val_total", "val_recon", "val_KL", "val_pred", "test_mse"]:
    m = min(rows, key=lambda r: r[key])
    flag = "<-- matches sweet spot" if abs(m["step"] - ss["step"]) <= 1500 else ""
    print(f"  min {key:>10} @ step {m['step']:>6} (c~{m['c']:.3f})  {flag}", flush=True)
print(f"saved -> {OUT}", flush=True)
