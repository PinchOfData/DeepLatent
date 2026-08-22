"""Measure the BIAS of the joint topic->outcome coefficient estimator across realistic outcome noise.

This supersedes the earlier pinned-Sigma "amortization-gap" hypothesis, which was FALSIFIED by directly
measuring the supervised variational gap (audit/experiment_supervised_gap.py): at sigma=1.0 with LEARNED
Sigma the bound is tight (0.083 nats), per-doc refinement does not move the coefficient, and c=0.985 --
the estimator is essentially unbiased. The c=0.65-0.78 "attenuation" seen earlier was an ARTIFACT of
pinning Sigma=I (oracle knowledge of Sigma_true=I that also biases the outcome-coef scale).

So here we report the actual bias mean|b_hat - b_true| (and per-coefficient b_hat vs b_true) with
Sigma LEARNED (no oracle), across a realistic outcome-noise range sigma in {1.0, 0.5, 0.2}
(R^2 ~ 0.4..0.9). sigma=0.05 is omitted: near-deterministic y destabilizes the learned-Sigma model
(temperature collapse / topic sign-flip), an artificial edge case.

Same docs/topics across the sweep (x is sigma-independent); only the outcome noise changes:
    y = theta_true @ label_coeffs + N(0, sigma^2).
Joint model: labels_in_encoder=True, linear head, learned sigma^2 (noise_log_var), learned Sigma.
b_hat = the linear head's weight, Hungarian-aligned to true topics and centered (simplex identification).
"""
import json, os, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

SMOKE = os.environ.get("SIGMA_SMOKE", "") == "1"
N, K, C, VOCAB, L = (3000 if SMOKE else 100000), 5, 3, 200, 10
HIDDEN, COMP = [256, 256], 20
CHECKPOINTS = [400, 800] if SMOKE else [20000, 40000, 60000]
SIGMA_SWEEP = [1.0, 0.5, 0.2]   # realistic outcome noise (1.0 first). 0.05 omitted: near-deterministic y, learned-Sigma unstable
SEED = 1000
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0}, "prior": {"lr": 1e-4, "weight_decay": 0.0}}
PRED_ARGS = {"y": {"hidden_dims": [], "loss_weight": 1.0}}
OUT = "audit/results_bias_learnedcov.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda_fixed = np.random.default_rng(7).standard_normal((C + 1, K)) * 0.5
label_coeffs = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
true_b = label_coeffs - label_coeffs.mean()

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def center(v): return v - v.mean()
def fit_stats(b):
    mab = float(np.abs(b - true_b).mean()); slope = float((b @ true_b) / (true_b @ true_b))
    corr = float(np.corrcoef(b, true_b)[0, 1]); unexpl = float(np.sum((b - slope * true_b) ** 2) / np.sum(b ** 2))
    return {"mab": mab, "slope": slope, "corr": corr, "unexpl": unexpl}

# data: x and topics are sigma-independent; only y noise changes across the sweep
dft, df, tw, lam, lc = generate_documents(
    num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C, doc_topic_prior="logistic_normal",
    min_words=L, max_words=L, lambda_=lambda_fixed,
    label_type="regression", label_coeffs=label_coeffs, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
mu_y = true_theta @ label_coeffs            # noiseless outcome mean
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}

def make_corpus(y):
    df2 = df.copy(); df2["label"] = y
    c = Corpus(df2, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3",
               labels={"y": {"column": "label", "type": "regression"}})
    mm = c.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(mm):
        c.processed_modalities["text"]["bow"]["matrix"] = np.asarray(mm.todense(), np.float32)
    return c

print(f"outcome-coef BIAS | N={N}, {L} w/doc, {HIDDEN} MoG-20 | Sigma LEARNED (no oracle pin) | true_b={true_b.round(2).tolist()}")
print("measuring mean|b_hat - b_true| of the joint estimator across realistic outcome-noise sigma.\n", flush=True)

results = {"meta": {"true_b": true_b.tolist(), "sigma_sweep": SIGMA_SWEEP, "checkpoints": CHECKPOINTS}, "runs": {}}

def measure(model, corpus, step, sig, t0):
    th = model.get_doc_topic_distribution(corpus, num_samples=10).astype(np.float64)
    perm = [align(true_theta, th)[t] for t in range(K)]
    W = model.predictor.predictors["y"].neural_net["pred_0"].weight.detach().cpu().numpy().astype(np.float64)
    bh = center(np.array([W[0, perm[t]] for t in range(K)]))   # aligned + centered recovered coefficients
    fs = fit_stats(bh)
    nv = float(torch.exp(model.predictor.noise_log_var["y"]).detach().cpu())
    print(f"  [sig={sig:.2f}|{step:>6}] mean|bias|={fs['mab']:.3f} max|bias|={np.abs(bh-true_b).max():.3f} "
          f"c={fs['slope']:.3f} corr={fs['corr']:.3f} | sig2_hat={nv:.4f} (true {sig**2:.4f})  ({time.time()-t0:.0f}s)", flush=True)
    return {"step": step, **fs, "b_hat": bh.tolist(), "sigma2_hat": nv}

for sig in SIGMA_SWEEP:
    rng = np.random.default_rng(2024 + int(sig * 1000))
    y = mu_y + rng.normal(0, sig, N)
    corpus = make_corpus(y)
    print(f"=== sigma={sig} (sig^2={sig**2:.4f}) ===", flush=True)
    t0 = time.time()
    model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
                mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
                learn_prior_cov=True, labels_in_encoder=True, predictor_args=PRED_ARGS,
                encoder_args={"text_bow": {"hidden_dims": HIDDEN}}, decoder_args={"text_bow": {"hidden_dims": []}},
                batch_size=256, num_steps=CHECKPOINTS[0], num_workers=4, print_every_n_steps=10**9,
                optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
    traj = [measure(model, corpus, CHECKPOINTS[0], sig, t0)]
    for cp in CHECKPOINTS[1:]:
        model.num_steps = cp; model.train(corpus)
        traj.append(measure(model, corpus, cp, sig, t0))
    results["runs"][str(sig)] = traj
    json.dump(results, open(OUT, "w"), indent=2)
    print(flush=True)

print("=== outcome-coef BIAS (learned Sigma), final checkpoint ===")
print(f"true_b = {true_b.round(3).tolist()}")
print(f"{'sigma':>6} {'sig2_hat':>9} {'mean|bias|':>10} {'max|bias|':>9} {'c':>7} {'corr':>7}")
for sig in SIGMA_SWEEP:
    r = results["runs"][str(sig)][-1]
    bh = np.array(r["b_hat"])
    print(f"{sig:>6} {r['sigma2_hat']:>9.4f} {r['mab']:>10.3f} {np.abs(bh-true_b).max():>9.3f} "
          f"{r['slope']:>7.3f} {r['corr']:>7.3f}", flush=True)
print("\nper-coefficient b_hat vs true_b (final checkpoint):")
for sig in SIGMA_SWEEP:
    bh = np.array(results["runs"][str(sig)][-1]["b_hat"])
    print(f"  sigma={sig}: b_hat={bh.round(3).tolist()}  bias={(bh-true_b).round(3).tolist()}", flush=True)
print(f"saved -> {OUT}", flush=True)
