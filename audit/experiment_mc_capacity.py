"""Is the small-N joint leakage an ENCODER-CAPACITY problem (memorization) or a direct y-CHANNEL problem?

At N=10k the joint estimator leaked: sigma2_hat collapsed to 0.08 << 1.0 (true noise floor), proving the
encoder peeks at y, and the recovered coef attenuated to c=0.56 (worse than two-step). Hypothesis under test
(user's): the [128,128] encoder simply has too much capacity for 10k obs and memorizes y. Competing mechanism:
labels_in_encoder feeds y as a DIRECT input, so even a linear encoder can leak via one weight W_y, gated only
by the recon/KL regularization (weak here: L=10 word docs barely constrain theta).

TEST: same DGP (N=10k, sigma=1.0), sweep encoder depth DOWN, JOINT arm only, watch sigma2_hat and c.
  capacity story  => sigma2_hat -> 1.0 and c -> 1 as the encoder shrinks.
  channel story   => sigma2_hat stays collapsed even for a linear encoder.
"""
import json, os, time, numpy as np, scipy.sparse, tempfile, torch
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

N, K, C, VOCAB, L = 10000, 5, 3, 200, 10
SIGMA, COMP, SEED = 1.0, 10, 1000
HIDDENS = [[], [16], [32], [64], [128, 128]]      # linear -> deep
CHECKPOINTS = [6000, 12000]
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0}, "prior": {"lr": 1e-4, "weight_decay": 0.0}}
PRED_ARGS = {"y": {"hidden_dims": [], "loss_weight": 1.0}}
OUT = "audit/results_mc_capacity.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda_fixed = np.random.default_rng(7).standard_normal((C + 1, K)) * 0.5
label_coeffs = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
true_b = label_coeffs - label_coeffs.mean()

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def center(v): return v - v.mean()
def stats(b):
    return (float(np.abs(b - true_b).mean()), float((b @ true_b) / (true_b @ true_b)),
            float(np.corrcoef(b, true_b)[0, 1]))

# one fixed DGP draw, shared across encoder sizes (only the encoder changes)
dft, df, tw, lam, lc = generate_documents(
    num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C, doc_topic_prior="logistic_normal",
    min_words=L, max_words=L, lambda_=lambda_fixed,
    label_type="regression", label_coeffs=label_coeffs, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
y = df["label"].values.astype(np.float64)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}
corpus = Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3",
                labels={"y": {"column": "label", "type": "regression"}})
mm = corpus.processed_modalities["text"]["bow"]["matrix"]
if scipy.sparse.issparse(mm):
    corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(mm.todense(), np.float32)

ob, *_ = np.linalg.lstsq(true_theta, y, rcond=None)
o_mab, o_c, _ = stats(center(ob))
print(f"capacity sweep | N={N} sigma={SIGMA} MoG-{COMP} | true_b={true_b.round(2).tolist()} | "
      f"oracle OLS c={o_c:.3f} mab={o_mab:.3f}")
print(f"leakage signature = sigma2_hat << {SIGMA**2:.2f} (true noise floor). watch it vs encoder size.\n", flush=True)

def enc_params(model):
    return sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)

def measure(model, step, hid, t0, ep):
    th = model.get_doc_topic_distribution(corpus, num_samples=10).astype(np.float64)
    perm = [align(true_theta, th)[t] for t in range(K)]
    W = model.predictor.predictors["y"].neural_net["pred_0"].weight.detach().cpu().numpy().astype(np.float64)
    bh = center(np.array([W[0, perm[t]] for t in range(K)]))
    mab, c, corr = stats(bh)
    nv = float(torch.exp(model.predictor.noise_log_var["y"]).detach().cpu())
    print(f"  [{str(hid):>11}|{step:>6}] sig2_hat={nv:.3f} (true {SIGMA**2:.2f})  c={c:.3f} mab={mab:.3f} "
          f"corr={corr:.3f} | enc_params={ep} ({ep/N:.1f}/obs)  ({time.time()-t0:.0f}s)", flush=True)
    return {"step": step, "sigma2_hat": nv, "c": c, "mab": mab, "corr": corr, "b_hat": bh.tolist()}

results = {"meta": {"N": N, "sigma": SIGMA, "true_b": true_b.tolist(), "oracle_c": o_c}, "runs": {}}
for hid in HIDDENS:
    t0 = time.time()
    model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
                mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
                learn_prior_cov=True, labels_in_encoder=True, predictor_args=PRED_ARGS,
                encoder_args={"text_bow": {"hidden_dims": hid}}, decoder_args={"text_bow": {"hidden_dims": []}},
                batch_size=256, num_steps=CHECKPOINTS[0], num_workers=4, print_every_n_steps=10**9,
                optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
    ep = enc_params(model)
    traj = [measure(model, CHECKPOINTS[0], hid, t0, ep)]
    for cp in CHECKPOINTS[1:]:
        model.num_steps = cp; model.train(corpus)
        traj.append(measure(model, cp, hid, t0, ep))
    results["runs"][str(hid)] = {"enc_params": ep, "traj": traj}
    json.dump(results, open(OUT, "w"), indent=2)
    print(flush=True)

print("=== encoder capacity vs leakage (final checkpoint) ===")
print(f"{'hidden':>12} {'enc_params':>10} {'/obs':>6} {'sig2_hat':>9} {'c':>7} {'mab':>7}")
for hid in HIDDENS:
    r = results["runs"][str(hid)]; f = r["traj"][-1]
    print(f"{str(hid):>12} {r['enc_params']:>10} {r['enc_params']/N:>6.1f} {f['sigma2_hat']:>9.3f} "
          f"{f['c']:>7.3f} {f['mab']:>7.3f}", flush=True)
print(f"\noracle c={o_c:.3f}. capacity story => sig2_hat rises to ~{SIGMA**2:.2f} & c->1 as encoder shrinks.")
print(f"saved -> {OUT}", flush=True)
