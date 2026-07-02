"""Does ENCODER DROPOUT slow the temperature slide (prior-Sigma deflation + coefficient overshoot)?

Mechanism under test (user's suggestion): the slide is PRIOR overfitting, but the prior overfits to
the AGGREGATE of the encoder's posteriors. Hidden-layer dropout keeps q(z|x) jittered/wider during
training -> the aggregated posterior stays wider -> Sigma-hat has less to tighten around -> the slide
should slow (or its sweet spot shift later). Also the ProdLDA folklore (Srivastava & Sutton 2017:
high lr + momentum + dropout against topic-VAE collapse) finally gets a mechanistic test here.

Setting: C=25 words/doc, N=100k — the regime where the slide is fastest (fixed 60k steps landed at
Sigma-hat ~0.36, head c=1.313, snis_x c=1.340 in the C-sweep). Arms: dropout in {0.0, 0.2, 0.5}
(encoder hidden layers only; decoder untouched; dropout OFF in all readouts via eval()).

Trajectories at checkpoints [10k..60k]: prior Sigma-hat diag, head c, sigma2_hat, y=0-readout OLS c;
SNIS E[theta|x]-readout OLS c at {20k, 40k, 60k}. Incremental training via model.num_steps = cp;
model.train(corpus) (the experiment_mc_calibrate.py checkpoint pattern).
DROPSLIDE_SMOKE=1 for plumbing check.
"""
import os, json, time, gc, numpy as np, scipy.sparse, tempfile, torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents

SMOKE = os.environ.get("DROPSLIDE_SMOKE") == "1"
DROPOUTS = json.loads(os.environ.get("DROPSLIDE_P", "[0.0, 0.2, 0.5]"))
K, C_COV, VOCAB, C_WORDS = 5, 3, 200, 25
HIDDEN, COMP = [256, 256], 20
SEED, SIGMA, INFLATE, S = 1000, 1.0, 1.5, 128
OPTIM = {"main": {"lr": 1e-3, "weight_decay": 0.0}, "prior": {"lr": 1e-4, "weight_decay": 0.0}}
PRED_ARGS = {"y": {"hidden_dims": [], "loss_weight": 1.0}}
if SMOKE:
    N, CKPTS, SNIS_AT, ENC_S, S = 3000, [1000, 2000], [2000], 10, 32
else:
    N, CKPTS, SNIS_AT, ENC_S = 100000, [10000, 20000, 30000, 40000, 50000, 60000], [20000, 40000, 60000], 20
OUT = "audit/results_dropout_slide.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lambda_fixed = np.random.default_rng(7).standard_normal((C_COV + 1, K)) * 0.5
label_coeffs = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
true_b = label_coeffs - label_coeffs.mean()

def align(tt, et):
    r, c = linear_sum_assignment(-(tt.T @ et)); return {int(i): int(j) for i, j in zip(r, c)}
def center(v): return v - v.mean()
def densify(c):
    mm = c.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(mm):
        c.processed_modalities["text"]["bow"]["matrix"] = np.asarray(mm.todense(), np.float32)

def ols_c(theta_hat, yv, perm):
    X = np.column_stack([theta_hat[:, perm[t]] for t in range(K)]).astype(np.float64)
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    bc = center(b)
    return float((bc @ true_b) / (true_b @ true_b)), float(np.abs(bc - true_b).mean())

print(f"{'[SMOKE] ' if SMOKE else ''}dropout slide | C={C_WORDS}w/doc N={N} dropouts={DROPOUTS} ckpts={CKPTS}", flush=True)
dft, df, tw, lam, lc = generate_documents(num_docs=N, num_topics=K, vocab_size=VOCAB, num_covs=C_COV,
    doc_topic_prior="logistic_normal", min_words=C_WORDS, max_words=C_WORDS, lambda_=lambda_fixed,
    label_type="regression", label_coeffs=label_coeffs, random_seed=SEED)
true_theta = dft[[f"Topic{i}" for i in range(K)]].values.astype(np.float64)
rng = np.random.default_rng(2024 + int(SIGMA * 1000))
y = (true_theta @ label_coeffs) + rng.normal(0, SIGMA, N)
df["label"] = y
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
mods = {"text": {"column": "doc_clean_0", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}
labels_cfg = {"y": {"column": "label", "type": "regression"}}
corpus = Corpus(df, modalities=mods, prevalence="~ cov_1 + cov_2 + cov_3", labels=labels_cfg); densify(corpus)

def snis_c(model, perm):
    model.encoder.eval()
    enc = model.encoder
    enc_key = list(enc.encoders.keys())[0]
    loader = torch.utils.data.DataLoader(corpus, batch_size=256, shuffle=False, num_workers=0)
    out = []
    with torch.no_grad():
        for data in loader:
            for k, v in data.items():
                if isinstance(v, torch.Tensor): data[k] = v.to(device)
            prevalence = data["M_prevalence_covariates"]
            lab0 = torch.zeros_like(data["M_labels"].float())
            x = torch.cat([data["modalities"]["text"]["bow"].to(device), prevalence, lab0], dim=1)
            content = data.get("M_content_covariates", None)
            B = prevalence.shape[0]
            mu_p, Sig = model.prior.get_prior_params(prevalence, return_full_cov=True)
            prior_dist = MultivariateNormal(mu_p.detach(), covariance_matrix=Sig.detach().unsqueeze(0).expand(B, -1, -1))
            _, _, info = enc({enc_key: x}, prevalence_covariates=prevalence)
            _, means, logvars, pi = info[-1]
            means, pi = means.detach(), pi.detach()
            lv_q = torch.clamp(logvars.detach(), -8.0, 8.0) + 2.0 * np.log(INFLATE)
            pi_t = ("mog", means, lv_q, pi)
            lws, ths = [], []
            for _ in range(S):
                z = enc._mog_sample(means, lv_q, pi)
                theta = model.latent_to_theta(z)
                lws.append(model._recon_loglik(theta, data, corpus, content)
                           + prior_dist.log_prob(z) - model._posterior_loglik(pi_t, z))
                ths.append(theta)
            w = torch.softmax(torch.stack(lws, dim=1), dim=1)
            out.append((w.unsqueeze(2) * torch.stack(ths, dim=1)).sum(dim=1).cpu().numpy())
    return ols_c(np.concatenate(out).astype(np.float64), y, perm)

results = {"smoke": SMOKE, "config": {"N": N, "C_words": C_WORDS, "dropouts": DROPOUTS,
           "ckpts": CKPTS, "snis_at": SNIS_AT, "S": S}, "true_b": true_b.tolist(), "arms": []}

for p_drop in DROPOUTS:
    print(f"\n===== ARM dropout={p_drop} =====", flush=True)
    arm = {"dropout": p_drop, "trajectory": []}
    t0 = time.time()
    model = GTM(train_data=corpus, n_topics=K, ae_type="vae", vi_type="mixture_of_gaussians",
                mixture_components=COMP, doc_topic_prior="logistic_normal", update_prior=True, w_prior=1.0,
                learn_prior_cov=True, labels_in_encoder=True, predictor_args=PRED_ARGS,
                encoder_args={"text_bow": {"hidden_dims": HIDDEN, "dropout": p_drop}},
                decoder_args={"text_bow": {"hidden_dims": []}},
                batch_size=256, num_steps=CKPTS[0], num_workers=0, print_every_n_steps=10**9,
                optim_args=OPTIM, return_best_model=False, ckpt_folder=tempfile.mkdtemp(), seed=SEED, device=device)
    for i, cp in enumerate(CKPTS):
        if i > 0:
            model.num_steps = cp
            model.train(corpus)
        model.encoder.eval()
        th_y0 = model.get_doc_topic_distribution(corpus, num_samples=ENC_S).astype(np.float64)
        perm = [align(true_theta, th_y0)[t] for t in range(K)]
        Wh = model.predictor.predictors["y"].neural_net["pred_0"].weight.detach().cpu().numpy().astype(np.float64)
        head_b = center(np.array([Wh[0, perm[t]] for t in range(K)]))
        with torch.no_grad():
            probe = torch.zeros(1, C_COV + 1, device=device); probe[0, 0] = 1.0
            _, Sig_f = model.prior.get_prior_params(probe, return_full_cov=True)
        rec = {"step": cp,
               "sigma_diag_mean": float(torch.diagonal(Sig_f).mean().cpu()),
               "head_c": float((head_b @ true_b) / (true_b @ true_b)),
               "sigma2_hat": float(torch.exp(model.predictor.noise_log_var["y"]).detach().cpu()),
               "y0_c": ols_c(th_y0, y, perm)[0]}
        if cp in SNIS_AT:
            rec["snis_c"], rec["snis_mab"] = snis_c(model, perm)
        arm["trajectory"].append(rec)
        snis_str = f" snis_c={rec['snis_c']:.3f}" if "snis_c" in rec else ""
        print(f"  [{cp:>6}] Sig~={rec['sigma_diag_mean']:.2f} head_c={rec['head_c']:.3f} "
              f"y0_c={rec['y0_c']:.3f}{snis_str} sig2={rec['sigma2_hat']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    results["arms"].append(arm)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"  arm dropout={p_drop} saved -> {OUT}", flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()

print("\n=== DROPOUT SLIDE SUMMARY (Sigma~ / head_c / y0_c @60k) ===")
for arm in results["arms"]:
    f = arm["trajectory"][-1]
    print(f"  dropout={arm['dropout']}: Sig~={f['sigma_diag_mean']:.2f} head_c={f['head_c']:.3f} "
          f"y0_c={f['y0_c']:.3f}" + (f" snis_c={f['snis_c']:.3f}" if "snis_c" in f else ""))
print(f"DROPOUT SWEEP COMPLETE — saved -> {OUT}", flush=True)
