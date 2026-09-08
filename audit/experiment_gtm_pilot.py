"""One-replication GTM pilot with a correctly specified linear-softmax BOW DGP.

The design (decoder, prevalence coefficients, full contrast covariance, outcome
coefficients) is fixed independently of the replication seed. Unlike the older
generate_documents-based experiments, counts follow softmax(theta @ decoder),
which is exactly the likelihood fitted by the current GTM implementation.

Joint and unsupervised arms learn the same covariate-informed full prior by
default. Outcome coefficients are centered and topics aligned by decoder profiles.
Post-fit OLS uses importance-weighted outcome-free posterior means, never the
supervised encoder with y simply set to zero. OLS intervals are diagnostic only:
they omit estimation of the first-stage model. No joint CIs are fabricated.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deeplatent_mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/deeplatent_numba")

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from deeplatent import Corpus, GTM
from deeplatent.utils import contrast_basis


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--words", type=int, default=25)
    p.add_argument("--topics", type=int, default=5)
    p.add_argument("--vocab", type=int, default=200)
    p.add_argument("--covariates", type=int, default=3)
    p.add_argument("--components", type=int, default=10)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--checkpoints", type=int, nargs="+", default=[2000, 4000, 8000, 16000])
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=9100)
    p.add_argument("--design-seed", type=int, default=777)
    p.add_argument("--sigma-y", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=0.45)
    p.add_argument("--anchors", type=int, default=0,
                   help="anchor words per topic: +anchor-logit in-topic, -anchor-logit off-topic")
    p.add_argument("--anchor-logit", type=float, default=4.0)
    p.add_argument("--prior-sd-scale", type=float, default=1.0,
                   help="multiplies the true contrast SDs (0.8..1.2); larger = sparser documents")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--prior-lr", type=float, default=1e-4)
    p.add_argument("--posterior-samples", type=int, default=1024)
    p.add_argument("--readout-batch", type=int, default=64)
    p.add_argument("--diagnostic-docs", type=int, default=256)
    p.add_argument("--two-step-prior", choices=["learned", "fixed"], default="learned")
    p.add_argument("--joint-prior-cov", choices=["learned", "fixed", "unit_trace"], default="learned",
                   help="fixed pins the joint arm's prior covariance at I; unit_trace learns the shape of "
                        "Sigma but rescales it to trace K-1 (removes only the scale gauge)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default="audit/results_gtm_pilot.json")
    return p


def fixed_design(a):
    rng = np.random.default_rng(a.design_seed)
    basis = contrast_basis(a.topics).numpy().astype(np.float64)
    prevalence = rng.normal(0, 0.5, (a.topics, a.covariates + 1))
    prevalence -= prevalence.mean(axis=0, keepdims=True)
    d = a.topics - 1
    sd = np.linspace(0.8, 1.2, d) * a.prior_sd_scale
    covariance = np.outer(sd, sd) * a.rho ** np.abs(np.arange(d)[:, None] - np.arange(d))
    # Finite topic-associated word logits; no hard zeros or imposed anchor constraints.
    decoder = rng.normal(0, 0.5, (a.topics, a.vocab))
    for k, words in enumerate(np.array_split(np.arange(a.vocab), a.topics)):
        decoder[k, words] += 3.0
    for k, words in enumerate(np.array_split(np.arange(a.vocab), a.topics)):
        anchors = words[: a.anchors]
        decoder[:, anchors] = -a.anchor_logit
        decoder[k, anchors] = a.anchor_logit
    decoder -= decoder.mean(axis=1, keepdims=True)
    beta = np.linspace(2.0, -2.0, a.topics)
    return dict(basis=basis, prevalence=prevalence, covariance=covariance,
                decoder=decoder, beta=beta)


def generate(a, design, n, seed):
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.binomial(1, 0.5, (n, a.covariates))])
    mean = x @ design["prevalence"].T @ design["basis"]
    eta = mean + rng.multivariate_normal(np.zeros(a.topics - 1), design["covariance"], n)
    theta = softmax(eta @ design["basis"].T, axis=1)
    probabilities = softmax(theta @ design["decoder"], axis=1)
    counts = np.array([rng.multinomial(a.words, row) for row in probabilities], dtype=np.float32)
    y = theta @ design["beta"] + rng.normal(0, a.sigma_y, n)
    return dict(x=x, eta=eta, theta=theta, counts=counts, y=y)


def make_corpus(a, data, with_y):
    words = [f"w{v:03d}" for v in range(a.vocab)]
    docs = [" ".join(word for word, count in zip(words, row) for _ in range(int(count)))
            for row in data["counts"]]
    frame = pd.DataFrame({"text": docs, "y": data["y"]})
    for j in range(a.covariates):
        frame[f"x{j + 1}"] = data["x"][:, j + 1]
    vec = CountVectorizer(vocabulary={word: j for j, word in enumerate(words)})
    modalities = {"text": {"column": "text", "views": {"bow": {"type": "bow", "vectorizer": vec}}}}
    kwargs = {"labels": {"y": {"column": "y", "type": "regression"}}} if with_y else {}
    corpus = Corpus(frame, modalities=modalities,
                    prevalence="~ " + " + ".join(f"x{j + 1}" for j in range(a.covariates)), **kwargs)
    observed = corpus.processed_modalities["text"]["bow"]["matrix"].toarray().astype(np.float32)
    assert np.array_equal(observed, data["counts"])
    assert np.array_equal(corpus.M_prevalence_covariates, data["x"])
    corpus.processed_modalities["text"]["bow"]["matrix"] = observed
    return corpus


def coefficient_stats(beta, truth):
    beta = np.asarray(beta, dtype=np.float64)
    beta = beta - beta.mean()
    error = beta - truth
    return {"beta": beta.tolist(), "error": error.tolist(),
            "mean_absolute_error": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "scale_ratio": float(beta @ truth / (truth @ truth)),
            "correlation": float(np.corrcoef(beta, truth)[0, 1])}


def ols(theta, y, truth):
    # Topic shares sum to one, so their columns already contain an intercept.
    x = np.asarray(theta, dtype=np.float64)
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ b
    bread = np.linalg.inv(x.T @ x)
    vcov = bread @ (x.T @ (x * resid[:, None] ** 2)) @ bread
    vcov *= len(y) / (len(y) - x.shape[1])
    center = np.eye(len(b)) - np.ones((len(b), len(b))) / len(b)
    vcov = center @ vcov @ center
    se = np.sqrt(np.maximum(np.diag(vcov), 0))
    stats = coefficient_stats(b, truth)
    stats.update(naive_hc1_se=se.tolist(), naive_hc1_vcov=vcov.tolist(),
                 naive_ci_low=(center @ b - 1.96 * se).tolist(),
                 naive_ci_high=(center @ b + 1.96 * se).tolist(),
                 design_condition_number=float(np.linalg.cond(x)),
                 intercept=float(b.mean()), residual_sd=float(np.std(resid)))
    return stats


@contextmanager
def diagnostic_context(model, seed):
    modules = [model.encoder, model.decoders, model.prior]
    if model.predictor is not None:
        modules.append(model.predictor)
    flags = [module.training for module in modules]
    devices = [torch.cuda.current_device()] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for module in modules:
            module.eval()
        with torch.no_grad():
            yield
    for module, flag in zip(modules, flags):
        module.train(flag)


def apply_unit_trace(prior):
    """Rescale the learned covariance to trace n_latent (shape learned, scale gauge pinned)."""
    base = type(prior)

    def sigma(self):
        raw = base.sigma.fget(self)
        return raw * (self.n_latent / torch.trace(raw))

    prior.__class__ = type("UnitTrace" + base.__name__, (base,), {"sigma": property(sigma)})


def decoder_matrix(model):
    return model.decoders["text_bow"].decoder["dec_0"].weight.detach().cpu().numpy().T.astype(np.float64)


def align_topics(model, design):
    fitted = decoder_matrix(model)
    fitted -= fitted.mean(axis=1, keepdims=True)
    truth = design["decoder"]
    # Normalize the topic-specific profiles to avoid using unknown theta or y for matching.
    ft = fitted - fitted.mean(axis=0, keepdims=True)
    tt = truth - truth.mean(axis=0, keepdims=True)
    similarity = (tt @ ft.T) / (np.linalg.norm(tt, axis=1)[:, None] * np.linalg.norm(ft, axis=1)[None, :])
    rows, cols = linear_sum_assignment(-similarity)
    assert np.array_equal(rows, np.arange(len(rows)))
    return cols, similarity[rows, cols]


def encoder_distribution(model, x, counts, y=None):
    parts = [counts, x]
    if model.labels_in_encoder:
        parts.append(torch.zeros((len(x), 1), device=model.device) if y is None else y[:, None])
    raw = model.encoder.encoders["text_bow"](torch.cat(parts, dim=1))
    means, logvars, weights = model.encoder._mog_unpack(raw)
    return MixtureSameFamily(Categorical(probs=weights), Independent(Normal(means, (0.5 * logvars).exp()), 1))


def posterior_readout(model, data, a, samples, seed, supervised=False, importance=True):
    """Read posterior means with proper proposal density and a defensive prior mix.

    Importance target includes y only when supervised=True. For y-free joint
    readout q(y=0) is just a proposal, never treated as the target posterior.
    Two disjoint halves diagnose simulation error in the posterior means.
    """
    out, half_a, half_b, ess_all, logz_all, elbo_all, interval_all = [], [], [], [], [], [], []
    device = model.device
    defensive = 0.15 if importance else 0.0
    with diagnostic_context(model, seed):
        for lo in range(0, len(data["y"]), a.readout_batch):
            sl = slice(lo, lo + a.readout_batch)
            x = torch.as_tensor(data["x"][sl], dtype=torch.float32, device=device)
            counts = torch.as_tensor(data["counts"][sl], device=device)
            y = torch.as_tensor(data["y"][sl], dtype=torch.float32, device=device)
            q = encoder_distribution(model, x, counts, y if supervised else None)
            z = q.sample((samples,))
            mu, covariance = model.prior.get_prior_params(x, return_full_cov=True)
            prior = torch.distributions.MultivariateNormal(mu, covariance_matrix=covariance)
            if defensive:
                from_prior = torch.rand((samples, len(x)), device=device) < defensive
                z = torch.where(from_prior[..., None], prior.sample((samples,)), z)
            theta = model.latent_to_theta(z.reshape(-1, model.n_latent)).reshape(samples, len(x), a.topics)
            if importance:
                logq = q.log_prob(z)
                logp = prior.log_prob(z)
                logproposal = torch.logaddexp(logq + math.log1p(-defensive), logp + math.log(defensive))
                logits = model.decoders["text_bow"](theta.reshape(-1, a.topics)).reshape(samples, len(x), -1)
                logtarget = logp + (counts[None] * logits.log_softmax(-1)).sum(-1)
                if supervised:
                    predictor = model.predictor.predictors["y"]
                    pred = predictor(theta.reshape(-1, a.topics), None).reshape(samples, len(x))
                    variance = model.predictor.noise_log_var["y"].exp()
                    logtarget += -0.5 * ((y[None] - pred) ** 2 / variance + variance.log() + math.log(2 * math.pi))
                lw = logtarget - logproposal
                weights = lw.softmax(0)
                ess_all.append((1 / weights.square().sum(0)).cpu().numpy())
                logz_all.append((lw.logsumexp(0) - math.log(samples)).cpu().numpy())
                # Valid lower bound for the defensive mixture proposal, labelled as such.
                elbo_all.append(lw.mean(0).cpu().numpy())
            else:
                lw = torch.zeros((samples, len(x)), device=device)
                weights = torch.full_like(lw, 1 / samples)
            pm = (weights[..., None] * theta).sum(0)
            mid = samples // 2
            pa = (lw[:mid].softmax(0)[..., None] * theta[:mid]).sum(0)
            pb = (lw[mid:].softmax(0)[..., None] * theta[mid:]).sum(0)
            out.append(pm.cpu().numpy())
            half_a.append(pa.cpu().numpy())
            half_b.append(pb.cpu().numpy())
            if not importance:
                interval_all.append(torch.quantile(theta, torch.tensor([0.05, 0.95], device=device), dim=0).cpu().numpy())
    means, first, second = np.vstack(out), np.vstack(half_a), np.vstack(half_b)
    diagnostic = {"samples": samples, "includes_y": supervised, "importance_weighted": importance,
                  "defensive_prior_weight": defensive,
                  "half_sample_mean_absolute_difference": float(np.abs(first - second).mean())}
    if importance:
        ess = np.concatenate(ess_all)
        diagnostic.update(ess_mean=float(ess.mean()), ess_median=float(np.median(ess)),
                          ess_p05=float(np.quantile(ess, 0.05)), fraction_ess_below_20=float((ess < 20).mean()),
                          log_evidence_estimate=float(np.concatenate(logz_all).mean()),
                          defensive_proposal_elbo=float(np.concatenate(elbo_all).mean()))
    else:
        diagnostic["interval_low"] = np.concatenate([v[0] for v in interval_all]).tolist()
        diagnostic["interval_high"] = np.concatenate([v[1] for v in interval_all]).tolist()
    return means, first, second, diagnostic


def plugin_mle_theta(model, data, a, steps=400, lr=0.1):
    """Naive two-step plug-in: per-document MLE of theta under the fitted decoder, no prior."""
    device = model.device
    counts = torch.as_tensor(data["counts"], device=device)
    eta = torch.zeros((len(counts), model.n_latent), device=device, requires_grad=True)
    opt = torch.optim.Adam([eta], lr=lr)
    decoder = model.decoders["text_bow"]
    was_training = decoder.training
    decoder.eval()
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            logits = decoder(model.latent_to_theta(eta))
            loss = -(counts * logits.log_softmax(-1)).sum()
            loss.backward(inputs=[eta])
            opt.step()
            with torch.no_grad():
                eta.clamp_(-12, 12)
    decoder.train(was_training)
    with torch.no_grad():
        return model.latent_to_theta(eta).cpu().numpy()


def checkpoint_record(model, data, a, design, joint, step):
    with diagnostic_context(model, a.seed + 30):
        perm, similarities = align_topics(model, design)
        record = {"step": step, "topic_permutation": perm.tolist(),
                  "decoder_profile_correlations": similarities.tolist(),
                  "recent_loss": float(np.mean(model.train_losses[-200:])),
                  "recent_reconstruction": float(np.mean(model.train_recon_losses[-200:])),
                  "recent_kl": float(np.mean(model.train_div_losses[-200:]))}
        if joint:
            head = model.predictor.predictors["y"].neural_net["pred_0"]
            beta = head.weight.detach().cpu().numpy()[0, perm]
            record["head"] = coefficient_stats(beta, design["beta"])
            record["outcome_noise_variance"] = float(model.predictor.noise_log_var["y"].exp())
        else:
            pm, _, _, _ = posterior_readout(model, data, a, 128, a.seed + 30, importance=False)
            record["ols_encoder"] = ols(pm[:, perm], data["y"], design["beta"])
            theta_mle = plugin_mle_theta(model, data, a)
            record["ols_plugin_mle"] = ols(theta_mle[:, perm], data["y"], design["beta"])
            record["ols_plugin_mle"]["share_rmse"] = float(np.sqrt(np.mean((theta_mle[:, perm] - data["theta"]) ** 2)))
            record["ols_plugin_mle"]["fraction_at_vertex"] = float((theta_mle.max(1) > 0.99).mean())
        if joint or a.two_step_prior == "learned":
            prev = model.get_prevalence_coefficients()[perm]
            basis = design["basis"]
            rotation = basis.T @ basis[perm]
            covariance = rotation @ model.prior.sigma.detach().cpu().numpy() @ rotation.T
            record.update(prevalence=prev.tolist(), covariance_contrast=covariance.tolist(),
                          prevalence_rmse=float(np.sqrt(np.mean((prev - design["prevalence"]) ** 2))),
                          covariance_relative_frobenius_error=float(np.linalg.norm(covariance - design["covariance"]) / np.linalg.norm(design["covariance"])),
                          covariance_eigenvalues=np.linalg.eigvalsh(covariance).tolist())
        assert np.isfinite(model.train_losses).all()
        record["finite_losses"] = True
    return record


def save_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def source_commit():
    if os.environ.get("GTM_SOURCE_COMMIT"):
        return os.environ["GTM_SOURCE_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return "unavailable (set GTM_SOURCE_COMMIT for an HPC snapshot without .git)"


def main():
    a = parser().parse_args()
    if a.n <= a.topics or a.topics < 3 or a.covariates < 1 or a.words < 1:
        raise ValueError("Need n > topics >= 3, at least one covariate and one word.")
    if a.checkpoints != sorted(set(a.checkpoints)) or a.checkpoints[0] < 1:
        raise ValueError("Checkpoints must be positive and strictly increasing.")
    if a.posterior_samples < 32 or a.posterior_samples % 2:
        raise ValueError("Use an even number of posterior samples, at least 32.")
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    design = fixed_design(a)
    data = generate(a, design, a.n, a.seed)
    validation = generate(a, design, a.diagnostic_docs, a.seed + 100000)
    # Verify DGP probability mapping against the package decoder and contrast map.
    counts = data["counts"]
    assert np.all(counts.sum(1) == a.words) and np.all(counts >= 0)
    assert np.allclose(data["theta"].sum(1), 1)
    assert np.linalg.eigvalsh(design["covariance"]).min() > 0
    start = time.time()
    result = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "config": vars(a), "replications": 1,
              "git_commit": source_commit(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "package_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((Path(__file__).resolve().parents[1] / "deeplatent").glob("*.py"))},
              "torch_version": torch.__version__,
              "design": {k: v.tolist() for k, v in design.items()},
              "oracle_true_shares": ols(data["theta"], data["y"], design["beta"]),
              "notes": ["DGP: multinomial(softmax(theta @ decoder)); fixed design across replications.",
                        "Reported coefficient errors are one-draw errors, not Monte Carlo bias.",
                        "Naive OLS intervals omit first-stage model uncertainty; no joint CIs estimated.",
                        "Checkpoint diagnostics preserve training RNG and do not select a checkpoint using truth.",
                        "Full covariance is learned in K-1 identifiable contrast coordinates."],
              "arms": {}}
    save_json(a.out, result)
    print(f"GTM pilot: n={a.n}, words={a.words}, K={a.topics}, covariates={a.covariates}, "
          f"MoG={a.components}, device={device}, checkpoints={a.checkpoints}", flush=True)
    print("Oracle coefficients:", np.round(result["oracle_true_shares"]["beta"], 3).tolist(), flush=True)
    for arm in ["joint", "two_step"]:
        joint = arm == "joint"
        corpus = make_corpus(a, data, joint)
        t0 = time.time()
        records = []
        result["arms"][arm] = {"checkpoints": records}
        with tempfile.TemporaryDirectory(prefix="gtm_pilot_") as ckpt:
            model = GTM(train_data=corpus, n_topics=a.topics, ae_type="vae",
                        vi_type="mixture_of_gaussians", mixture_components=a.components,
                        doc_topic_prior="logistic_normal",
                        update_prior=joint or a.two_step_prior == "learned",
                        learn_prior_cov=(a.joint_prior_cov == "learned") if joint else a.two_step_prior == "learned",
                        labels_in_encoder=joint,
                        predictor_args={"y": {"hidden_dims": [], "loss_weight": 1.0}} if joint else {},
                        encoder_args={"text_bow": {"hidden_dims": a.hidden, "dropout": 0.0}},
                        decoder_args={"text_bow": {"hidden_dims": [], "dropout": 0.0}},
                        w_prior=1.0, w_pred_loss=1.0, batch_size=a.batch, num_steps=0,
                        num_workers=0, print_every_n_steps=1000,
                        optim_args={"main": {"lr": a.lr, "weight_decay": 0.0},
                                    "prior": {"lr": a.prior_lr, "weight_decay": 0.0}},
                        return_best_model=False, ckpt_folder=ckpt, seed=a.seed, device=device)
            if joint and a.joint_prior_cov == "unit_trace":
                apply_unit_trace(model.prior)
            assert model.n_latent == a.topics - 1
            assert len(model.decoders["text_bow"].decoder) == 1
            # Check the exact DGP transform without altering model parameters.
            with torch.no_grad():
                transformed = model.latent_to_theta(torch.as_tensor(data["eta"][:8], dtype=torch.float32, device=device))
            assert np.allclose(transformed.cpu().numpy(), data["theta"][:8], atol=1e-6)
            for step in a.checkpoints:
                model.num_steps = step
                model.train(corpus)
                rec = checkpoint_record(model, data, a, design, joint, step)
                rec["elapsed_seconds"] = time.time() - t0
                records.append(rec)
                save_json(a.out, result)
                effects = rec["head" if joint else "ols_encoder"]
                if not joint:
                    mle = rec["ols_plugin_mle"]
                    print(f"{arm} @{step} plug-in MLE: beta={np.round(mle['beta'], 3).tolist()} scale={mle['scale_ratio']:.3f}, MAE={mle['mean_absolute_error']:.3f}", flush=True)
                print(f"{arm} @{step}: beta={np.round(effects['beta'], 3).tolist()} "
                      f"scale={effects['scale_ratio']:.3f}, MAE={effects['mean_absolute_error']:.3f}, "
                      f"prior_RMSE={rec.get('prevalence_rmse', float('nan')):.3f}, "
                      f"seconds={rec['elapsed_seconds']:.1f}", flush=True)
            # Preserve the trained model before potentially expensive readout diagnostics.
            checkpoint_path = str(Path(a.out).with_suffix("")) + f"_{arm}.ckpt"
            model.save_model(checkpoint_path)
            result["arms"][arm]["checkpoint_file"] = checkpoint_path
            save_json(a.out, result)
            perm = np.array(records[-1]["topic_permutation"])
            pm, first, second, diagnostics = posterior_readout(model, data, a, a.posterior_samples, a.seed + 50)
            postfit = ols(pm[:, perm], data["y"], design["beta"])
            postfit["posterior_diagnostics"] = diagnostics
            first_fit = ols(first[:, perm], data["y"], design["beta"])
            second_fit = ols(second[:, perm], data["y"], design["beta"])
            postfit["half_sample_beta_max_difference"] = float(np.max(np.abs(np.array(first_fit["beta"]) - second_fit["beta"])))
            postfit["topic_share_rmse"] = float(np.sqrt(np.mean((pm[:, perm] - data["theta"]) ** 2)))
            postfit["topic_share_correlations"] = [float(np.corrcoef(pm[:, perm[k]], data["theta"][:, k])[0, 1]) for k in range(a.topics)]
            result["arms"][arm]["ols_outcome_free_posterior"] = postfit
            # Encoder-versus-importance diagnostics on independently generated documents.
            vi, _, _, vi_diag = posterior_readout(model, validation, a, a.posterior_samples,
                                                  a.seed + 60, supervised=joint, importance=False)
            exactish, _, _, is_diag = posterior_readout(model, validation, a, a.posterior_samples,
                                                       a.seed + 61, supervised=joint)
            lower = np.array(vi_diag.pop("interval_low"))[:, perm]
            upper = np.array(vi_diag.pop("interval_high"))[:, perm]
            result["arms"][arm]["heldout_posterior"] = {
                "conditioning": "words, covariates, outcome" if joint else "words, covariates",
                "encoder_vs_importance_mean_absolute_difference": float(np.abs(vi - exactish).mean()),
                "encoder_topic_interval_90_fraction_containing_truth": ((validation["theta"] >= lower) & (validation["theta"] <= upper)).mean(0).tolist(),
                "importance_diagnostics": is_diag}
            result["arms"][arm]["elapsed_seconds"] = time.time() - t0
            save_json(a.out, result)
            print(f"{arm} outcome-free OLS: beta={np.round(postfit['beta'], 3).tolist()}, "
                  f"scale={postfit['scale_ratio']:.3f}, ESS median={diagnostics['ess_median']:.0f}, "
                  f"half-sample beta difference={postfit['half_sample_beta_max_difference']:.3f}", flush=True)
            del model
        del corpus
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["status"] = "complete"
    result["elapsed_seconds"] = time.time() - start
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(a.out, result)
    print(f"COMPLETE: {a.out} ({result['elapsed_seconds']:.1f} seconds)", flush=True)


if __name__ == "__main__":
    main()
