"""Replay a saved GTM pilot exactly and audit generalization during training.

Instrument the original training loop without changing its update boundaries or
RNG stream. The likelihood audit uses a fixed training subset and independently
generated held-out documents. Original coefficient checkpoints must reproduce.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deeplatent_mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/deeplatent_numba")

import numpy as np
from scipy.special import gammaln
import torch


def evaluate(model, data, args, harness, seed):
    """Per-document IWAE likelihoods and ELBO; constants included for BOW.

    Joint marginal likelihood includes the outcome. Its text-only target is also
    integrated, allowing a conditional outcome log score from the ratio of the
    two marginal likelihoods. Outcome-conditioned q is only an IS proposal.
    """
    values = {"nll": [], "negative_elbo": [], "text_nll": [], "ess": []}
    with harness.diagnostic_context(model, seed):
        for lo in range(0, len(data["y"]), args.batch):
            sl = slice(lo, lo + args.batch)
            x = torch.as_tensor(data["x"][sl], dtype=torch.float32, device=model.device)
            counts = torch.as_tensor(data["counts"][sl], dtype=torch.float32, device=model.device)
            y = torch.as_tensor(data["y"][sl], dtype=torch.float32, device=model.device)
            q = harness.encoder_distribution(model, x, counts, y if model.labels_in_encoder else None)
            z = q.sample((args.samples,))
            theta = model.latent_to_theta(z.reshape(-1, model.n_latent))
            logits = model.decoders["text_bow"](theta).reshape(args.samples, len(x), -1)
            mu, covariance = model.prior.get_prior_params(x, return_full_cov=True)
            prior = torch.distributions.MultivariateNormal(mu, covariance_matrix=covariance)
            constants = torch.as_tensor(data["multinomial_constant"][sl], dtype=torch.float32,
                                        device=model.device)
            text_weights = prior.log_prob(z) - q.log_prob(z)
            text_weights += (counts[None] * logits.log_softmax(-1)).sum(-1) + constants[None]
            weights = text_weights
            if model.labels_in_encoder:
                prediction = model.predictor.predictors["y"](theta, None).reshape(args.samples, len(x))
                variance = model.predictor.noise_log_var["y"].exp()
                weights = text_weights - 0.5 * ((y[None] - prediction).square() / variance
                                              + variance.log() + math.log(2 * math.pi))
            values["nll"].append((math.log(args.samples) - weights.logsumexp(0)).cpu().numpy())
            values["negative_elbo"].append(-weights.mean(0).cpu().numpy())
            values["text_nll"].append((math.log(args.samples) - text_weights.logsumexp(0)).cpu().numpy())
            values["ess"].append((1 / weights.softmax(0).square().sum(0)).cpu().numpy())
    return {key: np.concatenate(items) for key, items in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=1000)
    parser.add_argument("--eval-n", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("gtm_pilot", Path(__file__).with_name("experiment_gtm_pilot.py"))
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    reference = json.loads(args.reference.read_text())
    cfg = argparse.Namespace(**reference["config"])
    assert reference["status"] == "complete"
    torch.set_num_threads(cfg.threads)
    design = {key: np.asarray(value) for key, value in reference["design"].items()}
    original_data = h.generate(cfg, design, cfg.n, cfg.seed)
    indices = np.random.default_rng(cfg.seed + 70000).choice(cfg.n, min(args.eval_n, cfg.n), replace=False)
    training = {key: value[indices] for key, value in original_data.items()}
    heldout = h.generate(cfg, design, args.eval_n, cfg.seed + 200000)
    for data in (training, heldout):
        counts = data["counts"]
        data["multinomial_constant"] = gammaln(counts.sum(1) + 1) - gammaln(counts + 1).sum(1)
    result = {"status": "running", "reference": str(args.reference), "config": vars(cfg),
              "audit": {"every": args.every, "train_eval_n": len(indices), "heldout_n": args.eval_n,
                        "importance_samples": args.samples, "heldout_seed": cfg.seed + 200000,
                        "constants_included": True, "diagnostics_preserve_training_rng": True},
              "arms": {name: {"trajectory": [], "reproduced_checkpoints": []} for name in ("joint", "two_step")}}
    arrays = {}
    previous = {}
    started = time.time()
    original_step = h.GTM.step_batch
    original_record = h.checkpoint_record
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out.with_suffix("")
    checkpoint_dir.mkdir(exist_ok=True)

    def audited_step(model, *positional, **keywords):
        loss = original_step(model, *positional, **keywords)
        validation = keywords.get("validation", positional[2] if len(positional) > 2 else False)
        if validation or (model.steps + 1) % args.every:
            return loss
        step = model.steps + 1
        name = "joint" if model.labels_in_encoder else "two_step"
        rec = {"step": step}
        for split, data in (("train", training), ("heldout", heldout)):
            scores = evaluate(model, data, args, h, cfg.seed + 80000)
            key = (name, split)
            summary = {metric: float(value.mean()) for metric, value in scores.items()}
            summary["nll_se_across_documents"] = float(scores["nll"].std(ddof=1) / np.sqrt(len(scores["nll"])))
            if model.labels_in_encoder:
                summary["outcome_predictive_nll"] = float((scores["nll"] - scores["text_nll"]).mean())
            if key in previous:
                change = scores["nll"] - previous[key]
                summary["nll_change_since_previous"] = float(change.mean())
                summary["nll_change_paired_se"] = float(change.std(ddof=1) / np.sqrt(len(change)))
            previous[key] = scores["nll"]
            arrays[f"{name}_{split}_{step}_nll"] = scores["nll"]
            rec[split] = summary
        with h.diagnostic_context(model, cfg.seed + 80001):
            permutation, _ = h.align_topics(model, design)
            if model.labels_in_encoder:
                beta = model.predictor.predictors["y"].neural_net["pred_0"].weight.detach().cpu().numpy()[0, permutation]
                rec["head"] = h.coefficient_stats(beta, design["beta"])
            prevalence = model.get_prevalence_coefficients()[permutation]
            rotation = design["basis"].T @ design["basis"][permutation]
            covariance = rotation @ model.prior.sigma.detach().cpu().numpy() @ rotation.T
            rec["prevalence_rmse"] = float(np.sqrt(np.mean((prevalence - design["prevalence"]) ** 2)))
            rec["covariance_relative_error"] = float(np.linalg.norm(covariance - design["covariance"]) / np.linalg.norm(design["covariance"]))
        result["arms"][name]["trajectory"].append(rec)
        result["elapsed_seconds"] = time.time() - started
        h.save_json(args.out, result)
        print(f"AUDIT {name} @{step}: train NLL={rec['train']['nll']:.4f}, "
              f"heldout NLL={rec['heldout']['nll']:.4f}, "
              f"heldout change={rec['heldout'].get('nll_change_since_previous', float('nan')):+.5f}", flush=True)
        return loss

    def audited_checkpoint(model, data, settings, dgp, joint, step):
        rec = original_record(model, data, settings, dgp, joint, step)
        name = "joint" if joint else "two_step"
        expected = next(r for r in reference["arms"][name]["checkpoints"] if r["step"] == step)
        metric = "head" if joint else "ols_encoder"
        difference = float(np.max(np.abs(np.asarray(rec[metric]["beta"]) - expected[metric]["beta"])))
        if difference > 1e-6:
            raise AssertionError(f"Replay differs from original {name}@{step}: max coefficient difference={difference}")
        result["arms"][name]["reproduced_checkpoints"].append({"step": step, "max_coefficient_difference": difference})
        model.save_model(str(checkpoint_dir / f"{name}_step{step}.ckpt"))
        h.save_json(args.out, result)
        return rec

    h.GTM.step_batch = audited_step
    h.checkpoint_record = audited_checkpoint
    replay_path = args.out.with_name(args.out.stem + "_replay.json")
    replay_argv = ["experiment_gtm_pilot.py"]
    for key, value in vars(cfg).items():
        replay_argv.append("--" + key.replace("_", "-"))
        if key == "out":
            replay_argv.append(str(replay_path))
        elif isinstance(value, list):
            replay_argv.extend(str(v) for v in value)
        else:
            replay_argv.append(str(value))
    sys.argv = replay_argv
    try:
        h.main()
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        h.GTM.step_batch = original_step
        h.checkpoint_record = original_record
        result["elapsed_seconds"] = time.time() - started
        np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
        h.save_json(args.out, result)
    print(f"OVERFIT AUDIT COMPLETE: {args.out}", flush=True)


if __name__ == "__main__":
    main()
