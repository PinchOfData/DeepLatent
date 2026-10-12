"""Step 03: fit the joint topic-prediction model (model.py) for one prediction weight lam.

Data: the document-term matrix of step 02 and the monthly industrial-production growth g_s (100 x monthly log
change). The target of article month s is the average growth over months s+start .. s+start+window-1 (default:
the article month and the next two, i.e. the 3-month moving average centred on s+1). Months are split by article
month: training up to --train-end, validation up to --val-end, test after; a training (validation) month whose
target window reaches into the next period has no target, so no outcome crosses a split. The topic model trains on
the training months' records only; validation and test records are only encoded.

--features change feeds the head each month's attention minus its average over the previous --change-window months
(common.head_features) instead of the attention itself; months without that history have no target. In training
the trailing average is taken from the cache too, and the gradient flows through every month's batch records.

The prediction term takes each month's attention from a cache of every training record's theta (model.py), with the
gradient through the batch's records; the cache is refreshed for the batch records at every step and entirely at
every evaluation.

Every --eval-every steps (and at the end): the months' topic attention thetabar_t (average over each month's records
of E_q[theta], --eval-samples draws per record), the head's predictions and R^2 (validation and test R^2 are out of
sample: 1 - SSE / sum of squares around the TRAINING mean), the validation bound on the per-word perplexity (a
fixed sample of validation records), and a checkpoint on scratch. The selected model is the evaluated step with the
best validation R^2 among the steps from --select-after on (default: the end of KL annealing; the last step for
lam = 0); the test period is never used to choose anything. 04 compares runs both at their selected and at their
final steps. --two-step-head cuts the topics off from the prediction loss (the records' theta enter it detached):
the head is then fit on topics learned from the text alone, the two-step estimator with this model's own head,
training and step selection (lam then only scales the head's gradient, which Adam's normalization undoes).

--head ridge replaces the learned head by a ridge regression re-solved exactly at every step on the training months
(the forecasting equation of 04, with the penalty --head-penalty fixed in advance), on the inputs of months
t .. t-(--head-lags); the gradient reaches the topics through the solution. Months without the lags' history have no
target.

Writes results/<tag>/: config.json, metrics.csv (one row per evaluation), months.csv (month, split, target),
theta_<which>.npy (months x K attention), beta_<which>.npy (K x V topic-word probabilities), head_<which>.npz
(c, b and the standardization mbar, s),
predictions_<which>.csv and topic_words_<which>.csv for which in {final, selected}.

Usage: python 03_fit_joint.py --tag joint_k50_lam1e4 --lam 1e4 [--n-topics 50 --num-steps 20000 ...]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import common
from model import GPUCSR, JointTopicModel


def kmeans_topics(X, K: int, n_docs: int, seed: int) -> np.ndarray:
    """K x V word counts (+1) of K clusters of n_docs random rows of X: k-means on the L2-normalized tf-idf rows
    (cosine geometry), the start that avoids topic collapse in mixture decoders (app 1)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=min(n_docs, X.shape[0]), replace=False))
    S = X[idx].astype(np.float64)
    idf = np.log((1 + S.shape[0]) / (1 + np.asarray((S > 0).sum(axis=0)).ravel())) + 1.0
    Z = normalize(normalize(S, norm="l1") @ sp.diags(idf), norm="l2")
    labels = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=4096, n_init=3).fit_predict(Z)
    print("k-means cluster sizes:", np.bincount(labels, minlength=K).tolist(), flush=True)
    return np.vstack([np.asarray(S[labels == k].sum(axis=0)).ravel() for k in range(K)]) + 1.0


def top_words(beta: np.ndarray, background: np.ndarray, vocab: list, n: int = 15) -> pd.DataFrame:
    rows = []
    lift = np.log(beta) - np.log(background)[None, :]
    relevance = 0.6 * np.log(beta) + 0.4 * lift  # Sievert & Shirley (2014), lambda = 0.6
    for k in range(beta.shape[0]):
        rows.append({"topic": k,
                     "probable": ", ".join(vocab[j] for j in np.argsort(-beta[k])[:n]),
                     "relevant": ", ".join(vocab[j] for j in np.argsort(-relevance[k])[:n])})
    return pd.DataFrame(rows)


@torch.no_grad()
def month_attention(model, csr, month_of: torch.Tensor, n_months: int, n_samples: int, bs: int = 8192):
    model.eval()
    sums = torch.zeros(n_months, model.K, device=month_of.device)
    for s in range(0, csr.n_rows, bs):
        idx = torch.arange(s, min(s + bs, csr.n_rows), device=month_of.device)
        mu, logvar = model.encode(csr.rows(idx))
        sums.index_add_(0, month_of[idx], model.sample_theta(mu, logvar, n_samples))
    counts = torch.bincount(month_of, minlength=n_months).clamp_min(1).unsqueeze(1)
    model.train()
    return (sums / counts).cpu().numpy()


@torch.no_grad()
def record_theta(model, csr, n_rows: int, n_samples: int, dev, bs: int = 8192) -> torch.Tensor:
    """E_q[theta] (n_samples draws) of the first n_rows records: the training records' cache."""
    model.eval()
    out = torch.empty(n_rows, model.K, device=dev)
    for s in range(0, n_rows, bs):
        idx = torch.arange(s, min(s + bs, n_rows), device=dev)
        mu, logvar = model.encode(csr.rows(idx))
        out[s:s + len(idx)] = model.sample_theta(mu, logvar, n_samples)
    model.train()
    return out


@torch.no_grad()
def perplexity_bound(model, csr, idx: torch.Tensor, bs: int = 8192) -> float:
    """exp(sum of -ELBO / tokens) over the records idx: an upper bound on the held-out per-word perplexity."""
    model.eval()
    beta = model.beta()
    tot, tokens = 0.0, 0.0
    for s in range(0, len(idx), bs):
        x = csr.rows(idx[s:s + bs])
        mu, logvar = model.encode(x)
        theta = model.sample_theta(mu, logvar)
        tot += float((model.nll(x, theta, beta) + model.kl(mu, logvar)).sum())
        tokens += float(x.sum())
    model.train()
    return float(np.exp(tot / tokens))


def head_features_t(att: torch.Tensor, kind: str, window: int) -> torch.Tensor:
    """common.head_features for a torch tensor (rows without history are zeros: they carry no target)."""
    if kind == "level":
        return att
    prev = att.unfold(0, window, 1)[:-1].mean(-1)  # window i covers months i..i+window-1: month i+window's past
    return torch.cat([torch.zeros_like(att[:window]), att[window:] - prev])


def train_stats(theta, tr_months, floor: float = 1e-4):
    """Mean and standard deviation (floored) of the monthly attention over the training months with a target."""
    m = theta[tr_months].mean(0)
    s = theta[tr_months].std(0)
    return m, (s.clamp_min(floor) if torch.is_tensor(s) else np.maximum(s, floor))


def with_lags(z: np.ndarray, lags: int) -> np.ndarray:
    """The inputs of months t, t-1, ..., t-lags side by side (NaN without that history), as 04."""
    T, P = z.shape
    out = np.full((T, P * (lags + 1)), np.nan)
    for l in range(lags + 1):
        out[l:, l * P:(l + 1) * P] = z[:T - l]
    return out


def with_lags_t(z: torch.Tensor, lags: int) -> torch.Tensor:
    """with_lags for a torch tensor (zeros without history: those rows carry no target)."""
    T = z.shape[0]
    return torch.cat([torch.cat([torch.zeros_like(z[:l]), z[:T - l]]) for l in range(lags + 1)], dim=1)


def ridge_solve(Z, y, rho: float):
    """Ridge with an unpenalized intercept on the standardized inputs Z (numpy or torch): (c, b) with
    (Zc'Zc + rho I) c = Zc'yc, the solution of sklearn's Ridge(alpha=rho)."""
    zm, ym = Z.mean(0), y.mean()
    Zc, yc = Z - zm, y - ym
    if torch.is_tensor(Z):
        c = torch.linalg.solve(Zc.T @ Zc + rho * torch.eye(Z.shape[1], device=Z.device, dtype=Z.dtype), Zc.T @ yc)
    else:
        c = np.linalg.solve(Zc.T @ Zc + rho * np.eye(Z.shape[1]), Zc.T @ yc)
    return c, ym - zm @ c


def predict(theta: np.ndarray, head: dict) -> np.ndarray:
    """The head's prediction (standardized) from monthly attention, as model.predict."""
    return head["b"] + ((theta - head["mbar"]) / head["s"]) @ head["c"]


def reported_coefs(head: dict) -> np.ndarray:
    """c with its unidentified direction (s: s'z = 0 for the standardized shares) projected out."""
    c, s = head["c"], head["s"]
    return c - (c @ s) / (s @ s) * s


def r2(y, yhat, benchmark_mean):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(1 - ((y - yhat) ** 2).sum() / ((y - benchmark_mean) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lam", type=float, required=True, help="prediction weight (0 = topic model alone)")
    ap.add_argument("--n-topics", type=int, default=50)
    ap.add_argument("--train-end", default="1999-12")
    ap.add_argument("--val-end", default="2003-12")
    ap.add_argument("--target-start", type=int, default=0)
    ap.add_argument("--target-window", type=int, default=3)
    ap.add_argument("--num-steps", type=int, default=20000)
    ap.add_argument("--months-per-batch", type=int, default=0, help="0 = every training month in every step")
    ap.add_argument("--docs-per-month", type=int, default=32)
    ap.add_argument("--month-attention", choices=["cache", "batch"], default="cache",
                    help="the month term's attention: all records (a cache, gradient through the batch) or the "
                         "batch's records (with the subsampling correction)")
    ap.add_argument("--no-subsample-correction", action="store_true", help="--month-attention batch only")
    ap.add_argument("--two-step-head", action="store_true", help="theta detached in the prediction term")
    ap.add_argument("--features", choices=["level", "change"], default="level",
                    help="the head's inputs: the monthly attention, or its change from the previous months' average")
    ap.add_argument("--change-window", type=int, default=12)
    ap.add_argument("--head", choices=["learned", "ridge"], default="learned",
                    help="learned: the head's coefficients are parameters (Adam); ridge: re-solved at every step")
    ap.add_argument("--head-lags", type=int, default=0, help="ridge head: inputs of months t .. t-head_lags")
    ap.add_argument("--head-penalty", type=float, default=None, help="ridge head: its penalty (fixed)")
    ap.add_argument("--select-after", type=int, default=None, help="first step eligible for selection "
                    "(default: the end of KL annealing)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--prior-lr", type=float, default=1e-4)
    ap.add_argument("--encoder-hidden", type=int, nargs="+", default=[1024, 512, 256])
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="encoder dropout; 0 so that the cached and evaluated attention are those of the trained "
                         "encoder (0.1 brought no validation gain in the setup runs)")
    ap.add_argument("--kl-anneal", type=int, nargs=2, default=[0, 5000])
    ap.add_argument("--init-docs", type=int, default=200_000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-samples", type=int, default=8)
    ap.add_argument("--val-docs", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtm-dir", default=None, help="document-term matrix (default <data>/dtm, 1984-2011)")
    a = ap.parse_args()
    if a.select_after is None:
        a.select_after = a.kl_anneal[1]
    if a.head == "ridge":
        assert a.head_penalty is not None and a.features == "change", "--head ridge needs --head-penalty, change inputs"
    assert a.head == "ridge" or a.head_lags == 0, "lags need the ridge head"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = common.RESULTS / a.tag
    ckdir = common.CHECKPOINTS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    ckdir.mkdir(parents=True, exist_ok=True)

    dtm = Path(a.dtm_dir) if a.dtm_dir else common.DATA / "dtm"
    built = json.loads((dtm / "settings.json").read_text())
    vocab_end = built.get("train_end", built.get("vocab_end"))  # 02: the training period; w2_build_dtm: --vocab-end
    assert vocab_end <= a.train_end, f"the DTM's vocabulary uses records up to {vocab_end}, after the training period"
    X = sp.load_npz(dtm / "X.npz").tocsr()
    rec = pd.read_csv(dtm / "records.csv.gz", dtype={"month": str})
    vocab = (dtm / "vocab.txt").read_text().split("\n")[:-1]
    months = sorted(rec.month.unique())
    T = len(months)
    mpos = {m: i for i, m in enumerate(months)}
    month_of = rec.month.map(mpos).to_numpy()
    assert (np.diff(month_of) >= 0).all(), "records must be in date order"

    mt = common.month_targets(months, a.train_end, a.val_end, a.target_start, a.target_window)
    split, y_raw = mt.split.to_numpy(), mt.target.to_numpy()
    if a.features == "change":
        common.assert_consecutive(months)
        if a.month_attention != "cache":
            raise ValueError("--features change needs --month-attention cache")
    hist = ~np.isnan(with_lags(common.head_features(np.zeros((T, 1)), a.features, a.change_window)[0],
                               a.head_lags)).any(axis=1)
    y_raw = np.where(hist, y_raw, np.nan)  # months without the features' history have no target
    tr = (split == "train") & ~np.isnan(y_raw)
    y_mean, y_sd = float(y_raw[tr].mean()), float(y_raw[tr].std())
    y_std = (y_raw - y_mean) / y_sd
    mt.to_csv(out / "months.csv", index=False)

    train_months = np.flatnonzero(split == "train")
    n_month = np.bincount(month_of, minlength=T)
    m_start = np.concatenate([[0], np.cumsum(n_month)[:-1]])
    N_train = int(n_month[train_months].sum())
    T_y = int(tr.sum())
    n = a.docs_per_month
    M = a.months_per_batch or len(train_months)
    assert n_month[train_months].min() >= n, f"a training month has fewer than {n} records"
    print(f"{X.shape[0]:,} records, V = {X.shape[1]:,}, {T} months ({len(train_months)} training, {T_y} with a "
          f"target); {N_train:,} training records; target mean {y_mean:.3f}, sd {y_sd:.3f}", flush=True)

    Xtr = X[: N_train]  # records are in date order and the training months come first
    assert (month_of[:N_train] <= train_months.max()).all() and (month_of[N_train:] > train_months.max()).all()
    bg_counts = np.asarray(Xtr.sum(axis=0)).ravel() + 1.0
    background = bg_counts / bg_counts.sum()
    init = kmeans_topics(Xtr, a.n_topics, a.init_docs, a.seed)
    init = init / init.sum(axis=1, keepdims=True)
    log_bg = torch.tensor(np.log(background))
    model = JointTopicModel(X.shape[1], a.n_topics, a.encoder_hidden, a.dropout, log_bg,
                            torch.tensor(np.log(init) - np.log(background)[None, :])).to(dev)
    main_params = [p for n_, p in model.named_parameters() if not n_.startswith("prior_")]
    opt = torch.optim.Adam([{"params": main_params, "lr": a.lr},
                            {"params": [model.prior_mu, model.prior_L_raw], "lr": a.prior_lr}])

    csr = GPUCSR(X, dev)
    month_of_t = torch.as_tensor(month_of, device=dev)
    y_t = torch.as_tensor(np.nan_to_num(y_std, nan=0.0), dtype=torch.float32, device=dev)
    has_y = torch.as_tensor(~np.isnan(y_std) & (split == "train"), device=dev)
    start_t = torch.as_tensor(m_start, device=dev)
    count_t = torch.as_tensor(n_month, device=dev)
    train_t = torch.as_tensor(train_months, device=dev)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    val_rec = np.flatnonzero(split[month_of] == "val")
    val_idx = torch.as_tensor(np.sort(np.random.default_rng(a.seed).choice(val_rec, min(a.val_docs, len(val_rec)),
                                                                             replace=False)), device=dev)
    pred_scale = a.lam * T_y / N_train
    tr_t = torch.as_tensor(np.flatnonzero(tr), device=dev)
    use_cache = a.lam > 0 and a.month_attention == "cache"
    month_tr = month_of_t[:N_train]

    def refresh_cache():
        c_ = record_theta(model, csr, N_train, a.eval_samples, dev)
        return c_, torch.zeros(T, a.n_topics, device=dev).index_add_(0, month_tr, c_)

    if use_cache:
        cache, msum = refresh_cache()
    elif a.lam > 0:  # batch attention (level features): the standardization is refreshed at every evaluation
        mbar_t, s_t = train_stats(torch.as_tensor(month_attention(model, csr, month_of_t, T, a.eval_samples),
                                                  device=dev), tr_t)
    (out / "config.json").write_text(json.dumps({**vars(a), "N_train": N_train, "T_y": T_y, "V": X.shape[1],
                                                 "y_mean": y_mean, "y_sd": y_sd, "pred_scale": pred_scale}, indent=1))

    def evaluate(step):
        theta = month_attention(model, csr, month_of_t, T, a.eval_samples)
        feats = with_lags(common.head_features(theta, a.features, a.change_window)[0], a.head_lags)
        mbar, s = train_stats(feats, np.flatnonzero(tr))
        if a.head == "ridge":
            cr, br = ridge_solve((feats[tr] - mbar) / s, y_std[tr], a.head_penalty)
            c = {"c": cr, "b": float(br), "mbar": mbar, "s": s}
        else:
            c = {"c": model.head.detach().cpu().numpy(), "b": model.head_bias.item(), "mbar": mbar, "s": s}
        yhat = predict(feats, c) * y_sd + y_mean
        row = {"step": step, "val_perplexity_bound": perplexity_bound(model, csr, val_idx)}
        bench = y_raw[tr].mean()
        for s in ["train", "val", "test"]:
            m = (split == s) & ~np.isnan(y_raw)
            row[f"{s}_r2"] = r2(y_raw[m], yhat[m], bench) if a.lam > 0 else np.nan
            row[f"{s}_corr"] = float(np.corrcoef(y_raw[m], yhat[m])[0, 1]) if a.lam > 0 else np.nan
        torch.save({"model": model.state_dict(), "step": step}, ckdir / f"step_{step}.pt")
        return row, theta, c

    def save(which, step, theta, c):
        beta = model.beta().detach().cpu().numpy()
        np.save(out / f"theta_{which}.npy", theta)
        np.save(out / f"beta_{which}.npy", beta.astype(np.float32))
        np.savez(out / f"head_{which}.npz", **c)
        feats = with_lags(common.head_features(theta, a.features, a.change_window)[0], a.head_lags)
        pd.DataFrame({"month": months, "split": split, "target": y_raw,
                      "prediction": predict(feats, c) * y_sd + y_mean}).to_csv(out / f"predictions_{which}.csv",
                                                                               index=False)
        tw = top_words(beta, background, vocab)
        tw.insert(1, "share", theta[split == "train"].mean(0).round(4))
        coef = (c["c"].reshape(a.head_lags + 1, -1).sum(0) if a.head == "ridge"  # summed over lags
                else reported_coefs(c))
        tw.insert(2, "coef", coef.round(4))  # standardized outcome per sd of the topic's attention
        tw.to_csv(out / f"topic_words_{which}.csv", index=False)
        (out / f"{which}_step.txt").write_text(str(step))

    rows, best, t0 = [], (-np.inf, None), time.time()
    for step in range(1, a.num_steps + 1):
        mb = train_t if M == len(train_months) else train_t[torch.randperm(len(train_t), generator=g, device=dev)[:M]]
        rand = torch.rand(M, int(count_t[mb].max()), generator=g, device=dev)
        rand[torch.arange(rand.shape[1], device=dev)[None, :] >= count_t[mb][:, None]] = 2.0
        offs = torch.argsort(rand, dim=1)[:, :n]  # n distinct records of each month
        idx = (start_t[mb][:, None] + offs).reshape(-1)
        x = csr.rows(idx)
        mu, logvar = model.encode(x)
        theta = model.sample_theta(mu, logvar)
        kl_w = min(1.0, max(0.0, (step - a.kl_anneal[0]) / max(1, a.kl_anneal[1] - a.kl_anneal[0])))
        w = (len(train_months) * count_t[mb].float() / N_train).repeat_interleave(n)  # N_t / n per record, x M n
        rec_loss = model.nll(x, theta, model.beta())
        kl = model.kl(mu, logvar)
        loss = (w * (rec_loss + kl_w * kl)).mean()
        pred = torch.zeros((), device=dev)
        if a.lam > 0:
            th = theta.view(M, n, -1)
            if use_cache:
                new = theta.detach()
                msum.index_add_(0, month_of_t[idx], new - cache[idx])
                cache[idx] = new
                tb = (th.detach() if a.two_step_head else th).mean(1)
                att = msum / count_t.float().clamp_min(1).unsqueeze(1)
                att = att.index_add(0, mb, tb - tb.detach())  # value of the cache, gradient of the batch means
                feats = with_lags_t(head_features_t(att, a.features, a.change_window), a.head_lags)
                mbar_t, s_t = train_stats(feats.detach(), tr_t)  # constants in the gradient
                if a.head == "ridge":  # the exact ridge fit of the training months, re-solved on the current topics
                    Z = (feats[tr_t] - mbar_t) / s_t
                    cr, br = ridge_solve(Z, y_t[tr_t], a.head_penalty)
                    fit = torch.zeros_like(y_t).index_put((tr_t,), br + Z @ cr)
                    sq = (y_t[mb] - fit[mb]) ** 2
                else:
                    sq = (y_t[mb] - model.predict(feats[mb], mbar_t, s_t)) ** 2
            else:
                proj = model.predict(th.detach() if a.two_step_head else th, mbar_t, s_t)  # each record's part
                sq = (y_t[mb] - proj.mean(1)) ** 2
                if not a.no_subsample_correction:
                    sq = sq - (1 - n / count_t[mb].float()) * proj.var(1, unbiased=True) / n
            mask = has_y[mb].float()
            if mask.sum() > 0:
                pred = (sq * mask).sum() / mask.sum() / 2
                loss = loss + pred_scale * pred
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: recon {rec_loss.mean().item()}, "
                                     f"kl {kl.mean().item()}, pred {pred.item()}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step}: loss {loss.item():.2f} recon/word {(rec_loss.sum() / x.sum()).item():.3f} "
                  f"kl {kl.mean().item():.2f} pred {pred.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.num_steps:
            row, th_all, c = evaluate(step)
            if use_cache:
                cache, msum = refresh_cache()
            elif a.lam > 0:
                mbar_t, s_t = train_stats(torch.as_tensor(th_all, device=dev, dtype=torch.float32), tr_t)
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            print("  eval", {k: round(v, 4) for k, v in row.items()}, flush=True)
            score = row["val_r2"] if a.lam > 0 else step
            if step >= a.select_after and np.isfinite(score) and score > best[0]:
                best = (score, step)
                save("selected", step, th_all, c)
    save("final", a.num_steps, th_all, c)
    if best[1] is None:  # no eligible evaluation: the final step is the selected one
        save("selected", a.num_steps, th_all, c)
        best = (np.nan, a.num_steps)
    print(f"selected step {best[1]}; done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
