"""Wave 2, step 3: fit the joint topic-prediction model (model.py) for one window of the expanding-window evaluation
and one prediction weight lam.

The window ending in year E (w2_common.windows) trains on the records of the months up to E - val_years, chooses
the training step on the validation years (E - val_years + 1 .. E), and its model is then applied, frozen, to every
later month. The topic model is model.py's (mixture decoder, logistic-normal prior, k-means start); the head sees the
literature's inputs (w2_common.literature_inputs: backward moving average, linear trend removed and standardized on
the training months) of month t only (--head-lags 0), linear: yhat_t = b + c'x_t. The forecasting layer (w2_forecast:
a LASSO) uses months t .. t-6; giving the head all 7 x K inputs would let it fit the ~330 training months of the first
windows exactly (rank 7(K-1) = 343) and the outcome would only teach the topics noise. As in 03_fit_joint.py, every step holds n
records of every training month, the month's attention comes from a cache of every training record's theta with
the gradient through the batch records, and the objective is

    mean_i [ -ELBO_i ]  +  lam * (T_y / N) * mean_t [ (y_t - yhat_t)^2 / 2 ]      (y standardized on training months)

--head ridge replaces the learned head by a ridge regression re-solved exactly at every step on the training months,
on the same inputs as the forecasting layer (months t .. t-6), its penalty chosen every 50 steps by generalized
cross-validation (w2_common.ridge_gcv_t: no gradient through the choice; the gradient reaches the topics through
the solution). With the penalty, the head can see all 7 x K inputs without fitting the training months exactly.

Every --eval-every steps: the attention of every month of the sample (the frozen model applied to all records),
the head's R^2 by period, the validation perplexity bound, a checkpoint. The selected step is the best validation
R^2 from --select-after on (lam > 0; the last step for lam = 0).

Writes <results>/w2/<grid>/<tag>/: config.json, metrics.csv, months.csv, theta_<which>.npy (months x K, all months),
beta_<which>.npy, head_<which>.npz, predictions_<which>.csv, topic_words_<which>.csv for which in {selected, final}.

--inputs change uses w2_common.change_inputs (attention minus its trailing --change-window month mean, standardized)
instead of the literature's; --head-penalty fixes the ridge head's penalty instead of GCV (which picks no penalty
on these data: 2026-10-12 diagnostic).

Usage: python w2_fit_joint.py --dtm-dir <scratch>/data/dtm_w2 --grid k50 --window-end 1979 --lam 1e4 --seed 0
"""
import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import common
import w2_common
from model import GPUCSR, JointTopicModel

f03 = importlib.import_module("03_fit_joint")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtm-dir", required=True)
    ap.add_argument("--grid", required=True, help="results go to <results>/w2/<grid>/")
    ap.add_argument("--window-end", type=int, required=True)
    ap.add_argument("--val-years", type=int, default=5)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--n-topics", type=int, default=50)
    ap.add_argument("--target-start", type=int, default=0)
    ap.add_argument("--target-window", type=int, default=3)
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--inputs", choices=["literature", "change"], default="literature")
    ap.add_argument("--change-window", type=int, default=12)
    ap.add_argument("--head-penalty", type=float, default=None, help="ridge head: fixed penalty (default: GCV)")
    ap.add_argument("--lags", type=int, default=6, help="lags of the inputs (history required for a target)")
    ap.add_argument("--head-lags", type=int, default=0, help="lags the head itself sees (--head adam)")
    ap.add_argument("--head", choices=["adam", "ridge"], default="adam",
                    help="adam: a linear head learned with the topics; ridge: re-solved at every step, penalty by GCV")
    ap.add_argument("--num-steps", type=int, default=20000)
    ap.add_argument("--docs-per-month", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--prior-lr", type=float, default=1e-4)
    ap.add_argument("--encoder-hidden", type=int, nargs="+", default=[1024, 512, 256])
    ap.add_argument("--kl-anneal", type=int, nargs=2, default=[0, 5000])
    ap.add_argument("--select-after", type=int, default=None, help="default: the end of KL annealing")
    ap.add_argument("--init-docs", type=int, default=200_000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-samples", type=int, default=8)
    ap.add_argument("--val-docs", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.select_after is None:
        a.select_after = a.kl_anneal[1]
    if a.head == "ridge":
        a.head_lags = a.lags  # the forecasting layer's inputs
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"joint_lam{a.lam:g}_e{a.window_end}_s{a.seed}"
    out = common.RESULTS / "w2" / a.grid / tag
    ckdir = common.CHECKPOINTS / "w2" / a.grid / tag
    out.mkdir(parents=True, exist_ok=True)
    ckdir.mkdir(parents=True, exist_ok=True)

    dtm = Path(a.dtm_dir)
    X = sp.load_npz(dtm / "X.npz").tocsr()
    rec = pd.read_csv(dtm / "records.csv.gz", dtype={"month": str})
    vocab = (dtm / "vocab.txt").read_text().split("\n")[:-1]
    months = sorted(rec.month.unique())
    common.assert_consecutive(months)
    T, K = len(months), a.n_topics
    month_of = rec.month.map({m: i for i, m in enumerate(months)}).to_numpy()
    assert (np.diff(month_of) >= 0).all(), "records must be in date order"

    train_end, val_end = f"{a.window_end - a.val_years}-12", f"{a.window_end}-12"
    mt = common.month_targets(months, train_end, val_end, a.target_start, a.target_window)
    split = mt.split.to_numpy()
    hist = w2_common.history_mask(T, a.smooth, a.lags, a.inputs, a.change_window)
    y_raw = np.where(hist, mt.target.to_numpy(), np.nan)
    tr = (split == "train") & ~np.isnan(y_raw)
    y_mean, y_sd = float(y_raw[tr].mean()), float(y_raw[tr].std())
    y_std = (y_raw - y_mean) / y_sd
    mt.assign(target=y_raw).to_csv(out / "months.csv", index=False)

    train_months = np.flatnonzero(split == "train")
    n_month = np.bincount(month_of, minlength=T)
    m_start = np.concatenate([[0], np.cumsum(n_month)[:-1]])
    N_train = int(n_month[train_months].sum())
    T_y, n, M = int(tr.sum()), a.docs_per_month, len(train_months)
    assert n_month[train_months].min() >= n, f"a training month has fewer than {n} records"
    print(f"window ending {a.window_end}: training {months[train_months[0]]}..{train_end} ({M} months, {N_train:,} "
          f"records, {T_y} with a target), validation to {val_end}; {X.shape[0]:,} records, V = {X.shape[1]:,}",
          flush=True)

    Xtr = X[:N_train]
    assert (month_of[:N_train] <= train_months.max()).all() and (month_of[N_train:] > train_months.max()).all()
    bg = np.asarray(Xtr.sum(axis=0)).ravel() + 1.0
    background = bg / bg.sum()
    init = f03.kmeans_topics(Xtr, K, a.init_docs, a.seed)
    init = init / init.sum(axis=1, keepdims=True)
    model = JointTopicModel(X.shape[1], K, a.encoder_hidden, 0.0, torch.tensor(np.log(background)),
                            torch.tensor(np.log(init) - np.log(background)[None, :])).to(dev)
    P = K * (a.head_lags + 1)
    model.head = torch.nn.Parameter(torch.zeros(P, device=dev))  # on the literature inputs
    main_params = [p for n_, p in model.named_parameters() if not n_.startswith("prior_")]
    opt = torch.optim.Adam([{"params": main_params, "lr": a.lr},
                            {"params": [model.prior_mu, model.prior_L_raw], "lr": a.prior_lr}])

    csr = GPUCSR(X, dev)
    month_of_t = torch.as_tensor(month_of, device=dev)
    y_t = torch.as_tensor(np.nan_to_num(y_std, nan=0.0), dtype=torch.float32, device=dev)
    has_y = torch.as_tensor(tr, device=dev)
    start_t, count_t = torch.as_tensor(m_start, device=dev), torch.as_tensor(n_month, device=dev)
    train_t = torch.as_tensor(train_months, device=dev)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    val_rec = np.flatnonzero(split[month_of] == "val")
    val_idx = torch.as_tensor(np.sort(np.random.default_rng(a.seed).choice(val_rec, min(a.val_docs, len(val_rec)),
                                                                             replace=False)), device=dev)
    pred_scale = a.lam * T_y / N_train
    month_tr = month_of_t[:N_train]
    est = split == "train"
    est_t = torch.as_tensor(np.flatnonzero(est), device=dev)
    tr_t = torch.as_tensor(np.flatnonzero(tr), device=dev)
    (out / "config.json").write_text(json.dumps({**vars(a), "tag": tag, "train_end": train_end, "val_end": val_end,
                                                 "N_train": N_train, "T_y": T_y, "V": X.shape[1], "y_mean": y_mean,
                                                 "y_sd": y_sd, "pred_scale": pred_scale}, indent=1))

    def refresh_cache():
        c_ = f03.record_theta(model, csr, N_train, a.eval_samples, dev)
        return c_, torch.zeros(T, K, device=dev).index_add_(0, month_tr, c_)

    cache, msum = refresh_cache() if a.lam > 0 else (None, None)

    def predict(theta):
        Xin, ok = w2_common.make_inputs(theta, est, a.inputs, a.smooth, a.change_window, a.head_lags)
        if a.head == "ridge":
            c, b, rho = w2_common.ridge_gcv(Xin[tr], y_std[tr], **({"grid": [a.head_penalty]} if a.head_penalty
                                                                    else {}))
            head = {"c": c, "b": b, "rho": rho}
        else:
            c, b = model.head.detach().cpu().numpy(), model.head_bias.item()
            head = {"c": c, "b": b}
        return np.where(ok, b + np.nan_to_num(Xin) @ c, np.nan) * y_sd + y_mean, head

    def evaluate(step):
        theta = f03.month_attention(model, csr, month_of_t, T, a.eval_samples)
        yhat, head = predict(theta)
        row = {"step": step, "val_perplexity_bound": f03.perplexity_bound(model, csr, val_idx)}
        bench = y_raw[tr].mean()
        for s in ["train", "val", "test"]:
            m = (split == s) & ~np.isnan(y_raw) & ~np.isnan(yhat)
            row[f"{s}_r2"] = f03.r2(y_raw[m], yhat[m], bench) if a.lam > 0 and m.any() else np.nan
        torch.save({"model": model.state_dict(), "step": step}, ckdir / "last.pt")
        return row, theta, yhat, head

    def save(which, step, theta, yhat, head):
        beta = model.beta().detach().cpu().numpy()
        np.save(out / f"theta_{which}.npy", theta)
        np.save(out / f"beta_{which}.npy", beta.astype(np.float32))
        np.savez(out / f"head_{which}.npz", **head)
        pd.DataFrame({"month": months, "split": split, "target": y_raw, "prediction": yhat}).to_csv(
            out / f"predictions_{which}.csv", index=False)
        tw = f03.top_words(beta, background, vocab)
        tw.insert(1, "share", theta[est].mean(0).round(4))
        tw.to_csv(out / f"topic_words_{which}.csv", index=False)
        (out / f"{which}_step.txt").write_text(str(step))

    rows, best, t0, rho = [], (-np.inf, None), time.time(), None
    for step in range(1, a.num_steps + 1):
        mb = train_t
        rand = torch.rand(M, int(count_t[mb].max()), generator=g, device=dev)
        rand[torch.arange(rand.shape[1], device=dev)[None, :] >= count_t[mb][:, None]] = 2.0
        idx = (start_t[mb][:, None] + torch.argsort(rand, dim=1)[:, :n]).reshape(-1)
        x = csr.rows(idx)
        mu, logvar = model.encode(x)
        theta = model.sample_theta(mu, logvar)
        kl_w = min(1.0, max(0.0, (step - a.kl_anneal[0]) / max(1, a.kl_anneal[1] - a.kl_anneal[0])))
        w = (M * count_t[mb].float() / N_train).repeat_interleave(n)  # N_t / n per record, x M n
        rec_loss = model.nll(x, theta, model.beta())
        kl = model.kl(mu, logvar)
        loss = (w * (rec_loss + kl_w * kl)).mean()
        pred = torch.zeros((), device=dev)
        if a.lam > 0:
            new = theta.detach()
            msum.index_add_(0, month_of_t[idx], new - cache[idx])
            cache[idx] = new
            tb = theta.view(M, n, -1).mean(1)
            att = msum / count_t.float().clamp_min(1).unsqueeze(1)
            att = att.index_add(0, mb, tb - tb.detach())  # value of the cache, gradient of the batch means
            Xin = w2_common.make_inputs_t(att, est_t, a.inputs, a.smooth, a.change_window, a.head_lags)
            if a.head == "ridge":
                c, b, rho = w2_common.ridge_gcv_t(Xin[tr_t], y_t[tr_t],
                                                  rho=a.head_penalty or (None if step % 50 == 1 else rho))
                pred = ((y_t[tr_t] - (b + Xin[tr_t] @ c)) ** 2).mean() / 2
            else:
                yhat = model.head_bias + Xin[mb] @ model.head
                mask = has_y[mb].float()
                pred = (((y_t[mb] - yhat) ** 2) * mask).sum() / mask.sum() / 2
            loss = loss + pred_scale * pred
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            extra = f" rho {rho:.3g}" if a.lam > 0 and a.head == "ridge" else ""
            print(f"step {step}: loss {loss.item():.2f} recon/word {(rec_loss.sum() / x.sum()).item():.3f} "
                  f"kl {kl.mean().item():.2f} pred {pred.item():.4f}{extra} ({time.time() - t0:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.num_steps:
            row, th_all, yhat_all, head = evaluate(step)
            if a.lam > 0:
                cache, msum = refresh_cache()
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            print("  eval", {k: round(v, 4) for k, v in row.items()}, flush=True)
            score = row["val_r2"] if a.lam > 0 else step
            if step >= a.select_after and np.isfinite(score) and score > best[0]:
                best = (score, step)
                save("selected", step, th_all, yhat_all, head)
                torch.save({"model": model.state_dict(), "step": step}, ckdir / "selected.pt")
    save("final", a.num_steps, th_all, yhat_all, head)
    if best[1] is None:
        save("selected", a.num_steps, th_all, yhat_all, head)
        best = (np.nan, a.num_steps)
    print(f"selected step {best[1]}; done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
