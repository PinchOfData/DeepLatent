"""Step 08: the words of a saved step-05 model, read from the model itself.

    python 08_words_from_model.py --tag votes_text_103_118_baselines_v2 --step 80000

Loads the model saved at a checkpoint (summary.json's "model" path; --step warm_<N> for the
end of the warm start) and reports, for every word of the vocabulary:
  loading         the text decoder's word loading beta_w (log-odds of word w: baseline +
                  beta_w * position), when the model decodes the speeches (not --text
                  encode-only); positive: used more toward the Republican end
  encoder_effect  the speech encoder's average marginal effect of one more use of word w on its
                  position (the mean of its output), at each unit's observed speeches and prior
                  covariates, averaged over the units with speeches
both signed like the run's estimates (positive: toward DW-NOMINATE's conservative end). In a
log-linear word model a word's use rises or falls monotonically with the position, so no word
sits "at" the center; the center words listed are the most used words whose value is closest to
0 (among the --center-pool most used words): words whose use does not depend on the position.
Writes results/<tag>/words_model_<step>.csv and prints the lists.
"""

import argparse
import importlib
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import common as c

fit03 = importlib.import_module("03_fit_votes")
TEXT = "speech_terms"


def speech_encoder(state, key=TEXT):
    """The speech expert's MLP rebuilt from the saved encoder state dict (eval mode)."""
    from deeplatent.autoencoders import EncoderMLP

    prefix = next(k for k in state if k.endswith(f"{key}.encoder.enc_0.weight"))
    prefix = prefix[: -len("encoder.enc_0.weight")]
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    n = len({k.split(".")[1] for k in sub})
    dims = [sub["encoder.enc_0.weight"].shape[1]] + [
        sub[f"encoder.enc_{i}.weight"].shape[0] for i in range(n)]
    enc = EncoderMLP(encoder_dims=dims)
    enc.load_state_dict(sub)
    return enc.eval()


def show(df, col, top, pool):
    d = df.dropna(subset=[col])
    print(f"\n=== {col}: Republican end (largest)")
    print(", ".join(f"{w} ({v:+.3g}, n={n})" for w, v, n in
                    d.nlargest(top, col)[["word", col, "count"]].itertuples(index=False)))
    print(f"=== {col}: Democratic end (smallest)")
    print(", ".join(f"{w} ({v:+.3g}, n={n})" for w, v, n in
                    d.nsmallest(top, col)[["word", col, "count"]].itertuples(index=False)))
    common = d.nlargest(pool, "count")
    center = common.loc[common[col].abs().nsmallest(top).index]
    print(f"=== {col}: center (closest to 0 among the {pool:,} most used words)")
    print(", ".join(f"{w} ({v:+.2g}, n={n})" for w, v, n in
                    center[["word", col, "count"]].itertuples(index=False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--step", default="80000", help="checkpoint step, or warm_<N>")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--center-pool", type=int, default=2000)
    args = ap.parse_args()
    out = c.MODULE / "results" / args.tag
    record = json.loads((out / "summary.json").read_text())
    warm = args.step.startswith("warm_")
    step = int(args.step.removeprefix("warm_"))
    rec = next(r for r in record["checkpoints"]
               if r["step"] == step and ("phase" in r) == warm)
    sign = rec["sign"]
    ckpt = torch.load(rec["model"], map_location="cpu", weights_only=False)

    lo, hi = map(int, record["args"]["congresses"].split("-"))
    units, rc, votes = fit03.load(range(lo, hi + 1))
    house = c.DATA / "house"
    S = sp.load_npz(house / "speech_counts.npz").tocsr()[units["unit"].to_numpy()]
    vocab = [w for w in (house / "vocab.txt").read_text().split("\n") if w]
    from deeplatent import Corpus

    corpus = Corpus(units[["bioguide", "trend", "congress"]],
                    prevalence="~ 0 + C(bioguide) + C(bioguide):trend",
                    modalities={"speech": {"views": {"terms": {"type": "bow", "matrix": S,
                                                               "vocab": vocab}}}})
    assert list(corpus.prevalence_colnames) == list(ckpt["prevalence_colnames"]), \
        "prior covariates differ from the saved model's"
    Z = np.asarray(corpus.M_prevalence_covariates, dtype=np.float32)

    words = pd.DataFrame({"word": vocab, "count": np.asarray(S.sum(0)).ravel(),
                          "units_using": np.asarray((S > 0).sum(0)).ravel()})
    if TEXT in ckpt["decoders"]:
        W = ckpt["decoders"][TEXT]["decoder.dec_0.weight"].numpy()
        words["loading"] = sign * W[:, 0]
    enc = speech_encoder(ckpt["encoder"])
    rows = np.flatnonzero(np.asarray(S.sum(1)).ravel() > 0)
    grad = np.zeros(S.shape[1])
    for b in range(0, len(rows), 512):
        r = rows[b:b + 512]
        x = torch.tensor(np.hstack([S[r].toarray().astype(np.float32), Z[r]]),
                         requires_grad=True)
        enc(x)[:, 0].sum().backward()  # mean of the speech expert; units independent
        grad += x.grad[:, :S.shape[1]].sum(0).numpy()
    words["encoder_effect"] = sign * grad / len(rows)
    words.to_csv(out / f"words_model_{args.step}.csv", index=False)

    print(f"{args.tag}, {args.step}: {rec['model']}; {len(rows):,} units with speeches")
    for col in ("loading", "encoder_effect"):
        if col in words:
            show(words, col, args.top, args.center_pool)
    if "loading" in words:
        print(f"\ncorrelation of loading and encoder_effect over words: "
              f"{np.corrcoef(words['loading'], words['encoder_effect'])[0, 1]:+.3f}")
    print(f"\nwrote {out / f'words_model_{args.step}.csv'}")


if __name__ == "__main__":
    main()
