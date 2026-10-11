"""Paths and shared helpers for Application 1 (US Congress speech polarization).

Everything is relative to the module folder (on the HPC: /home/Gauthier/app1_us_congress,
synced from replication/app1_us_congress; the datasets live only there, not in git):
    code/, inputs/, slurm/   this code, the curated vocabulary, the job scripts
    data/raw/       hein-bound/, hein-daily/ (links to the Gentzkow-Shapiro-Taddy release in
                    ~/generalized_topic_model/data/us_congressional_record),
                    us_congress_speeches_1994_2024.csv (govinfo: speeches downloaded from govinfo.gov with the
                    congressional-record parser),
                    HSall_members.csv (Voteview), lid.176.bin (fasttext language id)
    data/speeches/  01: every speech, one parquet per (source, congress), sample flags as columns
    data/clean/     02: P&P-cleaned text of every English speech (the expensive step, run once)
    data/terms/<vocabulary>/   03: term counts per speech for a given vocabulary (cheap, rerunnable)
    results/        04-06 outputs
Sample restrictions (other than P&P's English filter) and vocabularies are applied after cleaning.
APP1_DATA_ROOT and APP1_RESULTS override the data and results folders.
"""
import importlib.util
import os
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
CODE = MODULE / "code"
INPUTS = MODULE / "inputs"
VENDOR = CODE / "vendor_pp"
VOCAB_PATH = INPUTS / "vocab_us.txt"

DATA_ROOT = Path(os.environ.get("APP1_DATA_ROOT", MODULE / "data"))
RAW = DATA_ROOT / "raw"
SPEECHES = DATA_ROOT / "speeches"
CLEAN = DATA_ROOT / "clean"
TERMS = DATA_ROOT / "terms"
RESULTS = Path(os.environ.get("APP1_RESULTS", MODULE / "results"))
# Model checkpoints (large: ~0.5 GB each) go to scratch on the HPC (slurm/05_fit.sbatch sets it).
CHECKPOINTS = Path(os.environ.get("APP1_CHECKPOINTS", RESULTS / "checkpoints"))

# Everything available is built and cleaned once: both editions of the Record where they overlap
# (bound 43rd-111th, daily 97th-114th) and govinfo (103rd-118th, 1994-2024).
ALL_TASKS = (
    [("bound", c) for c in range(43, 112)]
    + [("daily", c) for c in range(97, 115)]
    + [("govinfo", c) for c in range(103, 119)]
)  # SLURM array index -> (source, congress)

# Main build, as Gentzkow, Shapiro and Taddy (2019): bound edition where available (43rd-111th),
# daily edition thereafter (112th-114th), extended with govinfo for the 115th-118th (2017-2024).
MAIN_SOURCES = (
    [("bound", c) for c in range(43, 112)]
    + [("daily", c) for c in range(112, 115)]
    + [("govinfo", c) for c in range(115, 119)]
)


def task(i: int):
    """(source, congress) of SLURM array task i."""
    return ALL_TASKS[i]


def worker_tasks(worker: int, n_workers: int) -> list:
    """Task indices for one of n_workers cleaning jobs (the cluster runs at most 10 jobs per user):
    longest-first assignment by the size of each task's step-01 parquet, so workers finish together."""
    sizes = [(SPEECHES / f"{stem(s, c)}.parquet").stat().st_size for s, c in ALL_TASKS]
    load, assign = [0] * n_workers, [[] for _ in range(n_workers)]
    for i in sorted(range(len(ALL_TASKS)), key=lambda i: -sizes[i]):
        w = min(range(n_workers), key=lambda w: load[w])
        assign[w].append(i)
        load[w] += sizes[i]
    return assign[worker]


def stem(source: str, congress: int) -> str:
    return f"{source}/{congress:03d}"


def terms_dir(vocab_path=VOCAB_PATH) -> Path:
    """Counts for a vocabulary live under data/terms/<vocabulary file stem>/."""
    return TERMS / Path(vocab_path).stem


def as_bool(s):
    """Flag column from parquet (boolean) or CSV (True/False strings, missing) to a plain boolean."""
    return s.astype(str).str.lower().eq("true")


def gst_sample(meta):
    """Speeches in the sample of Gentzkow, Shapiro and Taddy (2019, Section 2): by a legislator
    identified by name, Republican or Democrat, with voting rights, not an Extension of Remarks.
    (Non-English speeches are already dropped by the P&P cleaning, step 02.)"""
    return (as_bool(meta["identified"]) & meta["party"].isin(["R", "D"]) & as_bool(meta["voting"])
            & ~as_bool(meta["extension"]))


def load_vocab(vocab_path=VOCAB_PATH) -> list:
    """A (1,2)-gram vocabulary in file order. The P&P curated US vocabulary is validated as in
    Codes/R/02_party_classifier_gst.R::load_frozen_vocab."""
    vocab = Path(vocab_path).read_text(encoding="utf-8").splitlines()
    assert all(v and v == v.strip() and v == v.lower() for v in vocab)
    assert len(set(vocab)) == len(vocab) and all(len(v.split(" ")) <= 2 for v in vocab)
    return vocab


def decoder_config(cfg: dict, content_colnames: list, init_topics=None) -> dict:
    """The GTM's text decoder from a fit configuration (05_fit_gtm.py arguments or config.json).

    "linear" (fits before 2026-10-08): word logits W2 W1 [theta, x], a hidden layer of
    --decoder-hidden units without activation or bias (topics add up inside the softmax).
    "sage": the STM's decoder (deeplatent.SAGEDecoder), a mixture of topics whose word
    distributions are softmax(m + kappa_k + x' kappa_cov [+ topic x party deviations]), with
    --interactions "none", "party" (one topic x party deviation for all congresses) or
    "congress_party" (one per congress), --l1 (L1 penalty on the deviations, in word counts over
    the corpus) or --jeffreys (SAGE's parameter-free Normal-Jeffreys prior on every deviation, estimated
    as in SCHOLAR: a ridge with a precision per coefficient, 1 until step --jeffreys-start, then
    1/deviation^2 refreshed every --jeffreys-every steps), --background "fixed" (log corpus frequencies) or
    "learned", and optional
    initial topic word distributions (05_fit_gtm.py --init kmeans; not needed to load a saved model)."""
    if cfg.get("decoder", "linear") == "linear":
        return {"hidden_dims": [cfg["decoder_hidden"]], "activation": None, "bias": False,
                "dropout": cfg.get("dropout", 0.0)}
    party = [n for n in content_colnames if n.endswith(":party_signed")]
    interactions = {"none": {}, "party": {"party_signed": party},
                    "congress_party": {n: [n] for n in party}}[cfg.get("interactions", "none")]
    out = {"type": "sage", "interactions": interactions, "l1": cfg.get("l1", 0.0),
           "background": cfg.get("background", "fixed")}
    if cfg.get("jeffreys"):  # SAGE's Normal-Jeffreys prior on all deviations, SCHOLAR-style (deeplatent.SAGEDecoder)
        # (the schedule only matters for training; configs before --jeffreys-start used one interval)
        out.update({"jeffreys": True, "jeffreys_every": cfg["jeffreys_every"],
                    "jeffreys_start": cfg.get("jeffreys_start", cfg["jeffreys_every"])})
    if init_topics is not None:
        out["init_topics"] = init_topics
    return out


class PrecomputedCounts:
    """Stands in for a fitted CountVectorizer in deeplatent.Corpus: the bow column holds row
    positions into the precomputed term count matrix, and transform() returns those rows
    (Corpus calls transform() on a vectorizer that has a vocabulary_)."""

    def __init__(self, counts, vocab: list):
        self.counts = counts.tocsr()
        self.vocab = list(vocab)
        self.vocabulary_ = {v: j for j, v in enumerate(self.vocab)}

    def transform(self, rows):
        import numpy as np

        return self.counts[np.asarray(rows, dtype=np.int64)]

    def get_feature_names_out(self):
        import numpy as np

        return np.asarray(self.vocab, dtype=object)


def load_pp_cleaning():
    """The vendored, unmodified P&P module Codes/python/scripts/01_clean_texts.py.

    It puts vendor_pp/ on sys.path itself, so its `from utils...` imports resolve to the
    vendored utils/text_processor.py and utils/congress.py (see vendor_pp/MD5SUMS). The
    fasttext model path is pointed at raw/lid.176.bin (compute nodes are offline)."""
    path = VENDOR / "scripts" / "01_clean_texts.py"
    spec = importlib.util.spec_from_file_location("pp_clean_texts", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pp_clean_texts"] = mod
    spec.loader.exec_module(mod)
    mod._LID_MODEL_PATH = RAW / "lid.176.bin"
    return mod
