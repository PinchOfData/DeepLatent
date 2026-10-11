#!/usr/bin/env python3
"""
Clean texts for the canonical GST left/right model across all datasets.

Datasets:
- fr_professions: French professions de foi
- fr_parliament: French parliamentary debates
- us_websites: US election websites
- us_congress: US congressional speeches
- us_tweets: US congressional tweets
- us_tv_ads: US TV ads

Usage:
    python3 <script> --dataset us_congress
    python3 <script> --dataset fr_parliament --sample-size 0
    python3 <script> --dataset all  # Process all datasets
"""

from __future__ import annotations

import argparse
import html
import re
import unicodedata
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.congress import build_sessions, assign_congress


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


# ============================================================================
# Language identification (fasttext lid.176)
# ============================================================================

# Retained in the historical topic-model checkpoint directory for compatibility.
_LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
_LID_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "Codes" / "python" / "ckpt" / "topic_model" / "lid.176.bin"
)
# ISO-639-1 codes expected by fasttext lid.176
_LANG_CODE = {"english": "en", "french": "fr"}

_lid_model = None


# spaCy NER labels to drop from cleaned tokens (people + places).
# en_core_web_lg uses PERSON/GPE/LOC; fr_core_news_lg uses PER/LOC.
DROP_ENT_LABELS = {
    "english": ["PERSON", "GPE", "LOC"],
    "french":  ["PER", "LOC"],
}

DEFAULT_N_PROCESS = 4


def _get_lid_model():
    """Load fasttext lid.176 once per process, downloading if missing."""
    global _lid_model
    if _lid_model is not None:
        return _lid_model
    try:
        import fasttext
    except ImportError as e:
        raise RuntimeError("fasttext not installed. Run: pip install fasttext") from e
    if not _LID_MODEL_PATH.exists():
        import urllib.request
        _LID_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading fasttext lid.176 to {_LID_MODEL_PATH} (~130MB)...")
        urllib.request.urlretrieve(_LID_MODEL_URL, _LID_MODEL_PATH)
    _lid_model = fasttext.load_model(str(_LID_MODEL_PATH))
    return _lid_model


def filter_by_language_raw(df: pd.DataFrame, text_col: str, lang: str) -> pd.DataFrame:
    """Keep only rows whose fasttext-predicted top language equals `lang`.

    Runs on raw text (before spaCy cleaning) so the language model sees
    function words and punctuation it was trained on.
    """
    if df.empty:
        return df
    model = _get_lid_model()
    texts = df[text_col].astype(str).str.replace("\n", " ", regex=False).tolist()
    labels, _ = model.predict(texts, k=1)
    preds = [lab[0].replace("__label__", "") for lab in labels]
    mask = pd.Series([p == lang for p in preds], index=df.index)
    n_drop = int((~mask).sum())
    print(f"  Language filter ({lang}): dropped {n_drop:,} of {len(df):,} "
          f"documents ({n_drop / max(len(df), 1) * 100:.2f}%)")
    return df.loc[mask].copy()


# ============================================================================
# Dataset configurations
# ============================================================================

@dataclass
class DatasetConfig:
    name: str
    input_path: Path
    output_path: Path
    text_col: str
    language: str  # "english" or "french"
    stopwords: set[str]
    has_congress: bool = False
    id_col: str | None = None  # If None, use index
    date_col: str | None = None  # For congress assignment
    date_format: str | None = None  # Explicit format for non-ISO source dates
    extra_cols: list[str] = field(default_factory=list)
    standard_professions_only: bool = False
    file_format: str = "csv"  # "csv" or "feather"
    keep_cols: list[str] | None = None  # If set, subset to these columns after reading
    # Dataset-specific raw-text preprocessor applied before spaCy. Returns
    # (text_for_spacy, sideband_tokens); sideband tokens are appended verbatim
    # to doc_clean after spaCy processing. Used for tweets to preserve
    # @mentions, #hashtags, and domains that spaCy would otherwise mangle.
    preprocessor: Callable[[str], tuple[str, list[str]]] | None = None


ROOT = _project_root()

# Note: DATASETS is defined after stopwords section below


# ============================================================================
# Stopwords
# ============================================================================

# General stopwords (applied to all datasets of that language)
STOPWORDS_EN_GENERAL = {
    "people", "thing", "time", "year", "make", "think", "want", "need",
    "go", "get", "know", "take", "tell", "talk", "say", "use", "way",
    "many", "more", "other", "let", "ask", "re", "ve",
}

STOPWORDS_FR_GENERAL: set[str] = set()

# Dataset-specific stopwords (combined with general stopwords)
_STOPWORDS_US_CONGRESS = {
    "speaker", "house", "senate", "act", "yield", "second", "minute",
    "amendment", "yea", "nay", "gentleman", "gentlewoman", "conference",
    "conferee", "thank", "vote", "quorum", "majority", "bill", "object",
    "unanimous", "consent", "distinguished", "nomination", "nominee",
    "debate", "roll", "call", "rollcall", "provision", "title", "rise",
    "support", "morning", "noon", "today", "tomorrow", "chairman",
    "resolution", "colleague", "report", "floor", "nominate", "committee",
    "subcommittee", "member", "program", "aisle", "waive", "clause",
    "section", "rule", "consideration", "legislation", "provide",
    "confirmation", "issue", "subsection", "adjourn", "adjournment",
    "motion", "objection", "confirm", "question", "session", "order",
    "proposition", "matter", "point", "germane", "bench", "ratification",
    "senator", "cloture", "designee", "agree", "disagree", "line",
    "strike", "insert", "paragraph", "following", "tribute", "sympathy",
    "page", "end", "begin", "number", "word", "requisite", "deem",
    "commend", "responder", "veto", "lieu", "oclock", "unanimousconsent",
    "subjectmatter", "germaneness", "b2", "cr",
}

_STOPWORDS_US_WEBSITES: set[str] = set()

_STOPWORDS_FR_PARLIAMENT: set[str] = set()

_STOPWORDS_FR_PROFESSIONS: set[str] = set()

_STOPWORDS_US_TWEETS: set[str] = set()

_STOPWORDS_US_TV_ADS: set[str] = set()

# Combined stopwords for each dataset
STOPWORDS_US_CONGRESS = STOPWORDS_EN_GENERAL | _STOPWORDS_US_CONGRESS
STOPWORDS_US_WEBSITES = STOPWORDS_EN_GENERAL | _STOPWORDS_US_WEBSITES
STOPWORDS_FR_PARLIAMENT = STOPWORDS_FR_GENERAL | _STOPWORDS_FR_PARLIAMENT
STOPWORDS_FR_PROFESSIONS = STOPWORDS_FR_GENERAL | _STOPWORDS_FR_PROFESSIONS
STOPWORDS_US_TWEETS = STOPWORDS_EN_GENERAL | _STOPWORDS_US_TWEETS
STOPWORDS_US_TV_ADS = STOPWORDS_EN_GENERAL | _STOPWORDS_US_TV_ADS


# ============================================================================
# Dataset-specific preprocessors
# ============================================================================

# Tweets need URLs, @mentions, and #hashtags stripped before spaCy sees them
# (spaCy's POS tagger mangles these, and the NLTK word list used to filter
# them out entirely). They're captured as a sideband and re-appended to
# doc_clean after spaCy cleaning so downstream topic models still see them.

_TWEET_TLDS = r"com|org|net|edu|gov|io|co|us|uk|de|fr|ca|info|biz"
_TWEET_URL_PATTERN = re.compile(
    rf"\b(?:https?://|www\.)\S+|\b\S+\.(?:{_TWEET_TLDS})\b\S*"
)
_TWEET_DOMAIN_PATTERN = re.compile(
    rf"\b(?:https?://|www\.)?\S*?([A-Za-z0-9.-]+\.(?:{_TWEET_TLDS}))\b"
)
_TWEET_RT_PATTERN = re.compile(r"^rt\s+")
_TWEET_MENTION_PATTERN = re.compile(r"@\w+")
_TWEET_HASHTAG_PATTERN = re.compile(r"#\w+")

# URL-shortener and tracking domains that carry no semantic content;
# dropped from the sideband so they don't crowd out real vocabulary.
_TWEET_DOMAIN_STOPLIST = {"t.co"}


def preprocess_tweet(text: str) -> tuple[str, list[str]]:
    """Strip RT prefix, URLs, @mentions, and #hashtags from a tweet.

    Returns (text_for_spacy, sideband_tokens) where sideband_tokens is the
    list of mentions, hashtags, and captured domains to re-append after
    spaCy cleaning.
    """
    text = str(text).lower().strip()
    # Decode HTML entities first (e.g. "&amp;" -> "&") so spaCy doesn't
    # later tag "amp" as a standalone noun.
    text = html.unescape(text)
    text = _TWEET_RT_PATTERN.sub("", text)

    domains = _TWEET_DOMAIN_PATTERN.findall(text)
    domains = [d for d in domains if d.lower() not in _TWEET_DOMAIN_STOPLIST]
    text = _TWEET_URL_PATTERN.sub(" ", text)

    mentions = _TWEET_MENTION_PATTERN.findall(text)
    text = _TWEET_MENTION_PATTERN.sub(" ", text)

    hashtags = _TWEET_HASHTAG_PATTERN.findall(text)
    text = _TWEET_HASHTAG_PATTERN.sub(" ", text)

    text = re.sub(r"\s+", " ", text).strip()
    sideband = mentions + hashtags + domains
    return text, sideband


# ============================================================================
# Dataset definitions (after stopwords)
# ============================================================================

DATASETS = {
    "us_congress": DatasetConfig(
        name="US Congress speeches",
        input_path=ROOT / "Data" / "us_congress" / "us_congress_speeches_1994_2024.csv",
        output_path=ROOT / "Data" / "processed" / "us_congress_clean.csv",
        text_col="text",
        language="english",
        stopwords=STOPWORDS_US_CONGRESS,
        has_congress=True,
        id_col="speaker_bioguide",
        date_col="date",
        extra_cols=["chamber"],
    ),
    "us_websites": DatasetConfig(
        name="US election websites",
        input_path=ROOT / "Data" / "processed" / "us_websites_classified.csv",
        output_path=ROOT / "Data" / "processed" / "us_websites_clean.csv",
        text_col="text_snap_content",
        language="english",
        stopwords=STOPWORDS_US_WEBSITES,
        id_col="candidate",
        date_col="year",
    ),
    "fr_parliament": DatasetConfig(
        name="French parliamentary debates",
        input_path=ROOT / "Data" / "french_parliament" / "collapsed_1958_2024.csv",
        output_path=ROOT / "Data" / "processed" / "fr_parliament_clean.csv",
        text_col="content",
        language="french",
        stopwords=STOPWORDS_FR_PARLIAMENT,
        id_col="speaker_name",
        date_col="date_sitting",
        # collapsed_1958_2024.csv stores date_sitting month-first ("07/01/1993" =
        # 1 July 1993): across 3.28M slash-formatted rows the first field never
        # exceeds 12 while the second does in 62% of them, and under %m/%d/%Y every
        # dated speech falls inside the speaker's MP term.  Parsing as %d/%m/%Y
        # (until 2026-08-25) yielded NaT for those 62% and swapped day/month for
        # the rest.  ~38k rows carry textual dates ("1er octobre 2002",
        # "Mardi 6 décembre 2005"); parse_date_column() handles them.
        date_format="%m/%d/%Y",
    ),
    "fr_professions": DatasetConfig(
        name="French professions de foi",
        input_path=ROOT / "Data" / "professions_de_foi" / "professions_de_foi.csv",
        output_path=ROOT / "Data" / "processed" / "fr_professions_clean.csv",
        text_col="text",
        language="french",
        stopwords=STOPWORDS_FR_PROFESSIONS,
        id_col="id_doc",
        date_col="year",
        extra_cols=["falc"],
        standard_professions_only=True,
    ),
    "us_tweets": DatasetConfig(
        name="US Congress tweets",
        input_path=ROOT / "Data" / "tweets" / "us_congress_tweets.csv",
        output_path=ROOT / "Data" / "processed" / "us_tweets_clean.csv",
        text_col="tweet_text",
        language="english",
        stopwords=STOPWORDS_US_TWEETS,
        has_congress=True,
        preprocessor=preprocess_tweet,
        id_col="id",
        date_col="time",
    ),
    "us_tv_ads": DatasetConfig(
        name="US TV ads",
        input_path=ROOT / "Data" / "tv_ads" / "ads-final.feather",
        output_path=ROOT / "Data" / "processed" / "us_tv_ads_clean.csv",
        text_col="text",
        language="english",
        stopwords=STOPWORDS_US_TV_ADS,
        file_format="feather",
        keep_cols=[
            "id", "race", "party", "dma", "state", "district2", "state2",
            "tone", "district", "primary", "sponsor", "year", "total_cost",
            "total_airings", "total_dmas", "text", "sponsor_or_positive_id",
            "negative_id", "sponsor_or_positive_name", "sponsor_or_positive_party",
            "negative_name", "negative_party", "outside_sponsored",
            "not_full_match", "weight",
        ],
    ),
}


# ============================================================================
# Date parsing
# ============================================================================

_FR_MONTHS = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
    "decembre": 12,
}
_FR_TEXT_DATE = re.compile(
    r"^(?:(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s+)?"
    r"(\d{1,2})(?:er)?\s+([a-z]+)\s+(\d{4})$"
)


def _french_textual_to_iso(value: object) -> str | None:
    """'1er octobre 2002' / 'Mardi 6 décembre 2005' -> '2002-10-01' / '2005-12-06'."""
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip().lower()
    match = _FR_TEXT_DATE.match(text)
    if match is None:
        return None
    day, month_name, year = match.groups()
    month = _FR_MONTHS.get(month_name)
    if month is None:
        return None
    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


def parse_date_column(values: pd.Series, date_format: str | None) -> pd.Series:
    """Parse a date column with the dataset's format, then rescue rows the format
    rejects but a French textual date ('9 decembre 1959') can express."""
    parsed = pd.to_datetime(values, format=date_format, errors="coerce", utc=True)
    missing = parsed.isna() & values.notna()
    if missing.any():
        rescued = values[missing].map(_french_textual_to_iso)
        parsed.loc[missing] = pd.to_datetime(rescued, format="%Y-%m-%d", errors="coerce", utc=True)
    return parsed


# ============================================================================
# Text Cleaner
# ============================================================================

class TextCleaner:
    def __init__(
        self,
        *,
        language: str,
        stopwords: set[str],
        pos_tags_to_keep: list[str] | None = None,
    ) -> None:
        from utils.text_processor import text_processor_new as text_processor

        spacy_model = "en_core_web_lg" if language == "english" else "fr_core_news_lg"

        self._stopwords = stopwords
        self._processor = text_processor(
            spacy_model,
            pos_tags_to_keep=pos_tags_to_keep or ["VERB", "NOUN", "PROPN", "ADJ"],
            drop_ent_labels=DROP_ENT_LABELS[language],
        )
        self._language = language

    def _filter_tokens(self, text: str) -> str:
        toks: list[str] = []
        for w in str(text).split():
            w_ = w.strip()
            if len(w_) < 3:
                continue
            if w_.lower() in self._stopwords:
                continue
            toks.append(w_)
        return " ".join(toks)

    def clean_df(
        self,
        df: pd.DataFrame,
        *,
        text_col: str,
        batch_size: int,
        n_process: int,
        min_tokens: int,
        show_progress: bool,
    ) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        doc_clean = self._processor.process_docs(
            df[text_col],
            batch_size=batch_size,
            n_process=n_process,
            progress_bar=bool(show_progress),
        )

        out = df.copy()
        out["doc_clean"] = doc_clean
        out = out.dropna(subset=["doc_clean"]).copy()
        if out.empty:
            return out

        out["doc_clean"] = out["doc_clean"].astype(str).apply(self._filter_tokens)
        out["word_count"] = out["doc_clean"].str.split().str.len()
        out = out[out["word_count"] >= int(min_tokens)].copy()
        return out


# ============================================================================
# Processing functions
# ============================================================================

def _append_csv(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def process_dataset(
    config: DatasetConfig,
    *,
    sample_size: int | None,
    min_tokens: int,
    batch_size: int,
    n_process: int,
    chunksize: int,
    start_from_scratch: bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"Processing: {config.name}")
    print(f"  Input: {config.input_path}")
    print(f"  Output: {config.output_path}")
    print(f"{'='*60}")

    if not config.input_path.exists():
        raise FileNotFoundError(
            f"Required input for {config.name} does not exist: {config.input_path}"
        )

    vocab_path = config.output_path.parent / f"{config.output_path.stem}_vocab.txt"

    if start_from_scratch:
        if config.output_path.exists():
            config.output_path.unlink()
        if vocab_path.exists():
            vocab_path.unlink()

    cleaner = TextCleaner(language=config.language, stopwords=config.stopwords)
    sessions = build_sessions() if config.has_congress else None
    vocab_counter: Counter[str] = Counter()

    # Read input data
    if config.file_format == "feather":
        full_df = pd.read_feather(config.input_path)
        if config.keep_cols:
            full_df = full_df[[c for c in config.keep_cols if c in full_df.columns]]
        chunks = [full_df]
    else:
        usecols = [config.text_col]
        if config.id_col:
            usecols.append(config.id_col)
        if config.date_col:
            usecols.append(config.date_col)
        usecols.extend(config.extra_cols)
        usecols = list(dict.fromkeys(usecols))
        chunks = pd.read_csv(config.input_path, usecols=usecols, chunksize=chunksize)

    total_seen = 0
    total_written = 0

    for chunk in chunks:
        if sample_size is not None and total_seen >= sample_size:
            break

        # Preserve original CSV row index
        chunk["raw_idx"] = chunk.index

        if config.standard_professions_only:
            falc = pd.to_numeric(chunk["falc"], errors="coerce")
            if falc.isna().any() or not falc.isin([0, 1]).all():
                raise ValueError("French profession `falc` must be coded 0/1")
            n_falc = int(falc.eq(1).sum())
            if n_falc:
                print(f"  Excluding {n_falc:,} FALC alternative documents in this chunk")
            chunk = chunk.loc[falc.eq(0)].copy()

        # Drop rows without text
        chunk = chunk.dropna(subset=[config.text_col]).copy()
        if chunk.empty:
            continue

        # Apply sample_size cap
        if sample_size is not None:
            remaining = max(sample_size - total_seen, 0)
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

        total_seen += len(chunk)

        # Drop documents whose raw text isn't in the dataset's target language.
        # Run before spaCy so we don't spend compute on docs we'll discard.
        lang_code = _LANG_CODE.get(config.language)
        if lang_code is not None:
            chunk = filter_by_language_raw(chunk, config.text_col, lang_code)
            if chunk.empty:
                continue

        # Parse date and assign congress if applicable
        if config.has_congress and config.date_col and sessions:
            chunk["date"] = parse_date_column(chunk[config.date_col], config.date_format)
            chunk["congress"] = assign_congress(chunk["date"], sessions)
            chunk = chunk.dropna(subset=["congress"]).copy()
            if chunk.empty:
                continue
        elif config.date_col:
            # Parse date for output — but only if it looks like a date string,
            # not a plain integer year column
            if config.date_col != "year":
                chunk["date"] = parse_date_column(chunk[config.date_col], config.date_format)

        # Dataset-specific preprocessing: rewrite text for spaCy and stash
        # sideband tokens (mentions/hashtags/domains) to re-append later.
        if config.preprocessor is not None:
            preprocessed = chunk[config.text_col].map(config.preprocessor)
            chunk[config.text_col] = [p[0] for p in preprocessed]
            chunk["_sideband"] = [p[1] for p in preprocessed]

        # Clean text
        cleaned = cleaner.clean_df(
            chunk,
            text_col=config.text_col,
            batch_size=batch_size,
            n_process=n_process,
            min_tokens=min_tokens,
            show_progress=True,
        )

        if cleaned.empty:
            continue

        # Re-append sideband tokens and refresh word_count for surviving rows.
        if "_sideband" in cleaned.columns:
            sideband_strs = cleaned["_sideband"].apply(
                lambda ts: " ".join(ts) if ts else ""
            )
            cleaned["doc_clean"] = (cleaned["doc_clean"] + " " + sideband_strs).str.strip()
            cleaned["word_count"] = cleaned["doc_clean"].str.split().str.len()

        # Select output columns: keep all input columns except original text, plus doc_clean and word_count
        out_cols = [
            c for c in cleaned.columns
            if c not in (config.text_col, "doc_clean", "word_count", "_sideband")
        ]
        out_cols.extend(["doc_clean", "word_count"])

        out_df = cleaned[out_cols].copy()
        _append_csv(config.output_path, out_df)
        total_written += len(out_df)

        # Accumulate vocabulary counts from this chunk's cleaned tokens.
        for doc in out_df["doc_clean"]:
            if doc:
                vocab_counter.update(str(doc).split())

        if total_seen % (chunksize * 5) == 0:
            print(f"Processed {total_seen:,} rows so far; wrote {total_written:,} cleaned rows...")

    print(f"Done. Processed {total_seen:,} input rows; wrote {total_written:,} cleaned rows.")
    print(f"Output -> {config.output_path}")

    if vocab_counter:
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with vocab_path.open("w", encoding="utf-8") as f:
            for word, count in vocab_counter.most_common():
                f.write(f"{word}\t{count}\n")
        print(
            f"Vocab -> {vocab_path} "
            f"({len(vocab_counter):,} unique / {sum(vocab_counter.values()):,} tokens)"
        )


# ============================================================================
# Main
# ============================================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean texts for the canonical GST model.")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()) + ["all"],
        required=True,
        help="Dataset to process, or 'all' for all datasets.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Rows to process per dataset. Use 0 for full dataset (default: 100000).",
    )
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-process", type=int, default=DEFAULT_N_PROCESS)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument(
        "--start-from-scratch",
        action="store_true",
        help="Delete existing output CSVs before running.",
    )

    return parser


def main() -> int:
    args = build_argparser().parse_args()

    if args.n_process < 1:
        raise ValueError("--n-process must be a positive integer")

    sample_size = args.sample_size if args.sample_size > 0 else None

    if args.dataset == "all":
        datasets_to_process = list(DATASETS.values())
    else:
        datasets_to_process = [DATASETS[args.dataset]]

    missing_inputs = [
        config.input_path for config in datasets_to_process
        if not config.input_path.exists()
    ]
    if missing_inputs:
        formatted = "\n  ".join(str(path) for path in missing_inputs)
        raise FileNotFoundError(
            "Cannot start cleaning because required input files are missing:\n  "
            + formatted
        )

    for config in datasets_to_process:
        process_dataset(
            config,
            sample_size=sample_size,
            min_tokens=args.min_tokens,
            batch_size=args.batch_size,
            n_process=args.n_process,
            chunksize=args.chunksize,
            start_from_scratch=args.start_from_scratch,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
