"""
Tests for validation over the whole test set (AUDIT_REPORT MED-2).

Validation used to score only the first batch of the test set (the loader was rebuilt at
every step and not shuffled). It now averages the per-document loss over every test
document, every ``validate_every_n_steps`` training steps, and ``patience`` counts
evaluations. These tests check:

1. ``evaluate`` equals the document-weighted mean of the per-batch losses (deterministic
   ``ae_type="ae"`` so it can be recomputed exactly; the last batch is partial);
2. evaluations happen at the configured steps and are recorded in ``val_steps``;
3. with test data, patience counts evaluations; without it, training steps (unchanged);
4. the VAE + mixture-of-Gaussians path used in applications runs and gives a finite loss.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader

from deeplatent import Corpus, GTM, generate_documents

N_TEST = 300  # 64 + 64 + 64 + 64 + 44: the last batch is partial
BATCH = 64


def _corpora(num_docs=900, num_topics=4, vocab=120, words=100, seed=0):
    _, df, *_ = generate_documents(
        num_docs, num_topics, vocab, doc_topic_prior="logistic_normal",
        min_words=words, max_words=words, random_seed=seed)
    vec = CountVectorizer().fit(df["doc_clean_0"])

    def corpus(part):
        return Corpus(part.reset_index(drop=True), modalities={
            "text": {"column": "doc_clean_0",
                     "views": {"bow": {"type": "bow", "vectorizer": vec}}}})

    return corpus(df.iloc[N_TEST:]), corpus(df.iloc[:N_TEST])


def _gtm(train, test, tmp_path, **kwargs):
    args = dict(
        train_data=train, test_data=test, n_topics=4, ae_type="ae", vi_type="mean_field",
        doc_topic_prior="logistic_normal",
        encoder_args={"text_bow": {"hidden_dims": [32]}},
        decoder_args={"text_bow": {"hidden_dims": [32]}},
        batch_size=BATCH, num_workers=0, print_every_n_steps=10**9,
        return_best_model=False, ckpt_folder=str(tmp_path), seed=42, device="cpu")
    args.update(kwargs)
    return GTM(**args)


def test_evaluate_averages_every_test_document(tmp_path):
    train, test = _corpora()
    tm = _gtm(train, test, tmp_path, num_steps=5)
    loader = DataLoader(test, batch_size=BATCH, shuffle=False, num_workers=0)

    per_batch, sizes = [], []
    for batch in loader:
        per_batch.append(tm.step_batch(batch, test, validation=True, verbose=False))
        sizes.append(batch["modalities"]["text"]["bow"].shape[0])
    assert sizes == [64, 64, 64, 64, 44]
    expected = np.average(per_batch, weights=sizes)

    got = tm.evaluate(loader, test)
    assert np.isclose(got, expected, rtol=1e-6)
    assert tm.loss == got
    # The old validation (first batch only) gives a different number.
    assert not np.isclose(per_batch[0], expected, rtol=1e-6)


def test_validation_every_n_steps(tmp_path):
    train, test = _corpora()
    tm = _gtm(train, test, tmp_path, num_steps=10, validate_every_n_steps=4)
    assert tm.val_steps == [4, 8]
    assert len(tm.val_losses) == len(tm.val_recon_losses) == len(tm.val_div_losses) == 2
    assert len(tm.train_losses) == 10


def test_patience_counts_evaluations_with_test_data(tmp_path):
    # A huge tolerance means only the first evaluation counts as an improvement, so with
    # patience=1 training stops at the second evaluation (step 6), not at step 2.
    train, test = _corpora()
    tm = _gtm(train, test, tmp_path, num_steps=100, validate_every_n_steps=3,
              patience=1, patience_tol=1e9)
    assert tm.val_steps == [3, 6]
    assert len(tm.train_losses) == 6


def test_patience_counts_steps_without_test_data(tmp_path):
    train, _ = _corpora()
    tm = _gtm(train, None, tmp_path, num_steps=100, patience=2, patience_tol=1e9)
    assert tm.val_losses == [] and tm.val_steps == []
    assert len(tm.train_losses) == 3


def test_vae_mixture_of_gaussians_validation(tmp_path):
    train, test = _corpora()
    tm = _gtm(train, test, tmp_path, ae_type="vae", vi_type="mixture_of_gaussians",
              mixture_components=3, num_steps=4, validate_every_n_steps=2,
              encoder_args={"text_bow": {"hidden_dims": [32], "dropout": 0.1}},
              decoder_args={"text_bow": {"hidden_dims": [32], "dropout": 0.1}})
    assert tm.val_steps == [2, 4]
    assert all(np.isfinite(v) for v in tm.val_losses)
    assert all(d > 0 for d in tm.val_div_losses)
