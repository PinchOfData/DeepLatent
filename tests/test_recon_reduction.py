"""
Tests for the per-document reconstruction scale that fixes posterior collapse
in the DeepLatent VAEs.

Reconstruction is the per-document negative log-likelihood (summed over the
tokens/votes within a document, averaged over the batch), so it shares the
per-document scale of the KL and ``w_prior=1`` is the true ELBO. These tests:

1. verify the BoW reconstruction equals the per-document NLL (sum over vocab / B),
   NOT the old per-token average (sum over vocab / total tokens);
2. verify the same for IdealPointNN vote reconstruction (sum over votes / B);
3. check that at ``w_prior=1`` the VAE keeps a healthy KL (i.e. does not collapse).

The scale tests use ``ae_type="ae"`` so the encoder/decoder forward is
deterministic (no sampling) and the reconstruction can be recomputed exactly.
"""
import numpy as np
import torch
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader

from deeplatent import Corpus, GTM, IdealPointNN, generate_documents, generate_ideal_points


def _text_corpus(num_docs=1200, num_topics=4, vocab=120, words=150, seed=0):
    _, df, *_ = generate_documents(
        num_docs, num_topics, vocab, doc_topic_prior="logistic_normal",
        min_words=words, max_words=words, random_seed=seed)
    vec = CountVectorizer()
    vec.fit(df["doc_clean_0"])
    corpus = Corpus(df, modalities={
        "text": {"column": "doc_clean_0",
                 "views": {"bow": {"type": "bow", "vectorizer": vec}}}})
    m = corpus.processed_modalities["text"]["bow"]["matrix"]
    if scipy.sparse.issparse(m):
        corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(
            m.todense(), dtype=np.float32)
    return corpus


def test_bow_reconstruction_is_per_document(tmp_path):
    """BoW recon == sum_vocab NLL / batch_size (per-document), not / total tokens."""
    num_topics = 4
    corpus = _text_corpus(num_topics=num_topics)
    tm = GTM(
        train_data=corpus, n_topics=num_topics, ae_type="ae", vi_type="mean_field",
        doc_topic_prior="logistic_normal",
        encoder_args={"text_bow": {"hidden_dims": [64]}},
        decoder_args={"text_bow": {"hidden_dims": [64]}},
        batch_size=128, num_steps=0, num_workers=0, print_every_n_steps=10**9,
        return_best_model=False, ckpt_folder=str(tmp_path), seed=42)
    tm.encoder.eval(); tm.decoders.eval()

    batch = next(iter(DataLoader(corpus, batch_size=128, shuffle=False, num_workers=0)))
    bow = batch["modalities"]["text"]["bow"].to(tm.device)
    B = bow.shape[0]
    total_tokens = float(bow.sum())

    tm.step_batch(batch, corpus, validation=True)
    got = float(tm.reconstruction_loss)

    with torch.no_grad():
        theta, _, _ = tm.encoder({"text_bow": bow}, prevalence_covariates=None)
        recon = tm.decoders["text_bow"](theta)
        nll = float(-(bow * torch.log_softmax(recon, dim=1)).sum())
    expected_per_document = nll / B
    expected_per_token = nll / total_tokens

    assert np.isclose(got, expected_per_document, rtol=1e-4), (
        f"recon {got:.4f} != per-document {expected_per_document:.4f}")
    # The per-document scale is ~doc-length times the per-token average.
    assert not np.isclose(got, expected_per_token, rtol=1e-2)
    assert np.isclose(got / expected_per_token, total_tokens / B, rtol=1e-4)


def test_idealpointnn_votes_reconstruction_is_per_document(tmp_path):
    """Vote recon == sum_votes NLL / batch_size (per-document), not / observed votes."""
    num_bills = 80
    _, df, *_ = generate_ideal_points(
        num_politicians=600, dim_ideal_points=1, num_bills=num_bills,
        num_survey_questions=2, doc_length=40, vocab_size=60,
        seed=0, progress_bar=False)
    corpus = Corpus(df, modalities={
        "vote": {"column": [f"vote_{i+1}" for i in range(num_bills)],
                 "views": {"responses": {"type": "vote"}}}})
    m = IdealPointNN(
        ae_type="ae", vi_type="mean_field", n_ideal_points=1, train_data=corpus,
        encoder_args={"vote_responses": {"hidden_dims": [32]}},
        decoder_args={"vote_responses": {"hidden_dims": []}},
        batch_size=128, num_steps=0, num_workers=0, print_every_n_steps=10**9,
        return_best_model=False, ckpt_folder=str(tmp_path), seed=7)
    m.encoder.eval(); m.decoders.eval()

    batch = next(iter(DataLoader(corpus, batch_size=128, shuffle=False, num_workers=0)))
    votes = batch["modalities"]["vote"]["responses"]["matrix"].to(m.device)
    mask = (~batch["modalities"]["vote"]["responses"]["mask"]).to(m.device)
    B = votes.shape[0]
    observed = float(mask.sum())

    m.step_batch(batch, corpus, validation=True)
    got = float(m.reconstruction_loss)

    with torch.no_grad():
        _, z, _ = m.encoder({"vote_responses": votes}, prevalence_covariates=None)
        recon = m.decoders["vote_responses"](z)
        losses = torch.nn.BCEWithLogitsLoss(reduction="none")(recon, votes)
        nll = float((losses * mask).sum())
    expected_per_document = nll / B
    expected_per_vote = nll / observed

    assert np.isclose(got, expected_per_document, rtol=1e-4), (
        f"recon {got:.4f} != per-document {expected_per_document:.4f}")
    assert np.isclose(got / expected_per_vote, observed / B, rtol=1e-4)


def test_vae_w1_does_not_collapse(tmp_path):
    """At w_prior=1 (the true ELBO) the VAE keeps a healthy, non-zero KL."""
    num_topics = 4
    corpus = _text_corpus(num_topics=num_topics)
    tm = GTM(
        train_data=corpus, n_topics=num_topics, ae_type="vae", vi_type="iaf",
        update_prior=False, doc_topic_prior="logistic_normal",
        optim_args={"main": {"lr": 1e-3, "weight_decay": 0.0}},
        encoder_args={"text_bow": {"hidden_dims": [64]}},
        decoder_args={"text_bow": {"hidden_dims": [64]}},
        w_prior=1.0, batch_size=128, num_steps=500, num_workers=0,
        print_every_n_steps=10**9, return_best_model=False,
        ckpt_folder=str(tmp_path), seed=42)
    # train_div_losses store w_prior * KL; w_prior=1 so they equal the raw KL.
    kl = float(np.mean(tm.train_div_losses[-100:]))
    assert kl > 0.5, f"posterior appears collapsed at w_prior=1 (KL={kl:.3f})"
