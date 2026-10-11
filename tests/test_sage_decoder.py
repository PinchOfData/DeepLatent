"""
Tests for SAGEDecoder, the STM's bag-of-words decoder (a mixture of topics whose word
distributions are SAGE log-linear models):

1. forward() equals the brute-force mixture sum_k theta_k softmax(m + kappa_k + x'kappa_cov
   + z'kappa_int[:, k]) and is normalized, with and without covariates and interactions;
2. topic_log_probs / covariate_log_probs match their definitions;
3. a GTM with the decoder trains through the mixture-of-Gaussians path, and its readouts
   (topic-word distributions, logit components) are consistent;
4. on mixture data from the package simulator it recovers the topics and topic shares;
5. the L1 penalty shrinks the deviations; a fixed background stays fixed, a learned one moves;
6. save_model / load_model restores the decoder (incl. the background and interactions);
7. interactions: content columns listed together form one covariate with topic-specific
   effects;
8. init_topics: the topics start at the given word distributions;
9. the Normal-Jeffreys prior (SCHOLAR-style): penalty and refresh formulas; refreshes at
   jeffreys_start and every jeffreys_every steps after it (both required); the training
   gradient adds prec * eta / N_train; validation neither penalizes nor refreshes;
   checkpoints keep the precisions and a mismatched checkpoint gives a clear error; it
   excludes the L1 penalty; on data with a few true party deviations it shrinks the null
   ones toward 0 while keeping the true ones, strong or weak (|z| ~ 2.5).
"""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer

from deeplatent import GTM, Corpus, SAGEDecoder, generate_documents


def _brute_force(dec, theta, x):
    """log sum_k theta_k beta_k(x), beta materialized per row."""
    out = []
    for t, xi in zip(theta, x):
        logits = dec.background + dec.kappa_topic  # K x V
        if dec.content_covariate_size:
            logits = logits + xi @ dec.kappa_cov
        if dec.n_interactions:
            z = xi @ dec.interactions
            logits = logits + torch.einsum("j,jkv->kv", z, dec.kappa_int)
        beta = F.softmax(logits, dim=1)
        out.append(torch.log(t @ beta))
    return torch.stack(out)


@pytest.mark.parametrize("p,J", [(0, 0), (3, 0), (3, 2)])
def test_forward_equals_brute_force(p, J):
    torch.manual_seed(0)
    K, V, N = 4, 50, 12
    A = None
    if J:
        A = np.zeros((p, J), dtype=np.float32)
        A[0, 0] = A[1, 1] = A[2, 1] = 1.0
    dec = SAGEDecoder(K, p, V, background_counts=np.random.default_rng(0).integers(0, 50, V),
                      interactions=A)
    with torch.no_grad():
        dec.kappa_cov.normal_(0, 0.5)
        dec.kappa_int.normal_(0, 0.5)
    theta = torch.softmax(torch.randn(N, K), dim=1)
    # Discrete covariates (as in applications), so rows share interaction values.
    x = torch.randint(0, 2, (N, p)).float() * torch.tensor([1.0, -1.0, 1.0][:p])
    got = dec(torch.cat([theta, x], dim=1))
    want = _brute_force(dec, theta, x)
    assert torch.allclose(got, want, atol=1e-5)
    assert torch.allclose(torch.logsumexp(got, dim=1), torch.zeros(N), atol=1e-5)
    # log_softmax / softmax of the output give log p / p back.
    assert torch.allclose(F.log_softmax(got, dim=1), got, atol=1e-5)


def test_topic_and_covariate_log_probs():
    torch.manual_seed(1)
    K, p, V = 3, 2, 40
    A = np.array([[1.0], [0.0]], dtype=np.float32)
    dec = SAGEDecoder(K, p, V, background_counts=np.arange(V), interactions=A)
    with torch.no_grad():
        dec.kappa_cov.normal_(0, 0.5)
        dec.kappa_int.normal_(0, 0.5)
    x = torch.tensor([1.0, -1.0])
    logits = dec.background + dec.kappa_topic + x @ dec.kappa_cov + dec.kappa_int[0] * 1.0
    assert torch.allclose(dec.topic_log_probs(x), F.log_softmax(logits, dim=1), atol=1e-6)
    # A one-hot theta gives that topic's distribution.
    theta = torch.eye(K)
    got = dec(torch.cat([theta, x.expand(K, -1)], dim=1))
    assert torch.allclose(got, dec.topic_log_probs(x), atol=1e-5)
    want = F.log_softmax(dec.background + dec.kappa_cov, dim=1)
    assert torch.allclose(dec.covariate_log_probs(), want, atol=1e-6)
    # Fixed background: the add-one smoothed log frequencies of the counts.
    counts = np.arange(V, dtype=np.float64)
    assert np.allclose(dec.background.numpy(), np.log((counts + 1) / (counts.sum() + V)), atol=1e-6)


def _sim_corpus(num_docs, K, V, words, seed, anchor_words=5, covs=0):
    _, df, tw, *_ = generate_documents(
        num_docs, K, V, num_covs=covs, doc_topic_prior="logistic_normal",
        min_words=words, max_words=words, random_seed=seed, anchor_words=anchor_words)
    return df, tw["lang0"]


def _vectorizer(df, V):
    # Fixed column order = the simulator's word index.
    return CountVectorizer(token_pattern=r"\S+", vocabulary=[f"lang0_word_{j}" for j in range(V)])


def _gtm(train, tmp_path, decoder_args, **kw):
    args = dict(
        train_data=train, n_topics=4, ae_type="vae", vi_type="mixture_of_gaussians",
        mixture_components=3, doc_topic_prior="logistic_normal", update_prior=True,
        encoder_args={"text_bow": {"hidden_dims": [64], "dropout": 0.0}},
        decoder_args={"text_bow": decoder_args}, batch_size=128, num_steps=50,
        num_workers=0, print_every_n_steps=10**9, return_best_model=False,
        ckpt_folder=str(tmp_path), seed=0, device="cpu")
    args.update(kw)
    return GTM(**args)


def test_gtm_trains_and_readouts(tmp_path):
    K, V = 4, 60
    df, _ = _sim_corpus(400, K, V, 40, seed=1)
    df["g"] = np.where(np.arange(len(df)) % 2 == 0, "a", "b")
    corpus = Corpus(df, modalities={"text": {"column": "doc_clean_0", "views": {
        "bow": {"type": "bow", "vectorizer": _vectorizer(df, V)}}}}, content="~ 0 + C(g)")
    tm = _gtm(corpus, tmp_path, {"type": "sage"})
    assert all(np.isfinite(tm.train_losses))
    beta = tm.get_topic_word_distribution(content_covariates=[0.5, 0.5])
    assert beta.shape == (K, V) and np.allclose(beta.sum(1), 1, atol=1e-5)
    comp = tm.get_word_logit_components()
    want = F.softmax(torch.as_tensor(comp["background"] + comp["topics"]
                                     + 0.5 * comp["covariates"].sum(0)), dim=1).numpy()
    assert np.allclose(beta, want, atol=1e-5)
    words = tm.get_topic_words(l_content_covariates=["C(g)[a]"], topK=3)
    assert len(words) == K and all(len(w) == 3 for w in words.values())
    cov = tm.get_covariate_word_distribution()
    assert cov.shape == (2, V) and np.allclose(cov.sum(1), 1, atol=1e-5)
    # The IWAE path (_recon_loglik) works with the normalized output.
    ll = tm.estimate_marginal_log_likelihood(corpus, n_samples=5)
    assert np.isfinite(np.asarray(ll)).all()


def _match(est, true):
    """Mean correlation of matched topic-word rows (Hungarian on -corr) and the matching."""
    c = np.corrcoef(est, true)[: len(est), len(est):]
    r, col = linear_sum_assignment(-c)
    return c[r, col].mean(), col


def test_recovers_mixture_topics(tmp_path):
    K, V = 4, 150
    df, true_beta = _sim_corpus(3000, K, V, 80, seed=2)
    corpus = Corpus(df, modalities={"text": {"column": "doc_clean_0", "views": {
        "bow": {"type": "bow", "vectorizer": _vectorizer(df, V)}}}})
    tm = _gtm(corpus, tmp_path, {"type": "sage"}, num_steps=3000)
    beta = tm.get_topic_word_distribution()
    corr, perm = _match(beta, true_beta)
    assert corr > 0.95, corr
    true_theta = generate_documents(
        3000, K, V, doc_topic_prior="logistic_normal", min_words=80, max_words=80,
        random_seed=2, anchor_words=5)[0].to_numpy()
    theta = tm.get_doc_topic_distribution(corpus, num_samples=10)
    r = [np.corrcoef(theta[:, k], true_theta[:, perm[k]])[0, 1] for k in range(K)]
    assert min(r) > 0.85, r


def test_l1_shrinks_and_background(tmp_path):
    K, V = 4, 80
    df, _ = _sim_corpus(500, K, V, 40, seed=3)
    corpus = Corpus(df, modalities={"text": {"column": "doc_clean_0", "views": {
        "bow": {"type": "bow", "vectorizer": _vectorizer(df, V)}}}})
    fixed = _gtm(corpus, tmp_path, {"type": "sage"}, num_steps=300)
    dec = fixed.decoders["text_bow"]
    counts = np.asarray(corpus.processed_modalities["text"]["bow"]["matrix"].sum(axis=0)).ravel()
    assert np.allclose(dec.background.cpu().numpy(), np.log((counts + 1) / (counts.sum() + V)), atol=1e-6)
    assert "background" not in dict(dec.named_parameters())
    l1 = _gtm(corpus, tmp_path, {"type": "sage", "l1": 200.0}, num_steps=300)
    assert l1.decoders["text_bow"].l1_norm().item() < 0.5 * dec.l1_norm().item()
    learned = _gtm(corpus, tmp_path, {"type": "sage", "background": "learned"}, num_steps=300)
    m = learned.decoders["text_bow"].background
    assert isinstance(m, torch.nn.Parameter)
    assert not np.allclose(m.detach().cpu().numpy(), dec.background.cpu().numpy(), atol=1e-4)


def test_save_load_roundtrip(tmp_path):
    K, V = 4, 60
    df, _ = _sim_corpus(300, K, V, 40, seed=4)
    df["g"] = np.where(np.arange(len(df)) % 3 == 0, "a", "b")
    df["s"] = np.where(np.arange(len(df)) % 2 == 0, 1.0, -1.0)
    corpus = Corpus(df, modalities={"text": {"column": "doc_clean_0", "views": {
        "bow": {"type": "bow", "vectorizer": _vectorizer(df, V)}}}}, content="~ 0 + C(g) + C(g):s")
    inter = {"s": ["C(g)[a]:s", "C(g)[b]:s"]}
    cfg = {"type": "sage", "interactions": inter, "background": "learned"}
    tm = _gtm(corpus, tmp_path, cfg, num_steps=100)
    path = str(tmp_path / "m.ckpt")
    tm.save_model(path)
    tm2 = _gtm(corpus, tmp_path / "b", cfg, num_steps=0, seed=5)
    tm2.load_model(path)
    x = [1.0, 0.0, 1.0, 0.0]
    assert np.allclose(tm.get_topic_word_distribution(content_covariates=x),
                       tm2.get_topic_word_distribution(content_covariates=x), atol=1e-6)


def test_interactions(tmp_path):
    K, V = 4, 60
    df, _ = _sim_corpus(400, K, V, 40, seed=5)
    df["g"] = np.where(np.arange(len(df)) % 2 == 0, "a", "b")
    df["s"] = np.where(np.arange(len(df)) % 4 < 2, 1.0, -1.0)
    corpus = Corpus(df, modalities={"text": {"column": "doc_clean_0", "views": {
        "bow": {"type": "bow", "vectorizer": _vectorizer(df, V)}}}}, content="~ 0 + C(g) + C(g):s")
    names = corpus.content_colnames
    assert names == ["C(g)[a]", "C(g)[b]", "C(g)[a]:s", "C(g)[b]:s"]
    # One topic x party effect shared by both groups ...
    tm = _gtm(corpus, tmp_path, {"type": "sage", "interactions": {"s": names[2:]}}, num_steps=200)
    dec = tm.decoders["text_bow"]
    assert dec.kappa_int.shape == (1, K, V)
    assert np.allclose(dec.interactions.cpu().numpy(), [[0], [0], [1], [1]])
    assert dec.kappa_int.abs().sum().item() > 0
    comp = tm.get_word_logit_components()
    assert comp["interactions"].shape == (1, K, V) and comp["interaction_names"] == ["s"]
    # ... or one per group.
    tm = _gtm(corpus, tmp_path, {"type": "sage", "interactions": {n: [n] for n in names[2:]}}, num_steps=50)
    assert tm.decoders["text_bow"].kappa_int.shape == (2, K, V)
    with pytest.raises(ValueError):
        _gtm(corpus, tmp_path, {"type": "sage", "interactions": {"x": ["nope"]}}, num_steps=0)
    with pytest.raises(ValueError):
        _gtm(corpus, tmp_path, {"type": "sage", "hidden_dims": [32]}, num_steps=0)


def test_init_topics():
    K, p, V = 3, 2, 30
    rng = np.random.default_rng(0)
    init = rng.dirichlet(np.full(V, 0.3), size=K) * 5.0  # rows need not sum to 1
    dec = SAGEDecoder(K, p, V, background_counts=rng.integers(1, 100, V), init_topics=init)
    got = dec.topic_log_probs(torch.zeros(p)).exp().detach().numpy()
    assert np.allclose(got, init / init.sum(1, keepdims=True), atol=1e-5)
    with pytest.raises(AssertionError):
        SAGEDecoder(K, p, V, init_topics=-init)


def test_jeffreys_penalty_and_refresh():
    torch.manual_seed(0)
    K, p, V = 3, 2, 30
    A = np.array([[0.0], [1.0]], dtype=np.float32)
    dec = SAGEDecoder(K, p, V, interactions=A, jeffreys=True, jeffreys_blocks=("topics", "interactions"),
                      jeffreys_strength=2.0, jeffreys_floor=1e-4, jeffreys_start=1, jeffreys_every=1)
    with torch.no_grad():
        dec.kappa_cov.normal_(0, 1.0)
        dec.kappa_int.normal_(0, 0.3)
        dec.kappa_int[0, 0, :5] = 0.0  # below the floor
    assert not hasattr(dec, "prec_kappa_cov")  # covariates not under the prior
    # Before the first refresh: precision 1 (variance-1 ridge).
    want = 0.5 * 2.0 * ((dec.kappa_topic ** 2).sum() + (dec.kappa_int ** 2).sum())
    assert torch.allclose(dec.jeffreys_penalty(), want)
    dec.refresh_jeffreys()
    assert torch.allclose(dec.prec_kappa_topic, 1 / dec.kappa_topic.detach().pow(2).clamp_min(1e-4))
    assert torch.allclose(dec.prec_kappa_int, 1 / dec.kappa_int.detach().pow(2).clamp_min(1e-4))
    assert torch.allclose(dec.prec_kappa_int[0, 0, :5], torch.full((5,), 1e4))
    # Right after a refresh each coefficient contributes 1/2 * strength * eta^2 / max(eta^2, floor):
    # 1/2 * strength above the floor, less below it.
    share = sum((k.detach() ** 2 / (k.detach() ** 2).clamp_min(1e-4)).sum() for k in (dec.kappa_topic, dec.kappa_int))
    assert abs(dec.jeffreys_penalty().item() - 0.5 * 2.0 * share.item()) < 1e-3
    # Gradient of the penalty: strength * prec * eta.
    dec.zero_grad()
    dec.jeffreys_penalty().backward()
    assert torch.allclose(dec.kappa_topic.grad, 2.0 * dec.prec_kappa_topic * dec.kappa_topic.detach(), rtol=1e-5)
    assert dec.kappa_cov.grad is None or torch.all(dec.kappa_cov.grad == 0)
    with pytest.raises(ValueError):
        SAGEDecoder(K, p, V, jeffreys=True, l1=1.0, jeffreys_start=1, jeffreys_every=1)
    with pytest.raises(ValueError):
        SAGEDecoder(K, p, V, jeffreys=True, jeffreys_blocks=("party",), jeffreys_start=1, jeffreys_every=1)
    with pytest.raises(ValueError):
        SAGEDecoder(K, p, V, jeffreys=True, jeffreys_start=0, jeffreys_every=1)
    with pytest.raises(ValueError):
        SAGEDecoder(K, p, V, jeffreys=True, jeffreys_start=10)  # jeffreys_every missing


def _party_corpus(seed, n_docs=3000, V=150, K=3, words=60, n_true=10, effect=1.0):
    """Documents from a K-topic mixture whose word distributions shift by party_signed * delta,
    delta nonzero (+-effect) for the first n_true words only."""
    rng = np.random.default_rng(seed)
    m = np.log(rng.dirichlet(np.full(V, 1.0)) + 1e-4)
    kap = rng.normal(0, 1.0, size=(K, V))
    delta = np.zeros(V)
    delta[:n_true] = effect * np.where(np.arange(n_true) % 2 == 0, 1, -1)
    party = np.where(rng.random(n_docs) < 0.5, 1.0, -1.0)
    theta = rng.dirichlet(np.full(K, 0.3), size=n_docs)
    docs = []
    for d in range(n_docs):
        logits = m + kap + party[d] * delta
        beta = np.exp(logits - logits.max(1, keepdims=True)); beta /= beta.sum(1, keepdims=True)
        w = rng.choice(V, size=words, p=theta[d] @ beta)
        docs.append(" ".join(f"lang0_word_{j}" for j in w))
    df = pd.DataFrame({"doc_clean_0": docs, "party": party})
    return Corpus(df, modalities={"text": {"column": "doc_clean_0", "views": {
        "bow": {"type": "bow", "vectorizer": _vectorizer(df, V)}}}}, content="~ 0 + party"), delta


def test_jeffreys_schedule_and_checkpoint(tmp_path):
    corpus, _ = _party_corpus(0, n_docs=300)
    cases = (({"jeffreys_start": 3, "jeffreys_every": 3}, [3, 6, 9]),
             ({"jeffreys_start": 4, "jeffreys_every": 2}, [4, 6, 8, 10]),
             ({"jeffreys_start": 2, "jeffreys_every": 5}, [2, 7]))
    for extra, want in cases:
        cfg = {"type": "sage", "jeffreys": True, **extra}
        tm = _gtm(corpus, tmp_path, cfg, num_steps=0, n_topics=3)
        dec = tm.decoders["text_bow"]
        calls = []
        orig = dec.refresh_jeffreys
        dec.refresh_jeffreys = lambda: (calls.append(tm.steps + 1), orig())[1]
        tm.num_steps = 10
        tm.train(corpus)
        assert calls == want, (extra, calls)
        del dec.refresh_jeffreys
    path = str(tmp_path / "j.ckpt")
    tm.save_model(path)
    tm2 = _gtm(corpus, tmp_path / "b", cfg, num_steps=0, n_topics=3, seed=1)
    tm2.load_model(path)
    for name in ("prec_kappa_topic", "prec_kappa_cov", "prec_kappa_int"):
        assert torch.allclose(getattr(tm2.decoders["text_bow"], name), getattr(dec, name))
    assert not torch.allclose(dec.prec_kappa_topic, torch.ones_like(dec.prec_kappa_topic))
    plain = _gtm(corpus, tmp_path / "c", {"type": "sage"}, num_steps=0, n_topics=3)
    with pytest.raises(ValueError, match="Jeffreys"):
        plain.load_model(path)
    plain.save_model(str(tmp_path / "plain.ckpt"))
    with pytest.raises(ValueError, match="Jeffreys"):
        tm2.load_model(str(tmp_path / "plain.ckpt"))


def test_jeffreys_gradient_scaling_and_validation(tmp_path):
    corpus, _ = _party_corpus(2, n_docs=300)
    tm = _gtm(corpus, tmp_path, {"type": "sage", "jeffreys": True, "jeffreys_start": 1, "jeffreys_every": 1},
              num_steps=0, n_topics=3, vi_type="mean_field")
    dec = tm.decoders["text_bow"]
    with torch.no_grad():
        dec.kappa_cov.normal_(0, 0.5)
    dec.refresh_jeffreys()
    for g in tm.optimizer.param_groups:
        g["lr"] = 0.0  # gradients only, parameters unchanged
    batch = next(iter(tm._data_loader(corpus)))
    grads, val = {}, {}
    for strength in (0.0, 1.0):
        dec.jeffreys_strength = strength
        torch.manual_seed(0)
        tm.step_batch(batch, corpus, validation=False)
        grads[strength] = dec.kappa_cov.grad.clone()
        prec = dec.prec_kappa_cov.clone()
        torch.manual_seed(0)
        val[strength] = tm.evaluate(tm._data_loader(corpus), corpus)
        assert torch.equal(dec.prec_kappa_cov, prec)  # validation does not refresh
    want = dec.prec_kappa_cov * dec.kappa_cov.detach() / tm.n_train_docs
    assert torch.allclose(grads[1.0] - grads[0.0], want, rtol=1e-4, atol=1e-9)
    assert val[0.0] == val[1.0]  # the penalty is not in the reported (validation) loss


def _jeffreys_vs_none(tmp_path, effect, seed):
    corpus, delta = _party_corpus(seed, effect=effect)
    true = delta != 0
    est = {}
    for name, extra in (("none", {}), ("jeffreys", {"jeffreys": True, "jeffreys_start": 500, "jeffreys_every": 100})):
        tm = _gtm(corpus, tmp_path / name, {"type": "sage", **extra}, num_steps=3000, n_topics=3,
                  vi_type="mean_field", kl_annealing_start=0, kl_annealing_end=300)
        assert tm.content_colnames == ["party"]
        row = tm.get_word_logit_components()["covariates"][0]
        est[name] = row - np.median(row)  # the softmax ignores a constant shift of the row
    return est, delta, true


def test_jeffreys_shrinks_null_party_deviations(tmp_path):
    est, delta, true = _jeffreys_vs_none(tmp_path, effect=1.0, seed=1)
    null_none, null_j = np.abs(est["none"][~true]).mean(), np.abs(est["jeffreys"][~true]).mean()
    true_none, true_j = np.abs(est["none"][true]).mean(), np.abs(est["jeffreys"][true]).mean()
    assert null_j < 0.3 * null_none, (null_j, null_none)
    assert true_j > 0.9 * true_none, (true_j, true_none)
    assert np.sign(est["jeffreys"][true]).tolist() == np.sign(delta[true]).tolist()


def test_jeffreys_keeps_weak_true_deviations(tmp_path):
    # A weak effect (0.15, |z| ~ 2.5 in this mixture) above the |z| >= 2 threshold: it must survive
    # the start of the refreshes (deviations start at 0), with the prior's shrinkage of borderline
    # effects (about 20% at |z| = 2.5).
    est, delta, true = _jeffreys_vs_none(tmp_path, effect=0.15, seed=1)
    true_none, true_j = np.abs(est["none"][true]).mean(), np.abs(est["jeffreys"][true]).mean()
    null_none, null_j = np.abs(est["none"][~true]).mean(), np.abs(est["jeffreys"][~true]).mean()
    assert true_j > 0.6 * true_none, (true_j, true_none)
    assert null_j < 0.3 * null_none, (null_j, null_none)
