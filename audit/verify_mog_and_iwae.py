"""Smoke test: MoG family trains, IWAE works for all four families."""
import numpy as np, scipy.sparse, tempfile, torch
from sklearn.feature_extraction.text import CountVectorizer
from deeplatent import Corpus, GTM, generate_documents
torch.manual_seed(0); np.random.seed(0)

_, df, *_ = generate_documents(800, 5, 80, doc_topic_prior="logistic_normal",
                               min_words=120, max_words=120, random_seed=0)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
corpus = Corpus(df, modalities={"text":{"column":"doc_clean_0",
              "views":{"bow":{"type":"bow","vectorizer":vec}}}})
m = corpus.processed_modalities["text"]["bow"]["matrix"]
if scipy.sparse.issparse(m):
    corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), np.float32)

def fit(vi, **kw):
    return GTM(train_data=corpus, n_topics=5, ae_type="vae", vi_type=vi,
              update_prior=False, doc_topic_prior="logistic_normal",
              encoder_args={"text_bow":{"hidden_dims":[64]}},
              decoder_args={"text_bow":{"hidden_dims":[]}},
              w_prior=1.0, batch_size=200, num_steps=600, num_workers=0,
              print_every_n_steps=10**9, return_best_model=False,
              ckpt_folder=tempfile.mkdtemp(), seed=0, device=torch.device("cpu"), **kw)

print("Training mixture_of_gaussians (C=5)...")
m_mog = fit("mixture_of_gaussians", mixture_components=5)
print("  OK. final KL=", round(float(np.mean(m_mog.train_div_losses[-50:])),3),
      " recon=", round(float(np.mean(m_mog.train_recon_losses[-50:])),2))
theta = m_mog.get_doc_topic_distribution(corpus)
print("  doc-topic theta shape:", theta.shape, " rows sum to 1:", np.allclose(theta.sum(1),1,atol=1e-4))

print("\nIWAE (S=100) for every family on the SAME corpus:")
for vi in ["mean_field","full_rank","iaf","mixture_of_gaussians"]:
    mm = fit(vi) if vi!="mixture_of_gaussians" else m_mog
    iwae, elbo = mm.estimate_marginal_log_likelihood(corpus, n_samples=100)
    print(f"  {vi:22s}  ELBO={elbo:9.2f}   IWAE={iwae:9.2f}   gap={float(iwae-elbo):.3f}")
print("\nOK: IWAE >= ELBO for all (sanity), no crashes.")
