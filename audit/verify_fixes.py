"""Confirm the audit fixes behave as intended."""
import torch, numpy as np
torch.manual_seed(0); np.random.seed(0)
def ok(b): return "PASS" if b else "FAIL"

# 1. learnable prior covariance now ~= I
from deeplatent.priors import LogisticNormalPrior, GaussianPrior
for cls, kw in [(LogisticNormalPrior, dict(prevalence_covariate_size=0, n_topics=5)),
                (GaussianPrior, dict(prevalence_covariate_size=0, n_dims=5))]:
    p = cls(**kw); S = p.sigma.detach()
    diag_ok = torch.allclose(S.diag(), torch.ones(5), atol=1e-3)
    offdiag_ok = (S - torch.diag(S.diag())).abs().max() < 1e-6
    print(f"[1] {cls.__name__:22s} Sigma_init diag={S.diag()[0]:.4f} (want 1.0)  "
          f"identity={ok(diag_ok and offdiag_ok)}")

# 2. encoder no longer drops the (mu, logvar) output layer
from deeplatent.autoencoders import EncoderMLP
e = EncoderMLP([10, 8, 2*5], dropout=0.9).train()   # huge dropout to expose any output drop
x = torch.ones(64, 10)
outs = torch.stack([e(x) for _ in range(300)])
zeroed_fraction = (outs == 0).float().mean().item()   # output units never zeroed by dropout
# With output dropout removed, exact zeros at the output are astronomically unlikely.
print(f"[2] EncoderMLP output zero-fraction at dropout=0.9: {zeroed_fraction:.4f}  "
      f"(want ~0)  {ok(zeroed_fraction < 1e-3)}")
# hidden dropout still active -> outputs still vary across passes (regularization intact)
varies = outs.std(0).mean().item() > 1e-3
print(f"    hidden-layer dropout still active (outputs vary): {ok(varies)}")

# 3. VAE + Dirichlet now raises
from deeplatent import Corpus, GTM, generate_documents
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse, tempfile
_, df, *_ = generate_documents(200, 3, 40, min_words=60, max_words=60, random_seed=0)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
corpus = Corpus(df, modalities={"text":{"column":"doc_clean_0",
              "views":{"bow":{"type":"bow","vectorizer":vec}}}})
mm = corpus.processed_modalities["text"]["bow"]["matrix"]
if scipy.sparse.issparse(mm):
    corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(mm.todense(), np.float32)
raised = False
try:
    GTM(train_data=corpus, n_topics=3, ae_type="vae", doc_topic_prior="dirichlet",
        encoder_args={"text_bow":{"hidden_dims":[16]}},
        decoder_args={"text_bow":{"hidden_dims":[]}},
        num_steps=0, num_workers=0, ckpt_folder=tempfile.mkdtemp(), seed=0)
except ValueError as ex:
    raised = True; msg = str(ex)
print(f"[3] GTM(ae_type='vae', dirichlet) raises ValueError: {ok(raised)}")
if raised: print("    msg:", msg[:80], "...")

# WAE + Dirichlet still allowed
allowed = True
try:
    GTM(train_data=corpus, n_topics=3, ae_type="wae", doc_topic_prior="dirichlet",
        encoder_args={"text_bow":{"hidden_dims":[16]}},
        decoder_args={"text_bow":{"hidden_dims":[]}},
        num_steps=0, num_workers=0, ckpt_folder=tempfile.mkdtemp(), seed=0)
except Exception as ex:
    allowed = False; print("    unexpected:", ex)
print(f"[3] GTM(ae_type='wae', dirichlet) still constructs: {ok(allowed)}")
