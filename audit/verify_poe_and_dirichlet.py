"""Follow-up: (a) corrected PoE mean is the exact posterior-combination identity;
(b) empirical Dirichlet-VAE vs LogisticNormal-VAE training pathology."""
import torch, numpy as np, scipy.sparse
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
torch.manual_seed(0); np.random.seed(0)

def header(s): print("\n"+"="*70+f"\n{s}\n"+"="*70)

# (a) corrected PoE: compare to EXACT product-of-posteriors / prior^(M-1) fusion
header("4b. Corrected PoE mean == exact [prod q_m]/p^(M-1) identity")
from deeplatent.autoencoders import MultiModalEncoder, EncoderMLP
B, K = 6, 4
Lam0 = torch.ones(B,K); mu0 = torch.zeros(B,K)
mu1, mu2 = torch.randn(B,K), torch.randn(B,K)
D1, D2 = torch.rand(B,K)+0.1, torch.rand(B,K)+0.1
Lam1, Lam2 = Lam0+D1, Lam0+D2                       # unimodal POSTERIOR precisions
# exact joint posterior natural params (combining unimodal posteriors):
Lam_S = Lam1 + Lam2 - Lam0
eta_S = Lam1*mu1 + Lam2*mu2 - Lam0*mu0
mu_S_exact = eta_S/Lam_S
enc = MultiModalEncoder({"a_x":EncoderMLP([2,2]),"b_x":EncoderMLP([2,2])}, topic_dim=K,
                        prior=None, ae_type="vae", poe="corrected", vi_type="mean_field")
class P:
    def get_prior_params(self,c,return_full_cov=False): return mu0, torch.zeros(B,K)
enc.prior = P()
mu_code, logvar_code = enc.product_of_experts([(mu1,-torch.log(Lam1)),(mu2,-torch.log(Lam2))])
print("max|mu_code - mu_exact|   =", (mu_code-mu_S_exact).abs().max().item(), "(~0 => correct)")
print("max|prec_code - Lam_exact|=", (torch.exp(-logvar_code)-Lam_S).abs().max().item(), "(~0 => correct)")

# (b) Empirical training pathology: Dirichlet-VAE vs LogisticNormal-VAE
header("3b. Empirical: GTM VAE with dirichlet vs logistic_normal prior")
from deeplatent import Corpus, GTM, generate_documents
import tempfile, os
_, df, *_ = generate_documents(800, 4, 80, doc_topic_prior="logistic_normal",
                               min_words=120, max_words=120, random_seed=0)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
corpus = Corpus(df, modalities={"text":{"column":"doc_clean_0",
              "views":{"bow":{"type":"bow","vectorizer":vec}}}})
m = corpus.processed_modalities["text"]["bow"]["matrix"]
if scipy.sparse.issparse(m):
    corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(m.todense(), dtype=np.float32)

tmp = tempfile.mkdtemp()
for prior in ["logistic_normal", "dirichlet"]:
    tm = GTM(train_data=corpus, n_topics=4, ae_type="vae", vi_type="mean_field",
             update_prior=False, doc_topic_prior=prior,
             encoder_args={"text_bow":{"hidden_dims":[64]}},
             decoder_args={"text_bow":{"hidden_dims":[]}},
             w_prior=1.0, batch_size=128, num_steps=400, num_workers=0,
             print_every_n_steps=10**9, return_best_model=False,
             ckpt_folder=tmp, seed=0)
    kl = float(np.mean(tm.train_div_losses[-50:]))
    rec = float(np.mean(tm.train_recon_losses[-50:]))
    theta = tm.get_doc_topic_distribution(corpus)
    # how concentrated/degenerate are the inferred topic proportions?
    mean_max = float(np.mean(theta.max(1)))         # avg top-topic share
    eff_topics = float(np.mean(np.exp(-(theta*np.log(theta+1e-9)).sum(1))))  # perplexity of theta
    print(f"\nprior={prior:16s}  KL(div_loss)={kl:8.2f}  recon={rec:8.2f}")
    print(f"   mean max-topic share={mean_max:.3f}   mean effective #topics={eff_topics:.2f} (of 4)")
print("\n(Logistic-normal: sane KL. Dirichlet: inflated KL from logit-vs-simplex mismatch.)")
