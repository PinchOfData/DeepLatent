"""Variational-family audit: correctness of each vi_type + tightness ordering +
an IWAE gap probe (the estimator the paper's claim needs and the package lacks)."""
import torch, numpy as np, scipy.sparse
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal, kl_divergence
from sklearn.feature_extraction.text import CountVectorizer
import tempfile
torch.manual_seed(0); np.random.seed(0)
def header(s): print("\n"+"="*72+f"\n{s}\n"+"="*72)

# ---------------------------------------------------------------------------
header("A. full_rank KL matches torch.distributions (exact, correlated Gaussian)")
B, K = 5, 4
# build a random lower-tri L with positive diag exactly as the encoder does
raw = torch.randn(B, K, K)
L = torch.tril(raw)
diag = torch.arange(K)
L[:, diag, diag] = F.softplus(L[:, diag, diag]) + 1e-4
mu = torch.randn(B, K)
Sigma = torch.bmm(L, L.transpose(1,2))
# code's closed form (diagonal prior N(0,I)): 0.5*(logdet_p - logdet_q - K + tr + quad)
logvar_p = torch.zeros(B, K); var_p = torch.exp(logvar_p)
logdet_q = 2*torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2)+1e-6), dim=1)
var_q_diag = torch.diagonal(Sigma, dim1=1, dim2=2)
trace = torch.sum(var_q_diag/var_p, dim=1)
quad = torch.sum((mu-0).pow(2)/var_p, dim=1)
logdet_p = torch.sum(logvar_p, dim=1)
kl_code = 0.5*(logdet_p - logdet_q - K + trace + quad)
q = MultivariateNormal(mu, scale_tril=L)
p = MultivariateNormal(torch.zeros(B,K), torch.eye(K).expand(B,K,K))
kl_ref = kl_divergence(q, p)
print("max |KL_full_rank_code - KL_torch| =", (kl_code-kl_ref).abs().max().item(), "(~0 => correct)")

# ---------------------------------------------------------------------------
header("B. IAF: strict autoregressivity + log|det J| vs autograd")
from deeplatent.autoencoders import IAF, MADE
iaf = IAF(dim=K, num_flows=3, use_permutations=True).eval()
z0 = torch.randn(1, K, requires_grad=True)
zk, logdet = iaf(z0)
J = torch.autograd.functional.jacobian(lambda z: iaf(z)[0].squeeze(0), z0).squeeze()
_, logabsdet = torch.linalg.slogdet(J)
print("IAF sum(log_sigma) =", round(logdet.item(),6), " autograd log|det J| =", round(logabsdet.item(),6))
# check the single-flow Jacobian is triangular under its permutation (autoregressive)
made = MADE(K); z = torch.randn(1,K,requires_grad=True)
Jm = torch.autograd.functional.jacobian(lambda z: made(z)[0].squeeze(0), z).squeeze()
upper_strict = torch.triu(Jm, diagonal=0).abs().max().item()  # mu_i must NOT depend on z_>=i
print("MADE mu Jacobian strictly-lower-triangular (upper incl diag ~0):", round(upper_strict,3))
# bounded scale range note
s = torch.linspace(-50,50,5)
print("MADE log_sigma range is bounded to ~[-2,2]; sigma in",
      [round(float(x),3) for x in torch.exp(-2.0+4.0*torch.sigmoid(torch.tensor([-1e9,1e9])))])

# ---------------------------------------------------------------------------
header("C. Tightness ordering: train GTM-VAE with each family on one corpus")
from deeplatent import Corpus, GTM, generate_documents
_, df, *_ = generate_documents(1000, 5, 100, doc_topic_prior="logistic_normal",
                               min_words=150, max_words=150, random_seed=1)
vec = CountVectorizer(); vec.fit(df["doc_clean_0"])
corpus = Corpus(df, modalities={"text":{"column":"doc_clean_0",
              "views":{"bow":{"type":"bow","vectorizer":vec}}}})
mm = corpus.processed_modalities["text"]["bow"]["matrix"]
if scipy.sparse.issparse(mm):
    corpus.processed_modalities["text"]["bow"]["matrix"] = np.asarray(mm.todense(), np.float32)

def train(vi):
    return GTM(train_data=corpus, n_topics=5, ae_type="vae", vi_type=vi,
               update_prior=False, doc_topic_prior="logistic_normal",
               encoder_args={"text_bow":{"hidden_dims":[128]}},
               decoder_args={"text_bow":{"hidden_dims":[]}},
               w_prior=1.0, batch_size=200, num_steps=1500, num_workers=0,
               print_every_n_steps=10**9, return_best_model=False,
               ckpt_folder=tempfile.mkdtemp(), seed=1, device=torch.device("cpu"))

results = {}
for vi in ["mean_field", "full_rank", "iaf"]:
    m = train(vi)
    rec = float(np.mean(m.train_recon_losses[-100:]))
    kl  = float(np.mean(m.train_div_losses[-100:]))
    results[vi] = (rec, kl, rec+kl, m)
    print(f"{vi:11s}  recon={rec:8.2f}  KL={kl:6.3f}  neg-ELBO={rec+kl:8.2f}")
print("\n(lower neg-ELBO = tighter bound; expect mean_field >= full_rank, iaf)")

# ---------------------------------------------------------------------------
header("D. IWAE gap probe for the mean_field model (machinery the package lacks)")
# log p(x) >= IWAE_S >= ELBO ; the gap quantifies the variational-family error.
m = results["mean_field"][3]
enc = m.encoder.encoders["text_bow"]; dec = m.decoders["text_bow"]
enc.eval(); dec.eval()
X = torch.tensor(np.asarray(corpus.processed_modalities["text"]["bow"]["matrix"][:200]), dtype=torch.float32)
def iwae(X, S):
    with torch.no_grad():
        h = enc(X); mu_q, logvar_q = torch.chunk(h, 2, dim=1)
        std = torch.exp(0.5*logvar_q)
        lws = []
        for _ in range(S):
            eps = torch.randn_like(std); z = mu_q + eps*std
            theta = F.softmax(z, dim=1)
            logits = dec(theta); logp = F.log_softmax(logits, dim=1)
            log_px_z = (X*logp).sum(1)                                  # multinomial loglik
            log_pz   = Normal(0,1).log_prob(z).sum(1)
            log_qz   = Normal(mu_q, std).log_prob(z).sum(1)
            lws.append(log_px_z + log_pz - log_qz)
        lw = torch.stack(lws, 1)                                        # [N,S]
        elbo = lw.mean(1)                                               # 1-sample ELBO (mean)
        iwae_S = torch.logsumexp(lw, 1) - np.log(S)                     # IWAE_S
    return elbo.mean().item(), iwae_S.mean().item()
e1, i1 = iwae(X, 1); e50, i50 = iwae(X, 200)
print(f"ELBO (per doc)          = {e1:8.2f}")
print(f"IWAE_200  (per doc)     = {i50:8.2f}   <-- tighter lower bound on log p(x)")
print(f"estimated gap log p(x) - ELBO  >=  {i50 - e1:6.2f} nats/doc")
print("\nThis IWAE estimator is ~15 lines and does not exist in the package; it is the")
print("direct way to MEASURE how tightly a family approximates the marginal likelihood.")
