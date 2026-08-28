"""Train a small multilingual VAE with exact corrected MoG-PoE fusion."""

import tempfile
import warnings

import numpy as np
import scipy.sparse
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader

from deeplatent import Corpus, GTM, generate_documents


torch.manual_seed(19)
np.random.seed(19)

N_TOPICS = 4
COMPONENTS = 2
MODALITIES = 2

_, frame, *_ = generate_documents(
    num_docs=240,
    num_topics=N_TOPICS,
    vocab_size=60,
    num_languages=MODALITIES,
    doc_topic_prior="logistic_normal",
    min_words=35,
    max_words=35,
    random_seed=19,
)

modalities = {}
for m in range(MODALITIES):
    column = f"doc_clean_{m}"
    vectorizer = CountVectorizer()
    vectorizer.fit(frame[column])
    modalities[f"lang{m}"] = {
        "column": column,
        "views": {"bow": {"type": "bow", "vectorizer": vectorizer}},
    }

corpus = Corpus(frame, modalities=modalities)
for m in range(MODALITIES):
    view = corpus.processed_modalities[f"lang{m}"]["bow"]
    if scipy.sparse.issparse(view["matrix"]):
        view["matrix"] = np.asarray(view["matrix"].todense(), dtype=np.float32)

print(
    "Training exact corrected MoG-PoE: "
    f"M={MODALITIES}, C={COMPONENTS}, expanded C^M={COMPONENTS ** MODALITIES}"
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", RuntimeWarning)
    model = GTM(
        train_data=corpus,
        n_topics=N_TOPICS,
        ae_type="vae",
        vi_type="mixture_of_gaussians",
        mixture_components=COMPONENTS,
        fusion="corrected_poe",
        update_prior=True,
        doc_topic_prior="logistic_normal",
        encoder_args={f"lang{m}_bow": {"hidden_dims": [32]} for m in range(MODALITIES)},
        decoder_args={f"lang{m}_bow": {"hidden_dims": []} for m in range(MODALITIES)},
        w_prior=1.0,
        batch_size=60,
        num_steps=80,
        num_workers=0,
        print_every_n_steps=10**9,
        return_best_model=False,
        ckpt_folder=tempfile.mkdtemp(prefix="deeplatent_mog_poe_"),
        seed=19,
        device=torch.device("cpu"),
    )

warning_text = "\n".join(str(item.message) for item in caught)
assert "grows exponentially" in warning_text

losses = np.asarray(model.train_losses, dtype=float)
assert losses.size > 0 and np.isfinite(losses).all()

inputs = {
    f"lang{m}_bow": torch.as_tensor(
        corpus.processed_modalities[f"lang{m}"]["bow"]["matrix"][:16],
        dtype=torch.float32,
    )
    for m in range(MODALITIES)
}
model.encoder.eval()
with torch.no_grad():
    theta, z, posterior_info = model.encoder(inputs)

tag, means, scale_trils, weights = posterior_info[-1]
assert tag == "mog_full"
assert means.shape == (16, COMPONENTS**MODALITIES, N_TOPICS - 1)
assert scale_trils.shape == (
    16,
    COMPONENTS**MODALITIES,
    N_TOPICS - 1,
    N_TOPICS - 1,
)
assert torch.isfinite(z).all() and torch.isfinite(theta).all()
assert torch.isfinite(means).all() and torch.isfinite(scale_trils).all()
assert torch.isfinite(weights).all()
assert torch.allclose(weights.sum(1), torch.ones(16), atol=1e-5)
assert torch.allclose(theta.sum(1), torch.ones(16), atol=1e-5)

# A reconstruction-only training step must reach each modality's mixing logits.
# Temporarily bypass random modality masking so both modality encoders participate,
# and set w_prior=0 so any logit gradient can only come from the likelihood term.
gradient_batch = next(
    iter(DataLoader(corpus, batch_size=60, shuffle=False, num_workers=0))
)
old_fusion, old_w_prior = model.fusion, model.w_prior
model.fusion = "all_modalities_gradient_check"
model.w_prior = 0.0
model.step_batch(gradient_batch, corpus, validation=False)
model.fusion, model.w_prior = old_fusion, old_w_prior

logit_gradient_norms = []
for modality_encoder in model.encoder.encoders.values():
    final_layer = list(modality_encoder.encoder.values())[-1]
    logit_grad = final_layer.bias.grad[-COMPONENTS:]
    assert logit_grad is not None and torch.isfinite(logit_grad).all()
    logit_gradient_norms.append(float(logit_grad.abs().sum()))
assert all(norm > 1e-10 for norm in logit_gradient_norms)

iwae, elbo = model.estimate_marginal_log_likelihood(corpus, n_samples=10)
assert np.isfinite(iwae) and np.isfinite(elbo)

print(
    f"OK: {len(losses)} finite training losses; fused shape={tuple(means.shape)}; "
    f"reconstruction logit gradients={logit_gradient_norms}; "
    f"ELBO={float(elbo):.3f}; IWAE_10={float(iwae):.3f}"
)
