# DeepLatent

`DeepLatent` is a unified **latent variable modeling framework** for analyzing large **multimodal** and **multilingual** datasets. It relies on **variational inference using deep neural networks** for estimation.

The package currently supports:

* **Generic latent factor models** 
* **Topic models:** The latent variables are a mixture of topics within documents.
* **Ideal point models**: The latent variables are interpreted as ideological dimensions.

---

## 🌟 Key Features

* **Multilingual and multimodal support**

  * Learn topics / ideal points across multiple modalities (e.g., texts and images, texts and votes, etc.)
  * Learn the weight of each modality in determining the latent variables per observation

* **Flexible metadata handling**:

  * `prevalence`: covariates that influence the latent variables
  * `content`: covariates that influence the response variables conditional on the latent variables (e.g., topic-word distributions)
  * `labels`: outcomes for classification or regression tasks
  * `prediction`: additional predictors for the labels

* **Flexible input/output representations**:

  * Document embeddings (for texts, images, audio-visual data)
  * Word frequencies (BoW)
  * Raw images
  * Discrete choice data 
  * Voting records

---

## 📦 Models

### `GTM` (Generalized Topic Model)

* Learns **topics on the simplex**
* Supports `dirichlet` or `logistic_normal` priors (optionally conditioned on covariates)

### `IdealPointNN`

* Learns **unconstrained latent variables** (ℝ️ⁿ) for ideal point modeling
* Designed for **political texts, images, audio and video recordings**, **surveys**, and **votes**
* Uses a  `gaussian` prior (optionally conditioned on covariates)

---

## Installation

```
pip install -r requirements.txt
```

---

## 🚀 Getting Started

### 1. Prepare Your Data with `Corpus()`

Supports text, embeddings, votes, and survey questions:

```python
import sys
sys.path.append('../src/')

from corpus import Corpus

modalities = {
    "text": {
        "column": "doc_clean",
        "views": {
            "bow": {
                "type": "bow",
                "vectorizer": CountVectorizer()
            }
        }
    },
    "image": {
        "column": "image_path",
        "views": {
            "embedding": {
                "type": "embedding",
                "embed_fn": my_image_embedder
            }
        }
    }
}

my_dataset = Corpus(df, modalities=modalities)
```

Optionally include metadata:

* `prevalence`, `content`, `labels`, `prediction`

---

### 2. Train a Model

#### For Topic Models:

```python
from models import GTM

model = GTM(
    n_topics=20, 
    doc_topic_prior="logistic_normal",
    ae_type="wae"
)
```

#### For Ideal Point Models:

```python
from models import IdealPointNN

model = IdealPointNN(
    n_ideal_points=1, # one-dimensional ideal point model
    ae_type="vae"
)
```

#### 🔧 Common Options

| Argument         | Description                                  |
| ---------------- | -------------------------------------------- |
| `ae_type`        | `"wae"` (Wasserstein autoencoder) or `"vae"` (variational autoencoder) |
| `fusion`         | `"poe"` (Product of Experts), `"moe_gating"` (Mixture of Experts), or `"moe_average"` (Simple averaging across modalities) |
| `update_prior`   | Learn a structured prior conditioned on `prevalence` covariates                    |
| `w_prior`        | Strength of prior alignment for `wae`              |
| `w_pred_loss`    | Weight of supervised loss predicting `label`                   |
| `kl_annealing_*` | Strength of prior alignment for `vae`. Helps preventing posterior collapse.    |

---

## 🔍 Analysis and Utilities

### 📚 Topic Models (`GTM`)

* `get_topic_words()` – top words per topic
* `get_covariate_words()` – word shifts by `content` covariates
* `get_top_docs()` – representative documents
* `get_topic_word_distribution()` – topic-word matrix
* `get_covariate_word_distribution()` – word shift matrix
* `plot_topic_word_distribution()` – word clouds / bar plots
* `visualize_docs()` – document embeddings (UMAP, t-SNE, PCA)
* `visualize_words()` – word embeddings
* `visualize_topics()` – topic embeddings

### 👤 Ideal Point Models (`IdealPointNN`)

* `get_ideal_points()` – ℝ️ⁿ latent space
* `get_predictions()` – supervised output
* `get_modality_weights()` – fusion weights (PoE or gating)

---

## 📁 Tutorials

Check out the [example notebooks](notebooks/) to get started.

Download sample data to run some notebooks:
[Congressional Speeches CSV](https://www.dropbox.com/scl/fi/ojshavj5azk4jt7a4p3ap/us_congress_speeches_sample.csv?rlkey=x3x86kc9pb94kuu1c8yze5u3l&st=awtc4wr2&dl=1)

---

## 📖 References

* **Deep Latent Variable Models for Unstructured Data**
*, Germain Gauthier, Philine Widmer, Elliott Ash (2025)*

* **The Neural Ideal Point Model**
*, Germain Gauthier, Hugo Subtil, Philine Widmer (2025)*

---

## ⚠️ Disclaimer

This package is under active development 🚧 — feedback and contributions welcome!
