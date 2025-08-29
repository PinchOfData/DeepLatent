import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
from scipy.stats import norm

# Simulating topics

def generate_multilingual_docs_vectorized(
    true_doc_topic_matrix: np.ndarray,
    topic_word_matrices: Dict[str, np.ndarray],
    min_words: int,
    max_words: int,
    languages: List[str],
) -> Dict[str, List[str]]:
    num_docs = true_doc_topic_matrix.shape[0]
    doc_lengths = np.random.randint(min_words, max_words + 1, size=num_docs)
    max_length = np.max(doc_lengths)

    doc_data = {}

    for lang in languages:
        topic_word_matrix = topic_word_matrices[lang]
        word_probs = np.dot(true_doc_topic_matrix, topic_word_matrix)
        word_probs /= word_probs.sum(axis=1, keepdims=True)

        # Vectorized sampling
        random_values = np.random.random((num_docs, max_length))
        cumulative_probs = np.cumsum(word_probs, axis=1)
        word_indices = np.argmax(random_values[:, :, np.newaxis] < cumulative_probs[:, np.newaxis, :], axis=2)

        # Mask out padding
        mask = np.arange(max_length)[np.newaxis, :] < doc_lengths[:, np.newaxis]
        words = np.core.defchararray.add(f"{lang}_word_", word_indices.astype(str))
        docs = np.array([' '.join(doc[mask[i]]) for i, doc in enumerate(words)])
        doc_data[f"doc_clean_{languages.index(lang)}"] = docs.tolist()

    return doc_data

def generate_documents(
    num_docs: int,
    num_topics: int,
    vocab_size: int,
    num_covs: int = 0,
    num_languages: int = 1,
    lambda_: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    doc_topic_prior: str = 'logistic_normal',
    min_words: int = 100,
    max_words: int = 100,
    random_seed: int = 42,
    label_type: Optional[str] = None,  # 'classification' or 'regression'
    label_coeffs: Optional[np.ndarray] = None,
) -> Tuple[
    pd.DataFrame,                 # df_topic_dist
    pd.DataFrame,                 # df
    Optional[Dict[str, np.ndarray]],  # topic_word_matrices
    Optional[np.ndarray],         # lambda_
    Optional[np.ndarray]          # label_coeffs
]:
    """
    Generate synthetic multilingual documents with latent topic structure.

    Args:
        num_docs: number of documents to generate
        num_topics: number of topics
        vocab_size: vocabulary size per language
        num_covs: number of binary covariates (plus intercept)
        num_languages: number of parallel languages to simulate
        lambda_: regression weights for prevalence model (shape: (num_covs+1, num_topics))
        sigma: covariance matrix for logistic normal prior (shape: (num_topics, num_topics))
        doc_topic_prior: 'logistic_normal' or 'dirichlet'
        min_words: minimum document length
        max_words: maximum document length
        random_seed: reproducibility
        label_type: type of label to simulate ('classification' or 'regression')
        label_coeffs: coefficients for label simulation (if None, random coefficients are generated)

    Returns:
        - DataFrame with true document-topic distributions
        - DataFrame with documents and covariates
        - (Optional) dictionary of topic-word matrices per language
    """
    np.random.seed(random_seed)

    topicnames = [f"Topic{i}" for i in range(num_topics)]
    cov_names = [f"cov_{i}" for i in range(num_covs + 1)]
    languages = [f"lang{i}" for i in range(num_languages)]

    # Prevalence covariates
    if num_covs > 0:
        M_prevalence_covariates = np.zeros((num_docs, num_covs + 1), dtype=int)
        M_prevalence_covariates[:, 0] = 1  # intercept
        for i in range(num_covs):
            M_prevalence_covariates[:, i + 1] = np.random.randint(2, size=num_docs)

        if lambda_ is None:
            # Generate covariate effects for K dimensions
            lambda_ = np.random.randn(num_covs + 1, num_topics) * 0.5
        if sigma is None:
            # Covariance matrix for K dimensions
            sigma = np.eye(num_topics)
    else:
        if sigma is None:
            # Covariance matrix for K dimensions
            sigma = np.eye(num_topics)
        M_prevalence_covariates = None

    # Topic proportions
    if doc_topic_prior == 'dirichlet':
        if num_covs > 0:
            # Generate alpha for K dimensions
            alpha = np.exp(M_prevalence_covariates @ lambda_)
        else:
            alpha = np.full((num_docs, num_topics), 0.1)

        true_doc_topic_matrix = np.array([np.random.dirichlet(a) for a in alpha])

    elif doc_topic_prior == 'logistic_normal':
        if num_covs > 0:
            # Generate mean for K dimensions
            mean_k = M_prevalence_covariates @ lambda_
        else:
            mean_k = np.zeros((num_docs, num_topics))

        # Sample from K dimensional multivariate normal
        z_samples = np.array([
            np.random.multivariate_normal(m, sigma) for m in mean_k
        ])
        
        # Apply softmax to get simplex
        true_doc_topic_matrix = np.exp(z_samples)
        true_doc_topic_matrix /= true_doc_topic_matrix.sum(axis=1, keepdims=True)

    # Topic-word matrices per language
    topic_word_matrices = {
        lang: np.random.dirichlet([0.1] * vocab_size, num_topics)
        for lang in languages
    }

    # Generate multilingual documents
    doc_data = {f"doc_clean_{i}": [] for i in range(num_languages)}

    doc_data = generate_multilingual_docs_vectorized(
        true_doc_topic_matrix,
        topic_word_matrices,
        min_words,
        max_words,
        languages
    )

    df = pd.DataFrame(doc_data)

    if num_covs > 0:
        cov_df = pd.DataFrame(M_prevalence_covariates, columns=cov_names)
        df = pd.concat([df, cov_df], axis=1)

    df_topic_dist = pd.DataFrame(true_doc_topic_matrix, columns=topicnames)

    # Optional label simulation
    if label_type is not None:
        if label_coeffs is None:
            label_coeffs = np.random.randn(num_topics)

        logits = true_doc_topic_matrix @ label_coeffs

        if label_type == "classification":
            probs = 1 / (1 + np.exp(-logits))
            labels = np.random.binomial(1, probs)
        elif label_type == "regression":
            labels = logits + np.random.normal(0, 0.05, size=logits.shape[0])
        else:
            raise ValueError(f"Unsupported label_type: {label_type}")

        df["label"] = labels

    return df_topic_dist, df, topic_word_matrices, lambda_, label_coeffs


# Simulating ideal points

def generate_ideal_points(
    num_politicians: int,
    dim_ideal_points: int = 1,
    num_covs: int = 0,
    num_bills: int = 1000,
    num_survey_questions: int = 100,
    doc_length: int = 300,
    vocab_size: int = 500,
    label_type: Optional[str] = None,  # 'classification' or 'regression'
    label_coeffs: Optional[np.ndarray] = None,
    seed: int = 42,
    progress_bar: bool = True
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Simulate data for ideal point models with covariates and optional labels.

    Returns:
        - ideal_points: (N, D) matrix of true ideal points
        - df: DataFrame with documents, covariates, votes, surveys, and labels
        - word_matrix: (D, V) word distribution matrix
        - beta: (C+1, D) covariate effect matrix for ideal points (if num_covs > 0)
        - label_coeffs: coefficients used for label prediction (if label_type is given)
    """
    np.random.seed(seed)

    # 1. Covariates
    if num_covs > 0:
        X_covs = np.zeros((num_politicians, num_covs + 1))
        X_covs[:, 0] = 1  # intercept
        X_covs[:, 1:] = np.random.binomial(1, 0.5, size=(num_politicians, num_covs))
        beta = np.random.randn(num_covs + 1, dim_ideal_points) * 0.5
        ideal_points = X_covs @ beta + np.random.normal(0, 1, size=(num_politicians, dim_ideal_points))
    else:
        X_covs = None
        beta = None
        ideal_points = np.random.randn(num_politicians, dim_ideal_points)

    # 2. Speech word probabilities
    word_matrix = np.random.rand(dim_ideal_points, vocab_size)
    logits = ideal_points @ word_matrix
    word_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    # 3. Generate speeches
    if progress_bar:
        iterator = tqdm(range(num_politicians), desc="Speeches")
    else:
        iterator = range(num_politicians)
    speeches = [
        " ".join([f"word_{np.random.choice(vocab_size, p=word_probs[i])}" for _ in range(doc_length)])
        for i in iterator
    ]

    # 4. Generate votes
    bill_positions = np.random.randn(num_bills, dim_ideal_points)
    vote_probs = norm.cdf(ideal_points @ bill_positions.T)
    votes = np.random.binomial(1, vote_probs).astype(float)
    mask = np.random.rand(*votes.shape) < 0.3
    votes[mask] = np.nan

    # 5. Generate survey responses
    if progress_bar:
        iterator = tqdm(range(num_survey_questions), desc="Surveys")
    else:
        iterator = range(num_survey_questions)
    surveys = []
    for _ in iterator:
        K = np.random.randint(2, 6)
        cat_pos = np.random.randn(dim_ideal_points, K)
        logits = ideal_points @ cat_pos
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        responses = [np.random.choice(K, p=probs[i]) for i in range(num_politicians)]
        surveys.append(responses)
    surveys = np.array(surveys).T

    # 6. Generate labels
    if label_type is not None:
        if label_coeffs is None:
            label_coeffs = np.random.randn(dim_ideal_points)

        logit = ideal_points @ label_coeffs

        if label_type == "classification":
            probs = 1 / (1 + np.exp(-logit))
            labels = np.random.binomial(1, probs)
        elif label_type == "regression":
            labels = logit + np.random.normal(0, 0.5, size=logit.shape[0])
        else:
            raise ValueError(f"Invalid label_type: {label_type}")
    else:
        labels = None

    # 7. Build DataFrame
    df = pd.DataFrame({'doc_clean': speeches, 'i': np.arange(1, num_politicians + 1)})
    if num_covs > 0:
        df = pd.concat([df, pd.DataFrame(X_covs, columns=[f"cov_{i}" for i in range(num_covs + 1)])], axis=1)
    vote_cols = [f"vote_{i+1}" for i in range(num_bills)]
    survey_cols = [f"Q_{i+1}" for i in range(num_survey_questions)]
    df = pd.concat([df, pd.DataFrame(votes, columns=vote_cols), pd.DataFrame(surveys, columns=survey_cols)], axis=1)

    if labels is not None:
        df["label"] = labels

    return ideal_points, df, word_matrix, beta, label_coeffs
