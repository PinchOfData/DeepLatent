#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
DeepLatent - A unified latent variable modeling framework for analyzing 
large multimodal and multilingual datasets using variational inference 
with deep neural networks.
"""

__version__ = "0.1.1"
__author__ = "Germain Gauthier, Elliott Ash, Hugo Subtil, Philine Widmer"
__email__ = "germain.jean.gauthier@gmail.com"  

from .models import GTM, IdealPointNN, DeepLatent
from .corpus import Corpus
from .autoencoders import EncoderMLP, DecoderMLP, MultiModalEncoder, ImageEncoder, ImageDecoder
from .predictors import Predictor
from .priors import DirichletPrior, LogisticNormalPrior, GaussianPrior
from .utils import (
    compute_mmd_loss,
    top_k_indices_column,
    parse_modality_view,
    compute_dirichlet_likelihood,
    topic_diversity,
    check_max_local_length,
    bert_embeddings_from_list,
    text_processor,
)
from .simulations import (
    generate_multilingual_docs_vectorized,
    generate_documents,
    generate_ideal_points,
)

__all__ = [
    "GTM",
    "IdealPointNN", 
    "DeepLatent",
    "Corpus",
    "EncoderMLP",
    "DecoderMLP", 
    "MultiModalEncoder",
    "ImageEncoder",
    "ImageDecoder",
    "Predictor",
    "DirichletPrior",
    "LogisticNormalPrior", 
    "GaussianPrior",
    "compute_mmd_loss",
    "top_k_indices_column",
    "parse_modality_view",
    "compute_dirichlet_likelihood",
    "topic_diversity",
    "check_max_local_length",
    "bert_embeddings_from_list",
    "text_processor",
    "generate_multilingual_docs_vectorized",
    "generate_documents",
    "generate_ideal_points"
]