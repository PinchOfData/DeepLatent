#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from .autoencoders import (
    EncoderMLP,
    DecoderMLP,
    MultiModalEncoder,
    ImageEncoder,
    ImageDecoder,
)
from .predictors import Predictor, MultiLabelPredictor
from .priors import (
    DirichletPrior,
    LogisticNormalPrior,
    GaussianPrior,
    FixedDirichletPrior,
    FixedGaussianPrior,
    FixedLogisticNormalPrior,
)
from .utils import compute_mmd_loss, top_k_indices_column, parse_modality_view
from typing import Optional, List
from collections import OrderedDict


class DeepLatent:
    def __init__(
        self,
        train_data,
        test_data=None,
        n_factors=20,
        ae_type="wae",
        vi_type="iaf",
        latent_factor_prior="logistic_normal",
        update_prior=False,
        alpha=0.1,
        encoder_args={},
        decoder_args={},
        predictor_args={},
        labels_in_encoder=False,
        learn_prior_cov=True,
        fusion: str = "moe_average",
        gating_hidden_dim=None,
        num_flows=4,
        flow_hidden_dim=None,
        flow_use_permutations=True,
        flow_logscale_bound=2.0,
        mixture_components=10,
        num_steps=10000,
        batch_size=64,
        num_workers=4,
        optim_args=None,
        print_every_n_steps=1000,
        log_every_n_steps=float("inf"),
        print_topics=False,
        print_covariates=False,
        patience=float("inf"),
        patience_tol=1e-3,
        w_prior=1,
        w_pred_loss=1,
        kl_annealing_start=-1,
        kl_annealing_end=-1,
        free_bits=0.0,
        ckpt_folder="../ckpt",
        return_best_model=True,
        device=None,
        seed=42,
    ):
        """
        Args:
            train_data: a Corpus object
            test_data: a Corpus object
            n_factors: number of factors (topics / ideal points) to learn.
            ae_type: type of autoencoder. Either 'wae' (Wasserstein Autoencoder), 'vae' (Variational Autoencoder), or 'ae' (plain Autoencoder without prior).
            latent_factor_prior: prior on the document-topic distribution. Either 'dirichlet', 'logistic_normal', or 'gaussian'.
            alpha: concentration parameter of the Dirichlet prior (only used when latent_factor_prior='dirichlet' and update_prior=False).
            encoder_args: dictionary with the parameters for the encoder.
            decoder_args: dictionary with the parameters for the decoder.
            predictor_args: dictionary mapping label names to their predictor config.
                Each config can include: hidden_dims (list), dropout (float), activation (str),
                bias (bool), loss_weight (float). Example:
                {"sentiment": {"hidden_dims": [64], "loss_weight": 1.0}}
            fusion: type of fusion method to use. Options: 'moe_average' (equal weights), 'moe_gating' (learned gating), 'moe_learned' (learned fixed weights), 'poe' (uncorrected Product of Experts), 'corrected_poe' (corrected PoE matching true posterior p(z|x₁,...,xₘ) ∝ p(z) × ∏ₘ p(xₘ|z)).
            gating_hidden_dim: hidden dimension for gating mechanism (if used).
            num_flows: number of flows for IAF (if used).
            flow_hidden_dim: hidden dimension for IAF flows (if None, defaults to max(n_factors, 16)).
            flow_use_permutations: whether to use permutations between IAF flows for better expressivity.
            flow_logscale_bound: per-step bound b on the IAF affine log-scale, log_sigma in [-b, b]
                (sigma in [e^-b, e^b]). Larger b lets each flow step sharpen/spread the posterior more,
                but on US-congress data widening it did not improve held-out likelihood and b>=6
                diverged (NaN) without gradient clipping. Default 2.0 (stable); tune with care.
            num_steps: number of training steps to train the model.
            num_workers: number of workers for the data loaders.
            batch_size: batch size for training.
            optim_args: dictionary with the parameters for the optimizer. Can include 'main' for main parameters (encoder+decoders), 'predictor' for predictor parameters, 'prior' for prior parameters, and other Adam parameters like 'betas'. If None, uses default parameters.
            print_every_n_steps: number of steps between each print.
            log_every_n_steps: number of steps between each checkpoint.
            print_topics: whether to print topic words during training.
            print_covariates: whether to print covariate words during training.
            patience: number of steps to wait before stopping the training if the validation or training loss does not improve.
            patience_tol: tolerance for improvement in loss. Loss must improve by at least this amount to reset patience counter.
            w_prior: parameter to control the tightness of the encoder output with the document-topic prior. If set to None, w_prior is chosen automatically.
            w_pred_loss: parameter to control the weight given to the prediction task in the likelihood. Default is 1.
            kl_annealing_start: step at which to start the KL annealing.
            kl_annealing_end: step at which to end the KL annealing.
            free_bits: minimum KL divergence per latent dimension (prevents posterior collapse). Now properly applied per dimension for all vi_types including full_rank and iaf. Default is 0.0 (disabled).
            ckpt_folder: folder to save the checkpoints.
            return_best_model: if True, return the best model based on validation/training loss (default behavior). If False, return the model from the latest step.
            device: device to use for training.
            seed: random seed.
        """

        self.device = device or (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )

        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            np.random.seed(seed)

        self.n_factors = n_factors

        self.ae_type = ae_type
        assert ae_type in {"wae", "vae", "ae"}, f"Invalid ae_type: {ae_type}"
        self.vi_type = vi_type
        assert self.vi_type in {
            "mean_field",
            "full_rank",
            "iaf",
            "mixture_of_gaussians",
        }, f"Invalid vi_type: {vi_type}"

        # Logistic-normal latents use (K-1)-dim contrast coordinates: softmax is
        # invariant to shifts along the all-ones logit direction, so a K-dim latent
        # carries K parameter directions the likelihood cannot identify (and which
        # drift during training). theta = softmax(z @ V.T) with V = contrast_basis(K).
        # Gaussian and Dirichlet latents have no softmax gauge and keep K dims.
        self.contrast = latent_factor_prior == "logistic_normal"
        if self.contrast and n_factors < 2:
            raise ValueError(
                f"latent_factor_prior='logistic_normal' requires n_factors >= 2, got {n_factors}"
            )
        self.n_latent = n_factors - 1 if self.contrast else n_factors

        self.latent_factor_prior = latent_factor_prior
        self.update_prior = update_prior
        self.alpha = alpha

        # A Dirichlet prior has no closed-form KL in the encoder's logit space
        # (the posterior q(z|x) is Gaussian over logits, the Dirichlet lives on the
        # simplex). The VAE path would otherwise compare incommensurable quantities.
        # Use the Dirichlet prior with the WAE objective (MMD on simplex samples),
        # or use the 'logistic_normal' prior for a VAE topic model.
        if ae_type == "vae" and latent_factor_prior == "dirichlet":
            raise ValueError(
                "ae_type='vae' is not supported with a Dirichlet prior: the Dirichlet "
                "has no closed-form KL in the encoder's logit space. Use ae_type='wae' "
                "with latent_factor_prior='dirichlet', or latent_factor_prior="
                "'logistic_normal' with ae_type='vae'."
            )
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.print_every_n_steps = print_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.patience = patience
        self.patience_tol = patience_tol
        self.w_prior = w_prior
        self.w_pred_loss = w_pred_loss
        self.kl_annealing_start = kl_annealing_start
        self.kl_annealing_end = kl_annealing_end
        self.free_bits = free_bits
        self.ckpt_folder = ckpt_folder
        self.return_best_model = return_best_model
        self.print_topics = print_topics
        self.print_covariates = print_covariates
        self.fusion = fusion
        self.gating_hidden_dim = gating_hidden_dim
        self.num_flows = num_flows
        self.flow_hidden_dim = flow_hidden_dim
        self.flow_use_permutations = flow_use_permutations
        self.flow_logscale_bound = flow_logscale_bound
        self.mixture_components = mixture_components

        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)

        self.prevalence_covariate_size = (
            train_data.M_prevalence_covariates.shape[1] if train_data.prevalence else 0
        )
        self.content_covariate_size = (
            train_data.M_content_covariates.shape[1] if train_data.content else 0
        )
        self.prediction_covariate_size = (
            train_data.M_prediction.shape[1] if train_data.prediction else 0
        )
        self.labels_info = train_data.labels_info if train_data.labels_info else {}
        self.predictor_args = predictor_args

        # When labels_in_encoder is on, the outcome(s) y are concatenated to the
        # (non-image) encoder input so q(theta | x, y) can approximate the true
        # supervised posterior p(theta | x, y). The true y is used during training;
        # at inference (y unknown) the label block is filled with zeros.
        self.label_size = (
            train_data.M_labels.shape[1]
            if (
                train_data.labels_info
                and getattr(train_data, "M_labels", None) is not None
            )
            else 0
        )
        self.labels_in_encoder = bool(labels_in_encoder and self.label_size > 0)

        self.content_colnames = train_data.content_colnames or []
        self.prevalence_colnames = train_data.prevalence_colnames or []
        self.id2token = train_data.id2token

        # ENCODERS
        if not encoder_args:
            raise ValueError(
                "encoder_args is empty. You must specify at least one encoder configuration."
            )

        encoders = OrderedDict()
        for key, config in encoder_args.items():
            mod, view = parse_modality_view(key)
            view_data = train_data.processed_modalities[mod][view]
            view_type = view_data["type"]

            if view_type == "image":
                if self.labels_in_encoder:
                    raise NotImplementedError(
                        "labels_in_encoder is not supported for image modalities yet."
                    )
                # Image-specific encoder with CNN
                # Head sizes use the latent dim (K-1 contrast coords for logistic_normal).
                if ae_type == "vae":
                    if self.vi_type == "mean_field":
                        final_dim = self.n_latent * 2
                    elif self.vi_type == "full_rank":
                        final_dim = (
                            self.n_latent + (self.n_latent * (self.n_latent + 1)) // 2
                        )
                    elif self.vi_type == "iaf":
                        final_dim = self.n_latent * 2
                    elif self.vi_type == "mixture_of_gaussians":
                        final_dim = self.mixture_components * (2 * self.n_latent + 1)
                    else:
                        raise ValueError(f"Invalid vi_type: {self.vi_type}")
                else:  # wae or ae
                    final_dim = self.n_latent

                # input_size is now required in config
                if "input_size" not in config:
                    raise ValueError(
                        f"input_size must be specified in encoder_args for image modality '{key}'"
                    )

                encoders[key] = ImageEncoder(
                    input_size=config["input_size"],
                    input_channels=config.get("input_channels", 3),
                    latent_dim=final_dim,
                    prevalence_covariate_size=self.prevalence_covariate_size,
                    hidden_dims=config.get("hidden_dims", [32, 64, 128, 256]),
                    fc_hidden_dims=config.get("fc_hidden_dims", [512]),
                    dropout=config.get("dropout", 0.1),
                    activation=config.get("activation", "relu"),
                    use_batch_norm=config.get("use_batch_norm", True),
                )

            elif view_type in {"bow", "embedding", "vote"}:
                input_dim = view_data["matrix"].shape[1]
            elif view_type == "discrete_choice":
                sample_key = next(k for k in view_data if k != "type")
                input_dim = sum(
                    view_data[q]["matrix"].shape[1] for q in view_data if q != "type"
                )
            else:
                raise ValueError(f"Unsupported view_type: {view_type}")

            # Create MLP encoders for non-image modalities
            # Head sizes use the latent dim (K-1 contrast coords for logistic_normal).
            if view_type != "image":
                if ae_type == "vae":
                    if self.vi_type == "mean_field":
                        final_dim = self.n_latent * 2
                    elif self.vi_type == "full_rank":
                        final_dim = (
                            self.n_latent + (self.n_latent * (self.n_latent + 1)) // 2
                        )
                    elif self.vi_type == "iaf":
                        final_dim = self.n_latent * 2  # base params, flows in encoder
                    elif self.vi_type == "mixture_of_gaussians":
                        final_dim = self.mixture_components * (2 * self.n_latent + 1)
                    else:
                        raise ValueError(f"Invalid vi_type: {self.vi_type}")
                else:  # wae or ae
                    final_dim = self.n_latent

                extra_in = self.prevalence_covariate_size + (
                    self.label_size if self.labels_in_encoder else 0
                )
                dims = (
                    [input_dim + extra_in] + config.get("hidden_dims", []) + [final_dim]
                )

                encoders[key] = EncoderMLP(
                    encoder_dims=dims,
                    encoder_non_linear_activation=config.get("activation", "relu"),
                    encoder_bias=config.get("bias", True),
                    dropout=config.get("dropout", 0.0),
                )

        if self.fusion == "moe_average":
            moe_type, gating, poe = "average", False, False
        elif self.fusion == "moe_gating":
            moe_type, gating, poe = "gating", True, False
        elif self.fusion == "moe_learned":
            moe_type, gating, poe = "learned_weights", False, False
        elif self.fusion == "poe":
            moe_type, gating, poe = "average", False, "uncorrected"
        elif self.fusion == "corrected_poe":
            moe_type, gating, poe = "average", False, "corrected"
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion}")

        # DECODERS
        if not decoder_args:
            raise ValueError(
                "decoder_args is empty. You must specify at least one decoder configuration."
            )

        self.decoders = nn.ModuleDict()
        for key, config in decoder_args.items():
            mod, view = parse_modality_view(key)
            view_data = train_data.processed_modalities[mod][view]
            view_type = view_data["type"]

            if view_type == "image":
                # Image-specific decoder with CNN
                # output_size is now required in config
                if "output_size" not in config:
                    raise ValueError(
                        f"output_size must be specified in decoder_args for image modality '{key}'"
                    )

                self.decoders[key] = ImageDecoder(
                    output_size=config["output_size"],
                    latent_dim=n_factors,
                    content_covariate_size=self.content_covariate_size,
                    output_channels=config.get("output_channels", 3),
                    hidden_dims=config.get("hidden_dims", [256, 128, 64, 32]),
                    fc_hidden_dims=config.get("fc_hidden_dims", [512]),
                    dropout=config.get("dropout", 0.1),
                    activation=config.get("activation", "relu"),
                    use_batch_norm=config.get("use_batch_norm", True),
                ).to(self.device)

            elif view_type == "discrete_choice":
                sub_decoders = nn.ModuleDict()
                for question, subview in view_data.items():
                    if question == "type":
                        continue
                    output_dim = subview["matrix"].shape[1]
                    dims = (
                        [n_factors + self.content_covariate_size]
                        + config.get("hidden_dims", [])
                        + [output_dim]
                    )
                    sub_decoders[question] = DecoderMLP(
                        decoder_dims=dims,
                        decoder_non_linear_activation=config.get("activation", None),
                        decoder_bias=config.get("bias", False),
                        dropout=config.get("dropout", 0.0),
                    ).to(self.device)
                self.decoders[key] = sub_decoders
            else:
                output_dim = view_data["matrix"].shape[1]
                dims = (
                    [n_factors + self.content_covariate_size]
                    + config.get("hidden_dims", [])
                    + [output_dim]
                )
                self.decoders[key] = DecoderMLP(
                    decoder_dims=dims,
                    decoder_non_linear_activation=config.get("activation", None),
                    decoder_bias=config.get("bias", False),
                    dropout=config.get("dropout", 0.0),
                ).to(self.device)

        # PRIOR
        if self.ae_type == "ae":
            # Plain autoencoder - no prior needed
            self.prior = None
        elif not update_prior:
            if latent_factor_prior == "dirichlet":
                self.prior = FixedDirichletPrior(
                    self.prevalence_covariate_size, n_factors, alpha=alpha
                )
            elif latent_factor_prior == "logistic_normal":
                self.prior = FixedLogisticNormalPrior(
                    self.prevalence_covariate_size, n_factors
                )
            elif latent_factor_prior == "gaussian":
                self.prior = FixedGaussianPrior(
                    prevalence_covariate_size=self.prevalence_covariate_size,
                    n_dims=n_factors,
                )
            else:
                raise ValueError(
                    f"Fixed prior not supported for: {latent_factor_prior}"
                )
        else:
            if latent_factor_prior == "dirichlet":
                self.prior = DirichletPrior(self.prevalence_covariate_size, n_factors)
            elif latent_factor_prior == "logistic_normal":
                self.prior = LogisticNormalPrior(
                    self.prevalence_covariate_size,
                    n_factors,
                    learn_cov=learn_prior_cov,
                )
            elif latent_factor_prior == "gaussian":
                self.prior = GaussianPrior(
                    prevalence_covariate_size=self.prevalence_covariate_size,
                    n_dims=n_factors,
                )
            else:
                raise ValueError(f"Unrecognized prior: {latent_factor_prior}")

        # ENCODER - initialized after prior so self.prior is available
        self.encoder = MultiModalEncoder(
            encoders=encoders,
            topic_dim=self.n_latent,
            n_topics=n_factors if self.contrast else None,
            prior=self.prior if ae_type == "vae" else None,
            gating=gating,
            gating_hidden_dim=gating_hidden_dim,
            ae_type=ae_type,
            poe=poe,
            vi_type=self.vi_type,
            num_flows=num_flows,
            flow_hidden_dim=self.flow_hidden_dim,
            flow_use_permutations=self.flow_use_permutations,
            flow_logscale_bound=self.flow_logscale_bound,
            moe_type=moe_type,
            mixture_components=self.mixture_components,
        ).to(self.device)

        # PREDICTOR (multi-label)
        if self.labels_info:
            self.predictor = MultiLabelPredictor(
                input_dim=n_factors + self.prediction_covariate_size,
                labels_info=self.labels_info,
                predictor_configs=predictor_args,
            ).to(self.device)
        else:
            self.predictor = None

        # OPTIMIZER with different parameter groups
        # Get encoder parameters, excluding prior if it will be updated separately
        encoder_params = []
        if update_prior and self.prior is not None:
            # Collect prior parameter ids to exclude them from encoder params
            prior_param_ids = {id(p) for p in self.prior.parameters()}
            encoder_params = [
                p for p in self.encoder.parameters() if id(p) not in prior_param_ids
            ]
        else:
            encoder_params = list(self.encoder.parameters())

        main_params = encoder_params + list(self.decoders.parameters())
        predictor_params = []
        if self.predictor is not None:
            predictor_params = list(self.predictor.parameters())

        # Set up optimizer configuration
        if optim_args is None:
            optim_args = {
                "main": {"lr": 1e-3, "weight_decay": 0.0},
                # Prior is trained on a SLOWER timescale than the encoder/decoder on purpose:
                # the topics must form first (driven by reconstruction), then mean_net catches
                # up to the true covariate->prevalence map. Verified empirically that raising
                # the prior lr to 1e-3 plateaus covariate-coefficient recovery at ~0.35, while
                # lr=1e-4 climbs to ~0.95 (N=10k, 80 words). Keep the slow prior lr.
                # No weight_decay on the prior: it shrinks mean_net's covariate coefficients
                # toward zero, biasing prevalence-effect estimates. Slow lr alone gives the
                # two-timescale benefit without that bias.
                "prior": {"lr": 1e-4, "weight_decay": 0.0},
            }

        # Extract configurations for each parameter group
        main_config = optim_args.get("main")
        prior_config = optim_args.get("prior")
        predictor_config = optim_args.get("predictor")

        # Extract global optimizer settings (applied to all parameter groups if not overridden)
        global_config = {
            k: v
            for k, v in optim_args.items()
            if k not in ["main", "prior", "predictor"]
        }
        default_global = {"eps": 1e-8}

        # Merge global defaults with user-provided global config
        for key, default_val in default_global.items():
            if key not in global_config:
                global_config[key] = default_val

        # Apply global config to parameter groups if not already specified
        for config in [main_config, prior_config, predictor_config]:
            if config is not None:
                for key, default_val in global_config.items():
                    if key not in config:
                        config[key] = default_val

        # Create parameter groups
        param_groups = []

        # Add main parameters (encoder + decoders)
        if main_config is not None:
            param_groups.append({"params": main_params, **main_config})
        else:
            # If no main config specified, add main params to first group with global config
            param_groups.append({"params": main_params, **global_config})

        # Add predictor parameters if they exist and have separate config
        if self.predictor is not None and predictor_config is not None:
            param_groups.append({"params": predictor_params, **predictor_config})
        elif self.predictor is not None and predictor_config is None:
            # If no separate predictor config, add predictor params to main group
            param_groups[0]["params"].extend(predictor_params)

        # Add prior parameters if they exist and prior is updatable
        if update_prior and self.prior is not None:
            prior_params = list(self.prior.parameters())
            if prior_config is not None:
                param_groups.append({"params": prior_params, **prior_config})
            else:
                # If no prior config specified, use global config
                param_groups.append({"params": prior_params, **global_config})

        # Create optimizer (no need to pass global_config again since it's in param_groups)
        self.optimizer = torch.optim.Adam(param_groups)

        self.steps = 0
        self.loss = np.inf
        self.reconstruction_loss = np.inf
        self.divergence_loss = np.inf
        self.prediction_loss = np.inf

        # Loss tracking for visualization
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.val_recon_losses = []
        self.train_div_losses = []
        self.val_div_losses = []
        self.train_pred_losses = []
        self.val_pred_losses = []

        # Move all model components to the specified device
        self.to(self.device)

        self.train(train_data, test_data)

    def train(self, train_data, test_data=None):
        """
        Train the model.
        """

        train_data_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        if test_data is not None:
            test_data_loader = DataLoader(
                test_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        counter = 0
        if self.return_best_model:
            self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))

        if self.steps == 0:
            best_loss = np.inf
            best_step = -1

        else:
            best_loss = self.loss
            best_step = self.steps

        # Create infinite iterator for training data
        train_iter = iter(train_data_loader)

        for step in range(self.steps, self.num_steps):
            # Get next batch, restart iterator if needed
            try:
                train_data_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_data_loader)
                train_data_batch = next(train_iter)

            # Training step
            training_loss = self.step_batch(
                train_data_batch, train_data, validation=False
            )

            # Record training losses
            self.train_losses.append(training_loss)
            self.train_recon_losses.append(
                self.reconstruction_loss.item()
                if isinstance(self.reconstruction_loss, torch.Tensor)
                else self.reconstruction_loss
            )
            self.train_div_losses.append(
                self.divergence_loss.item()
                if isinstance(self.divergence_loss, torch.Tensor)
                else self.divergence_loss
            )
            self.train_pred_losses.append(
                self.prediction_loss.item()
                if isinstance(self.prediction_loss, torch.Tensor)
                else self.prediction_loss * self.w_pred_loss
            )

            # Validation step (if test data is available)
            if test_data is not None:
                # Use first batch of test data for validation
                test_iter = iter(test_data_loader)
                test_data_batch = next(test_iter)
                validation_loss = self.step_batch(
                    test_data_batch, test_data, validation=True
                )

                # Record validation losses
                self.val_losses.append(validation_loss)
                self.val_recon_losses.append(
                    self.reconstruction_loss.item()
                    if isinstance(self.reconstruction_loss, torch.Tensor)
                    else self.reconstruction_loss
                )
                self.val_div_losses.append(
                    self.divergence_loss.item()
                    if isinstance(self.divergence_loss, torch.Tensor)
                    else self.divergence_loss
                )
                self.val_pred_losses.append(
                    self.prediction_loss.item()
                    if isinstance(self.prediction_loss, torch.Tensor)
                    else self.prediction_loss * self.w_pred_loss
                )

            # Determine current loss for patience checking
            if test_data is not None:
                current_loss = validation_loss
            else:
                current_loss = training_loss

            # Check for improvement
            if current_loss < best_loss - self.patience_tol:
                best_loss = current_loss
                best_step = step
                counter = 0
                if self.return_best_model:
                    self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))
            else:
                counter += 1

            # Early stopping check
            if counter >= self.patience:
                if self.return_best_model:
                    self.load_model("{}/best_model.ckpt".format(self.ckpt_folder))
                    print(
                        f"\nEarly stopping at step {step + 1}. Reverting to step {best_step + 1}"
                    )
                else:
                    print(
                        f"\nEarly stopping at step {step + 1}. Using model from latest step."
                    )
                break

            # Logging
            if (step + 1) % self.log_every_n_steps == 0:
                save_name = f'{self.ckpt_folder}/M_K{self.n_factors}_{self.latent_factor_prior}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_{step+1}.ckpt'
                self.save_model(save_name)

            self.steps = step + 1

    def _append_encoder_labels(self, x, target_labels=None):
        """Append the outcome block to a (non-image) encoder input.

        Active only when ``labels_in_encoder`` is on. During training the true
        labels are concatenated so q(theta | x, y) can match p(theta | x, y); at
        inference time y is unknown, so the block is filled with zeros.
        """
        if not self.labels_in_encoder:
            return x
        if target_labels is not None:
            lab = target_labels.to(x.device).float()
        else:
            lab = torch.zeros(x.size(0), self.label_size, device=x.device)
        return torch.cat([x, lab], dim=1)

    def step_batch(self, data_batch, dataset, validation=False, num_samples=1):
        """
        Train the model for one batch (single step).
        """
        if validation:
            self.encoder.eval()
            self.decoders.eval()
            if self.labels_info:
                self.predictor.eval()
            if self.prior is not None:
                self.prior.eval()
        else:
            self.encoder.train()
            self.decoders.train()
            if self.labels_info:
                self.predictor.train()
            if self.prior is not None:
                self.prior.train()

        with torch.no_grad() if validation else torch.enable_grad():
            data = data_batch
            # Initialize loss components
            prediction_loss = 0.0

            # Sample random subset of modalities for training with PoE (use all for validation)
            available_modalities = list(self.encoder.encoders.keys())
            if (
                not validation
                and len(available_modalities) > 1
                and self.fusion in ["poe", "corrected_poe"]
            ):
                # Randomly sample which modalities to use (at least 1)
                # Each modality has independent 0.5 probability of being selected
                active_modalities = [
                    mod for mod in available_modalities if torch.rand(1).item() > 0.5
                ]
                # Ensure at least one modality is selected
                if len(active_modalities) == 0:
                    active_modalities = [
                        available_modalities[
                            torch.randint(len(available_modalities), (1,)).item()
                        ]
                    ]
            else:
                # Use all modalities for validation, non-PoE fusion, or single-modality models
                active_modalities = None

            if not validation:
                self.optimizer.zero_grad()

            # Move all tensors to device
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(self.device)

            prevalence_covariates = data.get("M_prevalence_covariates", None)
            content_covariates = data.get("M_content_covariates", None)
            prediction_covariates = data.get("M_prediction", None)
            target_labels = data.get("M_labels", None)

            # -------------------- ENCODER INPUT --------------------
            modality_inputs = {}
            for key in self.encoder.encoders.keys():
                mod, view = parse_modality_view(key)
                view_type = dataset.processed_modalities[mod][view]["type"]

                if view_type == "image":
                    # Images are already tensors from corpus lazy loading
                    x = data["modalities"][mod][view].to(self.device)
                    modality_inputs[key] = (
                        x  # Don't concatenate covariates for images - handled in encoder
                    )
                elif view_type in {"bow", "embedding"}:
                    x = data["modalities"][mod][view].to(self.device)
                elif view_type == "vote":
                    x = data["modalities"][mod][view]["matrix"].to(self.device)
                elif view_type == "discrete_choice":
                    # concatenate all questions into one input vector
                    question_tensors = [
                        data["modalities"][mod][view][q].to(self.device)
                        for q in data["modalities"][mod][view]
                        if q != "type"
                    ]
                    x = torch.cat(question_tensors, dim=-1)
                else:
                    raise ValueError(f"Unsupported view type: {view_type}")

                # For non-image modalities, concatenate covariates as before
                if view_type != "image":
                    if prevalence_covariates is not None:
                        x = torch.cat([x, prevalence_covariates], dim=1)
                    x = self._append_encoder_labels(x, target_labels)

                    modality_inputs[key] = x

            theta_q, z, mu_logvar = self.encoder(
                modality_inputs,
                active_modalities=active_modalities,
                prevalence_covariates=(
                    prevalence_covariates.to(self.device)
                    if prevalence_covariates is not None
                    else None
                ),
            )

            if self.latent_factor_prior in {"dirichlet", "logistic_normal"}:
                doc_latents = theta_q  # simplex (deterministic for ae_type="ae")
            else:
                doc_latents = z  # real-valued (deterministic for ae_type="ae")

            # A hard categorical MoG draw has no pathwise gradient with respect to
            # mixture probabilities. For the VAE likelihood, Rao-Blackwellize over the
            # component index instead: draw once from every Gaussian component and
            # weight every conditional NLL by its (differentiable) mixture probability.
            # This is the exact sum_c pi_c E_{N_c}[log p(data | z)] estimator described
            # in VARIATIONAL_FAMILIES.md. The ordinary sampled `doc_latents` remains the
            # public encoder output, while training uses these component-wise latents.
            mog_component_z = None
            mog_component_latents = None
            mog_component_weights = None
            if self.ae_type == "vae" and self.vi_type == "mixture_of_gaussians":
                mog_info = mu_logvar if isinstance(mu_logvar, tuple) else mu_logvar[-1]
                _, mog_means, mog_covariance_params, mog_component_weights = mog_info
                mog_component_z = self.encoder._mog_component_samples(
                    mog_means, mog_covariance_params
                )  # [B,C,D]
                B_mog, C_mog, D_mog = mog_component_z.shape
                if self.latent_factor_prior in {"dirichlet", "logistic_normal"}:
                    mog_component_latents = self.encoder.latent_to_theta(
                        mog_component_z.reshape(B_mog * C_mog, D_mog)
                    ).reshape(B_mog, C_mog, -1)
                else:
                    mog_component_latents = mog_component_z

            # -------------------- DECODERS --------------------
            # Reconstruction is the per-document negative log-likelihood (summed over
            # the tokens/votes/features within a document, averaged over the batch).
            # This matches the per-document KL below, so the total loss is exactly the
            # (negative) ELBO at w_prior=1 -- a valid lower bound on the log-likelihood.
            # Per-token averaging would down-weight the likelihood by ~document length
            # relative to the KL and cause posterior collapse at w_prior=1.
            reconstruction_loss = 0.0
            if mog_component_latents is not None:
                B_mog, C_mog, latent_dim_mog = mog_component_latents.shape
                decoder_doc_latents = mog_component_latents.reshape(
                    B_mog * C_mog, latent_dim_mog
                )
                if content_covariates is not None:
                    decoder_content_covariates = (
                        content_covariates.unsqueeze(1)
                        .expand(-1, C_mog, -1)
                        .reshape(B_mog * C_mog, -1)
                    )
                    theta_input = torch.cat(
                        [decoder_doc_latents, decoder_content_covariates], dim=1
                    )
                else:
                    decoder_content_covariates = None
                    theta_input = decoder_doc_latents
            else:
                decoder_doc_latents = doc_latents
                decoder_content_covariates = content_covariates
                theta_input = (
                    torch.cat([doc_latents, content_covariates], dim=1)
                    if content_covariates is not None
                    else doc_latents
                )

            for key, decoder in self.decoders.items():
                # Skip inactive modalities during training with modality masking
                if active_modalities is not None and key not in active_modalities:
                    continue

                mod, view = parse_modality_view(key)
                view_type = dataset.processed_modalities[mod][view]["type"]
                modality_data = data["modalities"][mod][view]

                if view_type == "image":
                    # Image reconstruction
                    target_images = modality_data.to(self.device)

                    # For image decoder, we need to pass content covariates separately
                    reconstructed_images = decoder(
                        decoder_doc_latents, decoder_content_covariates
                    )

                    # Per-document Gaussian NLL: summed over pixels, averaged over the batch.
                    if mog_component_weights is not None:
                        target_expanded = (
                            target_images.unsqueeze(1)
                            .expand(-1, C_mog, *target_images.shape[1:])
                            .reshape(B_mog * C_mog, *target_images.shape[1:])
                        )
                        component_nll = (
                            (reconstructed_images - target_expanded)
                            .pow(2)
                            .reshape(B_mog, C_mog, -1)
                            .sum(-1)
                        )
                        recon_loss = (
                            mog_component_weights * component_nll
                        ).sum() / B_mog
                    else:
                        recon_loss = (
                            F.mse_loss(
                                reconstructed_images, target_images, reduction="sum"
                            )
                            / target_images.shape[0]
                        )
                    reconstruction_loss += recon_loss

                elif view_type == "discrete_choice":
                    for question, question_decoder in decoder.items():
                        if question == "type":
                            continue
                        x_out = modality_data[question].to(self.device)
                        logits = question_decoder(theta_input)
                        targets = x_out.argmax(dim=-1)
                        if mog_component_weights is not None:
                            targets_expanded = (
                                targets.unsqueeze(1)
                                .expand(-1, C_mog)
                                .reshape(B_mog * C_mog)
                            )
                            component_nll = F.cross_entropy(
                                logits, targets_expanded, reduction="none"
                            ).reshape(B_mog, C_mog)
                            reconstruction_loss += (
                                mog_component_weights * component_nll
                            ).sum() / B_mog
                        else:
                            reconstruction_loss += F.cross_entropy(logits, targets)

                else:
                    target = (
                        modality_data.to(self.device)
                        if view_type in {"bow", "embedding"}
                        else modality_data["matrix"].to(self.device)
                    )
                    recon = decoder(theta_input)

                    if view_type == "bow":
                        if mog_component_weights is not None:
                            log_probs = F.log_softmax(
                                recon.reshape(B_mog, C_mog, -1), dim=2
                            )
                            component_nll = -(target.unsqueeze(1) * log_probs).sum(-1)
                            recon_loss = (
                                mog_component_weights * component_nll
                            ).sum() / B_mog
                        else:
                            log_probs = F.log_softmax(recon, dim=1)
                            # Per-document multinomial NLL: summed over the vocabulary,
                            # averaged over the batch.
                            recon_loss = (
                                -torch.sum(target * log_probs) / target.shape[0]
                            )

                    elif view_type == "embedding":
                        # Per-document Gaussian NLL: summed over feature dims, averaged over batch.
                        if mog_component_weights is not None:
                            component_nll = (
                                (recon.reshape(B_mog, C_mog, -1) - target.unsqueeze(1))
                                .pow(2)
                                .sum(-1)
                            )
                            recon_loss = (
                                mog_component_weights * component_nll
                            ).sum() / B_mog
                        else:
                            recon_loss = (
                                F.mse_loss(recon, target, reduction="sum")
                                / target.shape[0]
                            )

                    elif view_type == "vote":
                        mask = ~modality_data["mask"].to(self.device)
                        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
                        if mog_component_weights is not None:
                            recon_reshaped = recon.reshape(B_mog, C_mog, -1)
                            losses = loss_fn(
                                recon_reshaped,
                                target.unsqueeze(1).expand_as(recon_reshaped),
                            )
                            component_nll = (losses * mask.unsqueeze(1)).sum(-1)
                            recon_loss = (
                                mog_component_weights * component_nll
                            ).sum() / B_mog
                        else:
                            losses = loss_fn(recon, target)
                            # Per-document Bernoulli NLL: summed over observed votes,
                            # averaged over the batch.
                            recon_loss = torch.sum(losses * mask) / target.shape[0]

                    else:
                        raise ValueError(f"Unsupported view type: {view_type}")

                    reconstruction_loss += recon_loss

            # -------------------- PRIOR / MMD / KL Divergence --------------------
            if self.ae_type == "ae":
                # Plain autoencoder - no prior loss
                divergence_loss = torch.tensor(0.0, device=self.device)
            else:
                # WAE or VAE - compute prior-related losses
                mmd_loss = 0.0
                for _ in range(num_samples):
                    theta_prior = self.prior.sample(
                        N=doc_latents.shape[0],
                        M_prevalence_covariates=prevalence_covariates,
                    ).to(self.device)
                    mmd_loss += compute_mmd_loss(
                        doc_latents, theta_prior, device=self.device
                    )

                if self.steps < self.kl_annealing_start:
                    beta = 0.0
                elif self.steps > self.kl_annealing_end:
                    beta = self.w_prior
                else:
                    # Linear KL annealing schedule
                    span = self.kl_annealing_end - self.kl_annealing_start
                    t = (self.steps - self.kl_annealing_start) / span  # 0→1
                    beta = self.w_prior * t

            if self.ae_type == "vae":
                # Expect a single fused posterior tuple in mu_logvar_fused
                mu_logvar_fused = (
                    mu_logvar if isinstance(mu_logvar, tuple) else mu_logvar[-1]
                )

                # Check if prior has full covariance (learned priors)
                has_full_cov = hasattr(self.prior, "sigma") and not isinstance(
                    self.prior,
                    (FixedGaussianPrior, FixedLogisticNormalPrior, FixedDirichletPrior),
                )

                if self.vi_type == "iaf":
                    # mu_q, logvar_q from encoder; z0 is the actual sample that was transformed;
                    # flow returns zk, log_det_j (sum over steps and dims)
                    mu_q, logvar_q, z0, zk, log_det_j = mu_logvar_fused

                    # log q0(z0) - use the actual z0 that was transformed
                    log_q_z0 = -0.5 * (
                        torch.sum(logvar_q, dim=1)
                        + torch.sum((z0 - mu_q) ** 2 / torch.exp(logvar_q), dim=1)
                        + z0.size(1) * np.log(2 * np.pi)
                    )

                    # log p(zk) - handle full covariance prior
                    if has_full_cov:
                        mu_p, Sigma_p = self.prior.get_prior_params(
                            prevalence_covariates, return_full_cov=True
                        )
                        B = zk.shape[0]
                        D = zk.shape[1]

                        # Expand for batch
                        Sigma_p_batch = Sigma_p.unsqueeze(0).expand(
                            B, -1, -1
                        )  # [B, D, D]
                        Sigma_p_inv = torch.inverse(Sigma_p_batch)

                        # Multivariate Gaussian log-pdf: log p(zk)
                        logdet_p = torch.logdet(Sigma_p_batch)  # [B]
                        diff = (zk - mu_p).unsqueeze(2)  # [B, D, 1]
                        quad_term = (
                            torch.bmm(
                                torch.bmm(diff.transpose(1, 2), Sigma_p_inv), diff
                            )
                            .squeeze(-1)
                            .squeeze(-1)
                        )  # [B]
                        log_p_zk = -0.5 * (logdet_p + quad_term + D * np.log(2 * np.pi))
                    else:
                        # Diagonal prior
                        mu_p, logvar_p = self.prior.get_prior_params(
                            prevalence_covariates
                        )
                        var_p = torch.exp(logvar_p)
                        log_p_zk = -0.5 * (
                            torch.sum(logvar_p, dim=1)
                            + torch.sum((zk - mu_p) ** 2 / var_p, dim=1)
                            + zk.size(1) * np.log(2 * np.pi)
                        )

                    # Monte Carlo KL (note the minus sign on log_det_j)
                    kl_per_sample = log_q_z0 - log_det_j - log_p_zk  # [B]

                    # For free_bits, approximate per-dimension KL
                    if self.free_bits > 0.0:
                        # Approximate per-dimension KL using base distribution
                        # This is an approximation since flow transforms don't preserve per-dimension structure
                        var_q = torch.exp(logvar_q)  # [B, D]

                        if has_full_cov:
                            # Use diagonal approximation for full covariance prior
                            Sigma_p_diag = (
                                torch.diagonal(Sigma_p, dim1=0, dim2=1)
                                .unsqueeze(0)
                                .expand(B, -1)
                            )  # [B, D]
                            kl_per_dim_base = 0.5 * (
                                -logvar_q
                                + torch.log(Sigma_p_diag)
                                - 1
                                + var_q / Sigma_p_diag
                                + (mu_q - mu_p).pow(2) / Sigma_p_diag
                            )  # [B, D]
                        else:
                            # Diagonal prior - exact per-dimension KL for base distribution
                            kl_per_dim_base = 0.5 * (
                                logvar_p
                                - logvar_q
                                - 1
                                + var_q / var_p
                                + (mu_q - mu_p).pow(2) / var_p
                            )  # [B, D]

                        # Distribute the flow correction (log_det_j) equally across dimensions
                        # This is an approximation, but better than ignoring per-dimension structure
                        flow_correction_per_dim = log_det_j.unsqueeze(1) / zk.size(
                            1
                        )  # [B, 1] -> [B, D]
                        kl_per_dim_approx = (
                            kl_per_dim_base - flow_correction_per_dim
                        )  # [B, D]

                        kl_raw = kl_per_dim_approx  # [B, D] for free_bits processing
                        kl_raw_total = kl_per_sample  # [B] store total for reference
                    else:
                        kl_raw = kl_per_sample  # [B] for regular processing

                elif self.vi_type == "mean_field":
                    mu_q, logvar_q = mu_logvar_fused

                    if has_full_cov:
                        # Full covariance prior, diagonal posterior
                        mu_p, Sigma_p = self.prior.get_prior_params(
                            prevalence_covariates, return_full_cov=True
                        )
                        var_q = torch.exp(
                            logvar_q
                        )  # [B, D] - diagonal posterior covariance
                        B, D = mu_q.shape

                        # Expand Sigma_p for batch processing
                        Sigma_p_batch = Sigma_p.unsqueeze(0).expand(
                            B, -1, -1
                        )  # [B, D, D]
                        Sigma_p_inv = torch.inverse(Sigma_p_batch)

                        # Log determinants
                        logdet_p = torch.logdet(Sigma_p_batch)  # [B]
                        logdet_q = torch.sum(
                            logvar_q, dim=1
                        )  # [B] - sum of log diagonal elements

                        # Trace term: tr(Σ_p^-1 Σ_q) where Σ_q is diagonal
                        trace_term = torch.sum(
                            torch.diagonal(Sigma_p_inv, dim1=1, dim2=2) * var_q, dim=1
                        )  # [B]

                        # Quadratic term: (μ_q - μ_p)^T Σ_p^-1 (μ_q - μ_p)
                        diff = (mu_q - mu_p).unsqueeze(2)  # [B, D, 1]
                        quad_term = (
                            torch.bmm(
                                torch.bmm(diff.transpose(1, 2), Sigma_p_inv), diff
                            )
                            .squeeze(-1)
                            .squeeze(-1)
                        )  # [B]

                        kl_raw = 0.5 * (
                            logdet_p - logdet_q - D + trace_term + quad_term
                        )  # [B]
                        kl = kl_raw.mean()
                    else:
                        # Diagonal prior
                        mu_p, logvar_p = self.prior.get_prior_params(
                            prevalence_covariates
                        )
                        var_q = torch.exp(logvar_q)
                        var_p = torch.exp(logvar_p)

                        # Compute per-dimension KL
                        kl_per_dim = 0.5 * (
                            logvar_p
                            - logvar_q
                            - 1
                            + var_q / var_p
                            + (mu_q - mu_p).pow(2) / var_p
                        )  # [B, D]
                        kl_raw = kl_per_dim  # Keep tensor for free_bits processing

                elif self.vi_type == "full_rank":
                    mu_q, L = mu_logvar_fused  # L lower-triangular with positive diag
                    B, D, _ = L.shape
                    Sigma_q = torch.bmm(L, L.transpose(1, 2))  # [B, D, D]
                    logdet_q = 2 * torch.sum(
                        torch.log(torch.diagonal(L, dim1=1, dim2=2) + 1e-6), dim=1
                    )  # [B]

                    if has_full_cov:
                        # Full covariance prior
                        mu_p, Sigma_p = self.prior.get_prior_params(
                            prevalence_covariates, return_full_cov=True
                        )

                        # Expand prior covariance for batch
                        Sigma_p_batch = Sigma_p.unsqueeze(0).expand(
                            B, -1, -1
                        )  # [B, D, D]
                        Sigma_p_inv = torch.inverse(Sigma_p_batch)

                        # KL divergence components
                        logdet_p = torch.logdet(Sigma_p_batch)  # [B]

                        # Trace term: tr(Σ_p^-1 Σ_q)
                        trace_term = torch.sum(
                            torch.diagonal(
                                torch.bmm(Sigma_p_inv, Sigma_q), dim1=1, dim2=2
                            ),
                            dim=1,
                        )  # [B]

                        # Quadratic term: (μ_q - μ_p)^T Σ_p^-1 (μ_q - μ_p)
                        diff = (mu_q - mu_p).unsqueeze(2)  # [B, D, 1]
                        quad_term = (
                            torch.bmm(
                                torch.bmm(diff.transpose(1, 2), Sigma_p_inv), diff
                            )
                            .squeeze(-1)
                            .squeeze(-1)
                        )  # [B]

                        kl_raw = 0.5 * (
                            logdet_p - logdet_q - D + trace_term + quad_term
                        )  # [B]

                        # For free_bits, approximate per-dimension KL using diagonal elements
                        # This is an approximation since full_rank KL can't be exactly decomposed per dimension
                        if self.free_bits > 0.0:
                            # Use diagonal elements of Sigma_q and Sigma_p to approximate per-dimension contributions
                            diag_q = torch.diagonal(Sigma_q, dim1=1, dim2=2)  # [B, D]
                            diag_p_inv = torch.diagonal(
                                Sigma_p_inv, dim1=1, dim2=2
                            )  # [B, D]

                            # Approximate per-dimension KL using diagonal terms
                            kl_per_dim_approx = 0.5 * (
                                -torch.log(diag_q)
                                + torch.log(1.0 / diag_p_inv)
                                - 1
                                + diag_q * diag_p_inv
                                + (mu_q - mu_p).pow(2) * diag_p_inv
                            )  # [B, D]
                    else:
                        # Diagonal prior - can compute exact per-dimension KL
                        mu_p, logvar_p = self.prior.get_prior_params(
                            prevalence_covariates
                        )
                        var_p = torch.exp(logvar_p)
                        var_q_diag = torch.diagonal(Sigma_q, dim1=1, dim2=2)  # [B, D]

                        # Per-dimension KL for diagonal prior
                        kl_per_dim_approx = 0.5 * (
                            logvar_p
                            - torch.log(var_q_diag)
                            - 1
                            + var_q_diag / var_p
                            + (mu_q - mu_p).pow(2) / var_p
                        )  # [B, D]

                        # Total KL for backward compatibility
                        logdet_p = torch.sum(logvar_p, dim=1)  # diag prior
                        trace_term = torch.sum(var_q_diag / var_p, dim=1)
                        diff = mu_q - mu_p
                        quad_term = torch.sum(diff.pow(2) / var_p, dim=1)
                        kl_raw = 0.5 * (
                            logdet_p - logdet_q - D + trace_term + quad_term
                        )  # [B]

                    # Store both total and per-dimension approximations for free_bits processing
                    kl_raw_total = kl_raw  # [B]
                    if self.free_bits > 0.0:
                        kl_raw = kl_per_dim_approx  # [B, D] for free_bits processing
                    else:
                        kl_raw = kl_raw_total  # [B] for regular processing

                elif self.vi_type == "mixture_of_gaussians":
                    # MC estimate of KL(sum_c pi_c N_c || p(z)), Rao-Blackwellized over
                    # the component index. Corrected MoG-PoE components can have full
                    # covariance; ordinary MoG/MoE components remain diagonal.
                    _, means, covariance_params, pi = mu_logvar_fused
                    B_, C_, D_ = means.shape
                    z_c = (
                        mog_component_z
                        if mog_component_z is not None
                        else self.encoder._mog_component_samples(
                            means, covariance_params
                        )
                    )  # [B,C,K]

                    # log q(z_c) = logsumexp_{c'} [log pi_c' + log N_c'(z_c)]
                    log_N = self.encoder._mog_component_log_prob(
                        z_c, means, covariance_params
                    )  # [B,C,C]
                    log_pi = torch.log(pi + 1e-12).unsqueeze(1)  # [B,1,C]
                    log_q = torch.logsumexp(log_pi + log_N, dim=2)  # [B,C]

                    # log p(z_c) under the prior
                    if has_full_cov:
                        mu_p, Sigma_p = self.prior.get_prior_params(
                            prevalence_covariates, return_full_cov=True
                        )
                        Sigma_p_inv = torch.inverse(Sigma_p)  # [K,K]
                        logdet_p = torch.logdet(Sigma_p)  # scalar
                        diff = z_c - mu_p.unsqueeze(1)  # [B,C,K]
                        quad = torch.einsum("bck,kj,bcj->bc", diff, Sigma_p_inv, diff)
                        log_p = -0.5 * (
                            quad + logdet_p + D_ * np.log(2 * np.pi)
                        )  # [B,C]
                    else:
                        mu_p, logvar_p = self.prior.get_prior_params(
                            prevalence_covariates
                        )
                        var_p = torch.exp(logvar_p).unsqueeze(1)  # [B,1,K]
                        log_p = -0.5 * (
                            (z_c - mu_p.unsqueeze(1)) ** 2 / var_p
                            + logvar_p.unsqueeze(1)
                            + np.log(2 * np.pi)
                        ).sum(
                            -1
                        )  # [B,C]

                    kl_raw = (pi * (log_q - log_p)).sum(dim=1)  # [B]

                # Apply free_bits if specified (now truly per dimension for all vi_types)
                if self.free_bits > 0.0:
                    if kl_raw.dim() == 2:  # [B, D] case - per dimension KL available
                        kl_raw = torch.max(
                            kl_raw, torch.tensor(self.free_bits, device=self.device)
                        )
                    else:  # [B] case - fallback to old behavior (shouldn't happen with new implementation)
                        min_kl_total = self.free_bits * self.n_latent
                        kl_raw = torch.max(
                            kl_raw, torch.tensor(min_kl_total, device=self.device)
                        )

                # Now take the mean (and sum over dimensions if needed)
                if (
                    kl_raw.dim() == 2
                ):  # [B, D] case - sum over dimensions, then mean over batch
                    kl = kl_raw.sum(dim=1).mean()
                else:  # [B] case - just mean over batch
                    kl = kl_raw.mean()

                divergence_loss = beta * kl
            elif self.ae_type == "wae":
                divergence_loss = mmd_loss * self.w_prior

            # -------------------- PREDICTION (multi-label) --------------------
            if target_labels is not None and self.labels_info:
                if mog_component_latents is not None:
                    prediction_latents = mog_component_latents.reshape(
                        B_mog * C_mog, -1
                    )
                    prediction_covariates_expanded = (
                        prediction_covariates.unsqueeze(1)
                        .expand(-1, C_mog, -1)
                        .reshape(B_mog * C_mog, -1)
                        if prediction_covariates is not None
                        else None
                    )
                    predictions = self.predictor(
                        prediction_latents, prediction_covariates_expanded
                    )
                else:
                    predictions = self.predictor(doc_latents, prediction_covariates)
                prediction_loss = 0.0
                self.per_label_losses = {}

                for label_name, label_info in self.labels_info.items():
                    pred = predictions[label_name]
                    label_type = label_info["type"]
                    start_idx = label_info["start_idx"]

                    # Extract target for this label
                    target = target_labels[:, start_idx]

                    # Compute loss based on label type
                    if label_type == "regression":
                        # Proper Gaussian NLL with a learned noise variance sigma^2, so this is
                        # the true log p(y|theta) and the ELBO at loss_weight=1 is the correct
                        # joint likelihood. (Bare MSE is this with sigma^2 fixed at 0.5, which
                        # under-weights an informative y and attenuates the topic->y coefficients.)
                        log_var = self.predictor.noise_log_var[label_name].clamp(
                            -12.0, 4.0
                        )
                        if mog_component_weights is not None:
                            sq_err = (
                                pred.reshape(B_mog, C_mog) - target.unsqueeze(1)
                            ).pow(2)
                            mean_sq_err = (mog_component_weights * sq_err).sum() / B_mog
                        else:
                            mean_sq_err = (pred.squeeze() - target).pow(2).mean()
                        label_loss = 0.5 * (
                            torch.exp(-log_var) * mean_sq_err
                            + log_var
                            + np.log(2 * np.pi)
                        )
                    elif label_type == "binary":
                        if mog_component_weights is not None:
                            component_nll = F.binary_cross_entropy_with_logits(
                                pred.reshape(B_mog, C_mog),
                                target.unsqueeze(1).expand(-1, C_mog),
                                reduction="none",
                            )
                            label_loss = (
                                mog_component_weights * component_nll
                            ).sum() / B_mog
                        else:
                            label_loss = F.binary_cross_entropy_with_logits(
                                pred.squeeze(), target
                            )
                    elif label_type == "multiclass":
                        if mog_component_weights is not None:
                            target_expanded = (
                                target.long()
                                .unsqueeze(1)
                                .expand(-1, C_mog)
                                .reshape(B_mog * C_mog)
                            )
                            component_nll = F.cross_entropy(
                                pred, target_expanded, reduction="none"
                            ).reshape(B_mog, C_mog)
                            label_loss = (
                                mog_component_weights * component_nll
                            ).sum() / B_mog
                        else:
                            label_loss = F.cross_entropy(pred, target.long())
                    else:
                        raise ValueError(f"Unknown label type: {label_type}")

                    # Apply per-label loss weight
                    config = self.predictor_args.get(label_name, {})
                    loss_weight = config.get("loss_weight", 1.0)
                    weighted_loss = label_loss * loss_weight

                    self.per_label_losses[label_name] = label_loss.item()
                    prediction_loss = prediction_loss + weighted_loss
            else:
                prediction_loss = 0.0
                self.per_label_losses = {}

            # -------------------- TOTAL LOSS --------------------
            loss = (
                reconstruction_loss
                + divergence_loss
                + prediction_loss * self.w_pred_loss
            )

            self.loss = loss
            self.reconstruction_loss = reconstruction_loss
            self.divergence_loss = divergence_loss
            self.prediction_loss = prediction_loss

            if not validation:
                loss.backward()
                self.optimizer.step()

            # Print progress if needed
            if (self.steps + 1) % self.print_every_n_steps == 0:
                msg = (
                    f"Step {(self.steps+1):>6d}"
                    f"\tMean {'Validation' if validation else 'Training'} Loss:{loss.item():<.7f}"
                    f"\nRec Loss:{reconstruction_loss.item():<.7f}"
                    f"\nDivergence Loss:{divergence_loss.item():<.7f}"
                    f"\nPred Loss:{prediction_loss * self.w_pred_loss:<.7f}\n"
                )
                print(msg)

                if self.print_topics == True:
                    print(
                        "\n".join(
                            [
                                "{}: {}".format(str(k), str(v))
                                for k, v in self.get_topic_words(topK=5).items()
                            ]
                        )
                    )

                if self.print_covariates == True and self.content_covariate_size > 0:
                    print(
                        "\n".join(
                            [
                                "{}: {}".format(str(k), str(v))
                                for k, v in self.get_covariate_words(topK=5).items()
                            ]
                        )
                    )

        return loss.item()

    def plot_training_loss(
        self,
        components=None,
        save_path=None,
        figsize=None,
        smoothing_window=None,
        all_components=None,
    ):
        """
        Plot the training (and validation if available) loss curves.

        Args:
            components: List of loss components to plot. Options: 'total', 'reconstruction',
                       'divergence', 'prediction'. If None, defaults based on all_components.
                       Examples: ['total'], ['reconstruction'], ['total', 'reconstruction']
            save_path: Path to save the plot. If None, the plot is displayed but not saved.
            figsize: Figure size as (width, height) tuple. Auto-determined if None.
            smoothing_window: If provided, apply moving average smoothing with this window size.
            all_components: If True, shows all components.

        Examples:
            >>> # Plot only total loss
            >>> model.plot_training_loss(components=['total'])
            >>>
            >>> # Plot only reconstruction loss
            >>> model.plot_training_loss(components=['reconstruction'])
            >>>
            >>> # Plot total and reconstruction
            >>> model.plot_training_loss(components=['total', 'reconstruction'])
            >>>
            >>> # Plot all components
            >>> model.plot_training_loss(all_components=True)
            >>>
            >>> # With smoothing and save
            >>> model.plot_training_loss(components=['total'], save_path='loss.png', smoothing_window=50)
        """
        if len(self.train_losses) == 0:
            print("No training losses recorded. Train the model first.")
            return None

        # Handle backward compatibility with all_components
        if components is None:
            if all_components is True:
                components = ["total", "reconstruction", "divergence", "prediction"]
            elif all_components is False:
                components = ["total"]
            else:
                components = ["total"]

        # Validate components
        valid_components = {"total", "reconstruction", "divergence", "prediction"}
        components = [c.lower() for c in components]
        invalid = set(components) - valid_components
        if invalid:
            raise ValueError(
                f"Invalid components: {invalid}. Valid options: {valid_components}"
            )

        def smooth(data, window):
            """Apply moving average smoothing"""
            if window is None or window <= 1:
                return data
            kernel = np.ones(window) / window
            return np.convolve(data, kernel, mode="valid")

        steps = np.arange(1, len(self.train_losses) + 1)

        # Determine subplot layout
        n_plots = len(components)
        if n_plots == 1:
            nrows, ncols = 1, 1
            if figsize is None:
                figsize = (10, 6)
        elif n_plots == 2:
            nrows, ncols = 1, 2
            if figsize is None:
                figsize = (14, 5)
        elif n_plots == 3:
            nrows, ncols = 1, 3
            if figsize is None:
                figsize = (16, 5)
        else:  # 4 components
            nrows, ncols = 2, 2
            if figsize is None:
                figsize = (12, 8)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_plots > 1 else [axes]

        if n_plots > 1:
            fig.suptitle("Training Loss", fontsize=16)

        # Plot each requested component
        plot_idx = 0

        if "total" in components:
            ax = axes[plot_idx]
            train_loss = smooth(self.train_losses, smoothing_window)
            ax.plot(
                steps[: len(train_loss)],
                train_loss,
                label="Train",
                linewidth=2,
                alpha=0.8,
            )
            if len(self.val_losses) > 0:
                val_loss = smooth(self.val_losses, smoothing_window)
                ax.plot(
                    steps[: len(val_loss)],
                    val_loss,
                    label="Validation",
                    linewidth=2,
                    alpha=0.8,
                )
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
            ax.set_title("Total Loss", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        if "reconstruction" in components:
            ax = axes[plot_idx]
            train_recon = smooth(self.train_recon_losses, smoothing_window)
            ax.plot(
                steps[: len(train_recon)],
                train_recon,
                label="Train",
                linewidth=2,
                alpha=0.8,
            )
            if len(self.val_recon_losses) > 0:
                val_recon = smooth(self.val_recon_losses, smoothing_window)
                ax.plot(
                    steps[: len(val_recon)],
                    val_recon,
                    label="Validation",
                    linewidth=2,
                    alpha=0.8,
                )
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
            ax.set_title("Reconstruction Loss", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        if "divergence" in components:
            ax = axes[plot_idx]
            train_div = smooth(self.train_div_losses, smoothing_window)
            ax.plot(
                steps[: len(train_div)],
                train_div,
                label="Train",
                linewidth=2,
                alpha=0.8,
            )
            if len(self.val_div_losses) > 0:
                val_div = smooth(self.val_div_losses, smoothing_window)
                ax.plot(
                    steps[: len(val_div)],
                    val_div,
                    label="Validation",
                    linewidth=2,
                    alpha=0.8,
                )
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
            ax.set_title("Divergence Loss", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        if "prediction" in components:
            ax = axes[plot_idx]
            if len(self.train_pred_losses) > 0 and max(self.train_pred_losses) > 0:
                train_pred = smooth(self.train_pred_losses, smoothing_window)
                ax.plot(
                    steps[: len(train_pred)],
                    train_pred,
                    label="Train",
                    linewidth=2,
                    alpha=0.8,
                )
                if len(self.val_pred_losses) > 0:
                    val_pred = smooth(self.val_pred_losses, smoothing_window)
                    ax.plot(
                        steps[: len(val_pred)],
                        val_pred,
                        label="Validation",
                        linewidth=2,
                        alpha=0.8,
                    )
                ax.set_xlabel("Training Step")
                ax.set_ylabel("Loss")
                ax.set_title("Prediction Loss", fontweight="bold")
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Prediction Loss",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_title("Prediction Loss", fontweight="bold")
            plot_idx += 1

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def get_topic_words(self):
        """
        Get top words per topic (not implemented in base DeepLatent class).

        This method is a stub in the base class. Use the GTM subclass which
        implements this method for topic models with BOW decoders.

        Raises:
            NotImplementedError: This method should be called on GTM instances,
                                not on the base DeepLatent class.

        See Also:
            GTM.get_topic_words(): Implemented version for topic models
        """
        pass

    def latent_to_theta(self, z):
        """
        Canonical latent -> simplex map.

        For logistic-normal models, z holds (K-1)-dimensional contrast
        coordinates and theta = softmax(z @ V.T) with V the fixed orthonormal
        contrast basis; otherwise a plain softmax over the latent dimensions.
        Use this instead of hand-rolling F.softmax(z, dim=1) on raw latents.

        Args:
            z: Latent tensor [batch_size, n_latent]

        Returns:
            theta: Simplex tensor [batch_size, n_factors]
        """
        return self.encoder.latent_to_theta(z)

    def get_prevalence_coefficients(
        self, fold_bias: bool = True, to_numpy: bool = True
    ):
        """
        Centered per-topic prevalence coefficients of the learned prior.

        Lifts the prior mean_net's (K-1)-dim contrast-space coefficients to
        theta-logit (K) space via the contrast basis V. Columns of V sum to
        zero, so each covariate's coefficients are exactly centered across
        topics -- the unique centered representative of the softmax gauge
        class, directly comparable to centered ground-truth coefficients.

        Args:
            fold_bias: If True, add the mean_net bias into the column named
                'Intercept' in self.prevalence_colnames (patsy's default
                first column). If no Intercept column exists, warns and
                returns the bias unfolded.
            to_numpy: Return numpy arrays (True) or torch tensors (False).

        Returns:
            W: [n_factors, prevalence_covariate_size] coefficient matrix
               (bias folded into the Intercept column when fold_bias=True).

        Raises:
            ValueError: If the prior is not a learnable logistic-normal prior
                with prevalence covariates (requires update_prior=True).
        """
        if self.latent_factor_prior != "logistic_normal" or not hasattr(
            self.prior, "get_prevalence_coefficients"
        ):
            raise ValueError(
                "get_prevalence_coefficients requires latent_factor_prior="
                "'logistic_normal' with a learnable prior (update_prior=True)."
            )
        W, b = self.prior.get_prevalence_coefficients()
        W = W.clone()
        if fold_bias:
            if "Intercept" in self.prevalence_colnames:
                W[:, self.prevalence_colnames.index("Intercept")] += b
            else:
                warnings.warn(
                    "No 'Intercept' column among prevalence covariates; "
                    "returning coefficients with the bias unfolded."
                )
        if to_numpy:
            return W.cpu().numpy()
        return W

    def get_latent_factors(
        self,
        dataset,
        to_simplex: bool = True,
        num_workers: Optional[int] = None,
        to_numpy: bool = True,
        single_modality: Optional[str] = None,
        num_samples: int = 1,
        return_std: bool = False,
        return_samples: bool = False,
    ):
        """
        Get the latent factor distribution for each document in the corpus.

        For topic models (with Dirichlet/Logistic Normal priors), returns document-topic
        distributions on the simplex. For ideal point models (with Gaussian priors),
        returns unconstrained real-valued vectors.

        Args:
            dataset: A Corpus object containing the documents
            to_simplex: Whether to map the latent factors to the simplex.
                       For topic models, this returns the K-dimensional topic distribution.
                       If False, returns the raw latent, which for logistic-normal
                       models is the (K-1)-dimensional contrast coordinates
                       (use model.latent_to_theta to map them to the simplex).
                       For ideal points, this parameter is typically False.
            num_workers: Number of workers for the data loaders. If None, uses self.num_workers.
            to_numpy: Whether to return as a numpy array (True) or torch tensor (False).
            single_modality: If set, uses only this modality (e.g., "default_bow").
                           Useful for analyzing modality-specific representations.
            num_samples: Number of samples from the VAE encoder (only used for ae_type="vae").
                        Results are averaged across samples unless return_samples=True.
            return_std: Whether to return standard errors across samples (only for VAE with num_samples > 1).
            return_samples: Whether to return all samples instead of mean. If True, returns
                          [batch_size, num_samples, latent_dim] tensor/array.

        Returns:
            If return_samples=True:
                Samples array [N, num_samples, latent_dim]
            If return_std=True:
                Tuple of (means [N, latent_dim], stds [N, latent_dim])
            Otherwise:
                Latent factors [N, latent_dim]

        Examples:
            >>> # Get document-topic distributions for a topic model
            >>> theta = model.get_latent_factors(corpus, to_simplex=True)
            >>>
            >>> # Get ideal points for an ideal point model
            >>> ideal_points = model.get_latent_factors(corpus, to_simplex=False)
            >>>
            >>> # Get representation from only text modality
            >>> text_theta = model.get_latent_factors(corpus, single_modality="default_bow")
        """
        """
        Get the topic distribution of each document in the corpus.

        Args:
            dataset: a Corpus object
            to_simplex: whether to map the topic distribution to the simplex. If False, returns latent logits.
            num_workers: number of workers for the data loaders.
            to_numpy: whether to return as a numpy array.
            single_modality: if set, uses only this modality (e.g., "default_bow")
            num_samples: number of samples from the VAE encoder (only used for VAE).
            return_std: whether to return standard errors across samples.
            return_samples: whether to return all samples instead of mean (and std). If True, returns [B, num_samples, D] tensor.
        """
        if num_workers is None:
            num_workers = self.num_workers

        self.encoder.eval()
        final_thetas = []
        final_stds = [] if return_std else None

        with torch.no_grad():
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            for data in data_loader:
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.to(self.device)

                prevalence_covariates = data.get("M_prevalence_covariates", None)

                # Prepare modality inputs
                if single_modality is not None:
                    # Single modality path - create input dict with only one modality
                    if single_modality not in self.encoder.encoders:
                        raise ValueError(
                            f"Modality '{single_modality}' not found in encoders"
                        )

                    mod, view = parse_modality_view(single_modality)
                    view_data = data["modalities"][mod][view]
                    view_type = dataset.processed_modalities[mod][view]["type"]

                    if view_type == "image":
                        x = view_data.to(self.device)
                        modality_inputs = {
                            single_modality: x
                        }  # Don't concatenate covariates for images
                    elif view_type in {"bow", "embedding"}:
                        x = view_data.to(self.device)
                    elif view_type == "vote":
                        x = view_data["matrix"].to(self.device)
                    elif view_type == "discrete_choice":
                        question_tensors = [
                            view_data[q].to(self.device)
                            for q in view_data
                            if q != "type"
                        ]
                        x = torch.cat(question_tensors, dim=-1)
                    else:
                        raise ValueError(f"Unsupported view type: {view_type}")

                    # For non-image modalities, concatenate covariates
                    if view_type != "image":
                        if prevalence_covariates is not None:
                            x = torch.cat([x, prevalence_covariates], dim=1)
                        x = self._append_encoder_labels(x)

                        modality_inputs = {single_modality: x}
                else:
                    # Multimodal path - prepare all modality inputs
                    modality_inputs = {}
                    for key in self.encoder.encoders.keys():
                        mod, view = parse_modality_view(key)
                        view_data = data["modalities"][mod][view]
                        view_type = dataset.processed_modalities[mod][view]["type"]

                        if view_type == "image":
                            # Images are handled separately - don't concatenate covariates
                            x = view_data.to(self.device)
                            modality_inputs[key] = x
                        elif view_type in {"bow", "embedding"}:
                            x = view_data.to(self.device)
                        elif view_type == "vote":
                            x = view_data["matrix"].to(self.device)
                        elif view_type == "discrete_choice":
                            question_tensors = [
                                view_data[q].to(self.device)
                                for q in view_data
                                if q != "type"
                            ]
                            x = torch.cat(question_tensors, dim=-1)
                        else:
                            raise ValueError(f"Unsupported view type: {view_type}")

                        # For non-image modalities, concatenate covariates
                        if view_type != "image":
                            if prevalence_covariates is not None:
                                x = torch.cat([x, prevalence_covariates], dim=1)
                            x = self._append_encoder_labels(x)

                            modality_inputs[key] = x

                # Use the MultiModalEncoder.forward() method with single_modality parameter
                if self.ae_type == "vae":
                    thetas = []
                    for _ in range(num_samples):
                        theta_q, z, _ = self.encoder(
                            modality_inputs,
                            single_modality=single_modality,
                            prevalence_covariates=prevalence_covariates,
                        )
                        thetas.append(theta_q if to_simplex else z)

                    samples = torch.stack(thetas, dim=1)  # [B, num_samples, D]

                    if return_samples:
                        # Return all samples instead of mean/std
                        theta_q = samples
                    else:
                        theta_q = samples.mean(dim=1)
                        if return_std:
                            theta_std = samples.std(dim=1)
                else:
                    theta_q, z, _ = self.encoder(
                        modality_inputs,
                        single_modality=single_modality,
                        prevalence_covariates=prevalence_covariates,
                    )
                    theta_q = theta_q if to_simplex else z
                    if return_samples:
                        # For non-VAE, expand to have num_samples dimension
                        theta_q = theta_q.unsqueeze(1).expand(-1, num_samples, -1)
                    elif return_std:
                        theta_std = torch.zeros_like(theta_q)

                final_thetas.append(theta_q)
                if return_std and not return_samples:
                    final_stds.append(theta_std)

        if to_numpy:
            final_thetas = [t.cpu().numpy() for t in final_thetas]
            if return_samples:
                # For samples, concatenate along batch dimension to get [total_batch, num_samples, D]
                final_thetas = np.concatenate(final_thetas, axis=0)
            else:
                final_thetas = np.concatenate(final_thetas, axis=0)
            if return_std and not return_samples:
                final_stds = [s.cpu().numpy() for s in final_stds]
                final_stds = np.concatenate(final_stds, axis=0)
        else:
            if return_samples:
                # For samples, concatenate along batch dimension to get [total_batch, num_samples, D]
                final_thetas = torch.cat(final_thetas, dim=0)
            else:
                final_thetas = torch.cat(final_thetas, dim=0)
            if return_std and not return_samples:
                final_stds = torch.cat(final_stds, dim=0)

        if return_samples:
            return final_thetas
        else:
            return (final_thetas, final_stds) if return_std else final_thetas

    def get_predictions(
        self,
        dataset,
        to_simplex=True,
        num_workers=None,
        to_numpy=True,
        num_samples: int = 1,
    ):
        """
        Predict labels for documents based on learned latent representations.

        Uses the predictor network to generate predictions from document latent factors.
        For binary classification, applies sigmoid. For multi-class classification,
        applies softmax. For regression, returns raw values.

        Args:
            dataset: A Corpus object containing the documents to predict
            to_simplex: Whether to map latent factors to simplex before prediction.
                       Should match the model's latent_factor_prior (True for topic models,
                       False for ideal point models).
            num_workers: Number of workers for data loaders. If None, uses self.num_workers.
            to_numpy: Whether to return as numpy array (True) or torch tensor (False).
            num_samples: Number of samples for VAE (only used if ae_type="vae").
                        Predictions are averaged across samples.

        Returns:
            Dict mapping label names to predictions:
            - regression: array of shape [N, 1]
            - binary: array of shape [N, 1] with probabilities (after sigmoid)
            - multiclass: array of shape [N, num_classes] with probabilities (after softmax)

        Raises:
            AttributeError: If model was not trained with labels (predictor is None)

        Examples:
            >>> predictions = model.get_predictions(test_corpus)
            >>> # Returns: {"sentiment": array, "category": array, "is_spam": array}
            >>>
            >>> # Binary classification
            >>> spam_probs = predictions["is_spam"]
            >>> spam_predictions = (spam_probs > 0.5).astype(int)
            >>>
            >>> # Multi-class classification
            >>> category_probs = predictions["category"]
            >>> category_predictions = category_probs.argmax(axis=1)
        """
        if not to_simplex and self.latent_factor_prior in {
            "logistic_normal",
            "dirichlet",
        }:
            # The predictor is trained on theta (n_factors dims); the raw latent has
            # n_latent dims (K-1 for logistic_normal), so feeding it is both a shape
            # mismatch and semantically wrong for simplex models.
            raise ValueError(
                "get_predictions(to_simplex=False) is not supported for simplex priors: "
                "the predictor is trained on theta (n_factors dims), not the raw latent."
            )
        if num_workers is None:
            num_workers = self.num_workers

        self.encoder.eval()
        self.predictor.eval()

        with torch.no_grad():
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            # Initialize dict to collect predictions per label
            final_predictions = {
                label_name: [] for label_name in self.labels_info.keys()
            }

            for data in data_loader:
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.to(self.device)

                prevalence_covariates = data.get("M_prevalence_covariates", None)
                prediction_covariates = data.get("M_prediction", None)

                # Prepare encoder input
                modality_inputs = {}
                for key in self.encoder.encoders.keys():
                    mod, view = parse_modality_view(key)
                    view_type = dataset.processed_modalities[mod][view]["type"]
                    view_data = data["modalities"][mod][view]

                    if view_type == "image":
                        # Images are handled separately - don't concatenate covariates
                        x = view_data.to(self.device)
                        modality_inputs[key] = x
                    elif view_type in {"bow", "embedding"}:
                        x = view_data.to(self.device)
                    elif view_type == "vote":
                        x = view_data["matrix"].to(self.device)
                    elif view_type == "discrete_choice":
                        question_tensors = [
                            view_data[q].to(self.device)
                            for q in view_data
                            if q != "type"
                        ]
                        x = torch.cat(question_tensors, dim=-1)
                    else:
                        raise ValueError(f"Unsupported view type: {view_type}")

                    # For non-image modalities, concatenate covariates
                    if view_type != "image":
                        if prevalence_covariates is not None:
                            x = torch.cat([x, prevalence_covariates], dim=1)
                        x = self._append_encoder_labels(x)

                        modality_inputs[key] = x

                if self.ae_type == "vae":
                    thetas = []
                    for _ in range(num_samples):
                        theta_q, z, _ = self.encoder(
                            modality_inputs, prevalence_covariates=prevalence_covariates
                        )
                        theta_q = theta_q if to_simplex else z
                        thetas.append(theta_q)
                    features = torch.stack(thetas, dim=1).mean(dim=1)
                else:
                    theta_q, z, _ = self.encoder(
                        modality_inputs, prevalence_covariates=prevalence_covariates
                    )
                    features = theta_q if to_simplex else z

                # Get predictions dict from multi-label predictor
                batch_predictions = self.predictor(features, prediction_covariates)

                # Apply appropriate activation and collect predictions
                for label_name, pred in batch_predictions.items():
                    label_type = self.labels_info[label_name]["type"]
                    if label_type == "binary":
                        pred = torch.sigmoid(pred)
                    elif label_type == "multiclass":
                        pred = torch.softmax(pred, dim=1)
                    # regression: no activation needed
                    final_predictions[label_name].append(pred)

            # Concatenate batches for each label
            for label_name in final_predictions.keys():
                if to_numpy:
                    final_predictions[label_name] = np.concatenate(
                        [p.cpu().numpy() for p in final_predictions[label_name]], axis=0
                    )
                else:
                    final_predictions[label_name] = torch.cat(
                        final_predictions[label_name], dim=0
                    )

        return final_predictions

    def get_modality_weights(
        self, dataset, num_workers: Optional[int] = None, to_numpy: bool = True
    ):
        """
        Returns modality weights per observation for multimodal models.

        The weights indicate how much each modality contributes to the final latent
        representation. The interpretation depends on the fusion method:

        - **moe_gating**: Softmax weights from the learned gating network
        - **poe (uncorrected)**: Normalized total precisions Λ_m (inverse variances)
        - **corrected_poe**: Normalized precision increments Δ_m = Λ_m - Λ_0
        - **moe_average**: Equal weights (1/M) for all modalities
        - **moe_learned**: Fixed learned mixture weights (same across observations)

        Mathematical formulations:
        - Gating: w_m = softmax(g(μ_1, σ_1, ..., μ_M, σ_M))
        - PoE: w_m ∝ Λ_m (precision of modality m's posterior)
        - Corrected PoE: w_m ∝ Δ_m (precision increment from prior to posterior)

        Args:
            dataset: A Corpus object containing the documents
            num_workers: Number of workers for data loaders. If None, uses self.num_workers.
            to_numpy: Whether to return as numpy array (True) or torch tensor (False).

        Returns:
            Modality weights [N, M] where N is number of documents and M is number of modalities.
            Each row sums to 1 (weights are normalized).

        Examples:
            >>> # Get modality importance for each document
            >>> weights = model.get_modality_weights(corpus)
            >>> print(f"Text weight: {weights[:, 0].mean():.3f}")
            >>> print(f"Image weight: {weights[:, 1].mean():.3f}")
        """
        if num_workers is None:
            num_workers = self.num_workers

        # A scalar precision-based modality weight is not defined for a Gaussian
        # mixture: MoE concatenates components, while corrected PoE assigns weights
        # to Cartesian component tuples. Report equal descriptive modality weights.
        if self.vi_type == "mixture_of_gaussians":
            N = len(dataset)
            M = len(self.encoder.encoders)
            w = np.full((N, M), 1.0 / M, dtype=np.float32)
            return w if to_numpy else torch.tensor(w, device=self.device)

        self.encoder.eval()
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        weights_list = []
        modality_names = list(self.encoder.encoders.keys())

        with torch.no_grad():
            for data in data_loader:
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.to(self.device)

                prevalence_covariates = data.get("M_prevalence_covariates", None)

                modality_inputs = {}
                mu_logvars = []
                zs = []

                for name in modality_names:
                    mod, view = parse_modality_view(name)
                    view_data = data["modalities"][mod][view]
                    view_type = dataset.processed_modalities[mod][view]["type"]

                    if view_type == "image":
                        x = view_data.to(self.device)
                        modality_inputs[name] = (
                            x  # Don't concatenate covariates for images
                        )
                    elif view_type in {"bow", "embedding"}:
                        x = view_data.to(self.device)
                    elif view_type == "vote":
                        x = view_data["matrix"].to(self.device)
                    elif view_type == "discrete_choice":
                        x = torch.cat(
                            [
                                view_data[q].to(self.device)
                                for q in view_data
                                if q != "type"
                            ],
                            dim=-1,
                        )
                    else:
                        raise ValueError(f"Unsupported view type: {view_type}")

                    # For non-image modalities, concatenate covariates
                    if view_type != "image":
                        if prevalence_covariates is not None:
                            x = torch.cat((x, prevalence_covariates), dim=1)
                        x = self._append_encoder_labels(x)

                        modality_inputs[name] = x

                    # Encode each modality separately for weight computation
                    encoder = self.encoder.encoders[name]
                    if isinstance(encoder, ImageEncoder):
                        z = encoder(
                            modality_inputs[name],
                            prevalence_covariates=prevalence_covariates,
                        )
                    else:
                        z = encoder(modality_inputs[name])

                    if self.ae_type == "vae":
                        if self.vi_type == "mean_field":
                            mu, logvar = torch.chunk(z, 2, dim=1)
                            mu_logvars.append((mu, logvar))
                            zs.append(mu)
                        elif self.vi_type == "full_rank":
                            D = self.n_latent
                            mu = z[:, :D]
                            L_flat = z[:, D:]
                            B = mu.size(0)
                            tril_indices = torch.tril_indices(D, D, device=z.device)
                            L = torch.zeros(B, D, D, device=z.device)
                            L[:, tril_indices[0], tril_indices[1]] = L_flat
                            # Stabilize diagonal
                            diag_idx = torch.arange(D, device=z.device)
                            L[:, diag_idx, diag_idx] = (
                                F.softplus(L[:, diag_idx, diag_idx]) + 1e-4
                            )
                            mu_logvars.append((mu, L))
                            zs.append(mu)
                        elif self.vi_type == "iaf":
                            mu, logvar = torch.chunk(z, 2, dim=1)
                            # Sample from base distribution
                            std = torch.exp(0.5 * logvar)
                            eps = torch.randn_like(std)
                            z0 = mu + eps * std
                            # Apply shared flow
                            zk, log_det_j = self.encoder.shared_flow(z0)
                            mu_logvars.append((mu, logvar, zk, log_det_j))
                            zs.append(zk)
                        else:
                            raise ValueError(f"Unsupported vi_type: {self.vi_type}")
                    else:
                        zs.append(z)

                    modality_inputs[name] = x

                B = zs[0].size(0)
                M = len(zs)

                if self.fusion == "moe_gating":
                    if self.ae_type == "vae":
                        if self.vi_type == "mean_field":
                            gate_input = torch.cat(
                                [
                                    torch.cat((mu, logvar), dim=1)
                                    for mu, logvar in mu_logvars
                                    if len(mu_logvars[0]) == 2
                                ],
                                dim=1,
                            )
                        elif self.vi_type == "full_rank":
                            gate_input = torch.cat(
                                [
                                    torch.cat((mu, L.view(mu.size(0), -1)), dim=1)
                                    for mu, L in mu_logvars
                                    if len(mu_logvars[0]) == 2
                                ],
                                dim=1,
                            )
                        elif self.vi_type == "iaf":
                            gate_input = torch.cat(
                                [
                                    torch.cat((mu, logvar), dim=1)
                                    for mu, logvar, _, _ in mu_logvars
                                ],
                                dim=1,
                            )
                    else:
                        gate_input = torch.cat(zs, dim=1)

                    weights = self.encoder.gate_net(gate_input)

                elif self.fusion in ["poe", "corrected_poe"]:
                    if self.ae_type == "vae":
                        if self.vi_type == "mean_field":
                            if self.fusion == "corrected_poe":
                                # For corrected PoE: weights based on precision INCREMENTS Δ_m
                                # Get prior precision Λ_0
                                has_full_cov = hasattr(
                                    self.prior, "sigma"
                                ) and not isinstance(
                                    self.prior,
                                    (
                                        FixedGaussianPrior,
                                        FixedLogisticNormalPrior,
                                        FixedDirichletPrior,
                                    ),
                                )
                                if has_full_cov:
                                    mu_p, Sigma_p = self.prior.get_prior_params(
                                        prevalence_covariates, return_full_cov=True
                                    )
                                    Sigma_p_batch = Sigma_p.unsqueeze(0).expand(
                                        B, -1, -1
                                    )
                                    Lambda_0_diag = 1.0 / (
                                        torch.diagonal(Sigma_p_batch, dim1=1, dim2=2)
                                        + 1e-8
                                    )
                                else:
                                    mu_p, logvar_p = self.prior.get_prior_params(
                                        prevalence_covariates
                                    )
                                    Lambda_0_diag = torch.exp(-logvar_p)

                                # Precision increments: Δ_m = Λ_m - Λ_0
                                deltas = []
                                for _, logvar in mu_logvars:
                                    if len(mu_logvars[0]) == 2:
                                        Lambda_m = torch.exp(-logvar)  # Total precision
                                        Delta_m = (
                                            Lambda_m - Lambda_0_diag
                                        )  # Precision increment
                                        Delta_m = torch.clamp(
                                            Delta_m, min=0.0
                                        )  # Ensure non-negative
                                        deltas.append(Delta_m)

                                if deltas:
                                    delta_stack = torch.stack(
                                        deltas, dim=1
                                    )  # (B, M, D)
                                    # Normalize across modalities
                                    delta_sum = delta_stack.sum(
                                        dim=1, keepdim=True
                                    )  # (B, 1, D)
                                    delta_normalized = delta_stack / (
                                        delta_sum + 1e-8
                                    )  # (B, M, D)
                                    # Average across dimensions
                                    weights = delta_normalized.mean(dim=2)  # (B, M)
                                else:
                                    weights = torch.full(
                                        (B, M), 1.0 / M, device=self.device
                                    )
                            else:
                                # Uncorrected PoE: weights based on total precisions Λ_m
                                precisions = [
                                    1.0 / torch.exp(logvar)
                                    for _, logvar in mu_logvars
                                    if len(mu_logvars[0]) == 2
                                ]
                                if precisions:
                                    precision_stack = torch.stack(
                                        precisions, dim=1
                                    )  # (B, M, D)
                                    # Normalize across modalities for each dimension
                                    precision_normalized = (
                                        precision_stack
                                        / precision_stack.sum(dim=1, keepdim=True)
                                    )  # (B, M, D)
                                    # Average across dimensions to get overall modality weight
                                    weights = precision_normalized.mean(dim=2)  # (B, M)
                                else:
                                    weights = torch.full(
                                        (B, M), 1.0 / M, device=self.device
                                    )

                        elif self.vi_type == "full_rank":
                            precisions = []
                            D = self.n_latent
                            I = torch.eye(D, device=self.device).unsqueeze(
                                0
                            )  # (1, D, D)
                            for _, L in mu_logvars:
                                if len(mu_logvars[0]) == 2:  # Not IAF flow
                                    B = L.size(0)
                                    Sigma_q = torch.bmm(
                                        L, L.transpose(1, 2)
                                    )  # (B, D, D)
                                    Sigma_inv = torch.linalg.inv(
                                        Sigma_q + 1e-4 * I.expand(B, -1, -1)
                                    )
                                    trace_prec = torch.diagonal(
                                        Sigma_inv, dim1=1, dim2=2
                                    ).sum(
                                        dim=1
                                    )  # (B,)
                                    precisions.append(trace_prec)
                            if precisions:
                                weights = torch.stack(precisions, dim=1)  # (B, M)
                                weights = weights / weights.sum(dim=1, keepdim=True)
                            else:
                                weights = torch.full(
                                    (B, M), 1.0 / M, device=self.device
                                )

                        elif self.vi_type == "iaf":
                            # For flows, use base distribution variances
                            precisions = [
                                1.0 / torch.exp(logvar)
                                for mu, logvar, _, _ in mu_logvars
                            ]
                            precision_stack = torch.stack(
                                precisions, dim=1
                            )  # (B, M, D)
                            weights = precision_stack.sum(dim=2)  # (B, M)
                            weights = weights / weights.sum(dim=1, keepdim=True)

                        else:
                            raise ValueError(f"Unsupported vi_type: {self.vi_type}")
                    else:
                        # For WAE, use equal weights
                        weights = torch.full((B, M), 1.0 / M, device=self.device)

                else:  # moe_average
                    weights = torch.full((B, M), 1.0 / M, device=self.device)

                weights_list.append(weights)

        weights_all = torch.cat(weights_list, dim=0)  # (N, M)
        return weights_all.cpu().numpy() if to_numpy else weights_all

    def get_mutual_information(
        self, dataset, num_workers: Optional[int] = None, to_numpy: bool = True
    ):
        """
        Computes pointwise mutual information I_q^{(i)}(X_m; Z) per observation and modality.

        For each observation i and modality m, computes:
            I_q^{(i)}(X_m; Z) = KL(q(z|x_m^{(i)}) || p(z))

        This measures how much information modality m provides about the latent variable
        for observation i, by computing how far the modality-specific posterior moves
        from the prior.

        Higher values indicate that the modality provides more information about the latent
        structure. A value of 0 would mean the modality provides no information (posterior
        equals prior).

        Mathematical definitions by variational inference type:
        - **Mean-field**: KL(N(μ, diag(σ²)) || N(μ_p, Σ_p))
        - **Full-rank**: KL(N(μ, Σ) || N(μ_p, Σ_p))
        - **IAF**: KL computed via the base distribution (flow transformations cancel in expectation)

        For Gaussian posteriors with Gaussian priors:
            KL = 0.5 * (tr(Σ_p^{-1} Σ_q) + (μ_q - μ_p)^T Σ_p^{-1} (μ_q - μ_p) - d - log(det(Σ_q)/det(Σ_p)))

        Args:
            dataset: A Corpus object containing the documents
            num_workers: Number of workers for data loaders. If None, uses self.num_workers.
            to_numpy: Whether to return as numpy array (True) or torch tensor (False).

        Returns:
            Pointwise mutual information matrix [N, M] where N is number of observations
            and M is number of modalities. All values are non-negative.

        Raises:
            ValueError: If model is not a VAE (ae_type != "vae")

        Examples:
            >>> # Analyze which modality is more informative
            >>> mi = model.get_mutual_information(corpus)
            >>> print(f"Average MI - Text: {mi[:, 0].mean():.3f}, Images: {mi[:, 1].mean():.3f}")
            >>>
            >>> # Find observations where images are most informative
            >>> image_informative = mi[:, 1] > mi[:, 0]
        """
        if num_workers is None:
            num_workers = self.num_workers

        if self.ae_type != "vae":
            raise ValueError(
                "Mutual information computation is only supported for VAE models. "
                f"Current ae_type is '{self.ae_type}'."
            )

        if self.vi_type == "mixture_of_gaussians":
            raise NotImplementedError(
                "get_mutual_information is not implemented for vi_type="
                "'mixture_of_gaussians' (the per-modality KL has no closed form). "
                "Use estimate_marginal_log_likelihood for held-out-likelihood diagnostics."
            )

        self.encoder.eval()
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        mi_list = []
        modality_names = list(self.encoder.encoders.keys())
        M = len(modality_names)

        with torch.no_grad():
            for data in data_loader:
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.to(self.device)

                prevalence_covariates = data.get("M_prevalence_covariates", None)
                labels = data.get("M_labels", None)

                # Prepare modality inputs and get individual posteriors
                modality_inputs = {}
                modality_posteriors = []

                for name in modality_names:
                    mod, view = parse_modality_view(name)
                    view_data = data["modalities"][mod][view]
                    view_type = dataset.processed_modalities[mod][view]["type"]

                    if view_type == "image":
                        x = view_data.to(self.device)
                        modality_inputs[name] = x
                    elif view_type in {"bow", "embedding"}:
                        x = view_data.to(self.device)
                    elif view_type == "vote":
                        x = view_data["matrix"].to(self.device)
                    elif view_type == "discrete_choice":
                        x = torch.cat(
                            [
                                view_data[q].to(self.device)
                                for q in view_data
                                if q != "type"
                            ],
                            dim=-1,
                        )
                    else:
                        raise ValueError(f"Unsupported view type: {view_type}")

                    # For non-image modalities, concatenate covariates
                    if view_type != "image":
                        if prevalence_covariates is not None:
                            x = torch.cat((x, prevalence_covariates), dim=1)
                        x = self._append_encoder_labels(x)
                        modality_inputs[name] = x

                    # Encode each modality separately to get individual posterior parameters
                    encoder = self.encoder.encoders[name]
                    if isinstance(encoder, ImageEncoder):
                        z = encoder(
                            modality_inputs[name],
                            prevalence_covariates=prevalence_covariates,
                        )
                    else:
                        z = encoder(modality_inputs[name])

                    # Extract posterior parameters based on vi_type
                    if self.vi_type == "mean_field":
                        mu, logvar = torch.chunk(z, 2, dim=1)
                        modality_posteriors.append((mu, logvar))
                    elif self.vi_type == "full_rank":
                        D = self.n_latent
                        mu = z[:, :D]
                        L_flat = z[:, D:]
                        B = mu.size(0)
                        tril_indices = torch.tril_indices(D, D, device=z.device)
                        L = torch.zeros(B, D, D, device=z.device)
                        L[:, tril_indices[0], tril_indices[1]] = L_flat
                        diag_idx = torch.arange(D, device=z.device)
                        L[:, diag_idx, diag_idx] = (
                            F.softplus(L[:, diag_idx, diag_idx]) + 1e-4
                        )
                        modality_posteriors.append((mu, L))
                    elif self.vi_type == "iaf":
                        mu, logvar = torch.chunk(z, 2, dim=1)
                        modality_posteriors.append((mu, logvar, name))
                    else:
                        raise ValueError(f"Unsupported vi_type: {self.vi_type}")

                B = modality_posteriors[0][0].size(0)
                mi_batch = torch.zeros(B, M, device=self.device)

                # Get prior parameters (per observation if prevalence covariates exist)
                # Check if prior has full covariance
                has_full_cov = hasattr(self.prior, "sigma") and not isinstance(
                    self.prior,
                    (FixedGaussianPrior, FixedLogisticNormalPrior, FixedDirichletPrior),
                )

                if has_full_cov:
                    mu_p, Sigma_p = self.prior.get_prior_params(
                        prevalence_covariates, return_full_cov=True
                    )
                    # Expand for batch if needed
                    if Sigma_p.dim() == 2:  # [D, D]
                        Sigma_p = Sigma_p.unsqueeze(0).expand(B, -1, -1)  # [B, D, D]
                    Sigma_p_inv = torch.inverse(Sigma_p)
                    logdet_p = torch.logdet(Sigma_p)  # [B] or scalar
                    if logdet_p.dim() == 0:
                        logdet_p = logdet_p.unsqueeze(0).expand(B)
                else:
                    mu_p, logvar_p = self.prior.get_prior_params(prevalence_covariates)
                    var_p = torch.exp(logvar_p)

                # Compute KL divergence for each modality: KL(q(z|x_m) || p(z))
                for m_idx, posterior in enumerate(modality_posteriors):
                    if self.vi_type == "mean_field":
                        mu_q, logvar_q = posterior
                        var_q = torch.exp(logvar_q)

                        if has_full_cov:
                            # KL(N(μ_q, diag(σ_q²)) || N(μ_p, Σ_p))
                            # = 0.5 * (tr(Σ_p^{-1} Σ_q) + (μ_q - μ_p)^T Σ_p^{-1} (μ_q - μ_p) - d - log(det(Σ_q)/det(Σ_p)))

                            # Trace term: tr(Σ_p^{-1} diag(σ_q²))
                            trace_term = torch.sum(
                                torch.diagonal(Sigma_p_inv, dim1=1, dim2=2) * var_q,
                                dim=1,
                            )  # [B]

                            # Quadratic term
                            diff = (mu_q - mu_p).unsqueeze(2)  # [B, D, 1]
                            quad_term = (
                                torch.bmm(
                                    torch.bmm(diff.transpose(1, 2), Sigma_p_inv), diff
                                )
                                .squeeze(-1)
                                .squeeze(-1)
                            )  # [B]

                            # Log determinant term
                            logdet_q = torch.sum(logvar_q, dim=1)  # [B]

                            kl = 0.5 * (
                                trace_term
                                + quad_term
                                - self.n_latent
                                - logdet_q
                                + logdet_p
                            )
                        else:
                            # KL(N(μ_q, diag(σ_q²)) || N(μ_p, diag(σ_p²)))
                            # = 0.5 * Σ_d (σ_q²/σ_p² + (μ_q - μ_p)²/σ_p² - 1 - log(σ_q²/σ_p²))
                            kl = 0.5 * torch.sum(
                                var_q / var_p
                                + (mu_q - mu_p).pow(2) / var_p
                                - 1
                                + logvar_p
                                - logvar_q,
                                dim=1,
                            )

                    elif self.vi_type == "full_rank":
                        mu_q, L_q = posterior
                        Sigma_q = torch.bmm(L_q, L_q.transpose(1, 2))  # [B, D, D]
                        logdet_q = 2 * torch.sum(
                            torch.log(torch.diagonal(L_q, dim1=1, dim2=2) + 1e-6), dim=1
                        )  # [B]

                        if has_full_cov:
                            # KL(N(μ_q, Σ_q) || N(μ_p, Σ_p))
                            # = 0.5 * (tr(Σ_p^{-1} Σ_q) + (μ_q - μ_p)^T Σ_p^{-1} (μ_q - μ_p) - d - log(det(Σ_q)/det(Σ_p)))

                            # Trace term
                            trace_term = torch.sum(
                                torch.diagonal(
                                    torch.bmm(Sigma_p_inv, Sigma_q), dim1=1, dim2=2
                                ),
                                dim=1,
                            )  # [B]

                            # Quadratic term
                            diff = (mu_q - mu_p).unsqueeze(2)  # [B, D, 1]
                            quad_term = (
                                torch.bmm(
                                    torch.bmm(diff.transpose(1, 2), Sigma_p_inv), diff
                                )
                                .squeeze(-1)
                                .squeeze(-1)
                            )  # [B]

                            kl = 0.5 * (
                                trace_term
                                + quad_term
                                - self.n_latent
                                - logdet_q
                                + logdet_p
                            )
                        else:
                            # KL(N(μ_q, Σ_q) || N(μ_p, diag(σ_p²)))
                            # = 0.5 * (tr(diag(1/σ_p²) Σ_q) + (μ_q - μ_p)^T diag(1/σ_p²) (μ_q - μ_p) - d - log(det(Σ_q)) + Σ log(σ_p²))

                            # Trace term: tr(diag(1/σ_p²) Σ_q)
                            diag_Sigma_q = torch.diagonal(
                                Sigma_q, dim1=1, dim2=2
                            )  # [B, D]
                            trace_term = torch.sum(diag_Sigma_q / var_p, dim=1)  # [B]

                            # Quadratic term
                            quad_term = torch.sum(
                                (mu_q - mu_p).pow(2) / var_p, dim=1
                            )  # [B]

                            # Log determinant terms
                            logdet_p_sum = torch.sum(logvar_p, dim=1)  # [B]

                            kl = 0.5 * (
                                trace_term
                                + quad_term
                                - self.n_latent
                                - logdet_q
                                + logdet_p_sum
                            )

                    elif self.vi_type == "iaf":
                        # For IAF, the KL is computed on the base distribution
                        # The flow transformations cancel out in expectation
                        mu_q, logvar_q, name_m = posterior
                        var_q = torch.exp(logvar_q)

                        if has_full_cov:
                            # KL using base distribution parameters
                            trace_term = torch.sum(
                                torch.diagonal(Sigma_p_inv, dim1=1, dim2=2) * var_q,
                                dim=1,
                            )
                            diff = (mu_q - mu_p).unsqueeze(2)
                            quad_term = (
                                torch.bmm(
                                    torch.bmm(diff.transpose(1, 2), Sigma_p_inv), diff
                                )
                                .squeeze(-1)
                                .squeeze(-1)
                            )
                            logdet_q = torch.sum(logvar_q, dim=1)
                            kl = 0.5 * (
                                trace_term
                                + quad_term
                                - self.n_latent
                                - logdet_q
                                + logdet_p
                            )
                        else:
                            # Diagonal prior
                            kl = 0.5 * torch.sum(
                                var_q / var_p
                                + (mu_q - mu_p).pow(2) / var_p
                                - 1
                                + logvar_p
                                - logvar_q,
                                dim=1,
                            )

                    else:
                        raise ValueError(f"Unsupported vi_type: {self.vi_type}")

                    mi_batch[:, m_idx] = kl

                mi_list.append(mi_batch)

        mi_all = torch.cat(mi_list, dim=0)  # (N, M)
        return mi_all.cpu().numpy() if to_numpy else mi_all

    def _posterior_loglik(self, info, z):
        """log q(z|x) for one drawn sample z, dispatched by variational family."""
        from torch.distributions import Normal, MultivariateNormal

        if self.vi_type == "mean_field":
            mu, logvar = info
            return Normal(mu, torch.exp(0.5 * logvar)).log_prob(z).sum(1)
        elif self.vi_type == "full_rank":
            mu, L = info
            return MultivariateNormal(mu, scale_tril=L).log_prob(z)
        elif self.vi_type == "iaf":
            mu, logvar, z0, zk, log_det_j = info
            log_q0 = Normal(mu, torch.exp(0.5 * logvar)).log_prob(z0).sum(1)
            return log_q0 - log_det_j  # change-of-variables density of the flow output
        elif self.vi_type == "mixture_of_gaussians":
            _, means, covariance_params, pi = info
            logN = self.encoder._mog_component_log_prob(
                z.unsqueeze(1), means, covariance_params
            ).squeeze(
                1
            )  # [B,C]
            return torch.logsumexp(torch.log(pi + 1e-12) + logN, dim=1)  # [B]
        else:
            raise ValueError(f"Unsupported vi_type: {self.vi_type}")

    def _recon_loglik(self, doc_latents, data, dataset, content_covariates):
        """Per-document log p(x|z) summed over modalities (NOT averaged over the batch)."""
        theta_input = (
            torch.cat([doc_latents, content_covariates], dim=1)
            if content_covariates is not None
            else doc_latents
        )
        total = None
        for key, decoder in self.decoders.items():
            mod, view = parse_modality_view(key)
            vt = dataset.processed_modalities[mod][view]["type"]
            md = data["modalities"][mod][view]
            if vt == "image":
                target = md.to(self.device)
                recon = decoder(doc_latents, content_covariates)
                ll = -((recon - target) ** 2).flatten(1).sum(1)
            elif vt == "discrete_choice":
                ll = 0
                for q, qdec in decoder.items():
                    if q == "type":
                        continue
                    x_out = md[q].to(self.device)
                    logits = qdec(theta_input)
                    ll = ll - F.cross_entropy(
                        logits, x_out.argmax(-1), reduction="none"
                    )
            else:
                target = (
                    md.to(self.device)
                    if vt in {"bow", "embedding"}
                    else md["matrix"].to(self.device)
                )
                recon = decoder(theta_input)
                if vt == "bow":
                    ll = (target * F.log_softmax(recon, dim=1)).sum(1)
                elif vt == "embedding":
                    ll = -((recon - target) ** 2).sum(1)
                elif vt == "vote":
                    mask = ~md["mask"].to(self.device)
                    bce = F.binary_cross_entropy_with_logits(
                        recon, target, reduction="none"
                    )
                    ll = -(bce * mask).sum(1)
                else:
                    raise ValueError(f"Unsupported view type: {vt}")
            total = ll if total is None else total + ll
        return total

    def estimate_marginal_log_likelihood(
        self,
        dataset,
        n_samples: int = 50,
        num_workers: Optional[int] = None,
        to_numpy: bool = True,
        reduce: str = "mean",
    ):
        """Importance-weighted (IWAE) estimate of the per-document marginal log-likelihood.

        For S samples z_s ~ q(z|x):
            log p(x) >= IWAE_S = logsumexp_s [ log p(x|z_s) + log p(z_s) - log q(z_s|x) ] - log S
                      >= ELBO  = mean_s   [ log p(x|z_s) + log p(z_s) - log q(z_s|x) ]
        IWAE_S is monotone increasing in S and converges to log p(x); the gap IWAE - ELBO
        and the gap of either to the true log p(x) measure how tightly the variational
        family approximates the marginal likelihood. Supports all four vi_types.

        Args:
            dataset: a Corpus object.
            n_samples: number of importance samples S.
            reduce: "mean" returns scalar corpus averages; "none" returns per-document arrays.

        Returns:
            (iwae, elbo) — scalars if reduce="mean", else arrays of shape [N].
        """
        from torch.distributions import Normal, MultivariateNormal

        if self.ae_type != "vae":
            raise ValueError("estimate_marginal_log_likelihood requires ae_type='vae'.")
        if num_workers is None:
            num_workers = self.num_workers

        self.encoder.eval()
        self.decoders.eval()
        if self.prior is not None:
            self.prior.eval()

        has_full_cov = hasattr(self.prior, "sigma") and not isinstance(
            self.prior,
            (FixedGaussianPrior, FixedLogisticNormalPrior, FixedDirichletPrior),
        )

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )
        iwae_all, elbo_all = [], []

        with torch.no_grad():
            for data in loader:
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(self.device)
                prevalence = data.get("M_prevalence_covariates", None)
                content = data.get("M_content_covariates", None)

                modality_inputs = {}
                for key in self.encoder.encoders.keys():
                    mod, view = parse_modality_view(key)
                    vt = dataset.processed_modalities[mod][view]["type"]
                    vd = data["modalities"][mod][view]
                    if vt == "image":
                        modality_inputs[key] = vd.to(self.device)
                        continue
                    elif vt in {"bow", "embedding"}:
                        x = vd.to(self.device)
                    elif vt == "vote":
                        x = vd["matrix"].to(self.device)
                    elif vt == "discrete_choice":
                        x = torch.cat(
                            [vd[q].to(self.device) for q in vd if q != "type"], dim=-1
                        )
                    else:
                        raise ValueError(f"Unsupported view type: {vt}")
                    if prevalence is not None:
                        x = torch.cat([x, prevalence], dim=1)
                    x = self._append_encoder_labels(x)
                    modality_inputs[key] = x

                B = (
                    prevalence.shape[0]
                    if prevalence is not None
                    else next(iter(modality_inputs.values())).shape[0]
                )
                if has_full_cov:
                    mu_p, Sigma_p = self.prior.get_prior_params(
                        prevalence, return_full_cov=True
                    )
                    prior_dist = MultivariateNormal(
                        mu_p, covariance_matrix=Sigma_p.unsqueeze(0).expand(B, -1, -1)
                    )
                else:
                    mu_p, logvar_p = self.prior.get_prior_params(prevalence)
                    prior_dist = Normal(mu_p, torch.exp(0.5 * logvar_p))

                lw = []
                for _ in range(n_samples):
                    theta, z, info = self.encoder(
                        modality_inputs, prevalence_covariates=prevalence
                    )
                    doc_latents = (
                        theta
                        if self.latent_factor_prior in {"dirichlet", "logistic_normal"}
                        else z
                    )
                    logpx = self._recon_loglik(doc_latents, data, dataset, content)
                    logpz = (
                        prior_dist.log_prob(z)
                        if has_full_cov
                        else prior_dist.log_prob(z).sum(1)
                    )
                    logqz = self._posterior_loglik(info[-1], z)
                    lw.append(logpx + logpz - logqz)

                lw = torch.stack(lw, dim=1)  # [B, S]
                iwae_all.append(torch.logsumexp(lw, dim=1) - np.log(n_samples))
                elbo_all.append(lw.mean(dim=1))

        iwae = torch.cat(iwae_all)
        elbo = torch.cat(elbo_all)
        if reduce == "mean":
            iwae, elbo = iwae.mean(), elbo.mean()
        if to_numpy:
            return (iwae.cpu().numpy(), elbo.cpu().numpy())
        return iwae, elbo

    def generate_samples(
        self,
        n_samples: int = 10,
        content_covariates: Optional[torch.Tensor] = None,
        prevalence_covariates: Optional[torch.Tensor] = None,
        modality_keys: Optional[List[str]] = None,
        temperature: float = 1.0,
        to_numpy: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Generate new samples using the trained decoders.

        Args:
            n_samples: Number of samples to generate
            content_covariates: Content covariates to condition generation [n_samples, content_dim]
                                If None, uses zeros
            prevalence_covariates: Prevalence covariates for prior sampling [n_samples, prevalence_dim]
                                   If None, uses zeros
            modality_keys: List of modality keys to generate. If None, generates all modalities
            temperature: Temperature for sampling (higher = more random, lower = more deterministic)
            to_numpy: Whether to return numpy arrays instead of tensors
            seed: Random seed for reproducible generation

        Returns:
            Dictionary mapping modality keys to generated samples
        """
        if seed is not None:
            torch.manual_seed(seed)

        self.encoder.eval()
        for decoder in self.decoders.values():
            if isinstance(decoder, nn.ModuleDict):
                for sub_decoder in decoder.values():
                    sub_decoder.eval()
            else:
                decoder.eval()
        if self.prior is not None:
            self.prior.eval()

        # Default to all modalities if not specified
        if modality_keys is None:
            modality_keys = list(self.decoders.keys())

        # Prepare covariates
        device = self.device

        if content_covariates is None:
            content_covariates = torch.zeros(
                n_samples, self.content_covariate_size, device=device
            )
        else:
            content_covariates = content_covariates.to(device)
            if content_covariates.size(0) != n_samples:
                raise ValueError(
                    f"content_covariates must have {n_samples} samples, got {content_covariates.size(0)}"
                )

        if prevalence_covariates is None:
            prevalence_covariates = torch.zeros(
                n_samples, self.prevalence_covariate_size, device=device
            )
        else:
            prevalence_covariates = prevalence_covariates.to(device)
            if prevalence_covariates.size(0) != n_samples:
                raise ValueError(
                    f"prevalence_covariates must have {n_samples} samples, got {prevalence_covariates.size(0)}"
                )

        generated_samples = {}

        with torch.no_grad():
            # Sample from prior (or provide latents for plain autoencoder)
            if self.ae_type == "ae":
                # Plain autoencoder - users need to provide latent codes
                raise ValueError(
                    "For plain autoencoders (ae_type='ae'), you need to provide the values for the latent codes. "
                    "Plain autoencoders don't have a learned prior distribution to sample from. "
                    "Please encode some data first to get latent representations, or provide specific latent codes manually."
                )

            elif self.latent_factor_prior in {"dirichlet", "logistic_normal"}:
                # For topic models, sample from prior and use as simplex
                doc_latents = self.prior.sample(
                    N=n_samples, M_prevalence_covariates=prevalence_covariates
                ).to(device)

                # Apply temperature scaling for topic models
                if temperature != 1.0:
                    doc_latents = F.softmax(
                        torch.log(doc_latents + 1e-8) / temperature, dim=1
                    )

            else:
                # For ideal point models, sample from Gaussian prior
                doc_latents = self.prior.sample(
                    N=n_samples, M_prevalence_covariates=prevalence_covariates
                ).to(device)

                # Apply temperature scaling
                if temperature != 1.0:
                    doc_latents = doc_latents * temperature

            # Prepare decoder input
            if self.content_covariate_size > 0:
                decoder_input = torch.cat([doc_latents, content_covariates], dim=1)
            else:
                decoder_input = doc_latents

            # Generate samples for each requested modality
            for key in modality_keys:
                if key not in self.decoders:
                    print(f"Warning: Modality '{key}' not found in decoders, skipping")
                    continue

                decoder = self.decoders[key]

                # Parse modality type for appropriate generation
                mod, view = parse_modality_view(key)

                if isinstance(decoder, ImageDecoder):
                    # Generate images
                    generated_images = decoder(doc_latents, content_covariates)

                    if to_numpy:
                        generated_images = generated_images.cpu().numpy()

                    generated_samples[key] = generated_images

                elif isinstance(decoder, nn.ModuleDict):
                    # Handle discrete choice (multiple sub-decoders)
                    generated_samples[key] = {}
                    for question, sub_decoder in decoder.items():
                        if question == "type":
                            continue
                        logits = sub_decoder(decoder_input)

                        # Apply temperature and sample
                        if temperature != 1.0:
                            logits = logits / temperature

                        probs = F.softmax(logits, dim=1)
                        samples = torch.multinomial(probs, 1).squeeze(1)

                        if to_numpy:
                            samples = samples.cpu().numpy()

                        generated_samples[key][question] = samples

                else:
                    # Handle other modalities (BOW, embedding, vote)
                    logits = decoder(decoder_input)

                    # Determine generation strategy based on modality type
                    if key.endswith("bow"):
                        # For BOW, apply temperature and sample from categorical
                        if temperature != 1.0:
                            logits = logits / temperature

                        probs = F.softmax(logits, dim=1)

                        # Sample word counts (simplified: sample once per document)
                        # For more realistic BOW, you might want to sample multiple words
                        samples = torch.multinomial(probs, 1).squeeze(1)

                        if to_numpy:
                            samples = samples.cpu().numpy()

                        generated_samples[key] = samples

                    elif key.endswith("embedding"):
                        # For embeddings, use the continuous output directly
                        if temperature != 1.0:
                            logits = logits * temperature

                        if to_numpy:
                            logits = logits.cpu().numpy()

                        generated_samples[key] = logits

                    elif key.endswith("vote"):
                        # For voting data, apply sigmoid and sample binary outcomes
                        if temperature != 1.0:
                            logits = logits / temperature

                        probs = torch.sigmoid(logits)
                        samples = torch.bernoulli(probs)

                        if to_numpy:
                            samples = samples.cpu().numpy()

                        generated_samples[key] = samples

                    else:
                        # Default: return raw logits/outputs
                        if temperature != 1.0:
                            logits = logits / temperature

                        if to_numpy:
                            logits = logits.cpu().numpy()

                        generated_samples[key] = logits

        return generated_samples

    def save_model(self, save_name):
        """
        Save the complete model state to disk.

        Saves all model components including encoder, decoders, predictor (if present),
        optimizer, and all hyperparameters. The saved checkpoint can be loaded later
        using load_model() to resume training or perform inference.

        Args:
            save_name: Path where the checkpoint will be saved (e.g., "checkpoints/model.ckpt")

        Saves:
            - Encoder state dict
            - Decoder state dicts (all modalities)
            - Predictor state dict (if labels were used)
            - Optimizer state dict
            - All model hyperparameters and training state

        Examples:
            >>> model.save_model("checkpoints/best_model.ckpt")
            >>> # Later, load with:
            >>> model.load_model("checkpoints/best_model.ckpt")
        """
        encoder_state_dict = self.encoder.state_dict()
        decoders_state_dict = {k: d.state_dict() for k, d in self.decoders.items()}
        predictor_state_dict = self.predictor.state_dict() if self.labels_info else None
        optimizer_state_dict = self.optimizer.state_dict()

        all_vars = vars(self)

        checkpoint = {}
        for key, value in all_vars.items():
            if key not in ["encoder", "decoders", "predictor", "optimizer"]:
                checkpoint[key] = value

        checkpoint["encoder"] = encoder_state_dict
        checkpoint["decoders"] = decoders_state_dict
        if self.labels_info:
            checkpoint["predictor"] = predictor_state_dict
        checkpoint["optimizer"] = optimizer_state_dict

        torch.save(checkpoint, save_name)

    def load_model(self, ckpt):
        """
        Load a saved model checkpoint from disk.

        Restores all model components including encoder, decoders, predictor (if present),
        optimizer, and all hyperparameters from a previously saved checkpoint.

        If the current model instance doesn't have a predictor but the checkpoint does,
        a predictor will be created automatically. Similarly, if the optimizer is missing,
        it will be reconstructed with default parameters.

        Args:
            ckpt: Path to the checkpoint file (e.g., "checkpoints/model.ckpt") or
                 checkpoint dictionary

        Loads:
            - Encoder state dict
            - Decoder state dicts (all modalities)
            - Predictor state dict (if present in checkpoint)
            - Optimizer state dict
            - All model hyperparameters and training state

        Examples:
            >>> model = DeepLatent(train_data, **config)
            >>> model.load_model("checkpoints/best_model.ckpt")
            >>> # Continue training or run inference
        """
        ckpt = torch.load(ckpt, map_location=self.device, weights_only=False)

        # Latent-parameterization guard. Since 0.2.0 logistic-normal latents are
        # (K-1)-dim contrast coordinates; pre-0.2.0 checkpoints (no "n_latent" key)
        # used K dims and cannot be loaded -- fail here with a clear message instead
        # of an opaque state-dict shape mismatch after self.prior was already swapped.
        ckpt_n_latent = ckpt.get("n_latent", ckpt.get("n_factors"))
        if ckpt_n_latent != self.n_latent:
            raise ValueError(
                f"Checkpoint latent dimension mismatch: checkpoint has n_latent="
                f"{ckpt_n_latent} but this model expects n_latent={self.n_latent}. "
                f"Pre-0.2.0 logistic-normal checkpoints use the removed K-dimensional "
                f"parameterization and cannot be loaded; re-train with >=0.2.0 or check "
                f"out the pre-0.2.0 commit to load this checkpoint."
            )

        for key, value in ckpt.items():
            if key not in ["encoder", "decoders", "predictor", "optimizer"]:
                setattr(self, key, value)

        self.encoder.load_state_dict(ckpt["encoder"])

        for key, state_dict in ckpt["decoders"].items():
            self.decoders[key].load_state_dict(state_dict)

        if self.labels_info and "predictor" in ckpt:
            if not hasattr(self, "predictor") or self.predictor is None:
                # Create MultiLabelPredictor from loaded labels_info and predictor_args
                self.predictor = MultiLabelPredictor(
                    input_dim=self.n_factors + self.prediction_covariate_size,
                    labels_info=self.labels_info,
                    predictor_configs=self.predictor_args,
                ).to(self.device)
            self.predictor.load_state_dict(ckpt["predictor"])

        if not hasattr(self, "optimizer"):
            # Create parameter groups with different parameter groups as in __init__
            main_params = list(self.encoder.parameters()) + list(
                self.decoders.parameters()
            )
            predictor_params = []
            if (
                self.labels_info
                and hasattr(self, "predictor")
                and self.predictor is not None
            ):
                predictor_params = list(self.predictor.parameters())

            # Use default optimizer configuration
            default_optim_args = {
                "main": {"lr": 1e-3, "weight_decay": 0.0},
                # Slow prior lr (two-timescale) but NO weight_decay: decay biases the
                # mean_net covariate coefficients toward zero. See __init__ for rationale.
                "prior": {"lr": 1e-4, "weight_decay": 0.0},
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            }

            main_config = default_optim_args["main"]
            prior_config = default_optim_args["prior"]
            predictor_config = default_optim_args.get("predictor")
            global_config = {
                k: v
                for k, v in default_optim_args.items()
                if k not in ["main", "prior", "predictor"]
            }

            # Apply global config to parameter groups
            for config in [main_config, prior_config, predictor_config]:
                if config is not None:
                    for key, default_val in global_config.items():
                        if key not in config:
                            config[key] = default_val

            # Create parameter groups
            param_groups = []

            # Add main parameters (encoder + decoders)
            param_groups.append({"params": main_params, **main_config})

            # Add predictor parameters if they exist (no separate config in default case)
            if predictor_params:
                param_groups[0]["params"].extend(predictor_params)

            # Add prior parameters if it's a learnable prior
            if (
                getattr(self, "update_prior", False)
                and hasattr(self, "prior")
                and self.prior is not None
            ):
                prior_params = list(self.prior.parameters())
                param_groups.append({"params": prior_params, **prior_config})

            self.optimizer = torch.optim.Adam(param_groups)

        self.optimizer.load_state_dict(ckpt["optimizer"])

    def to(self, device):
        """
        Move all model components to the specified device (CPU/GPU).

        This is a convenience method that moves the encoder, all decoders,
        predictor (if present), and prior (if present) to the specified device.
        Updates self.device to track the current device.

        Args:
            device: Target device (e.g., torch.device("cuda"), torch.device("cpu"),
                   or string like "cuda:0", "cpu")

        Returns:
            self: Returns the model instance for method chaining

        Examples:
            >>> model.to("cuda")  # Move to GPU
            >>> model.to("cpu")   # Move back to CPU
            >>> model.to(torch.device("cuda:1"))  # Move to specific GPU
        """
        self.encoder.to(device)
        self.decoders.to(device)
        if self.prior is not None:
            self.prior.to(device)
        if self.labels_info:
            self.predictor.to(device)
        self.device = device


class GTM(DeepLatent):
    """
    Generalized Topic Model (GTM) interface.
    """

    def __init__(
        self,
        *args,
        doc_topic_prior: str = "logistic_normal",
        n_topics: int = 10,
        **kwargs,
    ):
        # Check if ae_type is 'ae' (plain autoencoder)
        ae_type = kwargs.get("ae_type", "wae")

        if ae_type == "ae":
            pass
        else:
            assert doc_topic_prior in {
                "dirichlet",
                "logistic_normal",
            }, "GTM supports only 'dirichlet' or 'logistic_normal' priors (or ae_type='ae' for plain autoencoder)."

        self.n_topics = n_topics
        self.doc_topic_prior = doc_topic_prior
        self.topic_labels = [f"Topic_{i}" for i in range(n_topics)]

        super().__init__(
            latent_factor_prior=doc_topic_prior, n_factors=n_topics, *args, **kwargs
        )

    def get_doc_topic_distribution(
        self,
        dataset,
        to_numpy: bool = True,
        num_workers: Optional[int] = None,
        single_modality: Optional[str] = None,
        num_samples: int = 1,
        return_std: bool = False,
    ):
        """
        Returns the full K-dimensional document-topic distribution on the simplex.

        This is a convenience method specific to topic models (GTM). It calls
        get_latent_factors() with to_simplex=True to ensure the returned values
        represent valid probability distributions over topics.

        Args:
            dataset: A Corpus object containing the documents
            to_numpy: Whether to return as numpy array (True) or torch tensor (False).
            num_workers: Number of workers for data loaders. If None, uses self.num_workers.
            single_modality: If set, uses only this modality (e.g., "default_bow").
            num_samples: Number of samples from VAE encoder (only used for ae_type="vae").
            return_std: Whether to return standard errors across samples (only for VAE).

        Returns:
            If return_std=True:
                Tuple of (theta [N, K], std [N, K])
            Otherwise:
                Topic distributions theta [N, K] where each row sums to 1

        Examples:
            >>> # Get document-topic distributions
            >>> theta = model.get_doc_topic_distribution(corpus)
            >>> print(f"Document 0 topic distribution: {theta[0]}")
            >>> print(f"Sum check: {theta[0].sum()}")  # Should be 1.0
        """
        # Get K dimensional latent factors (these are already on simplex from priors)
        result = self.get_latent_factors(
            dataset=dataset,
            to_simplex=True,  # This returns full K-dimensional simplex from priors
            to_numpy=to_numpy,
            num_workers=num_workers,
            single_modality=single_modality,
            num_samples=num_samples,
            return_std=return_std,
        )

        # The priors already handle the softmax normalization for simplex, so we just return the result
        return result

    def get_topic_words(self, l_content_covariates=[], topK=8):
        """
        Get the top words per topic, potentially influenced by content covariates.

        Args:
            l_content_covariates: list of content covariate names to influence the topic-word distribution.
            topK: number of top words to return per topic.
        """
        for key in self.decoders:
            if key.endswith("bow"):
                decoder = self.decoders[key]
                id2token = self.id2token.get(key, {})
                break
        else:
            raise ValueError("No BOW decoder found — topic-words not available.")

        decoder.eval()
        with torch.no_grad():
            topic_words = {}
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(
                self.device
            )

            for k in l_content_covariates:
                idx = [i for i, l in enumerate(self.content_colnames) if l == k][0]
                idxes[:, (self.n_topics + idx)] += 1

            word_dist = decoder(idxes)
            word_dist = F.softmax(word_dist, dim=1)
            _, indices = torch.topk(word_dist, topK, dim=1)
            indices = indices.cpu().tolist()

            for topic_id in range(self.n_topics):
                topic_words[f"Topic_{topic_id}"] = [
                    id2token.get(idx, f"<UNK_{idx}>") for idx in indices[topic_id]
                ]
        return topic_words

    def get_covariate_words(self, topK=8):
        """
        Get the top words associated to a specific content covariate.

        Args:
            topK: number of top words to return per content covariate.
        """
        for key in self.decoders:
            if key.endswith("bow"):
                decoder = self.decoders[key]
                id2token = self.id2token.get(key, {})
                break
        else:
            raise ValueError("No BOW decoder found — covariate-words not available.")

        decoder.eval()
        with torch.no_grad():
            covariate_words = {}
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(
                self.device
            )
            word_dist = decoder(idxes)
            word_dist = F.softmax(word_dist, dim=1)
            _, indices = torch.topk(word_dist, topK, dim=1)
            indices = indices.cpu().tolist()
            for i in range(self.n_topics, self.n_topics + self.content_covariate_size):
                cov_name = self.content_colnames[i - self.n_topics]
                covariate_words[cov_name] = [
                    id2token.get(idx, f"<UNK_{idx}>") for idx in indices[i]
                ]
        return covariate_words

    def get_topic_word_distribution(self, l_content_covariates=[], to_numpy=True):
        """
        Get the topic-word distribution of each topic, potentially influenced by covariates.

        Args:
            l_content_covariates: list with the names of the content covariates to influence the topic-word distribution.
            to_numpy: whether to return the topic-word distribution as a numpy array.
        """
        for key in self.decoders:
            if key.endswith("bow"):
                decoder = self.decoders[key]
                break
        else:
            raise ValueError(
                "No BOW decoder found — topic-word distribution not available."
            )

        decoder.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(
                self.device
            )
            for k in l_content_covariates:
                try:
                    idx = self.content_colnames.index(k)
                    idxes[:, self.n_topics + idx] += 1
                except ValueError:
                    raise ValueError(
                        f"Content covariate '{k}' not found in content_colnames."
                    )
            topic_word_distribution = decoder(idxes)
            topic_word_distribution = F.softmax(topic_word_distribution, dim=1)

        return (
            topic_word_distribution[: self.n_topics, :].cpu().numpy()
            if to_numpy
            else topic_word_distribution[: self.n_topics, :]
        )

    def get_covariate_word_distribution(self, to_numpy=True):
        """
        Get the word distribution associated with each content covariate.

        For models with content covariates, this method returns the word distributions
        that are uniquely associated with each covariate (controlling for topics).
        This helps identify which words are characteristic of different document
        attributes (e.g., time periods, authors, genres).

        Args:
            to_numpy: Whether to return as numpy array (True) or torch tensor (False).

        Returns:
            Word distributions [num_covariates, vocab_size] as numpy array or tensor.
            Each row is a probability distribution over the vocabulary.

        Raises:
            ValueError: If no BOW decoder is found in the model.

        Examples:
            >>> # Get words associated with each covariate
            >>> covariate_dists = model.get_covariate_word_distribution()
            >>> top_words_cov0 = covariate_dists[0].argsort()[-10:]  # Top 10 words
        """
        """
        Get the covariate-word distribution of each topic.

        Args:
            to_numpy: whether to return the covariate-word distribution as a numpy array.
        """
        for key in self.decoders:
            if key.endswith("bow"):
                decoder = self.decoders[key]
                break
        else:
            raise ValueError(
                "No BOW decoder found — covariate-word distributions not available."
            )

        decoder.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(
                self.device
            )
            word_dist = decoder(idxes)
            word_dist = F.softmax(word_dist, dim=1)

        return (
            word_dist[self.n_topics :, :].cpu().numpy()
            if to_numpy
            else word_dist[self.n_topics :, :]
        )

    def get_top_docs(
        self,
        dataset,
        topic_id=None,
        l_content_covariates=[],
        return_df=False,
        topK=1,
        num_samples: int = 1,
    ):
        """
        Get the top documents for each topic (or a specific topic).

        Returns documents with the highest probability mass on each topic. This is
        useful for interpreting topics by examining representative documents.

        Args:
            dataset: A Corpus object containing the documents
            topic_id: If specified, returns top documents only for this topic.
                     If None, returns top documents for all topics.
            return_df: Whether to return results as a pandas DataFrame (True) or
                      dictionary (False).
            topK: Number of top documents to return per topic.
            num_samples: Number of samples from VAE encoder (only used for ae_type="vae").

        Returns:
            If return_df=True:
                DataFrame with columns: topic_id, doc_index, topic_weight, ...
            If return_df=False and topic_id is None:
                Dict mapping topic_id -> list of (doc_index, topic_weight) tuples
            If return_df=False and topic_id is specified:
                List of (doc_index, topic_weight) tuples for that topic

        Examples:
            >>> # Get top 5 documents for each topic
            >>> top_docs = model.get_top_docs(corpus, topK=5, return_df=True)
            >>>
            >>> # Get top 10 documents for topic 3
            >>> topic3_docs = model.get_top_docs(corpus, topic_id=3, topK=10)
        """
        """
        Get the most representative documents per topic.

        Args:
            dataset: a Corpus object
            topic_id: the topic to retrieve the top documents from. If None, the top documents for all topics are returned.
            return_df: whether to return the top documents as a DataFrame.
            topK: number of top documents to return per topic.
            num_samples: number of samples to draw for VAE inference (ignored for WAE).
        """

        view_column = list(dataset.modalities_config.values())[0]["column"]

        doc_topic_distribution = self.get_latent_factors(
            dataset, to_simplex=True, num_samples=num_samples
        )
        if l_content_covariates:
            idxes = [self.content_colnames.index(c) for c in l_content_covariates]
            mask = (dataset.M_content_covariates[:, idxes] > 0).all(axis=1)
        else:
            mask = np.ones(doc_topic_distribution.shape[0], dtype=bool)

        top_k_indices_df = pd.DataFrame(
            {
                f"Topic_{k}": np.argsort(doc_topic_distribution[:, k] * mask)[::-1][
                    :topK
                ]
                for k in range(doc_topic_distribution.shape[1])
            }
        )

        if not return_df:
            if topic_id is None:
                for topic_id in range(self.n_topics):
                    for i in top_k_indices_df[f"Topic_{topic_id}"]:
                        print(
                            f"Topic: {topic_id} | Document index: {i} | Topic share: {doc_topic_distribution[i, topic_id]:.4f}"
                        )
                        print(dataset.df[view_column].iloc[i])
                        print("\n")
            else:
                for i in top_k_indices_df[f"Topic_{topic_id}"]:
                    print(
                        f"Topic: {topic_id} | Document index: {i} | Topic share: {doc_topic_distribution[i, topic_id]:.4f}"
                    )
                    print(dataset.df[view_column].iloc[i])
                    print("\n")
        else:
            records = []
            for t_id in range(self.n_topics):
                for i in top_k_indices_df[f"Topic_{t_id}"]:
                    records.append(
                        {
                            "topic_id": t_id,
                            "doc_id": i,
                            "topic_share": doc_topic_distribution[i, t_id],
                            view_column: dataset.df[view_column].iloc[i],
                        }
                    )
            df = pd.DataFrame.from_records(records)
            if topic_id is not None:
                df = df[df["topic_id"] == topic_id].reset_index(drop=True)
            return df

    def plot_topic_word_distribution(
        self,
        topic_id,
        content_covariates=[],
        topK=100,
        plot_type="wordcloud",
        output_path=None,
        wordcloud_args={"background_color": "white"},
        plt_barh_args={"color": "grey"},
        plt_savefig_args={"dpi": 300},
    ):
        """
        Returns a wordcloud/barplot representation per topic.

        Args:
            topic_id: the topic to visualize.
            content_covariates: list with the names of the content covariates to influence the topic-word distribution.
            topK: number of top words to return per topic.
            plot_type: either 'wordcloud' or 'barplot'.
            output_path: path to save the plot.
            wordcloud_args: dictionary with the parameters for the wordcloud plot.
            plt_barh_args: dictionary with the parameters for the barplot plot.
            plt_savefig_args: dictionary with the parameters for the savefig function.
        """

        for key in self.decoders:
            if key.endswith("bow"):
                id2token = self.id2token.get(key, {})
                break

        topic_word_distribution = self.get_topic_word_distribution(
            content_covariates, to_numpy=False
        )
        vals, indices = torch.topk(topic_word_distribution, topK, dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()
        topic_words = [id2token[idx] for idx in indices[topic_id]]
        values = vals[topic_id]
        d = {}
        for i, w in enumerate(topic_words):
            d[w] = values[i]

        if plot_type == "wordcloud":
            wordcloud = WordCloud(**wordcloud_args).generate_from_frequencies(d)
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
        else:
            sorted_items = sorted(d.items(), key=lambda x: x[1])
            words = [item[0] for item in sorted_items]
            values = [item[1] * 100 for item in sorted_items]
            plt.figure(figsize=(8, len(words) // 2))
            plt.barh(words, values, **plt_barh_args)
            plt.xlabel("Probability")
            plt.ylabel("Words")
            plt.title("Words for {}".format(self.topic_labels[topic_id]))
            plt.show()

        if output_path is not None:
            plt.savefig(output_path, **plt_savefig_args)

    def visualize_docs(
        self,
        dataset,
        dimension_reduction="tsne",
        dimension_reduction_args={"random_state": 42},
        update_layout_args=dict(
            autosize=True,
            width=None,
            height=None,
            plot_bgcolor="white",
            paper_bgcolor="white",
        ),
        display=True,
        output_path=None,
        num_samples: int = 1,
    ):
        """
        Visualize the documents in the corpus based on their topic distribution.

        Args:
            dataset: a Corpus object
            dimension_reduction: dimensionality reduction technique. Either 'umap', 'tsne' or 'pca'.
            dimension_reduction_args: dictionary with the parameters for the dimensionality reduction technique.
            update_layout_args: dictionary with the parameters for the layout of the plot.
            display: whether to display the plot.
            output_path: path to save the plot.
        """

        matrix = self.get_latent_factors(
            dataset, to_simplex=True, num_samples=num_samples
        )
        most_prevalent_topics = np.argmax(matrix, axis=1)
        most_prevalent_topic_share = np.max(matrix, axis=1)

        if dimension_reduction == "umap":
            ModelLowDim = UMAP(n_components=2, **dimension_reduction_args)
        elif dimension_reduction == "tsne":
            ModelLowDim = TSNE(n_components=2, **dimension_reduction_args)
        elif dimension_reduction == "pca":
            ModelLowDim = PCA(n_components=2, **dimension_reduction_args)
        else:
            raise ValueError(
                f"Unsupported dimension_reduction '{dimension_reduction}'. "
                "Choose 'umap', 'tsne', or 'pca'."
            )

        EmbeddingsLowDim = ModelLowDim.fit_transform(matrix)

        labels = list(dataset.df["doc_clean"])

        deciles = np.percentile(most_prevalent_topic_share, np.arange(0, 100, 10))
        marker_sizes = np.zeros_like(most_prevalent_topic_share)
        for i in range(1, 10):
            marker_sizes[
                (most_prevalent_topic_share > deciles[i - 1])
                & (most_prevalent_topic_share <= deciles[i])
            ] = i

        trace = go.Scatter(
            x=EmbeddingsLowDim[:, 0],
            y=EmbeddingsLowDim[:, 1],
            mode="markers",
            text=labels,
            hoverinfo="text",
            marker=dict(
                size=marker_sizes,
                color=most_prevalent_topics,
                colorscale="Plasma",
                opacity=0.5,
            ),
        )
        annotations = []
        for i, topic_name in enumerate(self.topic_labels):
            annotations.append(
                dict(
                    x=EmbeddingsLowDim[most_prevalent_topics == i, 0].mean(),
                    y=EmbeddingsLowDim[most_prevalent_topics == i, 1].mean(),
                    xref="x",
                    yref="y",
                    text='<b> <span style="font-size: 16px;">'
                    + topic_name
                    + "</span> </b>",
                    showarrow=False,
                    ax=0,
                    ay=0,
                )
            )
        layout = go.Layout(hovermode="closest", annotations=annotations)
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout(**update_layout_args)
        if display:
            fig.show(config=dict(editable=True))
        if output_path is not None:
            fig.write_html(output_path, config=dict(editable=True))

    def visualize_words(
        self,
        dimension_reduction="tsne",
        dimension_reduction_args={"random_state": 42},
        update_layout_args=dict(
            autosize=True,
            width=None,
            height=None,
            plot_bgcolor="white",
            paper_bgcolor="white",
        ),
        display=True,
        output_path=None,
    ):
        """
        Visualize the words in the corpus based on their topic distribution.

        Args:
            dimension_reduction: dimensionality reduction technique. Either 'umap', 'tsne' or 'pca'.
            dimension_reduction_args: dictionary with the parameters for the dimensionality reduction technique.
            update_layout_args: dictionary with the parameters for the layout of the plot.
            display: whether to display the plot.
            output_path: path to save the plot.
        """

        matrix = self.get_topic_word_distribution().T
        most_prevalent_topics = np.argmax(matrix, axis=1)
        most_prevalent_topic_share = np.max(matrix, axis=1)

        if dimension_reduction == "umap":
            ModelLowDim = UMAP(n_components=2, **dimension_reduction_args)
        elif dimension_reduction == "tsne":
            ModelLowDim = TSNE(n_components=2, **dimension_reduction_args)
        elif dimension_reduction == "pca":
            ModelLowDim = PCA(n_components=2, **dimension_reduction_args)
        else:
            raise ValueError(
                f"Unsupported dimension_reduction '{dimension_reduction}'. "
                "Choose 'umap', 'tsne', or 'pca'."
            )

        EmbeddingsLowDim = ModelLowDim.fit_transform(matrix)

        labels = list(self.id2token.values())

        deciles = np.percentile(most_prevalent_topic_share, np.arange(0, 100, 10))
        marker_sizes = np.zeros_like(most_prevalent_topic_share)
        for i in range(1, 10):
            marker_sizes[
                (most_prevalent_topic_share > deciles[i - 1])
                & (most_prevalent_topic_share <= deciles[i])
            ] = i

        trace = go.Scatter(
            x=EmbeddingsLowDim[:, 0],
            y=EmbeddingsLowDim[:, 1],
            mode="markers",
            text=labels,
            hoverinfo="text",
            marker=dict(
                size=marker_sizes,
                color=most_prevalent_topics,
                colorscale="Plasma",
                opacity=0.5,
            ),
        )
        annotations = []
        top_words = [v for k, v in self.get_topic_words(topK=1).items()]
        for l in top_words:
            for word in l:
                annotations.append(
                    dict(
                        x=EmbeddingsLowDim[labels.index(word), 0],
                        y=EmbeddingsLowDim[labels.index(word), 1],
                        xref="x",
                        yref="y",
                        text='<b> <span style="font-size: 16px;">'
                        + word
                        + "</b> </span>",
                        showarrow=False,
                        ax=0,
                        ay=0,
                    )
                )
        layout = go.Layout(hovermode="closest", annotations=annotations)
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout(**update_layout_args)
        if display:
            fig.show(config=dict(editable=True))
        if output_path is not None:
            fig.write_html(output_path, config=dict(editable=True))

    def visualize_topics(
        self,
        dataset,
        dimension_reduction_args={},
        update_layout_args=dict(
            autosize=True,
            width=None,
            height=None,
            plot_bgcolor="white",
            paper_bgcolor="white",
        ),
        display=True,
        output_path=None,
        num_samples: int = 1,
    ):
        """
        Visualize the topics in the corpus based on their topic distribution.

        Args:
            dataset: a Corpus object
            dimension_reduction_args: dictionary with the parameters for the dimensionality reduction technique.
            update_layout_args: dictionary with the parameters for the layout of the plot.
            display: whether to display the plot.
            output_path: path to save the plot.
        """

        matrix = self.get_topic_word_distribution()
        doc_topic_dist = self.get_latent_factors(
            dataset, to_simplex=True, num_samples=num_samples
        )
        df = pd.DataFrame(doc_topic_dist)
        marker_sizes = np.array(df.mean()) * 1000
        ModelLowDim = PCA(n_components=2, **dimension_reduction_args)
        EmbeddingsLowDim = ModelLowDim.fit_transform(matrix)
        labels = [v for k, v in self.get_topic_words().items()]

        trace = go.Scatter(
            x=EmbeddingsLowDim[:, 0],
            y=EmbeddingsLowDim[:, 1],
            mode="markers",
            text=labels,
            hoverinfo="text",
            marker=dict(size=marker_sizes),
        )
        annotations = []
        for i, topic_name in enumerate(self.topic_labels):
            annotations.append(
                dict(
                    x=EmbeddingsLowDim[i, 0],
                    y=EmbeddingsLowDim[i, 1],
                    xref="x",
                    yref="y",
                    text='<b> <span style="font-size: 16px;">'
                    + topic_name
                    + "</span> </b>",
                    showarrow=False,
                    ax=0,
                    ay=0,
                )
            )
        layout = go.Layout(hovermode="closest", annotations=annotations)
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout(**update_layout_args)
        if display:
            fig.show(config=dict(editable=True))
        if output_path is not None:
            fig.write_html(output_path, config=dict(editable=True))


class IdealPointNN(DeepLatent):
    """
    Neural Ideal Point Model (IdealPointNN)
    """

    def __init__(
        self,
        *args,
        n_ideal_points: int = 1,
        **kwargs,
    ):

        self.n_ideal_points = n_ideal_points
        self.print_topics = False

        # For plain autoencoder, the prior won't be used anyway
        super().__init__(
            latent_factor_prior="gaussian", n_factors=n_ideal_points, *args, **kwargs
        )

    def get_ideal_points(
        self,
        dataset,
        to_numpy: bool = True,
        num_workers: Optional[int] = None,
        single_modality: Optional[str] = None,
        num_samples: int = 1,
        return_std: bool = False,
        return_samples: bool = False,
    ):
        """
        Returns unconstrained latent ideal points (z ∈ ℝⁿ).

        This is a convenience method specific to ideal point models. It calls
        get_latent_factors() with to_simplex=False to return real-valued vectors
        representing positions in policy/attribute space.

        For 1D ideal points, this represents a left-right political spectrum.
        For higher dimensions, each dimension can represent a different policy
        dimension or attribute.

        Args:
            dataset: A Corpus object containing the documents
            to_numpy: Whether to return as numpy array (True) or torch tensor (False).
            num_workers: Number of workers for data loaders. If None, uses self.num_workers.
            single_modality: If set, uses only this modality (e.g., "default_bow").
            num_samples: Number of samples from VAE encoder (only used for ae_type="vae").
            return_std: Whether to return standard errors across samples (only for VAE).
            return_samples: Whether to return all samples instead of mean. If True,
                          returns [batch_size, num_samples, n_dims] tensor/array.

        Returns:
            If return_samples=True:
                Samples array [N, num_samples, n_ideal_points]
            If return_std=True:
                Tuple of (ideal_points [N, n_ideal_points], stds [N, n_ideal_points])
            Otherwise:
                Ideal points [N, n_ideal_points] as real-valued vectors

        Examples:
            >>> # Get 1D ideal points (political positions)
            >>> positions = model.get_ideal_points(corpus)
            >>> liberal = positions < 0
            >>> conservative = positions > 0
            >>>
            >>> # Get 2D ideal points with uncertainty
            >>> points, stds = model.get_ideal_points(corpus, num_samples=10, return_std=True)
        """
        return self.get_latent_factors(
            dataset=dataset,
            to_simplex=False,
            to_numpy=to_numpy,
            num_workers=num_workers,
            single_modality=single_modality,
            num_samples=num_samples,
            return_std=return_std,
            return_samples=return_samples,
        )
