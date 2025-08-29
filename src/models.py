#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time
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
from autoencoders import EncoderMLP, DecoderMLP, MultiModalEncoder, ImageEncoder, ImageDecoder
from predictors import Predictor
from priors import DirichletPrior, LogisticNormalPrior, GaussianPrior, FixedDirichletPrior, FixedGaussianPrior, FixedLogisticNormalPrior
from utils import compute_mmd_loss, top_k_indices_column, parse_modality_view
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
        fixed_prior=True,  
        alpha=0.1, 
        encoder_args={},
        decoder_args={},
        predictor_args={},
        predictor_type="classifier",
        include_labels_in_encoder: bool = True,
        fusion: str = "moe_average",  
        gating_hidden_dim=None,
        num_flows=4,
        flow_hidden_dim=None,  
        flow_use_permutations=True,  
        initialization=False,
        num_epochs=1000,
        batch_size=64,
        num_workers=4,
        optim_args=None,
        print_every_n_epochs=1,
        print_every_n_batches=10000,
        log_every_n_epochs=10000,
        print_topics=False,
        patience=1,
        patience_tol=1e-3,
        w_prior=1,
        w_pred_loss=1,
        kl_annealing_start=-1,
        kl_annealing_end=-1,
        free_bits_lambda=None, 
        ckpt_folder="../ckpt",
        device=None,
        seed=42,
    ):         

        """
        Args:
            train_data: a Corpus object
            test_data: a Corpus object
            n_factors: number of factors (topics / ideal points) to learn.
            ae_type: type of autoencoder. Either 'wae' (Wasserstein Autoencoder) or 'vae' (Variational Autoencoder).
            latent_factor_prior: prior on the document-topic distribution. Either 'dirichlet', 'logistic_normal', or 'gaussian'.
            alpha: parameter of the Dirichlet prior (legacy, now unused)
            encoder_args: dictionary with the parameters for the encoder.
            decoder_args: dictionary with the parameters for the decoder.
            predictor_args: dictionary with the parameters for the predictor.
            predictor_type: type of predictor model. Either 'classifier' or 'regressor'.
            include_labels_in_encoder: whether to include labels in the encoder input.
            fusion: type of fusion method to use. Either 'moe_average', 'moe_gating', or 'poe'.
            gating_hidden_dim: hidden dimension for gating mechanism (if used).
            num_flows: number of flows for IAF (if used).
            flow_hidden_dim: hidden dimension for IAF flows (if None, defaults to max(n_factors, 16)).
            flow_use_permutations: whether to use permutations between IAF flows for better expressivity.
            num_epochs: number of epochs to train the model.
            num_workers: number of workers for the data loaders.
            batch_size: batch size for training.
            optim_args: dictionary with the parameters for the optimizer. Can include 'main' for main parameters, 'prior' for prior parameters, and other Adam parameters like 'betas'. If None, uses default parameters.
            print_every_n_epochs: number of epochs between each print.
            print_every_n_batches: number of batches between each print.
            log_every_n_epochs: number of epochs between each checkpoint.
            patience: number of epochs to wait before stopping the training if the validation or training loss does not improve.
            patience_tol: tolerance for improvement in loss. Loss must improve by at least this amount to reset patience counter.
            w_prior: parameter to control the tightness of the encoder output with the document-topic prior. If set to None, w_prior is chosen automatically.
            w_pred_loss: parameter to control the weight given to the prediction task in the likelihood. Default is 1.
            kl_annealing_start: epoch at which to start the KL annealing.
            kl_annealing_end: epoch at which to end the KL annealing.
            free_bits_lambda: Free bits threshold (e.g., 0.01). If None, no free bits constraint is applied. Free bits prevent posterior collapse by ensuring each latent dimension maintains a minimum KL divergence with the prior.
            ckpt_folder: folder to save the checkpoints.
            ckpt: checkpoint to load the model from.
            device: device to use for training.
            seed: random seed.
        """  

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cpu")
        )

        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            np.random.seed(seed)

        self.n_factors = n_factors

        self.ae_type = ae_type
        assert ae_type in {"wae", "vae"}, f"Invalid ae_type: {ae_type}"
        self.vi_type = vi_type
        assert self.vi_type in {"mean_field", "full_rank", "iaf"}, f"Invalid vi_type: {vi_type}"

        self.latent_factor_prior = latent_factor_prior
        self.fixed_prior = fixed_prior
        self.alpha = alpha
        self.initialization = initialization
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.print_every_n_epochs = print_every_n_epochs
        self.print_every_n_batches = print_every_n_batches
        self.log_every_n_epochs = log_every_n_epochs
        self.patience = patience
        self.patience_tol = patience_tol
        self.w_prior = w_prior
        self.w_pred_loss = w_pred_loss
        self.kl_annealing_start = kl_annealing_start
        self.kl_annealing_end = kl_annealing_end
        self.free_bits_lambda = free_bits_lambda
        self.ckpt_folder = ckpt_folder
        self.print_topics = print_topics
        self.predictor_type = predictor_type
        self.fusion = fusion
        self.gating_hidden_dim = gating_hidden_dim
        self.num_flows = num_flows
        self.flow_hidden_dim = flow_hidden_dim
        self.flow_use_permutations = flow_use_permutations
        self.include_labels_in_encoder = include_labels_in_encoder

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
        self.labels_size = (
            train_data.M_labels.shape[1] if train_data.labels else 0
        )

        self.content_colnames = train_data.content_colnames or []
        self.id2token = train_data.id2token

        # ENCODERS
        if not encoder_args:
            raise ValueError("encoder_args is empty. You must specify at least one encoder configuration.")

        encoders = OrderedDict()
        for key, config in encoder_args.items():
            mod, view = parse_modality_view(key)
            view_data = train_data.processed_modalities[mod][view]
            view_type = view_data["type"]

            if view_type == "image":
                # Image-specific encoder with CNN
                if ae_type == "vae":
                    if self.vi_type == "mean_field":
                        final_dim = n_factors * 2
                    elif self.vi_type == "full_rank":
                        final_dim = n_factors + (n_factors * (n_factors + 1)) // 2
                    elif self.vi_type == "iaf":
                        final_dim = n_factors * 2
                    else:
                        raise ValueError(f"Invalid vi_type: {self.vi_type}")
                else:
                    final_dim = n_factors
                
                encoders[key] = ImageEncoder(
                    input_channels=config.get("input_channels", 3),
                    latent_dim=final_dim,
                    prevalence_covariate_size=self.prevalence_covariate_size,
                    labels_size=self.labels_size,
                    include_labels=self.include_labels_in_encoder,
                    hidden_dims=config.get("hidden_dims", [32, 64, 128, 256]),
                    fc_hidden_dims=config.get("fc_hidden_dims", [512]),
                    dropout=config.get("dropout", 0.1),
                    activation=config.get("activation", "relu"),
                    use_batch_norm=config.get("use_batch_norm", True)
                )
                
            elif view_type in {"bow", "embedding", "vote"}:
                input_dim = view_data["matrix"].shape[1]
            elif view_type == "discrete_choice":
                sample_key = next(k for k in view_data if k != "type")
                input_dim = sum(view_data[q]["matrix"].shape[1] for q in view_data if q != "type")
            else:
                raise ValueError(f"Unsupported view_type: {view_type}")

            # Create MLP encoders for non-image modalities
            if view_type != "image":
                if ae_type == "vae":
                    if self.vi_type == "mean_field":
                        final_dim = n_factors * 2
                    elif self.vi_type == "full_rank":
                        final_dim = n_factors + (n_factors * (n_factors + 1)) // 2
                    elif self.vi_type == "iaf":
                        final_dim = n_factors * 2  # base params, flows in encoder
                    else:
                        raise ValueError(f"Invalid vi_type: {self.vi_type}")
                else:
                    final_dim = n_factors

                extra_in = self.prevalence_covariate_size + (self.labels_size if self.include_labels_in_encoder else 0)
                dims = [input_dim + extra_in] + config.get("hidden_dims", []) + [final_dim]

                encoders[key] = EncoderMLP(
                    encoder_dims=dims,
                    encoder_non_linear_activation=config.get("activation", "relu"),
                    encoder_bias=config.get("bias", True),
                    dropout=config.get("dropout", 0.0)
                )

        if self.fusion == "moe_average":
            moe_type, gating, poe = "average", False, False
        elif self.fusion == "moe_gating":
            moe_type, gating, poe = "gating", True, False
        elif self.fusion == "moe_learned":
            moe_type, gating, poe = "learned_weights", False, False
        elif self.fusion == "poe":
            moe_type, gating, poe = "average", False, True
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion}")

        self.encoder = MultiModalEncoder(
            encoders=encoders,
            topic_dim=n_factors,
            gating=gating,
            gating_hidden_dim=gating_hidden_dim,
            ae_type=ae_type,
            poe=poe,
            vi_type=self.vi_type,
            num_flows=num_flows,
            flow_hidden_dim=self.flow_hidden_dim,
            flow_use_permutations=self.flow_use_permutations,
            moe_type=moe_type
        ).to(self.device)

        # DECODERS
        if not decoder_args:
            raise ValueError("decoder_args is empty. You must specify at least one decoder configuration.")

        self.decoders = nn.ModuleDict()
        for key, config in decoder_args.items():
            mod, view = parse_modality_view(key)
            view_data = train_data.processed_modalities[mod][view]
            view_type = view_data["type"]

            if view_type == "image":
                # Image-specific decoder with CNN
                self.decoders[key] = ImageDecoder(
                    latent_dim=n_factors,
                    content_covariate_size=self.content_covariate_size,
                    output_channels=config.get("output_channels", 3),
                    hidden_dims=config.get("hidden_dims", [256, 128, 64, 32]),
                    output_size=config.get("output_size", (224, 224)),
                    fc_hidden_dims=config.get("fc_hidden_dims", [512]),
                    dropout=config.get("dropout", 0.1),
                    activation=config.get("activation", "relu"),
                    use_batch_norm=config.get("use_batch_norm", True)
                ).to(self.device)
                
            elif view_type == "discrete_choice":
                sub_decoders = nn.ModuleDict()
                for question, subview in view_data.items():
                    if question == "type":
                        continue
                    output_dim = subview["matrix"].shape[1]
                    dims = [n_factors + self.content_covariate_size] + config.get("hidden_dims", []) + [output_dim]
                    sub_decoders[question] = DecoderMLP(
                        decoder_dims=dims,
                        decoder_non_linear_activation=config.get("activation", None),
                        decoder_bias=config.get("bias", False),
                        dropout=config.get("dropout", 0.0)
                    ).to(self.device)
                self.decoders[key] = sub_decoders
            else:
                output_dim = view_data["matrix"].shape[1]
                dims = [n_factors + self.content_covariate_size] + config.get("hidden_dims", []) + [output_dim]
                self.decoders[key] = DecoderMLP(
                    decoder_dims=dims,
                    decoder_non_linear_activation=config.get("activation", None),
                    decoder_bias=config.get("bias", False),
                    dropout=config.get("dropout", 0.0)
                ).to(self.device)

        # PRIOR
        if fixed_prior:
            if latent_factor_prior == "dirichlet":
                self.prior = FixedDirichletPrior(
                    self.prevalence_covariate_size,
                    n_factors,
                    alpha=alpha
                )
            elif latent_factor_prior == "logistic_normal":
                self.prior = FixedLogisticNormalPrior(
                    self.prevalence_covariate_size,
                    n_factors
                )
            elif latent_factor_prior == "gaussian":
                self.prior = FixedGaussianPrior(
                    prevalence_covariate_size=self.prevalence_covariate_size,
                    n_dims=n_factors
                )
            else:
                raise ValueError(f"Fixed prior not supported for: {latent_factor_prior}")
        else:
            if latent_factor_prior == "dirichlet":
                self.prior = DirichletPrior(
                    self.prevalence_covariate_size,
                    n_factors
                )
            elif latent_factor_prior == "logistic_normal":
                self.prior = LogisticNormalPrior(
                    self.prevalence_covariate_size,
                    n_factors
                )
            elif latent_factor_prior == "gaussian":
                self.prior = GaussianPrior(
                    prevalence_covariate_size=self.prevalence_covariate_size,
                    n_dims=n_factors
                )
            else:
                raise ValueError(f"Unrecognized prior: {latent_factor_prior}")

        # PREDICTOR
        if self.labels_size != 0:
            if not predictor_args:
                raise ValueError("predictor_args is empty. You must specify at least one predictor configuration.")
            config = predictor_args.get("label", {})
            predictor_dims = [n_factors + self.prediction_covariate_size] + config.get("hidden_dims", []) + [1]
            self.predictor = Predictor(
                predictor_dims=predictor_dims,
                predictor_non_linear_activation=config.get("activation", "relu"),
                predictor_bias=config.get("bias", True),
                dropout=config.get("dropout", 0.0)
            ).to(self.device)
        else:
            self.predictor = None

        # OPTIMIZER with different parameter groups
        main_params = list(self.encoder.parameters()) + list(self.decoders.parameters())
        if self.predictor is not None:
            main_params += list(self.predictor.parameters())
            
        # Set up optimizer configuration
        if optim_args is None:
            optim_args = {
                "main": {"lr": 1e-3, "weight_decay": 0.0},
                "prior": {"lr": 1e-4, "weight_decay": 0.01}
            }
        
        # Extract configurations for each parameter group
        main_config = optim_args.get("main", {"lr": 1e-3, "weight_decay": 0.01})
        prior_config = optim_args.get("prior", {"lr": 1e-4, "weight_decay": 0.01})
        
        # Extract global optimizer settings (applied to all parameter groups if not overridden)
        global_config = {k: v for k, v in optim_args.items() 
                        if k not in ["main", "prior"]}
        default_global = {"betas": (0.9, 0.999), "eps": 1e-8}
        
        # Merge global defaults with user-provided global config
        for key, default_val in default_global.items():
            if key not in global_config:
                global_config[key] = default_val
        
        # Apply global config to parameter groups if not already specified
        for config in [main_config, prior_config]:
            for key, default_val in global_config.items():
                if key not in config:
                    config[key] = default_val
        
        # Create parameter groups
        if not fixed_prior:
            prior_params = list(self.prior.parameters())
            param_groups = [
                {'params': main_params, **main_config},
                {'params': prior_params, **prior_config}
            ]
        else:
            # For fixed priors, only optimize main parameters
            param_groups = [
                {'params': main_params, **main_config}
            ]
        
        # Create optimizer (no need to pass global_config again since it's in param_groups)
        self.optimizer = torch.optim.Adam(param_groups)

        self.epochs = 0
        self.loss = np.inf
        self.reconstruction_loss = np.inf
        self.divergence_loss = np.inf
        self.prediction_loss = np.inf

        # Move all model components to the specified device
        self.to(self.device)

        # No more initialization logic - everything is learned end-to-end
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
        self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))

        if self.epochs == 0:
            best_loss = np.inf
            best_epoch = -1

        else:
            best_loss = self.loss
            best_epoch = self.epochs

        for epoch in range(self.epochs, self.num_epochs):

            training_loss = self.epoch(train_data_loader, validation=False)

            if test_data is not None:
                validation_loss = self.epoch(test_data_loader, validation=True)

            if (epoch + 1) % self.log_every_n_epochs == 0:
                save_name = f'{self.ckpt_folder}/M_K{self.n_factors}_{self.latent_factor_prior}_{self.predictor_type}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_{self.epochs+1}.ckpt'
                self.save_model(save_name)

            # Stopping rule for the optimization routine
            if test_data is not None:
                if validation_loss < best_loss - self.patience_tol:
                    best_loss = validation_loss
                    best_epoch = self.epochs
                    self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))
                    counter = 0
                else:
                    counter += 1
            else:
                if training_loss < best_loss - self.patience_tol:
                    best_loss = training_loss
                    best_epoch = self.epochs
                    self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))
                    counter = 0
                else:
                    counter += 1

            if counter >= self.patience or (epoch + 1) == self.num_epochs:

                ckpt = "{}/best_model.ckpt".format(self.ckpt_folder)
                self.load_model(ckpt)

                print(
                    "\nStopping at Epoch {}. Reverting to Epoch {}".format(
                        epoch + 1, best_epoch + 1
                    )
                )
                break

            self.epochs += 1

    def epoch(self, data_loader, validation=False, num_samples=1):
        """
        Train the model for one epoch.
        """
        if validation:
            self.encoder.eval()
            self.decoders.eval()
            if self.labels_size != 0:
                self.predictor.eval()
            self.prior.eval()
        else:
            self.encoder.train()
            self.decoders.train()
            if self.labels_size != 0:
                self.predictor.train()
            self.prior.train()

        epochloss_lst = []

        with torch.no_grad() if validation else torch.enable_grad():
            for iter, data in enumerate(data_loader):
                # Initialize loss components
                prediction_loss = 0.0
                
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
                    view_type = data_loader.dataset.processed_modalities[mod][view]["type"]

                    if view_type == "image":
                        # Images are already tensors from corpus lazy loading
                        x = data["modalities"][mod][view].to(self.device)
                        modality_inputs[key] = x  # Don't concatenate covariates for images - handled in encoder
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
                        if self.include_labels_in_encoder and target_labels is not None:
                            # ensure labels are 2D; if class ids, one-hot encode to match self.labels_size
                            lab = target_labels
                            if lab.dim() == 1:  # class ids -> one-hot
                                lab = F.one_hot(lab.to(torch.int64), num_classes=self.labels_size).float()
                            x = torch.cat([x, lab], dim=1)

                        modality_inputs[key] = x

                theta_q, z, mu_logvar = self.encoder(
                    modality_inputs, 
                    prevalence_covariates=prevalence_covariates.to(self.device) if prevalence_covariates is not None else None,
                    labels=target_labels.to(self.device) if target_labels is not None and self.include_labels_in_encoder else None
                )

                if self.latent_factor_prior in {"dirichlet", "logistic_normal"}:
                    doc_latents = theta_q  # simplex
                else:
                    doc_latents = z      # real-valued

                # -------------------- DECODERS --------------------
                reconstruction_loss = 0.0
                theta_input = torch.cat([doc_latents, content_covariates], dim=1) if content_covariates is not None else doc_latents

                for key, decoder in self.decoders.items():
                    mod, view = parse_modality_view(key)
                    view_type = data_loader.dataset.processed_modalities[mod][view]["type"]
                    modality_data = data["modalities"][mod][view]

                    if view_type == "image":
                        # Image reconstruction
                        target_images = modality_data.to(self.device)
                        
                        # For image decoder, we need to pass content covariates separately
                        reconstructed_images = decoder(doc_latents, content_covariates)
                        
                        # Use MSE loss for image reconstruction (could also use perceptual loss)
                        recon_loss = F.mse_loss(reconstructed_images, target_images)
                        reconstruction_loss += recon_loss
                        
                    elif view_type == "discrete_choice":
                        for question, question_decoder in decoder.items():
                            if question == "type":
                                continue
                            x_out = modality_data[question].to(self.device)
                            logits = question_decoder(theta_input)
                            targets = x_out.argmax(dim=-1)
                            reconstruction_loss += F.cross_entropy(logits, targets)

                    else:
                        target = modality_data.to(self.device) if view_type in {"bow", "embedding"} else modality_data["matrix"].to(self.device)
                        recon = decoder(theta_input)

                        if view_type == "bow":
                            log_probs = F.log_softmax(recon, dim=1)
                            recon_loss = -torch.sum(target * log_probs) / torch.sum(target)

                        elif view_type == "embedding":
                            recon_loss = F.mse_loss(recon, target)

                        elif view_type == "vote":
                            mask = ~modality_data["mask"].to(self.device)
                            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
                            losses = loss_fn(recon, target)
                            recon_loss = torch.sum(losses * mask) / torch.sum(mask)

                        else:
                            raise ValueError(f"Unsupported view type: {view_type}")

                        reconstruction_loss += recon_loss

                # -------------------- PRIOR / MMD / KL Divergence --------------------
                mmd_loss = 0.0
                for _ in range(num_samples):
                    theta_prior = self.prior.sample(
                        N=doc_latents.shape[0],
                        M_prevalence_covariates=prevalence_covariates,
                        epoch=self.epochs
                    ).to(self.device)
                    mmd_loss += compute_mmd_loss(doc_latents, theta_prior, device=self.device)

                if self.epochs < self.kl_annealing_start:
                    beta = 0.0
                elif self.epochs > self.kl_annealing_end:
                    beta = self.w_prior
                else:
                    span = self.kl_annealing_end - self.kl_annealing_start
                    t = (self.epochs - self.kl_annealing_start) / span  # 0→1
                    # Sigmoid annealing: smooth S-shaped curve
                    # Shift and scale sigmoid to go from 0 to 1
                    sigmoid_input = 12 * (t - 0.5)  # Scale and center around 0.5
                    beta = self.w_prior * torch.sigmoid(torch.tensor(sigmoid_input)).item()

                if self.ae_type == "vae":
                    # Expect a single fused posterior tuple in mu_logvar_fused
                    mu_logvar_fused = mu_logvar if isinstance(mu_logvar, tuple) else mu_logvar[-1]
                    
                    # Check if prior has full covariance (learned priors)
                    has_full_cov = hasattr(self.prior, 'sigma') and not isinstance(self.prior, (FixedGaussianPrior, FixedLogisticNormalPrior, FixedDirichletPrior))
                    
                    if self.vi_type == "iaf":
                        # mu_q, logvar_q from encoder; z0 is the actual sample that was transformed; 
                        # flow returns zk, log_det_j (sum over steps and dims)
                        mu_q, logvar_q, z0, zk, log_det_j = mu_logvar_fused

                        # log q0(z0) - use the actual z0 that was transformed
                        log_q_z0 = -0.5 * (
                            torch.sum(logvar_q, dim=1)
                            + torch.sum((z0 - mu_q)**2 / torch.exp(logvar_q), dim=1)
                            + z0.size(1) * np.log(2*np.pi)
                        )

                        # log p(zk) - handle full covariance prior
                        if has_full_cov:
                            mu_p, Sigma_p = self.prior.get_prior_params(prevalence_covariates, return_full_cov=True)
                            B = zk.shape[0]
                            D = zk.shape[1]
                            
                            # Expand for batch
                            Sigma_p_batch = Sigma_p.unsqueeze(0).expand(B, -1, -1)  # [B, D, D]
                            Sigma_p_inv = torch.inverse(Sigma_p_batch)
                            
                            # Multivariate Gaussian log-pdf: log p(zk)
                            logdet_p = torch.logdet(Sigma_p_batch)  # [B]
                            diff = (zk - mu_p).unsqueeze(2)  # [B, D, 1]
                            quad_term = torch.bmm(torch.bmm(diff.transpose(1, 2), Sigma_p_inv), diff).squeeze(-1).squeeze(-1)  # [B]
                            log_p_zk = -0.5 * (logdet_p + quad_term + D * np.log(2*np.pi))
                        else:
                            # Diagonal prior
                            mu_p, logvar_p = self.prior.get_prior_params(prevalence_covariates)
                            var_p = torch.exp(logvar_p)
                            log_p_zk = -0.5 * (
                                torch.sum(logvar_p, dim=1)
                                + torch.sum((zk - mu_p)**2 / var_p, dim=1)
                                + zk.size(1) * np.log(2*np.pi)
                            )

                        # Monte Carlo KL (note the minus sign on log_det_j)
                        kl_per_sample = log_q_z0 - log_det_j - log_p_zk  # [B]
                        
                        # Apply free bits if specified
                        if self.free_bits_lambda is not None:
                            # For IAF, approximate per-dimension free bits using base distribution
                            var_q = torch.exp(logvar_q)  # [B, D]
                            kl_per_dim_base = 0.5 * (var_q + mu_q.pow(2) - 1 - logvar_q)  # [B, D]
                            kl_per_dim_clamped = torch.clamp(kl_per_dim_base, min=self.free_bits_lambda)
                            kl = kl_per_dim_clamped.sum(dim=1).mean()
                        else:
                            kl = kl_per_sample.mean()
                        
                    elif self.vi_type == "mean_field":
                        mu_q, logvar_q = mu_logvar_fused
                        
                        if has_full_cov:
                            # Full covariance prior, diagonal posterior
                            mu_p, Sigma_p = self.prior.get_prior_params(prevalence_covariates, return_full_cov=True)
                            var_q = torch.exp(logvar_q)  # [B, D] - diagonal posterior covariance
                            B, D = mu_q.shape
                            
                            # Expand Sigma_p for batch processing
                            Sigma_p_batch = Sigma_p.unsqueeze(0).expand(B, -1, -1)  # [B, D, D]
                            Sigma_p_inv = torch.inverse(Sigma_p_batch)
                            
                            # Log determinants
                            logdet_p = torch.logdet(Sigma_p_batch)  # [B]
                            logdet_q = torch.sum(logvar_q, dim=1)  # [B] - sum of log diagonal elements
                            
                            # Trace term: tr(Σ_p^-1 Σ_q) where Σ_q is diagonal
                            trace_term = torch.sum(torch.diagonal(Sigma_p_inv, dim1=1, dim2=2) * var_q, dim=1)  # [B]
                            
                            # Quadratic term: (μ_q - μ_p)^T Σ_p^-1 (μ_q - μ_p)
                            diff = (mu_q - mu_p).unsqueeze(2)  # [B, D, 1]
                            quad_term = torch.bmm(torch.bmm(diff.transpose(1, 2), Sigma_p_inv), diff).squeeze(-1).squeeze(-1)  # [B]
                            
                            kl_raw = 0.5 * (logdet_p - logdet_q - D + trace_term + quad_term)  # [B]
                            
                            # Apply free bits if specified
                            if self.free_bits_lambda is not None:
                                # Approximate per-dimension KL using diagonal elements
                                kl_per_dim = 0.5 * (var_q + mu_q.pow(2) - 1 - logvar_q)  # [B, D]
                                kl_per_dim_clamped = torch.clamp(kl_per_dim, min=self.free_bits_lambda)
                                kl = kl_per_dim_clamped.sum(dim=1).mean()
                            else:
                                kl = kl_raw.mean()
                        else:
                            # Diagonal prior
                            mu_p, logvar_p = self.prior.get_prior_params(prevalence_covariates)
                            var_q = torch.exp(logvar_q)
                            var_p = torch.exp(logvar_p)
                            
                            # Compute per-dimension KL
                            kl_per_dim = 0.5 * (
                                logvar_p - logvar_q - 1 + var_q / var_p + (mu_q - mu_p).pow(2) / var_p
                            )  # [B, D]
                            
                            # Apply free bits if specified
                            if self.free_bits_lambda is not None:
                                kl_per_dim_clamped = torch.clamp(kl_per_dim, min=self.free_bits_lambda)
                                kl = kl_per_dim_clamped.sum(dim=1).mean()
                            else:
                                kl = kl_per_dim.sum(dim=1).mean()
                            
                    elif self.vi_type == "full_rank":
                        mu_q, L = mu_logvar_fused  # L lower-triangular with positive diag
                        B, D, _ = L.shape
                        Sigma_q = torch.bmm(L, L.transpose(1, 2))  # [B, D, D]
                        logdet_q = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=1, dim2=2) + 1e-6), dim=1)  # [B]
                        
                        if has_full_cov:
                            # Full covariance prior
                            mu_p, Sigma_p = self.prior.get_prior_params(prevalence_covariates, return_full_cov=True)
                            
                            # Expand prior covariance for batch
                            Sigma_p_batch = Sigma_p.unsqueeze(0).expand(B, -1, -1)  # [B, D, D]
                            Sigma_p_inv = torch.inverse(Sigma_p_batch)
                            
                            # KL divergence components
                            logdet_p = torch.logdet(Sigma_p_batch)  # [B]
                            
                            # Trace term: tr(Σ_p^-1 Σ_q)
                            trace_term = torch.sum(torch.diagonal(torch.bmm(Sigma_p_inv, Sigma_q), dim1=1, dim2=2), dim=1)  # [B]
                            
                            # Quadratic term: (μ_q - μ_p)^T Σ_p^-1 (μ_q - μ_p)
                            diff = (mu_q - mu_p).unsqueeze(2)  # [B, D, 1]
                            quad_term = torch.bmm(torch.bmm(diff.transpose(1, 2), Sigma_p_inv), diff).squeeze(-1).squeeze(-1)  # [B]
                            
                            kl_raw = 0.5 * (logdet_p - logdet_q - D + trace_term + quad_term)  # [B]
                        else:
                            # Diagonal prior
                            mu_p, logvar_p = self.prior.get_prior_params(prevalence_covariates)
                            var_p = torch.exp(logvar_p)
                            logdet_p = torch.sum(logvar_p, dim=1)  # diag prior
                            trace_term = torch.sum(torch.diagonal(Sigma_q, dim1=1, dim2=2) / var_p, dim=1)
                            diff = mu_q - mu_p
                            quad_term = torch.sum(diff.pow(2) / var_p, dim=1)
                            kl_raw = 0.5 * (logdet_p - logdet_q - D + trace_term + quad_term)  # [B]
                        
                        # Apply free bits if specified (use already computed Sigma_q)
                        if self.free_bits_lambda is not None:
                            var_q_diag = torch.diagonal(Sigma_q, dim1=1, dim2=2)  # [B, D]
                            kl_per_dim = 0.5 * (var_q_diag + mu_q.pow(2) - 1 - torch.log(var_q_diag + 1e-8))
                            kl_per_dim_clamped = torch.clamp(kl_per_dim, min=self.free_bits_lambda)
                            kl = kl_per_dim_clamped.sum(dim=1).mean()
                        else:
                            kl = kl_raw.mean()
                    
                    divergence_loss = beta * kl
                else:
                    divergence_loss = mmd_loss * self.w_prior

                # -------------------- PREDICTION --------------------
                if target_labels is not None:
                    predictions = self.predictor(doc_latents, prediction_covariates)
                    if self.predictor_type == "classifier":
                        target_labels = target_labels.squeeze().to(torch.int64)
                        prediction_loss = F.cross_entropy(predictions, target_labels)
                    elif self.predictor_type == "regressor":
                        prediction_loss = F.mse_loss(predictions, target_labels)
                else:
                    prediction_loss = 0.0

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

                epochloss_lst.append(loss.item())

                if (iter + 1) % self.print_every_n_batches == 0:
                    msg = (
                        f"Epoch {(self.epochs+1):>3d}\tIter {(iter+1):>4d}"
                        f"\tMean {'Validation' if validation else 'Training'} Loss:{loss.item():<.7f}"
                        f"\nRec Loss:{reconstruction_loss.item():<.7f}"
                        f"\nDivergence Loss:{divergence_loss.item():<.7f}"
                        f"\nPred Loss:{prediction_loss * self.w_pred_loss:<.7f}\n"
                    )
                    print(msg)

        # -------------------- END OF EPOCH --------------------
        if (self.epochs + 1) % self.print_every_n_epochs == 0:
            avg_loss = sum(epochloss_lst) / len(epochloss_lst)
            print(
                f"\nEpoch {(self.epochs+1):>3d}\tMean {'Validation' if validation else 'Training'} Loss:{avg_loss:<.7f}\n"
            )

            if self.print_topics==True:
                print(
                    "\n".join(
                        [
                            "{}: {}".format(str(k), str(v))
                            for k, v in self.get_topic_words(topK=5).items()
                        ]
                    )
                )

        return sum(epochloss_lst)
    
    def get_topic_words(self):
        pass

    def get_latent_factors(
        self,
        dataset,
        to_simplex: bool = True,
        num_workers: Optional[int] = None,
        to_numpy: bool = True,
        single_modality: Optional[str] = None,
        num_samples: int = 1,
        return_std: bool = False,
    ):
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
                        raise ValueError(f"Modality '{single_modality}' not found in encoders")
                    
                    mod, view = parse_modality_view(single_modality)
                    view_data = data["modalities"][mod][view]
                    view_type = dataset.processed_modalities[mod][view]["type"]

                    if view_type == "image":
                        x = view_data.to(self.device)
                        modality_inputs = {single_modality: x}  # Don't concatenate covariates for images
                    elif view_type in {"bow", "embedding"}:
                        x = view_data.to(self.device)
                    elif view_type == "vote":
                        x = view_data["matrix"].to(self.device)
                    elif view_type == "discrete_choice":
                        question_tensors = [
                            view_data[q].to(self.device)
                            for q in view_data if q != "type"
                        ]
                        x = torch.cat(question_tensors, dim=-1)
                    else:
                        raise ValueError(f"Unsupported view type: {view_type}")

                    # For non-image modalities, concatenate covariates
                    if view_type != "image":
                        if prevalence_covariates is not None:
                            x = torch.cat([x, prevalence_covariates], dim=1)

                        labels = data.get("M_labels", None)
                        if self.include_labels_in_encoder and labels is not None:
                            lab = labels
                            if lab.dim() == 1:
                                lab = F.one_hot(lab.to(torch.int64), num_classes=self.labels_size).float()
                            x = torch.cat([x, lab], dim=1)

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
                                for q in view_data if q != "type"
                            ]
                            x = torch.cat(question_tensors, dim=-1)
                        else:
                            raise ValueError(f"Unsupported view type: {view_type}")

                        # For non-image modalities, concatenate covariates
                        if view_type != "image":
                            if prevalence_covariates is not None:
                                x = torch.cat([x, prevalence_covariates], dim=1)

                            labels = data.get("M_labels", None)
                            if self.include_labels_in_encoder and labels is not None:
                                lab = labels
                                if lab.dim() == 1:
                                    lab = F.one_hot(lab.to(torch.int64), num_classes=self.labels_size).float()
                                x = torch.cat([x, lab], dim=1)

                            modality_inputs[key] = x

                # Use the MultiModalEncoder.forward() method with single_modality parameter
                if self.ae_type == "vae":
                    thetas = []
                    for _ in range(num_samples):
                        theta_q, z, _ = self.encoder(
                            modality_inputs, 
                            single_modality=single_modality,
                            prevalence_covariates=prevalence_covariates,
                            labels=data.get("M_labels", None) if self.include_labels_in_encoder else None
                        )
                        thetas.append(theta_q if to_simplex else z)
                    
                    samples = torch.stack(thetas, dim=1)  # [B, num_samples, D]
                    theta_q = samples.mean(dim=1)
                    
                    if return_std:
                        theta_std = samples.std(dim=1)
                else:
                    theta_q, z, _ = self.encoder(
                        modality_inputs, 
                        single_modality=single_modality,
                        prevalence_covariates=prevalence_covariates,
                        labels=data.get("M_labels", None) if self.include_labels_in_encoder else None
                    )
                    theta_q = theta_q if to_simplex else z
                    if return_std:
                        theta_std = torch.zeros_like(theta_q)

                final_thetas.append(theta_q)
                if return_std:
                    final_stds.append(theta_std)

        if to_numpy:
            final_thetas = [t.cpu().numpy() for t in final_thetas]
            final_thetas = np.concatenate(final_thetas, axis=0)
            if return_std:
                final_stds = [s.cpu().numpy() for s in final_stds]
                final_stds = np.concatenate(final_stds, axis=0)
        else:
            final_thetas = torch.cat(final_thetas, dim=0)
            if return_std:
                final_stds = torch.cat(final_stds, dim=0)

        return (final_thetas, final_stds) if return_std else final_thetas

    def get_predictions(self, dataset, to_simplex=True, num_workers=None, to_numpy=True, num_samples: int = 1):
        """
        Predict the labels of the documents in the corpus based on topic proportions.

        Args:
            dataset: a Corpus object
            to_simplex: whether to map the topic distribution to the simplex. If False, the topic distribution is returned in the logit space.
            num_workers: number of workers for the data loaders.
            to_numpy: whether to return the predictions as a numpy array.
            num_samples: number of samples for VAE (only used if ae_type == 'vae').
        """
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
            final_predictions = []
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
                            for q in view_data if q != "type"
                        ]
                        x = torch.cat(question_tensors, dim=-1)
                    else:
                        raise ValueError(f"Unsupported view type: {view_type}")
                    
                    # For non-image modalities, concatenate covariates
                    if view_type != "image":
                        if prevalence_covariates is not None:
                            x = torch.cat([x, prevalence_covariates], dim=1)

                        labels = data.get("M_labels", None)
                        if self.include_labels_in_encoder and labels is not None:
                            lab = labels
                            if lab.dim() == 1:
                                lab = F.one_hot(lab.to(torch.int64), num_classes=self.labels_size).float()
                            x = torch.cat([x, lab], dim=1)

                        modality_inputs[key] = x

                if self.ae_type == "vae":
                    thetas = []
                    for _ in range(num_samples):
                        theta_q, z, _ = self.encoder(
                            modality_inputs,
                            prevalence_covariates=prevalence_covariates,
                            labels=data.get("M_labels", None) if self.include_labels_in_encoder else None
                        )
                        theta_q = theta_q if to_simplex else z
                        thetas.append(theta_q)
                    features = torch.stack(thetas, dim=1).mean(dim=1)
                else:
                    theta_q, z, _ = self.encoder(
                        modality_inputs,
                        prevalence_covariates=prevalence_covariates,
                        labels=data.get("M_labels", None) if self.include_labels_in_encoder else None
                    )
                    features = theta_q if to_simplex else z

                predictions = self.predictor(features, prediction_covariates)
                if self.predictor_type == "classifier":
                    predictions = torch.softmax(predictions, dim=1)

                final_predictions.append(predictions)

            if to_numpy:
                final_predictions = [p.cpu().numpy() for p in final_predictions]
                final_predictions = np.concatenate(final_predictions, axis=0)
            else:
                final_predictions = torch.cat(final_predictions, dim=0)

        return final_predictions

    def get_modality_weights(
        self,
        dataset,
        num_workers: Optional[int] = None,
        to_numpy: bool = True
    ):
        """
        Returns modality weights per observation.
        
        For moe_gating: softmax weights from the gating network.
        For poe: normalized precisions (inverse variances).
        For moe_average: equal weights.
        
        Returns:
            weights: tensor of shape (N, M) where M is number of modalities
        """
        if num_workers is None:
            num_workers = self.num_workers

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
                        modality_inputs[name] = x  # Don't concatenate covariates for images
                    elif view_type in {"bow", "embedding"}:
                        x = view_data.to(self.device)
                    elif view_type == "vote":
                        x = view_data["matrix"].to(self.device)
                    elif view_type == "discrete_choice":
                        x = torch.cat([view_data[q].to(self.device) for q in view_data if q != "type"], dim=-1)
                    else:
                        raise ValueError(f"Unsupported view type: {view_type}")

                    # For non-image modalities, concatenate covariates
                    if view_type != "image":
                        if prevalence_covariates is not None:
                            x = torch.cat((x, prevalence_covariates), dim=1)
                        labels = data.get("M_labels", None)
                        if self.include_labels_in_encoder and labels is not None:
                            lab = labels
                            if lab.dim() == 1:
                                lab = F.one_hot(lab.to(torch.int64), num_classes=self.labels_size).float()
                            x = torch.cat([x, lab], dim=1)
                        
                        modality_inputs[name] = x

                    # Encode each modality separately for weight computation
                    encoder = self.encoder.encoders[name]
                    if isinstance(encoder, ImageEncoder):
                        z = encoder(modality_inputs[name], prevalence_covariates=prevalence_covariates, labels=data.get("M_labels", None))
                    else:
                        z = encoder(modality_inputs[name])

                    if self.ae_type == "vae":
                        if self.vi_type == "mean_field":
                            mu, logvar = torch.chunk(z, 2, dim=1)
                            mu_logvars.append((mu, logvar))
                            zs.append(mu)
                        elif self.vi_type == "full_rank":
                            D = self.n_factors
                            mu = z[:, :D]
                            L_flat = z[:, D:]
                            B = mu.size(0)
                            tril_indices = torch.tril_indices(D, D, device=z.device)
                            L = torch.zeros(B, D, D, device=z.device)
                            L[:, tril_indices[0], tril_indices[1]] = L_flat
                            # Stabilize diagonal
                            diag_idx = torch.arange(D, device=z.device)
                            L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-4
                            mu_logvars.append((mu, L))
                            zs.append(mu)
                        elif self.vi_type == "iaf":
                            mu, logvar = torch.chunk(z, 2, dim=1)
                            # Sample from base distribution
                            std = torch.exp(0.5 * logvar)
                            eps = torch.randn_like(std)
                            z0 = mu + eps * std
                            # Apply flow
                            flow = self.encoder.flows[name]
                            zk, log_det_j = flow(z0)
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
                            gate_input = torch.cat([torch.cat((mu, logvar), dim=1) 
                                                  for mu, logvar in mu_logvars if len(mu_logvars[0]) == 2], dim=1)
                        elif self.vi_type == "full_rank":
                            gate_input = torch.cat([torch.cat((mu, L.view(mu.size(0), -1)), dim=1) 
                                                  for mu, L in mu_logvars if len(mu_logvars[0]) == 2], dim=1)
                        elif self.vi_type == "iaf":
                            gate_input = torch.cat([torch.cat((mu, logvar), dim=1) 
                                                  for mu, logvar, _, _ in mu_logvars], dim=1)
                    else:
                        gate_input = torch.cat(zs, dim=1)

                    weights = self.encoder.gate_net(gate_input)

                elif self.fusion == "poe":
                    if self.ae_type == "vae":
                        if self.vi_type == "mean_field":
                            precisions = [1.0 / torch.exp(logvar) for _, logvar in mu_logvars 
                                        if len(mu_logvars[0]) == 2]
                            if precisions:
                                precision_stack = torch.stack(precisions, dim=1)  # (B, M, D)
                                weights = precision_stack.sum(dim=2)  # (B, M)
                            else:
                                weights = torch.full((B, M), 1.0 / M, device=self.device)

                        elif self.vi_type == "full_rank":
                            precisions = []
                            D = self.n_factors
                            I = torch.eye(D, device=self.device).unsqueeze(0)  # (1, D, D)
                            for _, L in mu_logvars:
                                if len(mu_logvars[0]) == 2:  # Not IAF flow
                                    B = L.size(0)
                                    Sigma_q = torch.bmm(L, L.transpose(1, 2))  # (B, D, D)
                                    Sigma_inv = torch.linalg.inv(Sigma_q + 1e-4 * I.expand(B, -1, -1))
                                    trace_prec = torch.diagonal(Sigma_inv, dim1=1, dim2=2).sum(dim=1)  # (B,)
                                    precisions.append(trace_prec)
                            if precisions:
                                weights = torch.stack(precisions, dim=1)  # (B, M)
                                weights = weights / weights.sum(dim=1, keepdim=True)
                            else:
                                weights = torch.full((B, M), 1.0 / M, device=self.device)

                        elif self.vi_type == "iaf":
                            # For flows, use base distribution variances
                            precisions = [1.0 / torch.exp(logvar) for mu, logvar, _, _ in mu_logvars]
                            precision_stack = torch.stack(precisions, dim=1)  # (B, M, D)
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

    def generate_samples(
        self,
        n_samples: int = 10,
        content_covariates: Optional[torch.Tensor] = None,
        prevalence_covariates: Optional[torch.Tensor] = None,
        modality_keys: Optional[List[str]] = None,
        temperature: float = 1.0,
        to_numpy: bool = True,
        seed: Optional[int] = None
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
        self.prior.eval()
        
        # Default to all modalities if not specified
        if modality_keys is None:
            modality_keys = list(self.decoders.keys())
        
        # Prepare covariates
        device = self.device
        
        if content_covariates is None:
            content_covariates = torch.zeros(n_samples, self.content_covariate_size, device=device)
        else:
            content_covariates = content_covariates.to(device)
            if content_covariates.size(0) != n_samples:
                raise ValueError(f"content_covariates must have {n_samples} samples, got {content_covariates.size(0)}")
        
        if prevalence_covariates is None:
            prevalence_covariates = torch.zeros(n_samples, self.prevalence_covariate_size, device=device)
        else:
            prevalence_covariates = prevalence_covariates.to(device)
            if prevalence_covariates.size(0) != n_samples:
                raise ValueError(f"prevalence_covariates must have {n_samples} samples, got {prevalence_covariates.size(0)}")
        
        generated_samples = {}
        
        with torch.no_grad():
            # Sample from prior
            if self.latent_factor_prior in {"dirichlet", "logistic_normal"}:
                # For topic models, sample from prior and use as simplex
                doc_latents = self.prior.sample(
                    N=n_samples,
                    M_prevalence_covariates=prevalence_covariates,
                    epoch=self.epochs
                ).to(device)
                
                # Apply temperature scaling for topic models
                if temperature != 1.0:
                    doc_latents = F.softmax(torch.log(doc_latents + 1e-8) / temperature, dim=1)
                
            else:
                # For ideal point models, sample from Gaussian prior
                doc_latents = self.prior.sample(
                    N=n_samples,
                    M_prevalence_covariates=prevalence_covariates,
                    epoch=self.epochs
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
        encoder_state_dict = self.encoder.state_dict()
        decoders_state_dict = {k: d.state_dict() for k, d in self.decoders.items()}
        predictor_state_dict = self.predictor.state_dict() if self.labels_size != 0 else None
        optimizer_state_dict = self.optimizer.state_dict()

        all_vars = vars(self)

        checkpoint = {}
        for key, value in all_vars.items():
            if key not in ["encoder", "decoders", "predictor", "optimizer"]:
                checkpoint[key] = value

        checkpoint["encoder"] = encoder_state_dict
        checkpoint["decoders"] = decoders_state_dict
        if self.labels_size != 0:
            checkpoint["predictor"] = predictor_state_dict
        checkpoint["optimizer"] = optimizer_state_dict

        torch.save(checkpoint, save_name)

    def load_model(self, ckpt):
        """
        Helper function to load the model.
        """
        ckpt = torch.load(ckpt, map_location=self.device, weights_only=False)

        for key, value in ckpt.items():
            if key not in ["encoder", "decoders", "predictor", "optimizer"]:
                setattr(self, key, value)

        self.encoder.load_state_dict(ckpt["encoder"])

        for key, state_dict in ckpt["decoders"].items():
            self.decoders[key].load_state_dict(state_dict)

        if self.labels_size != 0 and "predictor" in ckpt:
            if not hasattr(self, "predictor"):
                # Create a basic predictor with default parameters if it doesn't exist
                # The exact architecture will be determined from the checkpoint state_dict
                predictor_dims = [self.n_factors + self.prediction_covariate_size, self.labels_size]
                self.predictor = Predictor(
                    predictor_dims=predictor_dims,
                    predictor_non_linear_activation="relu",
                    predictor_bias=True,
                    dropout=0.0,
                ).to(self.device)
            self.predictor.load_state_dict(ckpt["predictor"])

        if not hasattr(self, "optimizer"):
            # Create parameter groups with different parameter groups as in __init__
            main_params = list(self.encoder.parameters()) + list(self.decoders.parameters())
            if self.labels_size != 0 and hasattr(self, "predictor"):
                main_params += list(self.predictor.parameters())
            
            # Use default optimizer configuration
            default_optim_args = {
                "main": {"lr": 1e-3, "weight_decay": 0.0},
                "prior": {"lr": 1e-4, "weight_decay": 0.01},
                "betas": (0.9, 0.999),
                "eps": 1e-8
            }
            
            main_config = default_optim_args["main"]
            prior_config = default_optim_args["prior"]
            global_config = {k: v for k, v in default_optim_args.items() 
                           if k not in ["main", "prior"]}
            
            # Apply global config to parameter groups
            for config in [main_config, prior_config]:
                for key, default_val in global_config.items():
                    if key not in config:
                        config[key] = default_val
            
            # Add prior parameters if it's a learnable prior
            if not getattr(self, "fixed_prior", False) and hasattr(self, "prior"):
                prior_params = list(self.prior.parameters())
                param_groups = [
                    {'params': main_params, **main_config},
                    {'params': prior_params, **prior_config}
                ]
            else:
                param_groups = [
                    {'params': main_params, **main_config}
                ]
                
            self.optimizer = torch.optim.Adam(param_groups)

        self.optimizer.load_state_dict(ckpt["optimizer"])

    def to(self, device):
        """
        Move the model to a different device.
        """
        self.encoder.to(device)
        self.decoders.to(device)
        self.prior.to(device)
        if self.labels_size != 0:
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
        assert doc_topic_prior in {"dirichlet", "logistic_normal"}, \
            "GTM supports only 'dirichlet' or 'logistic_normal' priors."

        self.n_topics = n_topics
        self.doc_topic_prior = doc_topic_prior
        self.topic_labels = [f"Topic_{i}" for i in range(n_topics)]

        super().__init__(
            latent_factor_prior=doc_topic_prior,
            n_factors=n_topics,  # Learn K factors
            *args,
            **kwargs
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
        Returns the full K-dimensional document-topic distribution.

        Args:
            dataset: a Corpus object
            to_numpy: whether to return as a numpy array.
            num_workers: number of workers for the data loaders.
            single_modality: if set, uses only this modality (e.g., "default_bow")
            num_samples: number of samples from the VAE encoder (only used for VAE).
            return_std: whether to return standard errors across samples.
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
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(self.device)

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
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(self.device)
            word_dist = decoder(idxes)
            word_dist = F.softmax(word_dist, dim=1)
            _, indices = torch.topk(word_dist, topK, dim=1)
            indices = indices.cpu().tolist()
            for i in range(self.n_topics, self.n_topics + self.content_covariate_size):
                cov_name = self.content_colnames[i - self.n_topics]
                covariate_words[cov_name] = [id2token.get(idx, f"<UNK_{idx}>") for idx in indices[i]]
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
            raise ValueError("No BOW decoder found — topic-word distribution not available.")

        decoder.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(self.device)
            for k in l_content_covariates:
                try:
                    idx = self.content_colnames.index(k)
                    idxes[:, self.n_topics + idx] += 1
                except ValueError:
                    raise ValueError(f"Content covariate '{k}' not found in content_colnames.")
            topic_word_distribution = decoder(idxes)
            topic_word_distribution = F.softmax(topic_word_distribution, dim=1)

        return topic_word_distribution[:self.n_topics, :].cpu().numpy() if to_numpy else topic_word_distribution[:self.n_topics, :]

    def get_covariate_word_distribution(self, to_numpy=True):
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
            raise ValueError("No BOW decoder found — covariate-word distributions not available.")

        decoder.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topics + self.content_covariate_size).to(self.device)
            word_dist = decoder(idxes)
            word_dist = F.softmax(word_dist, dim=1)

        return word_dist[self.n_topics:, :].cpu().numpy() if to_numpy else word_dist[self.n_topics:, :]

    def get_top_docs(self, dataset, topic_id=None, return_df=False, topK=1, num_samples: int = 1):
        """
        Get the most representative documents per topic.

        Args:
            dataset: a Corpus object
            topic_id: the topic to retrieve the top documents from. If None, the top documents for all topics are returned.
            return_df: whether to return the top documents as a DataFrame.
            topK: number of top documents to return per topic.
            num_samples: number of samples to draw for VAE inference (ignored for WAE).
        """
        doc_topic_distribution = self.get_latent_factors(
            dataset, to_simplex=True, num_samples=num_samples
        )

        top_k_indices_df = pd.DataFrame(
            {
                f"Topic_{col}": top_k_indices_column(
                    doc_topic_distribution[:, col], topK
                )
                for col in range(doc_topic_distribution.shape[1])
            }
        )

        if not return_df:
            if topic_id is None:
                for topic_id in range(self.n_topics):
                    for i in top_k_indices_df[f"Topic_{topic_id}"]:
                        print(
                            f"Topic: {topic_id} | Document index: {i} | Topic share: {doc_topic_distribution[i, topic_id]:.4f}"
                        )
                        print(dataset.df["doc"].iloc[i])
                        print("\n")
            else:
                for i in top_k_indices_df[f"Topic_{topic_id}"]:
                    print(
                        f"Topic: {topic_id} | Document index: {i} | Topic share: {doc_topic_distribution[i, topic_id]:.4f}"
                    )
                    print(dataset.df["doc"].iloc[i])
                    print("\n")
        else:
            records = []
            for t_id in range(self.n_topics):
                for i in top_k_indices_df[f"Topic_{t_id}"]:
                    records.append({
                        "topic_id": t_id,
                        "doc_id": i,
                        "topic_share": doc_topic_distribution[i, t_id],
                        "doc": dataset.df["doc"].iloc[i]
                    })
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

        matrix = self.get_latent_factors(dataset, to_simplex=True, num_samples=num_samples)
        most_prevalent_topics = np.argmax(matrix, axis=1)
        most_prevalent_topic_share = np.max(matrix, axis=1)

        if dimension_reduction == "umap":
            ModelLowDim = UMAP(n_components=2, **dimension_reduction_args)
        if dimension_reduction == "tsne":
            ModelLowDim = TSNE(n_components=2, **dimension_reduction_args)
        else:
            ModelLowDim = PCA(n_components=2, **dimension_reduction_args)

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
        if dimension_reduction == "tsne":
            ModelLowDim = TSNE(n_components=2, **dimension_reduction_args)
        else:
            ModelLowDim = PCA(n_components=2, **dimension_reduction_args)

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
        num_samples: int = 1
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
        doc_topic_dist = self.get_latent_factors(dataset, to_simplex=True, num_samples=num_samples)
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

        super().__init__(
            latent_factor_prior="gaussian",
            n_factors=n_ideal_points,
            *args,
            **kwargs
        )

    def get_ideal_points(
        self,
        dataset,
        to_numpy: bool = True,
        num_workers: Optional[int] = None,
        single_modality: Optional[str] = None,
        num_samples: int = 1,
        return_std: bool = False,
    ):
        """
        Returns unconstrained latent ideal points (z ∈ ℝⁿ).
        Equivalent to get_latent_factors(to_simplex=False).

        Args:
            dataset: a Corpus object
            to_numpy: whether to return as a numpy array.
            num_workers: number of workers for the data loaders.
            single_modality: if set, uses only this modality (e.g., "default_bow")
            num_samples: number of samples from the VAE encoder (only used for VAE).
            return_std: whether to return standard errors across samples.
        """
        return self.get_latent_factors(
            dataset=dataset,
            to_simplex=False,
            to_numpy=to_numpy,
            num_workers=num_workers,
            single_modality=single_modality,
            num_samples=num_samples,
            return_std=return_std,
        )
