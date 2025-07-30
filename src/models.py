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
from autoencoders import EncoderMLP, DecoderMLP, MultiModalEncoder
from predictors import Predictor
from priors import DirichletPrior, LogisticNormalPrior, GaussianPrior
from utils import compute_mmd_loss, top_k_indices_column, parse_modality_view
import os
import torch
import numpy as np
from typing import Optional
from collections import OrderedDict

class DeepLatent:  
    def __init__(
        self,
        train_data,
        test_data=None,
        n_factors=20,
        ae_type="wae",
        latent_factor_prior="dirichlet",
        update_prior=False,
        alpha=0.1,
        prevalence_model_type="RidgeCV",
        prevalence_model_args={},
        tol=0.001,
        encoder_args={},
        decoder_args={},
        predictor_args={},
        predictor_type="classifier",
        fusion: str = "moe_average",  # Options: 'moe_average', 'moe_gating', 'poe'
        initialization=True,
        num_epochs=1000,
        batch_size=64,
        num_workers=4,
        optim_args=None,
        regularization=0.0,
        print_every_n_epochs=1,
        print_every_n_batches=10000,
        log_every_n_epochs=10000,
        print_topics=False,
        patience=1,
        w_prior=1,
        w_pred_loss=1,
        kl_annealing_start=0,
        kl_annealing_end=100,
        kl_annealing_max_beta=1.0,
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
            latent_factor_prior: prior on the document-topic distribution. Either 'dirichlet' or 'logistic_normal'.
            update_prior: whether to update the prior at each epoch to account for prevalence covariates.
            alpha: parameter of the Dirichlet prior (only used if update_prior=False)
            prevalence_model_type: type of model to estimate the prevalence of each topic. Either 'LinearRegression', 'RidgeCV', 'MultiTaskLassoCV', and 'MultiTaskElasticNetCV'.
            prevalence_model_args: dictionary with the parameters for the GLM on topic prevalence.
            tol: tolerance threshold to stop the MLE of the Dirichlet prior (only used if update_prior=True)
            encoder_args: dictionary with the parameters for the encoder.
            decoder_args: dictionary with the parameters for the decoder.
            predictor_args: dictionary with the parameters for the predictor.
            predictor_type: type of predictor model. Either 'classifier' or 'regressor'.
            fusion: type of fusion method to use. Either 'moe_average', 'moe_gating', or 'poe'.
            num_epochs: number of epochs to train the model.
            num_workers: number of workers for the data loaders.
            batch_size: batch size for training.
            optim_args: dictionary with the parameters for the optimizer. If None, uses default parameters.
            print_every_n_epochs: number of epochs between each print.
            print_every_n_batches: number of batches between each print.
            log_every_n_epochs: number of epochs between each checkpoint.
            patience: number of epochs to wait before stopping the training if the validation or training loss does not improve.
            w_prior: parameter to control the tightness of the encoder output with the document-topic prior. If set to None, w_prior is chosen automatically.
            w_pred_loss: parameter to control the weight given to the prediction task in the likelihood. Default is 1.
            kl_annealing_start: epoch at which to start the KL annealing.
            kl_annealing_end: epoch at which to end the KL annealing.
            kl_annealing_max_beta: maximum value of the KL annealing beta.
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
        self.latent_factor_prior = latent_factor_prior
        self.update_prior = update_prior
        self.alpha = alpha
        self.prevalence_model_type = prevalence_model_type
        self.prevalence_model_args = prevalence_model_args
        self.tol = tol
        self.initialization = initialization
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.regularization = regularization
        self.print_every_n_epochs = print_every_n_epochs
        self.print_every_n_batches = print_every_n_batches
        self.log_every_n_epochs = log_every_n_epochs
        self.patience = patience
        self.w_prior = w_prior
        self.w_pred_loss = w_pred_loss
        self.kl_annealing_start = kl_annealing_start
        self.kl_annealing_end = kl_annealing_end
        self.kl_annealing_max_beta = kl_annealing_max_beta
        self.ckpt_folder = ckpt_folder
        self.print_topics = print_topics
        self.predictor_type = predictor_type
        self.fusion = fusion

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

            if view_type in {"bow", "embedding", "vote"}:
                input_dim = view_data["matrix"].shape[1]
            elif view_type == "discrete_choice":
                sample_key = next(k for k in view_data if k != "type")
                input_dim = sum(view_data[q]["matrix"].shape[1] for q in view_data if q != "type")
            else:
                raise ValueError(f"Unsupported view_type: {view_type}")

            dims = [input_dim + self.prevalence_covariate_size] + config.get("hidden_dims", []) + [n_factors * 2 if ae_type == "vae" else n_factors]
            encoders[key] = EncoderMLP(
                encoder_dims=dims,
                encoder_non_linear_activation=config.get("activation", "relu"),
                encoder_bias=config.get("bias", True),
                dropout=config.get("dropout", 0.0)
            )

        if self.fusion == "moe_average":
            self.gating = False
            self.poe = False
        elif self.fusion == "moe_gating":
            self.gating = True
            self.poe = False
        elif self.fusion == "poe":
            self.gating = False
            self.poe = True
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion}")

        self.encoder = MultiModalEncoder(
            encoders=encoders,
            topic_dim=n_factors,
            gating=self.gating,
            ae_type=ae_type,
            poe=self.poe
        ).to(self.device)

        # DECODERS
        if not decoder_args:
            raise ValueError("decoder_args is empty. You must specify at least one decoder configuration.")

        self.decoders = nn.ModuleDict()
        for key, config in decoder_args.items():
            mod, view = parse_modality_view(key)
            view_data = train_data.processed_modalities[mod][view]
            view_type = view_data["type"]

            if view_type == "discrete_choice":
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
        if latent_factor_prior == "dirichlet":
            self.prior = DirichletPrior(
                update_prior,
                self.prevalence_covariate_size,
                n_factors,
                alpha,
                prevalence_model_args,
                tol,
                device=self.device
            )
        elif latent_factor_prior == "logistic_normal":
            self.prior = LogisticNormalPrior(
                self.prevalence_covariate_size,
                n_factors,
                prevalence_model_type,
                prevalence_model_args,
                device=self.device
            )
        elif latent_factor_prior == "gaussian":
            self.prior = GaussianPrior(
                prevalence_covariate_size=self.prevalence_covariate_size,
                n_dims=n_factors,
                model_type=prevalence_model_type,
                prevalence_model_args=prevalence_model_args,
                device=self.device
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

        all_params = list(self.encoder.parameters()) + list(self.decoders.parameters())
        if self.predictor is not None:
            all_params += list(self.predictor.parameters())

        self.optimizer = torch.optim.Adam(all_params, **(optim_args or {"lr": 1e-3, "betas": (0.9, 0.999)}))

        self.epochs = 0
        self.loss = np.inf
        self.reconstruction_loss = np.inf
        self.divergence_loss = np.inf
        self.prediction_loss = np.inf

        if self.initialization and self.update_prior:
            self.initialize(train_data, test_data)

        self.train(train_data, test_data)

    def initialize(self, train_data, test_data=None):
        """
        Train a rough initial model using Adam optimizer.
        Stops as soon as the validation loss stops improving (patience == 1).
        Saves the best model before the loss stopped improving.
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

        best_loss = np.inf
        counter = 0
        best_model_path = f"{self.ckpt_folder}/best_initial_model.ckpt"

        print('Initializing model...')

        for epoch in range(self.num_epochs):
            train_loss = self.epoch(train_data_loader, validation=False, initialization=True)

            if test_data is not None:
                val_loss = self.epoch(test_data_loader, validation=True, initialization=True)
                current_loss = val_loss
            else:
                current_loss = train_loss
                val_loss = None

            if (epoch + 1) % self.print_every_n_epochs == 0:
                msg = f"Epoch {epoch + 1:>3d} | Train Loss: {train_loss:.7f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.7f}"
                print(msg)

            loss_improved = current_loss < best_loss

            if loss_improved:
                best_loss = current_loss
                counter = 0  
                self.save_model(best_model_path)
            else:
                counter += 1

            if counter >= 1:
                print(f"Initialization completed in {epoch+1} epochs.")
                break

        self.load_model(best_model_path)

        if self.update_prior:
            if self.latent_factor_prior == "dirichlet":
                posterior_theta = self.get_latent_factors(train_data, to_numpy=True, num_samples=30)
                self.prior.update_parameters(
                    posterior_theta, train_data.M_prevalence_covariates
                )
            else:
                posterior_theta = self.get_latent_factors(
                    train_data, to_simplex=False, to_numpy=True, num_samples=30
                )
                self.prior.update_parameters(
                    posterior_theta, train_data.M_prevalence_covariates
                )             

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
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_epoch = self.epochs
                    self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))
                    counter = 0
                else:
                    counter += 1
            else:
                if training_loss < best_loss:
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

    def epoch(self, data_loader, validation=False, initialization=False, num_samples=1):
        """
        Train the model for one epoch.
        """
        if validation:
            self.encoder.eval()
            self.decoders.eval()
            if self.labels_size != 0:
                self.predictor.eval()
        else:
            self.encoder.train()
            self.decoders.train()
            if self.labels_size != 0:
                self.predictor.train()

        epochloss_lst = []
        all_topics = []
        all_prevalence_covariates = []

        with torch.no_grad() if validation else torch.enable_grad():
            for iter, data in enumerate(data_loader):
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

                    if view_type in {"bow", "embedding"}:
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

                    if prevalence_covariates is not None:
                        x = torch.cat([x, prevalence_covariates], dim=1)

                    modality_inputs[key] = x

                theta_q, z, mu_logvar = self.encoder(modality_inputs)

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

                    if view_type == "discrete_choice":
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

                # -------------------- PRIOR / MMD --------------------
                mmd_loss = 0.0
                for _ in range(num_samples):
                    theta_prior = self.prior.sample(
                        N=doc_latents.shape[0],
                        M_prevalence_covariates=prevalence_covariates,
                        epoch=self.epochs,
                        initialization=initialization
                    ).to(self.device)
                    mmd_loss += compute_mmd_loss(doc_latents, theta_prior, device=self.device)

                if self.epochs < self.kl_annealing_start:
                    beta = 0.0
                elif self.epochs > self.kl_annealing_end:
                    beta = self.kl_annealing_max_beta
                else:
                    progress = (self.epochs - self.kl_annealing_start) / (self.kl_annealing_end - self.kl_annealing_start)
                    beta = progress * self.kl_annealing_max_beta

                if self.ae_type == "vae" and self.update_prior:
                    kl_loss = 0.0

                    for (mu_q, logvar_q) in mu_logvar:
                        mu_p, _ = self.prior.get_prior_params(prevalence_covariates)
                        mu_p = mu_p.detach()  # (B, K)
                        Sigma = self.prior.sigma.detach()  # (K, K)
                        Sigma_inv = torch.linalg.inv(Sigma)  # (K, K)
                        logdet_Sigma = torch.logdet(Sigma)  # scalar

                        # q(z|x): diagonal covariance
                        sigma_q = torch.exp(0.5 * logvar_q)  # (B, K)
                        var_q = sigma_q ** 2  # (B, K)

                        # Trace term: tr(Σ⁻¹ Σ_q)
                        # Σ_q is diag(var_q) → trace is batch-wise dot product
                        trace_term = torch.sum(var_q @ Sigma_inv.T, dim=1)  # (B,)

                        # Quadratic term: (μ_q - μ_p)^T Σ⁻¹ (μ_q - μ_p)
                        diff = mu_q - mu_p  # (B, K)
                        quad_term = torch.sum((diff @ Sigma_inv) * diff, dim=1)  # (B,)

                        # Log determinant of q
                        logdet_q = torch.sum(logvar_q, dim=1)  # (B,)

                        # Full KL per sample
                        dim = mu_q.shape[1]
                        kl = 0.5 * (trace_term + quad_term - dim + logdet_Sigma - logdet_q)  # (B,)

                        # Free bits
                        kl = torch.maximum(kl, torch.tensor(0.1, device=kl.device))  # (B,)
                        kl_loss += kl.mean()  # scalar

                    divergence_loss = beta*kl_loss
                elif self.ae_type == "vae" and not self.update_prior:
                    kl_loss = 0.0
                    for mu, logvar in mu_logvar:
                        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                        kl = torch.max(kl, torch.tensor(0.1, device=kl.device))
                        kl_loss += kl.mean()                                 
                    divergence_loss = beta*kl_loss
                else:
                    divergence_loss = mmd_loss*self.w_prior

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

                # -------------------- L2 Regularization --------------------
                l2_norm = sum(torch.norm(param, p=2) for param in self.encoder.parameters())
                for decoder in self.decoders.values():
                    l2_norm += sum(torch.norm(param, p=2) for param in decoder.parameters())

                # -------------------- TOTAL LOSS --------------------
                loss = (
                    reconstruction_loss
                    + divergence_loss
                    + prediction_loss * self.w_pred_loss
                    + self.regularization * l2_norm
                )

                self.loss = loss
                self.reconstruction_loss = reconstruction_loss
                self.divergence_loss = divergence_loss
                self.prediction_loss = prediction_loss

                if not validation:
                    loss.backward()
                    self.optimizer.step()

                epochloss_lst.append(loss.item())

                if self.update_prior and not validation and not initialization:
                    if self.latent_factor_prior == "logistic_normal":
                        all_topics.append(z.detach().cpu())
                    else:
                        all_topics.append(doc_latents.detach().cpu())
                    all_prevalence_covariates.append(prevalence_covariates.detach().cpu())

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
        if (self.epochs + 1) % self.print_every_n_epochs == 0 and initialization == False:
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

        if self.update_prior and not validation and not initialization:
            all_topics = torch.cat(all_topics, dim=0).numpy()
            all_prevalence_covariates = torch.cat(all_prevalence_covariates, dim=0).numpy()
            self.prior.update_parameters(all_topics, all_prevalence_covariates)

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
        """
        if num_workers is None:
            num_workers = self.num_workers

        self.encoder.eval()
        final_thetas = []

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

                if single_modality is not None:
                    # Single modality path
                    mod, view = parse_modality_view(single_modality)
                    
                    view_data = data["modalities"][mod][view]
                    view_type = dataset.processed_modalities[mod][view]["type"]

                    if view_type in {"bow", "embedding"}:
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

                    if prevalence_covariates is not None:
                        x = torch.cat([x, prevalence_covariates], dim=1)

                    z = self.encoder.encoders[single_modality](x)

                    if self.ae_type == "vae":
                        mu, logvar = torch.chunk(z, 2, dim=1)
                        thetas = []
                        for _ in range(num_samples):
                            std = torch.exp(0.5 * logvar)
                            eps = torch.randn_like(std)
                            z_sampled = mu + eps * std
                            theta = F.softmax(z_sampled, dim=1) if to_simplex else z_sampled
                            thetas.append(theta)
                        theta_q = torch.stack(thetas, dim=1).mean(dim=1)
                    else:
                        theta_q = F.softmax(z, dim=1) if to_simplex else z

                else:
                    # Multimodal path
                    modality_inputs = {}
                    for key in self.encoder.encoders.keys():
                        mod, view = parse_modality_view(key)
                        
                        view_data = data["modalities"][mod][view]
                        view_type = dataset.processed_modalities[mod][view]["type"]

                        if view_type in {"bow", "embedding"}:
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

                        if prevalence_covariates is not None:
                            x = torch.cat([x, prevalence_covariates], dim=1)
                        modality_inputs[key] = x

                    if self.ae_type == "vae":
                        thetas = []
                        for _ in range(num_samples):
                            theta_q, z, _ = self.encoder(modality_inputs)
                            theta_q = theta_q if to_simplex else z
                            thetas.append(theta_q)
                        theta_q = torch.stack(thetas, dim=1).mean(dim=1)
                    else:
                        theta_q, z, _ = self.encoder(modality_inputs)
                        theta_q = theta_q if to_simplex else z

                final_thetas.append(theta_q)

            if to_numpy:
                final_thetas = [t.cpu().numpy() for t in final_thetas]
                final_thetas = np.concatenate(final_thetas, axis=0)
            else:
                final_thetas = torch.cat(final_thetas, dim=0)

        return final_thetas

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
                    x = data["modalities"][mod][view].to(self.device)
                    if prevalence_covariates is not None:
                        x = torch.cat([x, prevalence_covariates], dim=1)
                    modality_inputs[key] = x

                if self.ae_type == "vae":
                    thetas = []
                    for _ in range(num_samples):
                        theta_q, z, _ = self.encoder(modality_inputs)
                        theta_q = theta_q if to_simplex else z
                        thetas.append(theta_q)
                    features = torch.stack(thetas, dim=1).mean(dim=1)
                else:
                    theta_q, z, _ = self.encoder(modality_inputs)
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

                    if view_type in {"bow", "embedding"}:
                        x = view_data.to(self.device)
                    elif view_type == "vote":
                        x = view_data["matrix"].to(self.device)
                    elif view_type == "discrete_choice":
                        x = torch.cat([view_data[q].to(self.device) for q in view_data if q != "type"], dim=-1)
                    else:
                        raise ValueError(f"Unsupported view type: {view_type}")

                    if prevalence_covariates is not None:
                        x = torch.cat((x, prevalence_covariates), dim=1)

                    z = self.encoder.encoders[name](x)

                    if self.ae_type == "vae":
                        mu, logvar = torch.chunk(z, 2, dim=1)
                        mu_logvars.append((mu, logvar))
                        zs.append(mu)  # just for dimension alignment
                    else:
                        zs.append(z)

                    modality_inputs[name] = x

                B = zs[0].size(0)
                M = len(zs)

                if self.fusion == "moe_gating":
                    gate_input = torch.cat(
                        [torch.cat((mu, logvar), dim=1) for mu, logvar in mu_logvars],
                        dim=1
                    ) if self.ae_type == "vae" else torch.cat(zs, dim=1)

                    weights = self.encoder.gate_net(gate_input)  # shape (B, M)

                elif self.fusion == "poe":
                    precisions = [1.0 / torch.exp(logvar) for _, logvar in mu_logvars]
                    precision_stack = torch.stack(precisions, dim=1)  # shape (B, M, D)
                    weights = precision_stack.sum(dim=2)  # sum over latent dim → (B, M)
                    weights = weights / weights.sum(dim=1, keepdim=True)

                else:  # moe_average
                    weights = torch.full((B, M), 1.0 / M, device=self.device)

                weights_list.append(weights)

        weights_all = torch.cat(weights_list, dim=0)  # shape (N, M)
        return weights_all.cpu().numpy() if to_numpy else weights_all

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
                predictor_dims = [self.n_factors + self.prediction_covariate_size] + \
                                self.predictor_hidden_layers + [self.labels_size]
                self.predictor = Predictor(
                    predictor_dims=predictor_dims,
                    predictor_non_linear_activation=self.predictor_non_linear_activation,
                    predictor_bias=self.predictor_bias,
                    dropout=self.dropout,
                ).to(self.device)
            self.predictor.load_state_dict(ckpt["predictor"])

        if not hasattr(self, "optimizer"):
            all_params = list(self.encoder.parameters()) + list(self.decoders.parameters())
            if self.labels_size != 0:
                all_params += list(self.predictor.parameters())
            self.optimizer = torch.optim.Adam(all_params, **self.optim_args)

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
        doc_topic_prior: str = "dirichlet",
        n_topics: int = 10,
        **kwargs,
    ):
        assert doc_topic_prior in {"dirichlet", "logistic_normal"}, \
            "GTM supports only 'dirichlet' or 'logistic_normal' priors."

        super().__init__(
            latent_factor_prior=doc_topic_prior,
            n_factors=n_topics,
            *args,
            **kwargs
        )

        self.n_topics = n_topics
        self.doc_topic_prior = doc_topic_prior
        self.topic_labels = [f"Topic_{i}" for i in range(n_topics)]

    def get_doc_topic_distribution(
        self,
        dataset,
        to_numpy: bool = True,
        num_workers: Optional[int] = None,
        single_modality: Optional[str] = None,
        num_samples: int = 1,
    ):
        """
        Returns the document-topic distribution (on the simplex).
        Equivalent to get_latent_factors(to_simplex=True).
        """
        return self.get_latent_factors(
            dataset=dataset,
            to_simplex=True,
            to_numpy=to_numpy,
            num_workers=num_workers,
            single_modality=single_modality,
            num_samples=num_samples,
        )

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

        super().__init__(
            latent_factor_prior="gaussian",
            n_factors=n_ideal_points,
            *args,
            **kwargs
        )

        self.n_ideal_points = n_ideal_points
        self.print_topics = False

    def get_ideal_points(
        self,
        dataset,
        to_numpy: bool = True,
        num_workers: Optional[int] = None,
        single_modality: Optional[str] = None,
        num_samples: int = 1,
    ):
        """
        Returns unconstrained latent ideal points (z ∈ ℝⁿ).
        Equivalent to get_latent_factors(to_simplex=False).
        """
        return self.get_latent_factors(
            dataset=dataset,
            to_simplex=False,
            to_numpy=to_numpy,
            num_workers=num_workers,
            single_modality=single_modality,
            num_samples=num_samples,
        )
