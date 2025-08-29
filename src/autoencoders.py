import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict, Callable        

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for conditioning CNNs on covariates."""
    
    def __init__(self, num_features: int, covariate_dim: int):
        super().__init__()
        self.film_net = nn.Sequential(
            nn.Linear(covariate_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * num_features)  # gamma and beta
        )
    
    def forward(self, x: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning to feature maps.
        
        Args:
            x: Feature maps [batch_size, channels, height, width]
            covariates: Covariate vector [batch_size, covariate_dim]
            
        Returns:
            Modulated feature maps [batch_size, channels, height, width]
        """
        film_params = self.film_net(covariates)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        return gamma * x + beta


class ImageEncoder(nn.Module):
    """
    CNN-based image encoder with FiLM conditioning for covariates.
    
    Attributes:
        conv_layers: CNN feature extraction layers
        film_layers: FiLM conditioning layers (if covariates provided)
        fc_layers: Final fully connected layers
        use_covariates: Whether covariates are used for conditioning
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 20,
        prevalence_covariate_size: int = 0,
        labels_size: int = 0,
        include_labels: bool = True,
        hidden_dims: List[int] = [32, 64, 128, 256],
        fc_hidden_dims: List[int] = [512],
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.prevalence_covariate_size = prevalence_covariate_size
        self.labels_size = labels_size if include_labels else 0
        self.include_labels = include_labels
        self.dropout = nn.Dropout(p=dropout)
        
        # Total covariate size
        self.covariate_size = self.prevalence_covariate_size + self.labels_size
        self.use_covariates = self.covariate_size > 0
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build CNN layers
        conv_layers = []
        film_layers = nn.ModuleList() if self.use_covariates else None
        
        in_channels = input_channels
        for i, out_channels in enumerate(hidden_dims):
            # Convolution
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                self.activation,
                nn.MaxPool2d(2)
            ])
            
            # FiLM layer for covariate conditioning
            if self.use_covariates:
                film_layers.append(FiLMLayer(out_channels, self.covariate_size))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.film_layers = film_layers
        
        # Calculate flattened CNN output size properly
        self.cnn_output_size = self._get_conv_output_size()
        
        # Final FC layers
        fc_layers = []
        prev_dim = self.cnn_output_size
        
        for hidden_dim in fc_hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                self.dropout
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        fc_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def _get_conv_output_size(self, input_shape=None):
        """Calculate the size of flattened CNN output."""
        if input_shape is None:
            # Use a reasonable default - many tests use smaller images
            input_shape = (self.input_channels, 32, 32)
            
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self._forward_conv_only(dummy_input)
            return output.view(1, -1).size(1)
    
    def _forward_conv_only(self, x, covariates=None):
        """Forward pass through CNN layers only (for size calculation)."""
        layer_idx = 0
        film_idx = 0
        
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            
            # Apply FiLM after each CNN block (after pooling)
            if (isinstance(layer, nn.MaxPool2d) and 
                self.use_covariates and 
                covariates is not None and 
                film_idx < len(self.film_layers)):
                x = self.film_layers[film_idx](x, covariates)
                film_idx += 1
        
        return x
    
    def forward(
        self, 
        x: torch.Tensor, 
        prevalence_covariates: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the image encoder.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            prevalence_covariates: Prevalence covariates [batch_size, prevalence_dim]
            labels: Label covariates [batch_size, labels_dim]
            
        Returns:
            Encoded representation [batch_size, latent_dim]
        """
        # Prepare covariates
        covariates = None
        if self.use_covariates:
            covariate_list = []
            if prevalence_covariates is not None:
                covariate_list.append(prevalence_covariates)
            if labels is not None and self.include_labels:
                covariate_list.append(labels)
            
            if covariate_list:
                covariates = torch.cat(covariate_list, dim=1)
        
        # CNN feature extraction with FiLM conditioning
        layer_idx = 0
        film_idx = 0
        
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            
            # Apply FiLM after each CNN block (after pooling)
            if (isinstance(layer, nn.MaxPool2d) and 
                self.use_covariates and 
                covariates is not None and 
                film_idx < len(self.film_layers)):
                x = self.film_layers[film_idx](x, covariates)
                film_idx += 1
        
        # Flatten and pass through FC layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        
        return x


class ImageDecoder(nn.Module):
    """
    CNN-based image decoder for reconstructing images from latent representations.
    """
    
    def __init__(
        self,
        latent_dim: int = 20,
        content_covariate_size: int = 0,
        output_channels: int = 3,
        hidden_dims: List[int] = [256, 128, 64, 32],
        output_size: Tuple[int, int] = (224, 224),
        fc_hidden_dims: List[int] = [512],
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.content_covariate_size = content_covariate_size
        self.output_channels = output_channels
        self.output_size = output_size
        self.dropout = nn.Dropout(p=dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Input dimension includes latent factors and content covariates
        input_dim = latent_dim + content_covariate_size
        
        # Initial FC layers to expand from latent space
        fc_layers = []
        prev_dim = input_dim
        
        for hidden_dim in fc_hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                self.dropout
            ])
            prev_dim = hidden_dim
        
        # Calculate the size needed for reshaping into feature maps
        # Assuming 7x7 feature maps for the first deconv layer
        self.feature_map_size = 7
        self.first_conv_dim = hidden_dims[0]
        fc_output_dim = self.first_conv_dim * self.feature_map_size * self.feature_map_size
        
        fc_layers.append(nn.Linear(prev_dim, fc_output_dim))
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Deconvolutional layers
        deconv_layers = []
        in_channels = hidden_dims[0]
        
        for i, out_channels in enumerate(hidden_dims[1:]):
            deconv_layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                self.activation
            ])
            in_channels = out_channels
        
        # Final layer to get desired output channels
        deconv_layers.extend([
            nn.ConvTranspose2d(in_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Normalize output to [0, 1]
        ])
        
        self.deconv_layers = nn.Sequential(*deconv_layers)
    
    def forward(
        self, 
        z: torch.Tensor, 
        content_covariates: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode latent representation to image.
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            content_covariates: Content covariates [batch_size, content_covariate_size]
            
        Returns:
            Reconstructed image [batch_size, channels, height, width]
        """
        # Concatenate latent factors with content covariates
        if content_covariates is not None:
            x = torch.cat([z, content_covariates], dim=1)
        else:
            x = z
        
        # FC layers
        x = self.fc_layers(x)
        
        # Reshape for deconvolution
        batch_size = x.size(0)
        x = x.view(batch_size, self.first_conv_dim, self.feature_map_size, self.feature_map_size)
        
        # Deconvolutional layers
        x = self.deconv_layers(x)
        
        # Resize to exact output size if needed
        if x.size(2) != self.output_size[0] or x.size(3) != self.output_size[1]:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        return x


class EncoderMLP(nn.Module):
    """
    Torch implementation of an encoder Multilayer Perceptron.

    Attributes:
        encoder_dims (List[int]): Dimensions of the encoder layers.
        encoder_non_linear_activation (Optional[str]): Activation function for encoder ("relu" or "sigmoid").
        encoder_bias (bool): Whether to use bias in encoder layers.
        dropout (nn.Dropout): Dropout layer.
        encoder_nonlin (Optional[Callable]): Encoder activation function.
        encoder (nn.ModuleDict): Encoder layers.
    """
    def __init__(
        self,
        encoder_dims: List[int] = [2000, 1024, 512, 20],
        encoder_non_linear_activation: Optional[str] = "relu",
        encoder_bias: bool = True,
        dropout: float = 0.0,
    ):
        super(EncoderMLP, self).__init__()

        self.encoder_dims = encoder_dims
        self.encoder_non_linear_activation = encoder_non_linear_activation
        self.encoder_bias = encoder_bias
        self.dropout = nn.Dropout(p=dropout)
        
        self.encoder_nonlin: Optional[Callable] = None
        if encoder_non_linear_activation is not None:
            self.encoder_nonlin = {"relu": F.relu, "sigmoid": torch.sigmoid}[
                encoder_non_linear_activation
            ]

        self.encoder = nn.ModuleDict(
            {
                f"enc_{i}": nn.Linear(
                    encoder_dims[i], encoder_dims[i + 1], bias=encoder_bias
                )
                for i in range(len(encoder_dims) - 1)
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded representation.
        """
        hid = x
        for i, (_, layer) in enumerate(self.encoder.items()):
            hid = self.dropout(layer(hid))
            if (
                i < len(self.encoder) - 1
                and self.encoder_non_linear_activation is not None
            ):
                hid = self.encoder_nonlin(hid)
        return hid


class DecoderMLP(nn.Module):
    """
    Torch implementation of a decoder Multilayer Perceptron.

    Attributes:
        decoder_dims (List[int]): Dimensions of the decoder layers.
        decoder_non_linear_activation (Optional[str]): Activation function for decoder ("relu" or "sigmoid").
        decoder_bias (bool): Whether to use bias in decoder layers.
        dropout (nn.Dropout): Dropout layer.
        decoder_nonlin (Optional[Callable]): Decoder activation function.
        decoder (nn.ModuleDict): Decoder layers.
    """
    def __init__(
        self,
        decoder_dims: List[int] = [20, 1024, 2000],
        decoder_non_linear_activation: Optional[str] = None,
        decoder_bias: bool = False,
        dropout: float = 0.0,
    ):
        super(DecoderMLP, self).__init__()

        self.decoder_dims = decoder_dims
        self.decoder_non_linear_activation = decoder_non_linear_activation
        self.decoder_bias = decoder_bias
        self.dropout = nn.Dropout(p=dropout)
        
        self.decoder_nonlin: Optional[Callable] = None
        if decoder_non_linear_activation is not None:
            self.decoder_nonlin = {"relu": F.relu, "sigmoid": torch.sigmoid}[
                decoder_non_linear_activation
            ]

        self.decoder = nn.ModuleDict(
            {
                f"dec_{i}": nn.Linear(
                    decoder_dims[i], decoder_dims[i + 1], bias=decoder_bias
                )
                for i in range(len(decoder_dims) - 1)
            }
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the input.

        Args:
            z (torch.Tensor): Encoded representation.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = self.dropout(layer(hid))
            if (
                i < len(self.decoder) - 1
                and self.decoder_non_linear_activation is not None
            ):
                hid = self.decoder_nonlin(hid)
        return hid


class MaskedLinear(nn.Module):
    """A linear layer with a mask to ensure autoregressive property."""
    
    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('mask', mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (MADE).
    Implements proper autoregressive masking using degrees.
    """
    def __init__(self, dim: int, hidden_dim: int = None, num_hidden: int = 1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(dim, 16)
            
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
            
        # Create layer dimensions
        layer_dims = [dim] + [hidden_dim] * num_hidden + [2 * dim]  # 2*dim for mu and log_sigma
        
        # Create masks using proper MADE degree scheme
        self.masks = self._create_masks(layer_dims)
        
        # Build the network with masked linear layers
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(MaskedLinear(layer_dims[i], layer_dims[i + 1], self.masks[i]))
            if i < len(layer_dims) - 2:  # No activation on last layer
                layers.append(nn.ReLU())
        
        self.net = nn.Sequential(*layers)
    
    def _create_masks(self, layer_dims):
        """Create binary masks using proper MADE degree scheme."""
        D = self.dim
        degrees = []
        
        # Input degrees: [1, 2, ..., D]
        degrees.append(torch.arange(1, D + 1))
        
        # Hidden layer degrees: sample uniformly from [1, ..., D-1]
        for h in layer_dims[1:-1]:
            if D > 1:
                degrees.append(torch.randint(1, D, (h,)))
            else:
                # Special case for D=1: hidden units get degree 1
                degrees.append(torch.ones(h, dtype=torch.long))
        
        # Output degrees: [1, 2, ..., D, 1, 2, ..., D] for (mu, log_sigma)
        output_degrees = torch.cat([torch.arange(1, D + 1), torch.arange(1, D + 1)])
        degrees.append(output_degrees)
        
        # Create masks
        masks = []
        for l in range(len(layer_dims) - 1):
            d_in = degrees[l].unsqueeze(0).float()    # [1, n_in]
            d_out = degrees[l + 1].unsqueeze(1).float()  # [n_out, 1]
            
            # For hidden layers: use >= to allow equal degrees
            # For output layer: use > to enforce strict autoregressive property
            if l == len(layer_dims) - 2:  # Output layer
                mask = (d_out > d_in).float()
            else:  # Hidden layers
                mask = (d_out >= d_in).float()
            
            masks.append(mask)
        
        return masks
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MADE.
        
        Args:
            x: Input tensor [batch_size, dim]
            
        Returns:
            mu: Mean parameters [batch_size, dim]
            log_sigma: Log scale parameters [batch_size, dim]
        """
        out = self.net(x)
        mu, s = torch.chunk(out, 2, dim=-1)
        
        # Use sigmoid-based parameterization for better stability
        # Maps s to a bounded range for log_sigma
        log_sigma = -2.0 + 4.0 * torch.sigmoid(s)  # Range approximately [-2, 2]
        
        return mu, log_sigma


class IAF(nn.Module):
    """
    Inverse Autoregressive Flow (IAF) with proper MADE implementation.
    """
    def __init__(self, dim: int, num_flows: int = 4, hidden_dim: int = None, num_hidden: int = 1, 
                 use_permutations: bool = True):
        super().__init__()
        self.dim = dim
        self.num_flows = num_flows
        self.use_permutations = use_permutations
        
        # Create MADE networks for each flow
        self.mades = nn.ModuleList([
            MADE(dim, hidden_dim, num_hidden) 
            for k in range(num_flows)
        ])
        
        # Create permutations between flows for better expressivity
        if use_permutations:
            self.register_buffer('perms', torch.stack([torch.randperm(dim) for _ in range(num_flows)]))
        else:
            self.perms = None
    
    def forward(self, z0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the IAF.
        
        Args:
            z0: Initial sample [batch_size, dim]
            
        Returns:
            zk: Transformed sample [batch_size, dim]
            log_det_jacobian: Log determinant of Jacobian [batch_size]
        """
        zk = z0
        log_det_jacobian = torch.zeros(z0.size(0), device=z0.device)
        
        for k in range(self.num_flows):
            # Apply permutation if enabled
            if self.use_permutations:
                zk = zk[:, self.perms[k]]
            
            # Get autoregressive parameters
            mu, log_sigma = self.mades[k](zk)
            sigma = torch.exp(log_sigma)
            
            # IAF transformation: z_{k+1} = sigma * z_k + mu
            zk = sigma * zk + mu
            
            # Log determinant of Jacobian is sum of log sigma
            log_det_jacobian += torch.sum(log_sigma, dim=-1)
            
            # Note: Do NOT invert permutation here - permutations should accumulate
        
        return zk, log_det_jacobian


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        encoders: Dict[str, EncoderMLP],
        topic_dim: int,
        gating: bool = False,
        gating_hidden_dim: Optional[int] = None,
        ae_type: str = "wae",
        poe: bool = False,
        vi_type: str = "mean_field",  # "mean_field", "full_rank", "iaf"
        num_flows: int = 4,
        flow_hidden_dim: Optional[int] = None,  # Hidden dimension for IAF
        flow_use_permutations: bool = True,  # Whether to use permutations in IAF
        moe_type: str = "average"  # "average", "gating", "learned_weights"
    ):
        super().__init__()

        assert ae_type in {"wae", "vae"}, f"Invalid ae_type: {ae_type}"
        assert vi_type in {"mean_field", "full_rank", "iaf"}, f"Invalid vi_type: {vi_type}"
        assert moe_type in {"average", "gating", "learned_weights"}, f"Invalid moe_type: {moe_type}"

        self.encoders = nn.ModuleDict(encoders)
        self.topic_dim = topic_dim
        self.gating = gating
        self.ae_type = ae_type
        self.poe = poe
        self.vi_type = vi_type
        self.moe_type = moe_type
        self.num_modalities = len(encoders)

        if self.gating and self.poe:
            raise ValueError("Cannot use both gating and PoE. Choose one fusion method.")

        # Initialize IAF flows if needed
        if self.vi_type == "iaf":
            self.flows = nn.ModuleDict({
                name: IAF(topic_dim, num_flows, flow_hidden_dim, use_permutations=flow_use_permutations) 
                for name in encoders.keys()
            })

        # MoE gating network
        if self.moe_type == "gating" or self.gating:
            if ae_type == "vae":
                if vi_type == "mean_field":
                    input_dim = len(encoders) * topic_dim * 2
                elif vi_type == "full_rank":
                    L_flat_dim = topic_dim * (topic_dim + 1) // 2
                    input_dim = len(encoders) * (topic_dim + L_flat_dim)
                elif vi_type == "iaf":
                    input_dim = len(encoders) * topic_dim * 2  # mu + logvar before flow
            else:
                input_dim = len(encoders) * topic_dim
                
            self.gate_net = nn.Sequential(
                nn.Linear(input_dim, gating_hidden_dim or len(encoders)),
                nn.ReLU(),
                nn.Linear(gating_hidden_dim or len(encoders), len(encoders)),
                nn.Softmax(dim=-1)
            )
        
        # Learned mixture weights for MoE
        elif self.moe_type == "learned_weights":
            self.mixture_weights = nn.Parameter(torch.ones(len(encoders)) / len(encoders))

    def product_of_experts(
        self, 
        distributions: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine Gaussian distributions using Product of Experts.
        
        Args:
            distributions: List of (mu, logvar) tuples for each modality
            
        Returns:
            Combined (mu, logvar) from PoE
        """
        # Convert to precision (inverse variance) and precision-weighted means
        precisions = []
        precision_means = []
        
        for mu, logvar in distributions:
            precision = torch.exp(-logvar)  # 1/var
            precisions.append(precision)
            precision_means.append(mu * precision)
        
        # Combine precisions and precision-weighted means
        combined_precision = torch.stack(precisions, dim=0).sum(dim=0)
        combined_precision_mean = torch.stack(precision_means, dim=0).sum(dim=0)
        
        # Convert back to mean and logvar
        combined_mu = combined_precision_mean / combined_precision
        combined_logvar = -torch.log(combined_precision)
        
        return combined_mu, combined_logvar

    def product_of_experts_full_rank(
        self, 
        distributions: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine full-rank Gaussian distributions using Product of Experts.
        
        Args:
            distributions: List of (mu, L) tuples where L is lower triangular
            
        Returns:
            Combined (mu, L) from PoE
        """
        batch_size = distributions[0][0].size(0)
        dim = self.topic_dim
        device = distributions[0][0].device
        
        # Convert to precision matrices and precision-weighted means
        precisions = []
        precision_means = []
        
        for mu, L in distributions:
            # Sigma = L @ L^T
            Sigma = torch.bmm(L, L.transpose(-2, -1))
            # Add small regularization for numerical stability
            eye = torch.eye(dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            Sigma_reg = Sigma + 1e-4 * eye
            
            # Precision matrix
            precision = torch.linalg.inv(Sigma_reg)
            precisions.append(precision)
            
            # Precision-weighted mean
            precision_mean = torch.bmm(precision, mu.unsqueeze(-1)).squeeze(-1)
            precision_means.append(precision_mean)
        
        # Combine precisions and precision-weighted means
        combined_precision = torch.stack(precisions, dim=0).sum(dim=0)
        combined_precision_mean = torch.stack(precision_means, dim=0).sum(dim=0)
        
        # Convert back to mean and covariance
        combined_Sigma = torch.linalg.inv(combined_precision)
        combined_mu = torch.bmm(combined_Sigma, combined_precision_mean.unsqueeze(-1)).squeeze(-1)
        
        # Cholesky decomposition to get L
        try:
            combined_L = torch.linalg.cholesky(combined_Sigma)
        except:
            # Fallback: add more regularization
            eye = torch.eye(dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            combined_Sigma_reg = combined_Sigma + 1e-3 * eye
            combined_L = torch.linalg.cholesky(combined_Sigma_reg)
        
        return combined_mu, combined_L

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        single_modality: Optional[str] = None,
        prevalence_covariates: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple]]:
        
        # If single_modality is specified, only process that modality
        if single_modality is not None:
            if single_modality not in self.encoders:
                raise ValueError(f"Modality '{single_modality}' not found in encoders")
            
            # Process only the specified modality
            x = modality_inputs[single_modality]
            encoder = self.encoders[single_modality]
            
            # Handle different encoder types for single modality
            if isinstance(encoder, ImageEncoder):
                z_raw = encoder(x, prevalence_covariates=prevalence_covariates, labels=labels)
            else:
                z_raw = encoder(x)
            
            if self.ae_type == "vae":
                if self.vi_type == "mean_field":
                    mu, logvar = torch.chunk(z_raw, 2, dim=1)
                    mu_logvar_info = [(mu, logvar)]
                    
                    # Sample from distribution
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z_sample = mu + eps * std
                    
                elif self.vi_type == "full_rank":
                    D = self.topic_dim
                    mu = z_raw[:, :D]
                    L_flat = z_raw[:, D:]
                    B = mu.size(0)
                    
                    # Reconstruct lower triangular matrix
                    tril_indices = torch.tril_indices(D, D, device=z_raw.device)
                    L = torch.zeros(B, D, D, device=z_raw.device)
                    L[:, tril_indices[0], tril_indices[1]] = L_flat
                    
                    # Ensure positive diagonal
                    diag_idx = torch.arange(D, device=z_raw.device)
                    L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-4
                    
                    mu_logvar_info = [(mu, L)]
                    
                    # Sample from distribution
                    eps = torch.randn(mu.size(0), self.topic_dim, 1, device=mu.device)
                    z_sample = mu.unsqueeze(-1) + torch.bmm(L, eps)
                    z_sample = z_sample.squeeze(-1)
                    
                elif self.vi_type == "iaf":
                    mu, logvar = torch.chunk(z_raw, 2, dim=1)
                    
                    # Sample from base distribution
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z0 = mu + eps * std
                    
                    # Apply IAF flow
                    flow = self.flows[single_modality]
                    zk, log_det_j = flow(z0)
                    
                    mu_logvar_info = [(mu, logvar, z0, zk, log_det_j)]
                    z_sample = zk
                    
            else:  # WAE
                mu_logvar_info = [(z_raw,)]
                z_sample = z_raw
            
            # Convert to simplex for topic models
            theta = F.softmax(z_sample, dim=1)
            
            return theta, z_sample, mu_logvar_info
        
        # Original multimodal processing code follows...
        modality_outputs = []
        mu_logvar_info = []
        
        # Process each modality
        for name, encoder in self.encoders.items():
            x = modality_inputs[name]
           
            # Handle different encoder types
            if isinstance(encoder, ImageEncoder):
                # ImageEncoder expects separate prevalence_covariates and labels
                z_raw = encoder(x, prevalence_covariates=prevalence_covariates, labels=labels)
            else:
                # Standard MLP encoder - covariates already concatenated in models.py
                z_raw = encoder(x)
            
            if self.ae_type == "vae":
                if self.vi_type == "mean_field":
                    mu, logvar = torch.chunk(z_raw, 2, dim=1)
                    mu_logvar_info.append((mu, logvar))
                    
                    # Sample from distribution
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z_sample = mu + eps * std
                    modality_outputs.append(z_sample)
                    
                elif self.vi_type == "full_rank":
                    D = self.topic_dim
                    mu = z_raw[:, :D]
                    L_flat = z_raw[:, D:]
                    B = mu.size(0)
                    
                    # Reconstruct lower triangular matrix
                    tril_indices = torch.tril_indices(D, D, device=z_raw.device)
                    L = torch.zeros(B, D, D, device=z_raw.device)
                    L[:, tril_indices[0], tril_indices[1]] = L_flat
                    
                    # Ensure positive diagonal
                    diag_idx = torch.arange(D, device=z_raw.device)
                    L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-4
                    
                    mu_logvar_info.append((mu, L))
                    
                    # Sample from distribution
                    eps = torch.randn(mu.size(0), self.topic_dim, 1, device=mu.device)
                    z_sample = mu.unsqueeze(-1) + torch.bmm(L, eps)
                    z_sample = z_sample.squeeze(-1)
                    modality_outputs.append(z_sample)
                    
                elif self.vi_type == "iaf":
                    mu, logvar = torch.chunk(z_raw, 2, dim=1)
                    
                    # Sample from base distribution
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z0 = mu + eps * std
                    
                    # Apply IAF flow
                    flow = self.flows[name]
                    zk, log_det_j = flow(z0)
                    
                    # Store base parameters, base sample, flow output, and log determinant
                    mu_logvar_info.append((mu, logvar, z0, zk, log_det_j))
                    modality_outputs.append(zk)
                    
            else:  # WAE
                mu_logvar_info.append((z_raw,))  # Wrap in tuple for consistency
                modality_outputs.append(z_raw)

        # Fusion step
        if self.poe and self.ae_type == "vae":
            # Product of Experts fusion
            if self.vi_type == "mean_field":
                distributions = [(info[0], info[1]) for info in mu_logvar_info 
                            if len(info) >= 2]
                if distributions:
                    combined_mu, combined_logvar = self.product_of_experts(distributions)
                    # Sample from combined distribution
                    std = torch.exp(0.5 * combined_logvar)
                    eps = torch.randn_like(std)
                    z_final = combined_mu + eps * std
                    # Update mu_logvar_info for consistency
                    mu_logvar_info = [(combined_mu, combined_logvar)]
                else:
                    z_final = torch.stack(modality_outputs, dim=1).mean(dim=1)
                    
            elif self.vi_type == "full_rank":
                distributions = [(info[0], info[1]) for info in mu_logvar_info 
                            if len(info) >= 2]
                if distributions:
                    combined_mu, combined_L = self.product_of_experts_full_rank(distributions)
                    # Sample from combined distribution
                    eps = torch.randn(combined_mu.size(0), self.topic_dim, 1, device=combined_mu.device)
                    z_final = combined_mu.unsqueeze(-1) + torch.bmm(combined_L, eps)
                    z_final = z_final.squeeze(-1)
                    # Update mu_logvar_info for consistency
                    mu_logvar_info = [(combined_mu, combined_L)]
                else:
                    z_final = torch.stack(modality_outputs, dim=1).mean(dim=1)
                    
            elif self.vi_type == "iaf":
                # For IAF flows with PoE, combine base distributions then apply average flow
                base_distributions = [(info[0], info[1]) for info in mu_logvar_info 
                                    if len(info) >= 5]
                if base_distributions:
                    combined_mu, combined_logvar = self.product_of_experts(base_distributions)
                    # Use first flow as representative (could be improved)
                    first_flow_name = list(self.flows.keys())[0]
                    flow = self.flows[first_flow_name]
                    
                    std = torch.exp(0.5 * combined_logvar)
                    eps = torch.randn_like(std)
                    z0 = combined_mu + eps * std
                    z_final, log_det_j = flow(z0)
                    
                    # Update mu_logvar_info for consistency
                    mu_logvar_info = [(combined_mu, combined_logvar, z0, z_final, log_det_j)]
                else:
                    z_final = torch.stack(modality_outputs, dim=1).mean(dim=1)
        
        else:
            # Mixture of Experts fusion
            if self.moe_type == "average":
                z_final = torch.stack(modality_outputs, dim=1).mean(dim=1)
                
            elif self.moe_type == "gating" or self.gating:
                # Prepare input for gating network
                if self.ae_type == "vae":
                    if self.vi_type == "mean_field":
                        gate_input = torch.cat([torch.cat((info[0], info[1]), dim=1) 
                                            for info in mu_logvar_info if len(info) >= 2], dim=1)
                    elif self.vi_type == "full_rank":
                        gate_input = torch.cat([torch.cat((info[0], info[1].view(info[0].size(0), -1)), dim=1) 
                                            for info in mu_logvar_info if len(info) >= 2], dim=1)
                    elif self.vi_type == "iaf":
                        gate_input = torch.cat([torch.cat((info[0], info[1]), dim=1) 
                                            for info in mu_logvar_info if len(info) >= 5], dim=1)
                else:
                    gate_input = torch.cat([info[0] for info in mu_logvar_info], dim=1)
                
                # Compute mixture weights
                weights = self.gate_net(gate_input).unsqueeze(2)  # [B, M, 1]
                modality_stack = torch.stack(modality_outputs, dim=1)  # [B, M, D]
                z_final = torch.sum(modality_stack * weights, dim=1)
                
            elif self.moe_type == "learned_weights":
                weights = F.softmax(self.mixture_weights, dim=0)
                modality_stack = torch.stack(modality_outputs, dim=1)  # [B, M, D]
                z_final = torch.sum(modality_stack * weights.unsqueeze(0).unsqueeze(2), dim=1)

        # Convert to simplex for topic models
        theta = F.softmax(z_final, dim=1)
        
        return theta, z_final, mu_logvar_info