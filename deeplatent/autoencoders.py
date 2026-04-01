import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Tuple, Dict, Callable, Union
from .priors import FixedGaussianPrior, FixedLogisticNormalPrior, FixedDirichletPrior        

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for conditioning CNNs on covariates."""
    
    def __init__(self, num_features: int, covariate_dim: int, film_hidden_dim: int = 128):
        super().__init__()
        self.film_net = nn.Sequential(
            nn.Linear(covariate_dim, film_hidden_dim),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, 2 * num_features)  # gamma and beta
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
        input_size: Tuple[int, int],
        input_channels: int = 3,
        latent_dim: int = 20,
        prevalence_covariate_size: int = 0,
        labels_size: int = 0,
        include_labels: bool = True,
        hidden_dims: List[int] = [32, 64, 128, 256],
        fc_hidden_dims: List[int] = [512],
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = True,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        pool_size: int = 2,
        film_hidden_dim: int = 128
    ):
        super().__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.prevalence_covariate_size = prevalence_covariate_size
        self.labels_size = labels_size if include_labels else 0
        self.include_labels = include_labels
        self.dropout = nn.Dropout(p=dropout)
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding = padding if padding is not None else kernel_size // 2
        self.pool_size = pool_size
        self.film_hidden_dim = film_hidden_dim
        
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
                nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                self.activation,
                nn.MaxPool2d(self.pool_size)
            ])

            # FiLM layer for covariate conditioning
            if self.use_covariates:
                film_layers.append(FiLMLayer(out_channels, self.covariate_size, self.film_hidden_dim))
            
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
            input_shape = (self.input_channels, *self.input_size)
            
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
        output_size: Tuple[int, int],
        latent_dim: int = 20,
        content_covariate_size: int = 0,
        output_channels: int = 3,
        hidden_dims: List[int] = [256, 128, 64, 32],
        fc_hidden_dims: List[int] = [512],
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = True,
        deconv_kernel_size: int = 4,
        deconv_stride: int = 2,
        deconv_padding: int = 1
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.content_covariate_size = content_covariate_size
        self.output_channels = output_channels
        self.output_size = output_size
        self.hidden_dims = hidden_dims  # Store for use in calculation method
        self.dropout = nn.Dropout(p=dropout)
        self.deconv_kernel_size = deconv_kernel_size
        self.deconv_stride = deconv_stride
        self.deconv_padding = deconv_padding
        
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
        # Dynamically compute based on output size and number of upsampling layers
        self.feature_map_size = self._calculate_initial_feature_map_size()
        self.first_conv_dim = hidden_dims[0]
        fc_output_dim = self.first_conv_dim * self.feature_map_size * self.feature_map_size
        
        fc_layers.append(nn.Linear(prev_dim, fc_output_dim))
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Deconvolutional layers
        deconv_layers = []
        in_channels = hidden_dims[0]
        
        for i, out_channels in enumerate(hidden_dims[1:]):
            deconv_layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=self.deconv_kernel_size,
                                   stride=self.deconv_stride,
                                   padding=self.deconv_padding),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                self.activation
            ])
            in_channels = out_channels

        # Final layer to get desired output channels
        deconv_layers.extend([
            nn.ConvTranspose2d(in_channels, output_channels,
                               kernel_size=self.deconv_kernel_size,
                               stride=self.deconv_stride,
                               padding=self.deconv_padding),
            nn.Sigmoid()  # Normalize output to [0, 1]
        ])
        
        self.deconv_layers = nn.Sequential(*deconv_layers)
    
    def _calculate_initial_feature_map_size(self):
        """
        Calculate the initial feature map size needed to reach target output size.
        Works backwards from output size through the deconvolutional layers.
        """
        # Start with target output size
        h, w = self.output_size
        
        # Work backwards through each deconv layer
        # Each layer uses kernel_size=4, stride=2, padding=1
        # Formula: output_size = (input_size - 1) * stride - 2 * padding + kernel_size
        # Rearranged: input_size = (output_size + 2 * padding - kernel_size) / stride + 1
        
        # Count deconv layers (hidden_dims[1:] + final output layer)
        num_layers = len(self.hidden_dims) - 1 + 1

        current_h, current_w = h, w

        for _ in range(num_layers):
            # Reverse the deconv operation
            current_h = (current_h + 2 * self.deconv_padding - self.deconv_kernel_size) // self.deconv_stride + 1
            current_w = (current_w + 2 * self.deconv_padding - self.deconv_kernel_size) // self.deconv_stride + 1
        
        # Ensure minimum size of 1
        current_h = max(1, current_h)
        current_w = max(1, current_w)
        
        # Use square feature maps (take the larger dimension for better results)
        feature_map_size = max(current_h, current_w)
        
        # Clamp to reasonable bounds
        feature_map_size = max(1, min(feature_map_size, 32))  # Between 1 and 32
        
        return feature_map_size
    
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
        prior: Optional[nn.Module] = None,
        gating: bool = False,
        gating_hidden_dim: Optional[int] = None,
        ae_type: str = "wae",
        poe: Union[bool, str] = False,  # False, "uncorrected", or "corrected"
        vi_type: str = "mean_field",  # "mean_field", "full_rank", "iaf"
        num_flows: int = 4,
        flow_hidden_dim: Optional[int] = None,  # Hidden dimension for IAF
        flow_use_permutations: bool = True,  # Whether to use permutations in IAF
        moe_type: str = "average"  # "average", "gating", "learned_weights"
    ):
        super().__init__()

        assert ae_type in {"wae", "vae", "ae"}, f"Invalid ae_type: {ae_type}"
        assert vi_type in {"mean_field", "full_rank", "iaf"}, f"Invalid vi_type: {vi_type}"
        assert moe_type in {"average", "gating", "learned_weights"}, f"Invalid moe_type: {moe_type}"
        assert poe in {False, "uncorrected", "corrected"}, f"Invalid poe: {poe}"

        self.encoders = nn.ModuleDict(encoders)
        self.topic_dim = topic_dim
        self.prior = prior
        self.gating = gating
        self.ae_type = ae_type
        self.poe = poe
        self.vi_type = vi_type
        self.moe_type = moe_type
        self.num_modalities = len(encoders)

        if self.gating and self.poe:
            raise ValueError("Cannot use both gating and PoE. Choose one fusion method.")

        # Initialize IAF flow if needed (single shared flow for all modalities)
        if self.ae_type == "vae" and self.vi_type == "iaf":
            self.shared_flow = IAF(
                dim=topic_dim,
                num_flows=num_flows,
                hidden_dim=flow_hidden_dim if flow_hidden_dim else max(topic_dim, 16),
                use_permutations=flow_use_permutations
            )
        else:
            self.shared_flow = None

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
        distributions: List[Tuple[torch.Tensor, torch.Tensor]],
        prevalence_covariates: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine Gaussian distributions using Product of Experts.
        Supports both standard (uncorrected) and corrected PoE.
        
        Args:
            distributions: List of (mu, logvar) tuples for each modality
            prevalence_covariates: Covariates for prior parameters (needed for corrected PoE)
            
        Returns:
            Combined (mu, logvar) from PoE
        """
        # Uncorrected PoE (standard, backward compatible)
        if self.poe != "corrected" or self.prior is None:
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
        
        # Corrected PoE: Λ_S = Σ_m Λ_m - (|S|-1)Λ_0
        # Get prior parameters
        B = distributions[0][0].size(0)  # Batch size
        
        has_full_cov = hasattr(self.prior, 'sigma') and not isinstance(
            self.prior, (FixedGaussianPrior, FixedLogisticNormalPrior, FixedDirichletPrior)
        )
        
        if has_full_cov:
            mu_p, Sigma_p = self.prior.get_prior_params(prevalence_covariates, return_full_cov=True)
            Sigma_p_batch = Sigma_p.unsqueeze(0).expand(B, -1, -1)
            Lambda_0_diag = 1.0 / (torch.diagonal(Sigma_p_batch, dim1=1, dim2=2) + 1e-8)
        else:
            mu_p, logvar_p = self.prior.get_prior_params(prevalence_covariates)
            Lambda_0_diag = torch.exp(-logvar_p)
        
        # Ensure prior parameters are properly batched
        if mu_p.size(0) == 1 and B > 1:
            mu_p = mu_p.expand(B, -1)
            Lambda_0_diag = Lambda_0_diag.expand(B, -1)
        
        eta_0 = Lambda_0_diag * mu_p
        
        # Accumulate unimodal natural parameters
        # For corrected PoE: η_m = Λ_m * μ_m where Λ_m comes from (μ_m, logvar_m)
        num_modalities = len(distributions)
        Lambda_sum = torch.zeros_like(Lambda_0_diag)
        eta_sum = torch.zeros_like(eta_0)
        
        for mu_m, logvar_m in distributions:
            # Λ_m is encoded in logvar_m (already includes Λ_0 + Δ_m if corrected PoE)
            Lambda_m = torch.exp(-logvar_m)
            # Natural mean: η_m = Λ_m * μ_m
            eta_m = Lambda_m * mu_m
            Lambda_sum += Lambda_m
            eta_sum += eta_m
        
        # Corrected PoE formula
        Lambda_S = Lambda_sum - (num_modalities - 1) * Lambda_0_diag
        eta_S = eta_sum - (num_modalities - 1) * eta_0
        
        # Ensure positive precision
        Lambda_S = torch.clamp(Lambda_S, min=1e-8)
        
        # Fused posterior mean: μ_S = Λ_S^(-1) * η_S (precision-weighted average)
        mu_S = eta_S / Lambda_S
        logvar_S = -torch.log(Lambda_S)
        
        return mu_S, logvar_S

    def product_of_experts_full_rank(
        self, 
        distributions: List[Tuple[torch.Tensor, torch.Tensor]],
        prevalence_covariates: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine full-rank Gaussian distributions using Product of Experts.
        Supports both standard (uncorrected) and corrected PoE.
        
        Args:
            distributions: List of (mu, L) tuples where L is lower triangular
            prevalence_covariates: Covariates for prior parameters (needed for corrected PoE)
            
        Returns:
            Combined (mu, L) from PoE
        """
        batch_size = distributions[0][0].size(0)
        dim = self.topic_dim
        device = distributions[0][0].device
        eye = torch.eye(dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Uncorrected PoE (standard, backward compatible)
        if self.poe != "corrected" or self.prior is None:
            precisions = []
            precision_means = []
            
            for mu, L in distributions:
                # Sigma = L @ L^T
                Sigma = torch.bmm(L, L.transpose(-2, -1))
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
                combined_Sigma_reg = combined_Sigma + 1e-3 * eye
                combined_L = torch.linalg.cholesky(combined_Sigma_reg)
            
            return combined_mu, combined_L
        
        # Corrected PoE: Λ_S = Σ_m Λ_m - (|S|-1)Λ_0
        # Get prior precision matrix
        has_full_cov = hasattr(self.prior, 'sigma') and not isinstance(
            self.prior, (FixedGaussianPrior, FixedLogisticNormalPrior, FixedDirichletPrior)
        )
        
        if has_full_cov:
            mu_p, Sigma_p = self.prior.get_prior_params(prevalence_covariates, return_full_cov=True)
            Sigma_p_batch = Sigma_p.unsqueeze(0).expand(batch_size, -1, -1)
            Lambda_0 = torch.linalg.inv(Sigma_p_batch + 1e-6 * eye)
        else:
            mu_p, logvar_p = self.prior.get_prior_params(prevalence_covariates)
            Lambda_0_diag = torch.exp(-logvar_p)
            Lambda_0 = torch.diag_embed(Lambda_0_diag)
        
        # Ensure prior parameters are properly batched
        if mu_p.size(0) == 1 and batch_size > 1:
            mu_p = mu_p.expand(batch_size, -1)
            if Lambda_0.dim() == 2:  # [D, D] -> [B, D, D]
                Lambda_0 = Lambda_0.unsqueeze(0).expand(batch_size, -1, -1)
            elif Lambda_0.size(0) == 1:  # [1, D, D] -> [B, D, D]
                Lambda_0 = Lambda_0.expand(batch_size, -1, -1)
        
        eta_0 = torch.bmm(Lambda_0, mu_p.unsqueeze(-1)).squeeze(-1)
        
        # Accumulate modality natural parameters
        num_modalities = len(distributions)
        Lambda_sum = torch.zeros(batch_size, dim, dim, device=device)
        eta_sum = torch.zeros(batch_size, dim, device=device)
        
        for mu_m, L_m in distributions:
            Sigma_m = torch.bmm(L_m, L_m.transpose(-2, -1))
            Lambda_m = torch.linalg.inv(Sigma_m + 1e-6 * eye)
            eta_m = torch.bmm(Lambda_m, mu_m.unsqueeze(-1)).squeeze(-1)
            Lambda_sum += Lambda_m
            eta_sum += eta_m
        
        # Corrected PoE formula
        Lambda_S = Lambda_sum - (num_modalities - 1) * Lambda_0
        eta_S = eta_sum - (num_modalities - 1) * eta_0
        
        # Regularization
        Lambda_S = Lambda_S + 1e-6 * eye
        
        # Convert to mean parameterization
        Sigma_S = torch.linalg.inv(Lambda_S)
        mu_S = torch.bmm(Sigma_S, eta_S.unsqueeze(-1)).squeeze(-1)
        
        try:
            L_S = torch.linalg.cholesky(Sigma_S)
        except:
            Sigma_S_reg = Sigma_S + 1e-3 * eye
            L_S = torch.linalg.cholesky(Sigma_S_reg)
        
        return mu_S, L_S

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        single_modality: Optional[str] = None,
        active_modalities: Optional[List[str]] = None,
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
                    
                    # Apply shared IAF flow
                    zk, log_det_j = self.shared_flow(z0)
                    
                    mu_logvar_info = [(mu, logvar, z0, zk, log_det_j)]
                    z_sample = zk
                    
            else:  # WAE or plain AE
                mu_logvar_info = [(z_raw,)]
                z_sample = z_raw
            
            # Convert to simplex for topic models
            theta = F.softmax(z_sample, dim=1)
            
            return theta, z_sample, mu_logvar_info
        
        # Original multimodal processing code follows...
        modality_outputs = []
        mu_logvar_info = []
        
        # Determine which modalities to process
        if active_modalities is None:
            active_modalities = list(self.encoders.keys())
        
        # Process each active modality
        for name, encoder in self.encoders.items():
            # Skip inactive modalities
            if name not in active_modalities:
                continue
                
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
                    
                    # For corrected PoE with precision increments:
                    # Encoder outputs (μ_m, Δ_m) where Δ_m ≥ 0 is the precision increment
                    # The unimodal posterior is q_m(z|x_m) = N(μ_m, (Λ_0 + Δ_m)^(-1))
                    if self.poe == "corrected" and self.prior is not None:
                        # Get prior precision Λ_0
                        has_full_cov = hasattr(self.prior, 'sigma') and not isinstance(
                            self.prior, (FixedGaussianPrior, FixedLogisticNormalPrior, FixedDirichletPrior)
                        )
                        if has_full_cov:
                            mu_p, Sigma_p = self.prior.get_prior_params(prevalence_covariates, return_full_cov=True)
                            B = mu.size(0)
                            Sigma_p_batch = Sigma_p.unsqueeze(0).expand(B, -1, -1)
                            Lambda_0_diag = 1.0 / (torch.diagonal(Sigma_p_batch, dim1=1, dim2=2) + 1e-8)
                        else:
                            mu_p, logvar_p = self.prior.get_prior_params(prevalence_covariates)
                            Lambda_0_diag = torch.exp(-logvar_p)
                        
                        # Precision increment Δ_m from encoder (logvar encodes -log(Δ_m))
                        Delta_m = torch.exp(-logvar)
                        
                        # Total unimodal precision: Λ_m = Λ_0 + Δ_m
                        Lambda_m = Lambda_0_diag + Delta_m
                        
                        # Convert to logvar for distribution sampling
                        logvar_m = -torch.log(Lambda_m)
                        
                        # Store (μ_m, logvar_m) for PoE fusion
                        # Note: μ_m is the encoder's point estimate, NOT the fused posterior mean
                        mu_logvar_info.append((mu, logvar_m))
                        
                        # Sample from unimodal posterior
                        std = torch.exp(0.5 * logvar_m)
                        eps = torch.randn_like(std)
                        z_sample = mu + eps * std
                    else:
                        # Standard VAE (uncorrected PoE or no PoE)
                        mu_logvar_info.append((mu, logvar))
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
                    
                    # Store base parameters only - will apply shared flow after PoE fusion
                    mu_logvar_info.append((mu, logvar))
                    modality_outputs.append((mu, logvar))
                    
            else:  # WAE or plain AE
                mu_logvar_info.append((z_raw,))  # Wrap in tuple for consistency
                modality_outputs.append(z_raw)

        # Fusion step
        if self.poe and self.ae_type == "vae":
            # Product of Experts fusion (both corrected and uncorrected)
            if self.vi_type == "mean_field":
                distributions = [(info[0], info[1]) for info in mu_logvar_info 
                            if len(info) >= 2]
                if distributions:
                    combined_mu, combined_logvar = self.product_of_experts(
                        distributions,
                        prevalence_covariates=prevalence_covariates
                    )
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
                    combined_mu, combined_L = self.product_of_experts_full_rank(
                        distributions,
                        prevalence_covariates=prevalence_covariates
                    )
                    # Sample from combined distribution
                    eps = torch.randn(combined_mu.size(0), self.topic_dim, 1, device=combined_mu.device)
                    z_final = combined_mu.unsqueeze(-1) + torch.bmm(combined_L, eps)
                    z_final = z_final.squeeze(-1)
                    # Update mu_logvar_info for consistency
                    mu_logvar_info = [(combined_mu, combined_L)]
                else:
                    z_final = torch.stack(modality_outputs, dim=1).mean(dim=1)
                    
            elif self.vi_type == "iaf":
                # For IAF with PoE, combine base distributions then apply shared flow
                base_distributions = [(info[0], info[1]) for info in mu_logvar_info 
                                    if len(info) == 2]
                if base_distributions:
                    combined_mu, combined_logvar = self.product_of_experts(
                        base_distributions,
                        prevalence_covariates=prevalence_covariates
                    )
                    
                    # Sample from fused base distribution
                    std = torch.exp(0.5 * combined_logvar)
                    eps = torch.randn_like(std)
                    z0_fused = combined_mu + eps * std
                    
                    # Apply shared flow to fused base
                    z_final, log_det_j = self.shared_flow(z0_fused)
                    
                    # Update mu_logvar_info for consistency
                    mu_logvar_info = [(combined_mu, combined_logvar, z0_fused, z_final, log_det_j)]
                else:
                    # Fallback if no valid distributions
                    z_final = torch.zeros(prevalence_covariates.size(0) if prevalence_covariates is not None else 1, 
                                        self.topic_dim, device=prevalence_covariates.device if prevalence_covariates is not None else 'cpu')
        
        else:
            # Mixture of Experts fusion
            if self.moe_type == "average":
                # Fuse base distributions first, then apply flow
                if self.ae_type == "vae" and self.vi_type == "iaf":
                    # Average base distribution parameters (fusion)
                    avg_mu = torch.stack([info[0] for info in mu_logvar_info], dim=1).mean(dim=1)
                    avg_logvar = torch.stack([info[1] for info in mu_logvar_info], dim=1).mean(dim=1)
                    
                    # Sample from fused base distribution
                    std = torch.exp(0.5 * avg_logvar)
                    eps = torch.randn_like(std)
                    z0_fused = avg_mu + eps * std
                    
                    # Apply flow to fused sample
                    z_final, log_det_j = self.shared_flow(z0_fused)
                    
                    # Store fused flow information
                    mu_logvar_info = [(avg_mu, avg_logvar, z0_fused, z_final, log_det_j)]
                else:
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
                        # Use base distribution parameters for gating
                        gate_input = torch.cat([torch.cat((info[0], info[1]), dim=1) 
                                            for info in mu_logvar_info if len(info) >= 2], dim=1)
                else:
                    gate_input = torch.cat([info[0] for info in mu_logvar_info], dim=1)
                
                # Compute mixture weights
                weights = self.gate_net(gate_input).unsqueeze(2)  # [B, M, 1]
                
                # Fuse base distributions with learned weights, then apply flow
                if self.ae_type == "vae" and self.vi_type == "iaf":
                    # Weight and fuse base distribution parameters
                    fused_mu = torch.sum(torch.stack([info[0] for info in mu_logvar_info], dim=1) * weights, dim=1)
                    fused_logvar = torch.sum(torch.stack([info[1] for info in mu_logvar_info], dim=1) * weights, dim=1)
                    
                    # Sample from fused base distribution
                    std = torch.exp(0.5 * fused_logvar)
                    eps = torch.randn_like(std)
                    z0_fused = fused_mu + eps * std
                    
                    # Apply flow to fused sample
                    z_final, log_det_j = self.shared_flow(z0_fused)
                    
                    # Store fused flow information
                    mu_logvar_info = [(fused_mu, fused_logvar, z0_fused, z_final, log_det_j)]
                else:
                    modality_stack = torch.stack(modality_outputs, dim=1)  # [B, M, D]
                    z_final = torch.sum(modality_stack * weights, dim=1)
                
            elif self.moe_type == "learned_weights":
                weights = F.softmax(self.mixture_weights, dim=0)
                weights_expanded = weights.unsqueeze(0).unsqueeze(2)  # [1, M, 1]
                
                # Fuse base distributions with learned weights, then apply flow
                if self.ae_type == "vae" and self.vi_type == "iaf":
                    # Weight and fuse base distribution parameters
                    fused_mu = torch.sum(torch.stack([info[0] for info in mu_logvar_info], dim=1) * weights_expanded, dim=1)
                    fused_logvar = torch.sum(torch.stack([info[1] for info in mu_logvar_info], dim=1) * weights_expanded, dim=1)
                    
                    # Sample from fused base distribution
                    std = torch.exp(0.5 * fused_logvar)
                    eps = torch.randn_like(std)
                    z0_fused = fused_mu + eps * std
                    
                    # Apply flow to fused sample
                    z_final, log_det_j = self.shared_flow(z0_fused)
                    
                    # Store fused flow information
                    mu_logvar_info = [(fused_mu, fused_logvar, z0_fused, z_final, log_det_j)]
                else:
                    modality_stack = torch.stack(modality_outputs, dim=1)  # [B, M, D]
                    z_final = torch.sum(modality_stack * weights_expanded, dim=1)

        # Convert to simplex for topic models
        theta = F.softmax(z_final, dim=1)
        
        return theta, z_final, mu_logvar_info