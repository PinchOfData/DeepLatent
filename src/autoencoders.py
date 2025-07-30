import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Callable        

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
    
class MultiModalEncoder(nn.Module):
    """
    Product/Mixture of Experts encoder for multi-modal topic models.
    
    Supports WAE (default), VAE, and PoE via `ae_type` and `poe`.
    """
    def __init__(
        self,
        encoders: Dict[str, EncoderMLP],
        topic_dim: int,
        gating: bool = False,
        gating_hidden_dim: Optional[int] = None,
        ae_type: str = "wae",
        poe: bool = False
    ):
        super().__init__()

        assert ae_type in {"wae", "vae"}, f"Invalid ae_type: {ae_type}"
        self.encoders = nn.ModuleDict(encoders)
        self.topic_dim = topic_dim
        self.gating = gating
        self.ae_type = ae_type
        self.poe = poe

        if self.gating and self.poe:
            raise ValueError("Cannot use both gating and PoE. Choose one fusion method.")

        if self.gating:
            input_dim = len(encoders) * topic_dim
            if ae_type == "vae":
                input_dim *= 2  # because we'll concatenate mu and logvar
            if gating_hidden_dim:
                self.gate_net = nn.Sequential(
                    nn.Linear(input_dim, gating_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(gating_hidden_dim, len(encoders)),
                    nn.Softmax(dim=-1)
                )
            else:
                self.gate_net = nn.Sequential(
                    nn.Linear(input_dim, len(encoders)),
                    nn.Softmax(dim=-1)
                )

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        prevalence_covariates: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Returns:
            theta: Softmaxed topic proportions
            z: Latent representation before softmax
            mu_logvar: List of (mu, logvar) pairs for each modality (if VAE), else None
        """
        zs = []
        mu_logvar = [] if self.ae_type == "vae" else None

        for name, encoder in self.encoders.items():
            x = modality_inputs[name]
            if prevalence_covariates is not None:
                x = torch.cat((x, prevalence_covariates), dim=1)

            z = encoder(x)

            if z.dim() == 3 and z.shape[1] == 1:
                z = z.squeeze(1)

            if self.ae_type == "vae":
                mu, logvar = torch.chunk(z, 2, dim=1)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                mu_logvar.append((mu, logvar))

            zs.append(z)

        if self.ae_type == "vae" and self.poe:
            precisions = []
            precision_mu = []

            for mu, logvar in mu_logvar:
                var = torch.exp(logvar)
                precision = 1.0 / var
                precisions.append(precision)
                precision_mu.append(mu * precision)

            total_precision = torch.stack(precisions).sum(dim=0)
            fused_mu = torch.stack(precision_mu).sum(dim=0) / total_precision
            fused_var = 1.0 / total_precision
            fused_logvar = torch.log(fused_var)

            std = torch.sqrt(fused_var)
            eps = torch.randn_like(std)
            z_fused = fused_mu + eps * std

            z_moe = z_fused
            mu_logvar = [(fused_mu, fused_logvar)]

        elif self.gating:
            zs_stack = torch.stack(zs, dim=1)
            if self.ae_type == "vae":
                gate_input = torch.cat([torch.cat((mu, logvar), dim=1) for (mu, logvar) in mu_logvar], dim=1)
            else:
                gate_input = torch.cat(zs, dim=1)
            weights = self.gate_net(gate_input).unsqueeze(2)
            z_moe = torch.sum(zs_stack * weights, dim=1)

        else:
            zs_stack = torch.stack(zs, dim=1)
            z_moe = torch.mean(zs_stack, dim=1)

        theta = F.softmax(z_moe, dim=1)
        return theta, z_moe, mu_logvar
