import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union

class Predictor(nn.Module):
    """
    Torch implementation of a neural network for supervised learning.
    The neural network for prediction is a Multilayer Perceptron defined by users.
    """

    def __init__(
        self,
        predictor_dims: List[int] = [20, 512, 1024],
        predictor_non_linear_activation: Optional[str] = "relu",
        predictor_bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize the Predictor neural network.

        Args:
            predictor_dims (List[int]): List of integers, where the first element is the dimension of the input,
                                        and the last element is the dimension of the output.
            predictor_non_linear_activation (Optional[str]): 'relu' or 'sigmoid'. Non-linear activation function for the hidden layers.
            predictor_bias (bool): Whether to include bias terms in the hidden layers.
            dropout (float): Dropout rate.
        """
        super(Predictor, self).__init__()
        self.predictor_dims = predictor_dims
        self.predictor_non_linear_activation = predictor_non_linear_activation
        self.predictor_bias = predictor_bias
        self.dropout = nn.Dropout(p=dropout)

        if predictor_non_linear_activation is not None:
            self.encoder_nonlin = {"relu": F.relu, "sigmoid": torch.sigmoid}[
                predictor_non_linear_activation
            ]
        if predictor_non_linear_activation is not None:
            self.decoder_nonlin = {"relu": F.relu, "sigmoid": torch.sigmoid}[
                predictor_non_linear_activation
            ]

        self.neural_net = nn.ModuleDict(
            {
                f"pred_{i}": nn.Linear(
                    predictor_dims[i], predictor_dims[i + 1], bias=predictor_bias
                )
                for i in range(len(predictor_dims) - 1)
            }
        )

    def forward(self, x: torch.Tensor, M_prediction: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the Predictor neural network.

        Args:
            x (torch.Tensor): Input tensor.
            M_prediction (Optional[torch.Tensor]): Optional additional input tensor to be concatenated with x.

        Returns:
            torch.Tensor: Output tensor after passing through the neural network.
        """
        if M_prediction is not None:
            hid = torch.cat((x, M_prediction), dim=1)
        else:
            hid = x

        for i, (_, layer) in enumerate(self.neural_net.items()):
            hid = self.dropout(layer(hid))
            if (
                i < len(self.neural_net) - 1
                and self.predictor_non_linear_activation is not None
            ):
                hid = self.encoder_nonlin(hid)

        return hid