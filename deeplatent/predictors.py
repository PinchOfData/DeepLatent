import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Dict

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


class MultiLabelPredictor(nn.Module):
    """
    Multi-label predictor that creates independent MLP networks per label.

    Supports three label types:
    - regression: continuous output, 1 output unit
    - binary: binary classification, 1 output unit (logits)
    - multiclass: multi-class classification, num_classes output units (logits)
    """

    def __init__(
        self,
        input_dim: int,
        labels_info: Dict[str, Dict],
        predictor_configs: Dict[str, Dict],
    ):
        """
        Initialize the MultiLabelPredictor.

        Args:
            input_dim: Dimension of input features (n_factors + prediction_covariate_size)
            labels_info: Dict mapping label names to their metadata:
                {
                    "label_name": {
                        "type": "regression|binary|multiclass",
                        "num_classes": int or None,
                        "start_idx": int,
                        "end_idx": int,
                        "column": str
                    }
                }
            predictor_configs: Dict mapping label names to their predictor config:
                {
                    "label_name": {
                        "hidden_dims": [64, 32],
                        "dropout": 0.1,
                        "loss_weight": 1.0
                    }
                }
        """
        super(MultiLabelPredictor, self).__init__()

        self.labels_info = labels_info
        self.predictor_configs = predictor_configs
        self.predictors = nn.ModuleDict()

        for label_name, label_info in labels_info.items():
            config = predictor_configs.get(label_name, {})
            hidden_dims = config.get("hidden_dims", [])
            dropout = config.get("dropout", 0.0)
            activation = config.get("activation", "relu")
            bias = config.get("bias", True)

            # Determine output dimension based on label type
            label_type = label_info["type"]
            if label_type in {"regression", "binary"}:
                output_dim = 1
            elif label_type == "multiclass":
                output_dim = label_info["num_classes"]
            else:
                raise ValueError(f"Unknown label type: {label_type}")

            # Build predictor dims: input -> hidden layers -> output
            predictor_dims = [input_dim] + hidden_dims + [output_dim]

            # Create MLP for this label
            self.predictors[label_name] = Predictor(
                predictor_dims=predictor_dims,
                predictor_non_linear_activation=activation,
                predictor_bias=bias,
                dropout=dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        M_prediction: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all label predictors.

        Args:
            x: Input tensor of shape (batch_size, n_factors)
            M_prediction: Optional prediction covariates of shape (batch_size, prediction_covariate_size)

        Returns:
            Dict mapping label names to prediction tensors.
            Output shapes depend on label type:
            - regression/binary: (batch_size, 1)
            - multiclass: (batch_size, num_classes)
        """
        predictions = {}
        for label_name, predictor in self.predictors.items():
            predictions[label_name] = predictor(x, M_prediction)
        return predictions