"""
models.py

Defines the core Multi-Layer Perceptron (MLP) architecture used for regression.
"""

import torch.nn as nn


class MLP(nn.Module):
    """
    A fully-connected feedforward neural network with customizable hidden layers
    and activation functions.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    hidden_dims : list[int]
        Sizes of each hidden layer, in order.
    activations : list[str]
        Activation function names for each hidden layer. Supported:
        'relu', 'tanh', 'sigmoid', 'softplus'.
    out_dim : int
        Number of outputs (e.g., 2 for two-target regression).
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        activations: list[str],
        out_dim: int
    ):
        super().__init__()

        # Build a sequence of Linear → Activation → ... → Linear
        layers = []
        dims = [in_dim] + hidden_dims  # include input dimension first

        # Add each hidden layer and its activation
        for i, h in enumerate(hidden_dims):
            # Linear transformation from dims[i] → dims[i+1]
            layers.append(nn.Linear(dims[i], dims[i+1]))

            # Map activation name to PyTorch module
            act = activations[i].lower()
            if act == 'relu':
                layers.append(nn.ReLU())
            elif act == 'tanh':
                layers.append(nn.Tanh())
            elif act == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif act == 'softplus':
                layers.append(nn.Softplus())
            else:
                # Guard against unsupported activation names
                raise ValueError(f"Unknown activation '{activations[i]}'")

        # Final output layer (no activation)
        layers.append(nn.Linear(dims[-1], out_dim))

        # Combine all layers into a Sequential module
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, in_dim)
            Input feature tensor.

        Returns
        -------
        torch.Tensor, shape (batch_size, out_dim)
            Raw network outputs (no activation applied).
        """
        return self.net(x)
