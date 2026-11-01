"""
utils.py

Utility functions for device selection, loss creation, and dataset loading/normalization.
"""

import pandas as pd
import numpy as np
import torch
from torch import nn as _nn


def get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_loss_fn(loss_type: str, loss_weights: list[float], device: str):
    """
    Create a weighted loss function for multi-output regression.

    Parameters
    ----------
    loss_type : str
        'mse' for mean-squared error; otherwise uses Smooth L1 loss.
    loss_weights : list[float]
        Weight for each output; length must match number of outputs.
    device : str
        Device string for the weight tensor ('cpu' or 'cuda').

    Returns
    -------
    loss_fn : callable(pred, true) -> torch.Tensor
        Function computing the mean weighted loss across outputs.
    """
    weights = torch.tensor(loss_weights, device=device).view(1, -1)

    def loss_fn(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if loss_type == "mse":
            errors = (pred - true) ** 2
        else:
            errors = _nn.functional.smooth_l1_loss(pred, true, reduction="none")
        # Apply per-output weights and average over batch and outputs
        return (errors * weights).mean()

    return loss_fn


def load_dataset(
    csv_path: str,
    include_ratio_features: bool = True
):
    """
    Load and normalize a dataset for MLP training or evaluation.

    Parameters
    ----------
    csv_path : str
        Path to CSV file where first two columns are targets and remaining
        columns are features (last column is categorical).
    include_ratio_features : bool, default=True
        Whether to include the two per-GPI ratio features in inputs.

    Returns
    -------
    X_np : np.ndarray, shape (N, D)
        Normalized input feature matrix.
    y_norm : np.ndarray, shape (N, 2)
        Normalized target matrix (first two columns of the CSV).
    norms : dict
        {
            'y_mean': float,
            'y_std': float,
            'x_mean': dict(feature->mean),
            'x_std': dict(feature->std)
        }
    feature_names : list[str]
        Ordered list of input feature names corresponding to columns in X_np.
    """
    # Read raw data
    df = pd.read_csv(csv_path)

    # Split target and feature columns
    output_cols = df.columns[:2].tolist()
    input_cols = df.columns[2:].tolist()
    num_cols = input_cols[:-1]  # all except last are numeric
    cat_col = input_cols[-1]    # last column is categorical

    # Optionally drop ratio features
    ratio_cols = ["Proppant.per.GPI..lb.ft.", "Fluid.per.GPI..gal.ft."]
    if not include_ratio_features:
        num_cols = [c for c in num_cols if c not in ratio_cols]

    # Normalize targets
    y = df[output_cols].to_numpy(dtype=np.float32)
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std[y_std == 0] = 1.0
    y_norm = (y - y_mean) / y_std

    # Normalize numeric features
    X_num = df[num_cols].astype(float).copy()
    x_mean = X_num.mean()
    x_std = X_num.std().replace(0, 1.0)
    X_num = (X_num - x_mean) / x_std

    # One-hot encode categorical feature
    X_cat = pd.get_dummies(df[cat_col].astype(str), prefix=cat_col)

    # Combine numeric and categorical
    X = pd.concat([X_num, X_cat], axis=1)
    feature_names = X.columns.tolist()
    X_np = X.to_numpy(dtype=np.float32)

    norms = {
        'y_mean': float(y_mean[0, 0]),
        'y_std': float(y_std[0, 0]),
        'x_mean': x_mean.to_dict(),
        'x_std': x_std.to_dict(),
    }

    return X_np, y_norm, norms, feature_names
