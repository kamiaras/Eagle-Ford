"""
evaluation.py

Utilities to evaluate a trained MLP model on test data,
including normalization, metric computation, and diagnostic plots.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from .models import MLP
from .utils import get_device


def evaluate_and_plot_mlp(
    model_path: str,
    norms_json: str,
    hidden_dims: list[int],
    activations: list[str],
    test_csv: str,
    sample_index: int = 1,
    include_ratio_features: bool = True
):
    """
    Evaluate a trained MLP model on test data and generate diagnostic plots.

    Parameters
    ----------
    model_path : str
        Path to the saved PyTorch model checkpoint (.pth file).
    norms_json : str
        Path to JSON file containing feature and target normalization stats.
    hidden_dims : list[int]
        Sizes of the hidden layers in the MLP.
    activations : list[str]
        Activation functions for each hidden layer.
    test_csv : str
        Path to the test dataset CSV.
    sample_index : int, optional
        If provided, highlights this test sample on the scatter plot.
    include_ratio_features : bool, default True
        Whether to include the two per-GPI ratio features in the inputs.
    """
    # --- Load normalization parameters ---
    with open(norms_json, "r") as f:
        norms = json.load(f)
    y_mean = np.array(norms["y_mean"], dtype=np.float32)
    y_std = np.array(norms["y_std"], dtype=np.float32)
    feat_mean = norms["x_mean"]  # mapping feature → mean
    feat_std = norms["x_std"]    # mapping feature → stddev

    # --- Read test data and extract true outputs ---
    df = pd.read_csv(test_csv)
    output_cols = list(df.columns[:2])  # assume first two columns are targets
    y_true = df[output_cols].to_numpy(dtype=np.float32)

    # --- Prepare and normalize numeric inputs ---
    numeric_features = list(feat_mean.keys())
    if not include_ratio_features:
        numeric_features = [
            col for col in numeric_features
            if col not in ("Proppant.per.GPI..lb.ft.", "Fluid.per.GPI..gal.ft.")
        ]
    df_num = df[numeric_features].astype(float).copy()
    for col in numeric_features:
        df_num[col] = (df_num[col] - feat_mean[col]) / feat_std[col]

    # --- One-hot encode categorical feature (last column) ---
    cat_col = df.columns[-1]
    df_cat = pd.get_dummies(df[cat_col].astype(str), prefix=cat_col)

    # --- Combine features into final input matrix ---
    feature_order = numeric_features + list(df_cat.columns)
    X_full = pd.concat([df_num, df_cat], axis=1)
    X = X_full.reindex(columns=feature_order, fill_value=0)
    X_np = X.to_numpy(dtype=np.float32)

    # --- Instantiate model and load weights ---
    device = get_device()
    model = MLP(
        in_dim=X_np.shape[1],
        hidden_dims=hidden_dims,
        activations=activations,
        out_dim=y_true.shape[1]
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- Predict (normalized) and un-normalize ---
    with torch.no_grad():
        y_pred_norm = model(torch.from_numpy(X_np).to(device)).cpu().numpy()
    y_pred = y_pred_norm * y_std + y_mean

    # --- Compute and print R² for the first output ---
    r2_first = r2_score(y_true[:, 0], y_pred[:, 0])
    print(f"R² (output 1): {r2_first:.4f}")

    # --- Plot error histogram for the first output ---
    errors = y_pred[:, 0] - y_true[:, 0]
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=30, edgecolor="black")
    plt.title("Output 1 Error Histogram")
    plt.xlabel("Predicted − True")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # --- Plot true vs. predicted scatter for the first output ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6)
    mn = min(y_true[:, 0].min(), y_pred[:, 0].min())
    mx = max(y_true[:, 0].max(), y_pred[:, 0].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="gray")
    if sample_index is not None and 0 <= sample_index < len(y_true):
        plt.scatter(
            y_true[sample_index, 0],
            y_pred[sample_index, 0],
            color="red",
            s=80,
            label=f"Sample {sample_index}"
        )
        plt.legend()
    plt.title("Output 1: True vs. Predicted")
    plt.xlabel("True Output 1")
    plt.ylabel("Predicted Output 1")
    plt.tight_layout()
    plt.show()
