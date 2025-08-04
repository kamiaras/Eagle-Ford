"""
training.py

Cross-validation training routine for a 2-output MLP, with optional saving
of models, metadata, and diagnostics.

Function
--------
cross_validate_mlp:
    Performs k-fold CV on the dataset at csv_path, trains an MLP per fold,
    tracks losses and R² on the first output, plots diagnostics, and optionally
    saves results (models, norms, hyperparams, plot).
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime

from .models import MLP
from .utils import get_device, get_loss_fn, load_dataset


def cross_validate_mlp(
    csv_path: str,
    k_folds: int,
    layer_dims: list[int],
    activations: list[str],
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    random_state: int,
    loss_type: str = "mse",
    loss_weights: list[float] = [1.0, 1.0],
    include_ratio_features: bool = True,
    save_results: bool = False,
    save_name: str = None
):
    """
    Perform k-fold cross-validation for a 2-output MLP.

    Parameters
    ----------
    csv_path : str
        Path to the CSV containing features and targets.
    k_folds : int
        Number of CV folds.
    layer_dims : list[int]
        Sizes of hidden layers.
    activations : list[str]
        Activation names for each hidden layer.
    epochs : int
        Number of training epochs per fold.
    lr : float
        Learning rate for Adam optimizer.
    weight_decay : float
        L2 regularization coefficient.
    batch_size : int
        Mini-batch size for training.
    random_state : int
        Seed for reproducibility.
    loss_type : str, default="mse"
        Loss function type ('mse', 'huber', etc.).
    loss_weights : list[float], default=[1.0, 1.0]
        Weights for each output in the loss.
    include_ratio_features : bool, default=True
        Whether to include Proppant/Fluid ratio features.
    save_results : bool, default=False
        If True, saves model checkpoints, norms, hyperparams, and diagnostics.
    save_name : str, optional
        Base name for the output folder (required if save_results=True).

    Returns
    -------
    fold_results : list[dict]
        Per-fold dicts with train/val true and predicted outputs.
    histories : list[tuple]
        Per-fold training histories (train_losses, val_losses, train_r2s, val_r2s).
    models : list[nn.Module]
        Trained model for each fold.
    norms : dict
        Normalization statistics for inputs and outputs.
    feature_names : list[str]
        Names of input features used.
    """
    # 1) Set seeds for reproducibility
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    device = get_device()

    # 2) Load data and normalization: X_np, y_norm, norms, feature_names
    X_np, y_norm, norms, feature_names = load_dataset(
        csv_path, include_ratio_features
    )
    # 3) Prepare loss function
    loss_fn = get_loss_fn(loss_type, loss_weights, device)

    # 4) Set up k-fold splitter
    kf = KFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=random_state
    )

    fold_results = []
    histories = []
    models = []

    # 5) Loop over folds
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_np), start=1):
        # Split normalized data
        X_tr, X_va = X_np[tr_idx], X_np[va_idx]
        y_tr, y_va = y_norm[tr_idx], y_norm[va_idx]

        # Create DataLoaders
        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
        ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        ld_va = DataLoader(ds_va, batch_size=batch_size)

        # Instantiate model and optimizer
        model = MLP(
            in_dim=X_tr.shape[1],
            hidden_dims=layer_dims,
            activations=activations,
            out_dim=y_tr.shape[1]
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        tr_losses, va_losses, tr_r2s, va_r2s = [], [], [], []

        # 6) Training epochs
        for epoch in range(1, epochs + 1):
            # Train mode
            model.train()
            for xb, yb in ld_tr:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

            # Eval mode: compute losses and R² on first output
            model.eval()
            with torch.no_grad():
                p_tr = model(torch.from_numpy(X_tr).to(device))
                p_va = model(torch.from_numpy(X_va).to(device))
            l_tr = loss_fn(p_tr, torch.from_numpy(y_tr).to(device)).item()
            l_va = loss_fn(p_va, torch.from_numpy(y_va).to(device)).item()

            p_tr_np = p_tr.cpu().numpy()
            p_va_np = p_va.cpu().numpy()

            tr_losses.append(l_tr)
            va_losses.append(l_va)
            tr_r2s.append(r2_score(y_tr[:, 0], p_tr_np[:, 0]))
            va_r2s.append(r2_score(y_va[:, 0], p_va_np[:, 0]))

        # 7) Un-normalize first output and compute final R²
        y_mean, y_std = norms["y_mean"], norms["y_std"]
        yt_tr = y_tr[:, 0] * y_std + y_mean
        yp_tr = p_tr_np[:, 0] * y_std + y_mean
        yt_va = y_va[:, 0] * y_std + y_mean
        yp_va = p_va_np[:, 0] * y_std + y_mean

        r2_tr1 = r2_score(yt_tr, yp_tr)
        r2_va1 = r2_score(yt_va, yp_va)
        print(f"Fold {fold}: train R²1={r2_tr1:.4f}, val R²1={r2_va1:.4f}")

        # Store fold results
        fold_results.append({
            "fold": fold,
            "train_true": yt_tr,
            "train_pred": yp_tr,
            "val_true": yt_va,
            "val_pred": yp_va
        })
        histories.append((tr_losses, va_losses, tr_r2s, va_r2s))
        models.append(model)

    # 8) Diagnostic plots: losses and true-vs-pred scatter for each fold
    fig, axes = plt.subplots(
        k_folds, 3,
        figsize=(15, 4 * k_folds),
        squeeze=False
    )
    for i, ((tr_losses, va_losses, tr_r2s, va_r2s), fr) in enumerate(zip(histories, fold_results)):
        ax_loss, ax_tr_sc, ax_va_sc = axes[i]
        fold = fr["fold"]

        # Loss vs. Epoch
        ax_loss.plot(range(1, epochs + 1), tr_losses, label="train")
        ax_loss.plot(range(1, epochs + 1), va_losses, label="val")
        ax_loss.set_title(f"Fold {fold} Loss vs Epoch")
        ax_loss.set_ylabel("Weighted Loss")
        ax_loss.legend()

        # Train scatter
        mn = min(fr["train_true"].min(), fr["train_pred"].min())
        mx = max(fr["train_true"].max(), fr["train_pred"].max())
        ax_tr_sc.scatter(fr["train_true"], fr["train_pred"], s=20, alpha=0.6)
        ax_tr_sc.plot([mn, mx], [mn, mx], "k--")
        ax_tr_sc.set_title(f"Fold {fold} Out1 Train")
        ax_tr_sc.set_xlabel("True Out1")
        ax_tr_sc.set_ylabel("Pred Out1")

        # Val scatter
        mn = min(fr["val_true"].min(), fr["val_pred"].min())
        mx = max(fr["val_true"].max(), fr["val_pred"].max())
        ax_va_sc.scatter(fr["val_true"], fr["val_pred"], s=20, alpha=0.6)
        ax_va_sc.plot([mn, mx], [mn, mx], "k--")
        ax_va_sc.set_title(f"Fold {fold} Out1 Val")
        ax_va_sc.set_xlabel("True Out1")
        ax_va_sc.set_ylabel("Pred Out1")

    plt.tight_layout()
    plt.show()

    # 9) Optionally save models, hyperparams, norms, and diagnostics
    if save_results:
        if not save_name:
            raise ValueError("`save_name` must be provided when save_results=True")
        date_str = datetime.now().strftime("%Y%m%d")
        tag = f"{save_name}_{date_str}"
        os.makedirs(tag, exist_ok=True)

        # Save each fold’s state_dict
        for i, m in enumerate(models, start=1):
            path = os.path.join(tag, f"{tag}_fold{i}.pth")
            torch.save(m.state_dict(), path)

        # Save hyperparameters used
        meta = {
            "csv_path": csv_path,
            "k_folds": k_folds,
            "layer_dims": layer_dims,
            "activations": activations,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "random_state": random_state,
            "loss_type": loss_type,
            "loss_weights": loss_weights,
            "include_ratio_features": include_ratio_features
        }
        with open(os.path.join(tag, f"{tag}_hyperparams.json"), "w") as f:
            json.dump(meta, f, indent=4)

        # Save normalization stats
        with open(os.path.join(tag, f"{tag}_norms.json"), "w") as f:
            json.dump(norms, f, indent=4)

        # Save diagnostic plot
        plot_path = os.path.join(tag, f"{tag}_diagnostics.png")
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Saved diagnostic plot → {plot_path}")
        print(f"All results saved in folder: ./{tag}/")

    return fold_results, histories, models, norms, feature_names
