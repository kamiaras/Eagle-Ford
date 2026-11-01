"""
pf_ratio_surfaces.py

Build and visualize P/F-ratio response surfaces for a trained MLP model:

1. Load model hyperparameters, normalization stats, and weights.
2. Construct a normalized feature vector for a specific test sample.
3. Sweep Total.Fluid and Total.Propellant over a grid.
4. Predict Output₁ at each grid point, un-normalize, and assemble into a DataFrame.
5. Save the grid to CSV or plot as a static 3D surface.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .models import MLP
from .utils import get_device

# Column names for ratio features
RATIO_COLS = ["Proppant.per.GPI..lb.ft.", "Fluid.per.GPI..gal.ft."]


def _load_json(run_folder: str, suffix: str) -> dict:
    """
    Find and load the first JSON file in run_folder ending with _{suffix}.json.
    Raises FileNotFoundError if none is found.
    """
    for fn in os.listdir(run_folder):
        if fn.endswith(f"_{suffix}.json"):
            return json.load(open(os.path.join(run_folder, fn), "r"))
    raise FileNotFoundError(f"No *_{suffix}.json found in {run_folder}")


def _load_norms(run_folder: str) -> dict:
    """Load normalization stats (x_mean, x_std, y_mean, y_std)."""
    return _load_json(run_folder, "norms")


def _load_hyperparams(run_folder: str) -> dict:
    """Load model hyperparameters (layer_dims, activations, etc.)."""
    return _load_json(run_folder, "hyperparams")


def _load_model(
    run_folder: str,
    fold: int,
    in_dim: int,
    out_dim: int,
    hyper: dict,
    device: str
) -> torch.nn.Module:
    """
    Instantiate the MLP using saved hyperparams and load the fold-specific weights.
    """
    run_id = Path(run_folder).name
    model = MLP(in_dim, hyper["layer_dims"], hyper["activations"], out_dim).to(device)
    ckpt = os.path.join(run_folder, f"{run_id}_fold{fold}.pth")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _build_base_features(
    test_csv: str,
    sample_index: int,
    include_ratio_features: bool,
    norms: dict
):
    """
    Construct a dict of normalized features for all inputs except the two
    sweep variables (Total.Fluid and Total.Propellant).

    Returns
    -------
    base : dict
        Mapping from feature name → normalized value for this sample.
    feat_names : list[str]
        Ordered list of all features as the model expects them.
    """
    df = pd.read_csv(test_csv)
    row = df.iloc[sample_index]

    # Select numeric columns, optionally dropping ratio features
    num_feats = list(norms["x_mean"].keys())
    if not include_ratio_features:
        num_feats = [c for c in num_feats if c not in RATIO_COLS]
    x_mean, x_std = norms["x_mean"], norms["x_std"]

    # Normalize numeric features
    base = {
        c: (row[c] - x_mean[c]) / x_std[c] 
        for c in num_feats
    }

    # One-hot encode categorical column (last column in CSV)
    cat_col = df.columns[-1]
    all_dummies = pd.get_dummies(df[cat_col].astype(str), prefix=cat_col).columns
    sample_dum = pd.get_dummies([row[cat_col]], prefix=cat_col).iloc[0]

    # Ensure every dummy column is present
    for c in all_dummies:
        base[c] = sample_dum.get(c, 0)

    feat_names = num_feats + list(all_dummies)
    return base, feat_names


def generate_surface_data(
    run_folder: str,
    fold: int,
    test_csv: str,
    sample_index: int,
    fluid_range: tuple[float, float],
    prop_range: tuple[float, float],
    n_fluid: int = 50,
    n_prop: int = 50,
    include_ratio_features: bool = True
):
    """
    Create a grid of predicted Output₁ over (Total.Fluid, Total.Propellant).

    Returns
    -------
    fluid_vals : np.ndarray, shape (n_fluid,)
        Linspace between fluid_range.
    prop_vals : np.ndarray, shape (n_prop,)
        Linspace between prop_range.
    grid : pd.DataFrame
        Pivoted table with index=Total.Propellant, columns=Total.Fluid, values=predictions.
    """
    norms = _load_norms(run_folder)
    hyper = _load_hyperparams(run_folder)

    # Base normalized features for this sample (excluding the two sweep vars)
    base, feat_names = _build_base_features(
        test_csv, sample_index, include_ratio_features, norms
    )

    device = get_device()
    in_dim = len(feat_names)
    out_dim = len(norms["y_mean"])  # typically 2 outputs
    model = _load_model(run_folder, fold, in_dim, out_dim, hyper, device)

    # Create grid points
    fluid_vals = np.linspace(fluid_range[0], fluid_range[1], n_fluid)
    prop_vals  = np.linspace(prop_range[0],  prop_range[1],  n_prop)

    rows = []
    for f in fluid_vals:
        for p in prop_vals:
            # Copy base features and override the two sweep inputs
            vec = base.copy()
            vec["Total.Fluid"]      = (f - norms["x_mean"]["Total.Fluid"])      / norms["x_std"]["Total.Fluid"]
            vec["Total.Propellant"] = (p - norms["x_mean"]["Total.Propellant"]) / norms["x_std"]["Total.Propellant"]

            x_np = np.array([vec[n] for n in feat_names], dtype=np.float32)
            with torch.no_grad():
                y_norm = model(torch.from_numpy(x_np).to(device)).cpu().numpy()
            # Un-normalize first output
            pred = y_norm[0] * norms["y_std"][0] + norms["y_mean"][0]
            rows.append((f, p, pred))

    df_out = pd.DataFrame(rows, columns=["Total.Fluid", "Total.Propellant", "Predicted"])
    grid = df_out.pivot(index="Total.Propellant", columns="Total.Fluid", values="Predicted")
    return fluid_vals, prop_vals, grid


def save_3d_response_surfaces_csv(
    run_folder: str,
    fold: int,
    test_csv: str,
    sample_index: int,
    fluid_range: tuple[float, float],
    prop_range: tuple[float, float],
    n_fluid: int = 50,
    n_prop: int = 50,
    include_ratio_features: bool = True,
    out_dir: str = None
):
    """
    Generate surface data and save to CSV:
      {run_id}_fold{fold}_surface.csv
    The CSV uses "Total.Propellant ↓ / Total.Fluid →" as the index label.
    """
    fluid_vals, prop_vals, grid = generate_surface_data(
        run_folder, fold, test_csv, sample_index,
        fluid_range, prop_range, n_fluid, n_prop, include_ratio_features
    )

    run_id = Path(run_folder).name
    out_dir = out_dir or run_folder
    os.makedirs(out_dir, exist_ok=True)

    fname = f"{run_id}_fold{fold}_surface.csv"
    path = os.path.join(out_dir, fname)
    grid.to_csv(path, index_label="Total.Propellant ↓ / Total.Fluid →")
    print(f"Saved surface CSV → {path}")
    return grid


def plot_3d_response_surfaces(
    *args,
    elev: float = 30,
    azim: float = -60,
    **kwargs
):
    """
    Create a static Matplotlib 3D surface plot from generated surface data.

    Parameters elev, azim control the viewing angle.
    """
    fluid_vals, prop_vals, grid = generate_surface_data(*args, **kwargs)

    X, Y = np.meshgrid(fluid_vals, prop_vals)
    Z = grid.values

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Total.Fluid")
    ax.set_ylabel("Total.Propellant")
    ax.set_zlabel("Predicted Out1")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()
