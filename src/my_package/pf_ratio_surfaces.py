"""
sensitivity.py

Static P/F‐ratio response surface generation and plotting utilities.

This module provides:
  - plot_3d_response_pf_ratio_surface: 
      Build and display one Matplotlib 3D subplot per fluid type, overlaying
      the true sample point. Optionally saves per‐fluid CSVs and a combined PNG.
  - generate_pf_ratio_surface_grids_for_all_fluids:
      Produce pivoted DataFrames of predictions for each fluid type.
  - generate_pf_ratio_surface_data:
      Alias for generating a single fluid’s grid (maintained for compatibility).
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .io_surfaces import (
    _load_norms,
    _load_hyperparams,
    _load_model,
    _build_base_features,
    RATIO_COLS
)
from .utils import get_device


def plot_3d_response_pf_ratio_surface(
    run_folder: str,
    fold: int,
    test_csv: str,
    sample_index: int,
    fluid_range: tuple[float, float],
    pf_ratio_range: tuple[float, float],
    n_fluid: int = 50,
    n_ratio: int = 50,
    include_ratio_features: bool = True,
    elev: float = 30,
    azim: float = -60,
    save_csv: bool = False,
    save_plot: bool = False,
    out_dir: str = None,
    file_prefix: str = None
):
    """
    Plot static 3D P/F‐ratio surfaces for each fluid type.

    1. Loads norms and hyperparams for the run.
    2. Reads the test CSV and extracts the specified sample.
    3. Builds normalized feature names (numeric + one‐hot dummies).
    4. Creates Total.Fluid and P/F Ratio grids.
    5. For each fluid type:
         • Computes predicted Output₁ over the grid.
         • Optionally writes a CSV: `<prefix>_{fluid}_sample{idx}_pf_ratio_surface.csv`.
    6. Plots a 1×N subplot figure (N = number of fluid types):
         • Surface of predictions.
         • Red marker for the true sample on its matching fluid.
    7. Optionally saves the combined figure as PNG:
         `<prefix>_sample{idx}_pf_ratio_surface.png`.
    """
    # Load normalization stats & model hyperparams
    norms = _load_norms(run_folder)
    hyper = _load_hyperparams(run_folder)

    # Read test data and select the sample to overlay
    df = pd.read_csv(test_csv)
    sample = df.iloc[sample_index]
    cat_col = df.columns[-1]                   # fluid type column
    fluid_types = sorted(df[cat_col].astype(str).unique())

    # Build feature name lists
    numeric_feats = list(norms["x_mean"].keys())
    if not include_ratio_features:
        numeric_feats = [c for c in numeric_feats if c not in RATIO_COLS]
    dummy_feats = [f"{cat_col}_{ft}" for ft in fluid_types]
    feat_names = numeric_feats + dummy_feats

    # Precompute normalized numeric values for the sample
    base_numeric = {
        c: (float(sample[c]) - norms["x_mean"][c]) / norms["x_std"][c]
        for c in numeric_feats
    }

    # Instantiate model on appropriate device
    device = get_device()
    model = _load_model(
        run_folder,
        fold,
        in_dim=len(feat_names),
        out_dim=2,
        hyper=hyper,
        device=device
    )

    # Define sweep grids
    fluid_vals = np.linspace(fluid_range[0], fluid_range[1], n_fluid)
    ratio_vals = np.linspace(pf_ratio_range[0], pf_ratio_range[1], n_ratio)

    # Prepare output directory & filename prefix
    run_id = os.path.basename(os.path.normpath(run_folder))
    out_dir = out_dir or run_folder
    prefix = file_prefix or run_id
    os.makedirs(out_dir, exist_ok=True)

    # Compute and optionally save per‐fluid CSVs
    grids = {}
    for ft in fluid_types:
        Z = np.zeros((n_ratio, n_fluid), dtype=float)
        for i, r in enumerate(ratio_vals):
            for j, f in enumerate(fluid_vals):
                p = f * r
                vec = base_numeric.copy()
                # One‐hot for fluid type
                for d in dummy_feats:
                    vec[d] = 1.0 if d == f"{cat_col}_{ft}" else 0.0
                # Normalize swept inputs
                vec["Total.Fluid"] = (f - norms["x_mean"]["Total.Fluid"]) / norms["x_std"]["Total.Fluid"]
                vec["Total.Proppant.Volume"] = (p - norms["x_mean"]["Total.Proppant.Volume"]) / norms["x_std"]["Total.Proppant.Volume"]

                x_np = np.array([vec[n] for n in feat_names], dtype=np.float32)
                x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)
                with torch.no_grad():
                    y_norm = model(x_t).cpu().numpy()
                # Un-normalize first output
                Z[i, j] = float(y_norm[0, 0] * norms["y_std"] + norms["y_mean"])

        grids[ft] = Z

        if save_csv:
            csv_name = f"{prefix}_{ft}_sample{sample_index}_pf_ratio_surface.csv"
            csv_path = os.path.join(out_dir, csv_name)
            pd.DataFrame(Z, index=ratio_vals, columns=fluid_vals)\
              .to_csv(csv_path, index_label="P/F Ratio")
            print(f"Saved CSV for fluid {ft} → {csv_path}")

    # Plot subplots: one 3D surface per fluid type
    fig, axes = plt.subplots(
        1, len(fluid_types),
        figsize=(6 * len(fluid_types), 5),
        subplot_kw={"projection": "3d"}
    )
    for ax, ft in zip(np.atleast_1d(axes), fluid_types):
        Fg, Rg = np.meshgrid(fluid_vals, ratio_vals)
        ax.plot_surface(
            Fg, Rg, grids[ft],
            rstride=1, cstride=1,
            cmap="viridis", edgecolor="none", alpha=0.8
        )
        ax.set_title(ft)
        ax.set_xlabel("Total Fluid")
        ax.set_ylabel("P/F Ratio")
        ax.set_zlabel(df.columns[0])

        # Overlay actual sample if fluid matches
        if str(sample[cat_col]) == ft:
            sf = float(sample["Total.Fluid"])
            sp = float(sample["Total.Proppant.Volume"])
            sr = sp / sf
            y_true = float(sample[df.columns[0]])
            ax.scatter(sf, sr, y_true, color="red", s=50, label="Actual Sample")
            ax.legend()

    fig.suptitle(f"True fluid type: {sample[cat_col]}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_plot:
        plot_name = f"{prefix}_sample{sample_index}_pf_ratio_surface.png"
        plot_path = os.path.join(out_dir, plot_name)
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Saved combined plot → {plot_path}")

    plt.show()


def generate_pf_ratio_surface_grids_for_all_fluids(
    run_folder: str,
    fold: int,
    test_csv: str,
    sample_index: int,
    fluid_range: tuple[float, float],
    pf_ratio_range: tuple[float, float],
    n_fluid: int = 50,
    n_ratio: int = 50,
    include_ratio_features: bool = True
):
    """
    Produce pivoted DataFrames of predicted Output₁ for every fluid type.

    Returns
    -------
    fluid_vals : np.ndarray of length n_fluid
    ratio_vals : np.ndarray of length n_ratio
    grids      : dict[str, pd.DataFrame]
        Each DataFrame has index=P/F Ratio, columns=Total.Fluid, values=predictions.
    """
    norms = _load_norms(run_folder)
    hyper = _load_hyperparams(run_folder)
    df = pd.read_csv(test_csv)
    sample = df.iloc[sample_index]
    cat_col = df.columns[-1]
    fluid_types = sorted(df[cat_col].astype(str).unique())

    # Numeric + dummy feature ordering
    numeric_feats = list(norms["x_mean"].keys())
    if not include_ratio_features:
        numeric_feats = [c for c in numeric_feats if c not in RATIO_COLS]
    dummy_feats = [f"{cat_col}_{ft}" for ft in fluid_types]
    feat_names = numeric_feats + dummy_feats

    # Base normalized numeric for the sample
    base_numeric = {
        c: (float(sample[c]) - norms["x_mean"][c]) / norms["x_std"][c]
        for c in numeric_feats
    }

    device = get_device()
    model = _load_model(run_folder, fold, len(feat_names), 2, hyper, device)

    fluid_vals = np.linspace(fluid_range[0], fluid_range[1], n_fluid)
    ratio_vals = np.linspace(pf_ratio_range[0], pf_ratio_range[1], n_ratio)

    grids = {}
    for ft in fluid_types:
        rows = []
        for f in fluid_vals:
            for r in ratio_vals:
                p = f * r
                vec = base_numeric.copy()
                for d in dummy_feats:
                    vec[d] = 1.0 if d == f"{cat_col}_{ft}" else 0.0
                vec["Total.Fluid"] = (f - norms["x_mean"]["Total.Fluid"]) / norms["x_std"]["Total.Fluid"]
                vec["Total.Proppant.Volume"] = (p - norms["x_mean"]["Total.Proppant.Volume"]) / norms["x_std"]["Total.Proppant.Volume"]

                x_arr = np.array([vec[n] for n in feat_names], dtype=np.float32)
                x_t = torch.from_numpy(x_arr).unsqueeze(0).to(device)
                with torch.no_grad():
                    y_norm = model(x_t).cpu().numpy()
                pred = float(y_norm[0, 0] * norms["y_std"] + norms["y_mean"])
                rows.append((f, r, pred))

        df_out = pd.DataFrame(rows, columns=["Total.Fluid", "P/F Ratio", "Predicted"])
        grids[ft] = df_out.pivot(index="P/F Ratio", columns="Total.Fluid", values="Predicted")

    return fluid_vals, ratio_vals, grids


def generate_pf_ratio_surface_data(
    run_folder: str,
    fold: int,
    test_csv: str,
    sample_index: int,
    fluid_range: tuple[float, float],
    pf_ratio_range: tuple[float, float],
    n_fluid: int = 50,
    n_ratio: int = 50,
    include_ratio_features: bool = True
):
    """
    Legacy alias for generate_pf_ratio_surface_grids_for_all_fluids; returns
    (fluid_vals, ratio_vals, grid) for a single fluid type.
    """
    vals, ratios, grids = generate_pf_ratio_surface_grids_for_all_fluids(
        run_folder, fold, test_csv, sample_index,
        fluid_range, pf_ratio_range,
        n_fluid, n_ratio, include_ratio_features
    )
    # Return only the first fluid’s grid
    first_grid = next(iter(grids.values()))
    return vals, ratios, first_grid
