"""
sensitivity.py

Sensitivity analysis plots for a trained MLP’s first output (Output1),
showing how predictions change when varying Total Fluid or P/F ratio.

Functions
---------
- plot_sensitivity_auto_fluid:
    For each specified Total.Propellant value, sweep Total.Fluid from
    (propellant * fluid_ratio_min) up to propellant, predict Output1,
    and overlay the true sample point.

- plot_sensitivity_by_pf_ratio:
    For each specified P/F ratio, compute Propellant = Fluid / ratio,
    sweep Fluid over a list of values, predict Output1,
    and overlay the true sample point.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from .io_surfaces import _load_norms, _load_hyperparams, _load_model, _build_base_features
from .utils import get_device


def plot_sensitivity_auto_fluid(
    run_folder: str,
    fold: int,
    test_csv: str,
    sample_index: int,
    total_propellant_vals: list[float],
    include_ratio_features: bool = True,
    fluid_ratio_min: float = 0.4
):
    """
    Plot Output1 vs. Total Fluid for fixed Propellant levels.

    Parameters
    ----------
    run_folder : str
        Directory with the saved model, norms, and hyperparams JSONs.
    fold : int
        Cross-validation fold number to load the correct checkpoint.
    test_csv : str
        Path to the test CSV file.
    sample_index : int
        Row index in test_csv denoting the sample to analyze/overlay.
    total_propellant_vals : list[float]
        List of Total.Propellant values to analyze.
    include_ratio_features : bool, default=True
        Whether to include ratio-based features in the feature vector.
    fluid_ratio_min : float, default=0.4
        Minimum fraction of fluid to propellant when sweeping Total.Fluid.
    """
    # 1) Load normalization statistics and hyperparameters
    norms = _load_norms(run_folder)
    hyper = _load_hyperparams(run_folder)

    # 2) Build the normalized "base" feature dict for all non-swept inputs
    base, feat_names = _build_base_features(
        test_csv, sample_index, include_ratio_features, norms
    )

    # 3) Load the model for the specified fold
    device = get_device()
    in_dim = len(feat_names)
    out_dim = len(norms["y_mean"])  # number of outputs, usually 2
    model = _load_model(run_folder, fold, in_dim, out_dim, hyper, device)

    # 4) Read the true Output1 value for overlay on the plot
    df = pd.read_csv(test_csv)
    true_output1 = df.iloc[sample_index, 0]  # first column is Output1

    # 5) Generate curves for each propellant value
    plt.figure(figsize=(8, 5))
    for p in total_propellant_vals:
        # Sweep Fluid from p * fluid_ratio_min up to p
        fluids = np.linspace(p * fluid_ratio_min, p, 50)
        preds = []
        for f in fluids:
            vec = base.copy()
            # Normalize sweep inputs
            vec["Total.Propellant"] = (p - norms["x_mean"]["Total.Propellant"]) / norms["x_std"]["Total.Propellant"]
            vec["Total.Fluid"]      = (f - norms["x_mean"]["Total.Fluid"])      / norms["x_std"]["Total.Fluid"]
            # Build input tensor and predict
            x_in = torch.tensor([vec[n] for n in feat_names], dtype=torch.float32).to(device)
            with torch.no_grad():
                y_norm = model(x_in).cpu().numpy()
            preds.append(y_norm[0, 0] * norms["y_std"][0] + norms["y_mean"][0])
        plt.plot(fluids, preds, label=f"Propellant={p:.0f}")

    # 6) Overlay the true sample point
    true_fluid = df.loc[sample_index, "Total.Fluid"]
    plt.scatter(
        true_fluid, true_output1,
        color="red", s=80, label="True Sample"
    )

    plt.xlabel("Total Fluid")
    plt.ylabel("Predicted Output1")
    plt.title("Sensitivity: Output1 vs Total Fluid\nfor fixed Propellant levels")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sensitivity_by_pf_ratio(
    run_folder: str,
    fold: int,
    test_csv: str,
    sample_index: int,
    fluid_vals: list[float],
    include_ratio_features: bool = True,
    pf_ratio_vals: list[float] = [0.5, 1.0, 1.5]
):
    """
    Plot Output1 vs. Total Fluid for fixed P/F ratio values.

    Parameters
    ----------
    fluid_vals : list[float]
        Total.Fluid values to sweep.
    pf_ratio_vals : list[float], default=[0.5, 1.0, 1.5]
        P/F ratios to analyze; Propellant = Fluid / ratio.
    (Other parameters same as plot_sensitivity_auto_fluid)
    """
    # 1) Load normalization statistics and hyperparameters
    norms = _load_norms(run_folder)
    hyper = _load_hyperparams(run_folder)

    # 2) Build base features
    base, feat_names = _build_base_features(
        test_csv, sample_index, include_ratio_features, norms
    )

    # 3) Load model
    device = get_device()
    in_dim = len(feat_names)
    out_dim = len(norms["y_mean"])
    model = _load_model(run_folder, fold, in_dim, out_dim, hyper, device)

    # 4) Read true Output1 for overlay
    df = pd.read_csv(test_csv)
    true_output1 = df.iloc[sample_index, 0]

    # 5) Generate curves for each P/F ratio
    plt.figure(figsize=(8, 5))
    for r in pf_ratio_vals:
        preds = []
        for f in fluid_vals:
            p = f / r
            vec = base.copy()
            # Normalize
            vec["Total.Fluid"]      = (f - norms["x_mean"]["Total.Fluid"])      / norms["x_std"]["Total.Fluid"]
            vec["Total.Propellant"] = (p - norms["x_mean"]["Total.Propellant"]) / norms["x_std"]["Total.Propellant"]
            x_in = torch.tensor([vec[n] for n in feat_names], dtype=torch.float32).to(device)
            with torch.no_grad():
                y_norm = model(x_in).cpu().numpy()
            preds.append(y_norm[0, 0] * norms["y_std"][0] + norms["y_mean"][0])
        plt.plot(fluid_vals, preds, label=f"P/F Ratio={r:.2f}")

    # 6) Overlay true sample
    true_fluid = df.loc[sample_index, "Total.Fluid"]
    plt.scatter(
        true_fluid, true_output1,
        color="red", s=80, label="True Sample"
    )

    plt.xlabel("Total Fluid")
    plt.ylabel("Predicted Output1")
    plt.title("Sensitivity: Output1 vs Total Fluid\nfor fixed P/F ratios")
    plt.legend()
    plt.tight_layout()
    plt.show()
