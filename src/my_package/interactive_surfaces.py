"""
interactive_surfaces.py

Generate and display interactive 3D P/F ratio surfaces for each fluid type,
overlay the true test sample, and optionally save the figures as HTML.
"""

import os
import pandas as pd
from plotly.graph_objects import Figure, Surface, Scatter3d
from .pf_ratio_surfaces import generate_pf_ratio_surface_grids_for_all_fluids


def interactive_pf_ratio_surface(
    run_folder: str,
    fold: int,
    test_csv: str,
    sample_index: int,
    fluid_range: tuple[float, float],
    pf_ratio_range: tuple[float, float],
    n_fluid: int = 50,
    n_ratio: int = 50,
    include_ratio_features: bool = True,
    width: int = 800,
    height: int = 600,
    save_html: bool = False,
    out_dir: str = "out_dir",
    file_prefix: str = "file_prefix"
) -> None:
    """
    For each fluid type, display an interactive 3D surface of predicted Output₁
    vs. (Total Fluid, P/F Ratio), overlaying the true sample point.

    Parameters
    ----------
    run_folder : str
        Directory containing model outputs and grid data.
    fold : int
        Cross-validation fold number.
    test_csv : str
        Path to the test dataset CSV.
    sample_index : int
        Index of the test sample to overlay.
    fluid_range : tuple[float, float]
        Min and max Total Fluid values for the grid.
    pf_ratio_range : tuple[float, float]
        Min and max Proppant/Fluid ratio values for the grid.
    n_fluid : int, default 50
        Number of Total Fluid grid points.
    n_ratio : int, default 50
        Number of P/F ratio grid points.
    include_ratio_features : bool, default True
        Include ratio-based features when generating grids.
    width : int, default 800
        Figure width in pixels.
    height : int, default 600
        Figure height in pixels.
    save_html : bool, default False
        If True, save each interactive plot as HTML.
    out_dir : str, optional
        Directory to save HTML files (defaults to run_folder).
    file_prefix : str, optional
        Prefix for saved HTML filenames (defaults to run_folder basename).
    """
    # Derive run identifier and ensure output directory exists if saving HTML
    run_id = os.path.basename(os.path.normpath(run_folder))
    out_dir = out_dir or run_folder
    prefix = file_prefix or run_id
    if save_html:
        os.makedirs(out_dir, exist_ok=True)

    # 1) Generate prediction grids for each fluid type
    fluid_vals, ratio_vals, grids = generate_pf_ratio_surface_grids_for_all_fluids(
        run_folder, fold, test_csv, sample_index,
        fluid_range, pf_ratio_range,
        n_fluid, n_ratio, include_ratio_features
    )

    # 2) Load the test sample to overlay
    df = pd.read_csv(test_csv)
    sample = df.iloc[sample_index]
    cat_col = df.columns[-1]                    # assume last column is fluid type
    true_fluid = str(sample[cat_col])
    z_label = df.columns[0]                     # first column is the target
    total_fluid = float(sample["Total.Fluid"])
    total_prop = float(sample["Total.Proppant.Volume"])
    pf_ratio = total_prop / total_fluid
    true_output = float(sample[z_label])

    # 3) Plot one interactive figure per fluid type
    for fluid_type, grid in grids.items():
        fig = Figure()

        # Surface trace of predicted Output₁ over the grid
        fig.add_trace(Surface(
            x=fluid_vals,
            y=ratio_vals,
            z=grid.values,
            colorscale="Viridis",
            name=f"{fluid_type} Surface"
        ))

        # Overlay the true sample point on its matching fluid surface
        if fluid_type == true_fluid:
            fig.add_trace(Scatter3d(
                x=[total_fluid], y=[pf_ratio], z=[true_output],
                mode="markers",
                marker=dict(color="red", size=6),
                name="True Sample"
            ))

        # Configure titles and axis labels
        fig.update_layout(
            title=f"Interactive 3D Surface — {fluid_type} (True: {true_fluid})",
            scene=dict(
                xaxis_title="Total Fluid",
                yaxis_title="P/F Ratio",
                zaxis_title=z_label
            ),
            width=width,
            height=height
        )

        # Display in notebook or browser
        try:
            fig.show()
        except ValueError:
            fig.show(renderer="browser")

        # Save as HTML if requested
        if save_html:
            html_filename = f"{prefix}_{fluid_type}_sample{sample_index}.html"
            html_path = os.path.join(out_dir, html_filename)
            fig.write_html(html_path)
            print(f"Saved interactive HTML → {html_path}")
