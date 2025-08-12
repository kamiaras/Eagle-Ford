# Eagle Ford Neural-Surface Explorer

This repository provides tools to train, evaluate, and visualize neural-network models on Eagle Ford production data, including interactive 3D response surfaces and ratio-based analyses.

## Repository Structure

```
.
├── data
│   ├── raw
│   │   └── Eagle Ford Data(Eagle Ford).csv
│   └── processed
│       ├── Eagle Ford Data(Eagle Ford)_train_val.csv
│       ├── Eagle Ford Data(Eagle Ford)_test.csv
│       └── spliting.ipynb
├── notebooks
│   └── relu_withRatio
│       ├── Train_relu_withRatio.ipynb
│       ├── Test_relu_withRatio.ipynb
│       ├── plot3D-FluidVsPf.ipynb
│       ├── plot3D-FluidVsPf_output
│       │   ├── *.csv, *.png, *.html outputs
│       └── nnModelSaved_relu_withRatio_20250804/
│           ├── *_fold*.pth
│           ├── *_norms.json
│           └── *_hyperparams.json
└── src
    └── my_package
        ├── evaluation.py
        ├── interactive_surfaces.py
        ├── models.py
        ├── pf_ratio_surfaces.py
        ├── sensitivity.py
        ├── training.py
        ├── utils.py
        └── io_surfaces.py
```

* **data/raw/**: Original Eagle Ford CSV.
* **data/processed/**: Train/test splits and a notebook demonstrating the split.
* **notebooks/relu\_withRatio/**: End-to-end workflows:

  * **Train\_relu\_withRatio.ipynb**: Cross-validation training.
  * **Test\_relu\_withRatio.ipynb**: Evaluation and plotting on test data.
  * **plot3D-FluidVsPf.ipynb**: Builds and visualizes 3D surfaces; saves CSV/PNG/HTML.
  * **plot3D-FluidVsPf\_output/**: Precomputed surface outputs.
  * **nnModelSaved\_relu\_withRatio\_20250804/**: Saved model checkpoints and metadata.
* **src/my\_package/**: Python package:

  * **utils.py**: Device selection, loss factory, dataset loading/normalization.
  * **models.py**: `MLP` class for configurable feedforward networks.
  * **training.py**: `cross_validate_mlp()` for k-fold CV training, diagnostics, and optional saving.
  * **evaluation.py**: `evaluate_and_plot_mlp()` for test-set evaluation and plots.
  * **pf\_ratio\_surfaces.py**: Generate, save, and plot static P/F-ratio response surfaces.
  * **interactive\_surfaces.py**: Interactive Plotly 3D surfaces per fluid type, with HTML export.
  * **sensitivity.py**: Sensitivity plots of Output1 vs. Total Fluid or P/F ratio.
  * **io\_surfaces.py**: Internal helpers for loading norms, hyperparams, models, and building feature vectors.

---

## Getting Started

1. **Clone** this repository.
2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare data**

   ```bash
   cd data/processed
   # Use spliting.ipynb to generate train/test CSVs
   ```
4. **Train models**
   Open `notebooks/relu_withRatio/Train_relu_withRatio.ipynb`.
5. **Evaluate**
   Run `notebooks/relu_withRatio/Test_relu_withRatio.ipynb`.
6. **Visualize surfaces**
   Run `notebooks/relu_withRatio/plot3D-FluidVsPf.ipynb`.

---

## Dependencies

* Python ≥ 3.10
* PyTorch
* pandas, NumPy, scikit-learn
* matplotlib
* plotly (for interactive notebooks)
* ipywidgets (optional, for notebook interactivity)

---

## Data Splitting

**Script:** `data/processed/splitting.py`
Splits the raw CSV into train+validation and test sets:

1. Reads `data/raw/Eagle Ford Data(Eagle Ford).csv`.
2. Uses `sklearn.model_selection.train_test_split(test_size=0.15, random_state=42)`.
3. Writes:

   * `Eagle Ford Data(Eagle Ford)_train_val.csv`
   * `Eagle Ford Data(Eagle Ford)_test.csv`
4. Prints row counts and file paths.

---

## Notebooks

* **spliting.ipynb**: Demonstrates the train/test split process.
* **Train\_relu\_withRatio.ipynb**:

  * Loads processed data.
  * Normalizes features/targets.
  * Runs 5-fold CV training of an MLP (ReLU activations).
  * Saves per-fold `.pth` checkpoints, `*_norms.json`, `*_hyperparams.json`.
  * Plots loss curves and true vs. predicted scatter.
* **Test\_relu\_withRatio.ipynb**:

  * Loads a specific checkpoint and norms.
  * Evaluates on `*_test.csv`.
  * Prints R² and shows error histogram & scatter plot.
* **plot3D-FluidVsPf.ipynb**:

  * For a selected sample, sweeps Total Fluid & P/F ratio.
  * Uses `pf_ratio_surfaces.py` to generate a grid of predictions.
  * Creates static 3D plots and saves CSV/PNG, and interactive HTML via `interactive_surfaces.py`.

---

## Utilities (`src/my_package/utils.py`)

* **get\_device()**
  Returns `"cuda"` if a GPU is available, otherwise `"cpu"`.
* **get\_loss\_fn(loss\_type, loss\_weights, device)**
  Constructs a weighted multi-output loss (MSE or Smooth L1).
* **load\_dataset(csv\_path, include\_ratio\_features=True)**
  Reads a CSV (first two columns targets, last column categorical), normalizes inputs & outputs, returns `(X_np, y_norm, norms, feature_names)`.

---

## Model Definition (`src/my_package/models.py`)

* **Class**: `MLP(in_dim, hidden_dims, activations, out_dim)`

  * Builds a `nn.Sequential` of `Linear → Activation` layers.
  * Supported activations: `relu`, `tanh`, `sigmoid`, `softplus`.

---

## Cross-Validation Training (`src/my_package/training.py`)

**Function:** `cross_validate_mlp(csv_path, k_folds, layer_dims, activations, epochs, lr, weight_decay, batch_size, random_state, loss_type='mse', loss_weights=[1.0,1.0], include_ratio_features=True, save_results=False, save_name=None)`

Performs k-fold CV training:

1. Loads and normalizes data.
2. Sets up `KFold` splitting.
3. Trains an `MLP` per fold, tracking losses and R² on the first output.
4. Plots diagnostics (loss vs. epoch and true vs. pred scatter).
5. Optionally saves:

   * Model checkpoints (`.pth`),
   * `*_hyperparams.json`,
   * `*_norms.json`,
   * Diagnostic plot PNG in a timestamped folder.

---

## Model Evaluation & Plotting (`src/my_package/evaluation.py`)

**Function:** `evaluate_and_plot_mlp(model_path, norms_json, hidden_dims, activations, test_csv, sample_index=None, include_ratio_features=True)`

* Loads a saved MLP checkpoint and normalization stats.
* Applies model to `test_csv`.
* Prints R² for the first output.
* Displays:

  * Error histogram,
  * True vs. Predicted scatter (optionally highlighting one sample).

---

## P/F-Ratio Surface Generation (`src/my_package/pf_ratio_surfaces.py`)

* **generate\_surface\_data(...)**
  Returns `(fluid_vals, prop_vals, grid)` where `grid` is a pivoted DataFrame of predicted Output₁ over Total.Fluid × Total.proppant .
* **save\_3d\_response\_surfaces\_csv(...)**
  Calls `generate_surface_data` and writes `{run_id}_fold{fold}_surface.csv` with labeled rows/columns.
* **plot\_3d\_response\_surfaces(...)**
  Static Matplotlib 3D surface plot of the grid.

---

## Interactive P/F-Ratio Surfaces (`src/my_package/interactive_surfaces.py`)

**Function:** `interactive_pf_ratio_surface(run_folder, fold, test_csv, sample_index, fluid_range, pf_ratio_range, n_fluid=50, n_ratio=50, include_ratio_features=True, width=800, height=600, save_html=False, out_dir=None, file_prefix=None)`

* Generates interactive Plotly 3D surfaces per fluid type.
* Overlays the true test sample point.
* If `save_html=True`, writes `<prefix>_<fluid>_sample<idx>.html`.

---

## Sensitivity Analysis (`src/my_package/sensitivity.py`)

* **plot\_sensitivity\_auto\_fluid(...)**
  For fixed proppant  levels, sweeps Total.Fluid and plots Output1 curves with true-point overlay.
* **plot\_sensitivity\_by\_pf\_ratio(...)**
  For fixed P/F ratios, sweeps Total.Fluid and plots Output1 curves with true-point overlay.

