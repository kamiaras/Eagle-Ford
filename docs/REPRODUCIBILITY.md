# Reproduce the synthetic workflow

Use the command in the [README](../Readme.md). `requirements.txt` pins the direct dependencies tested with Python 3.12.14 on Windows. `requirements-lock-windows-py312.txt` records the installed dependency closure for that platform. JupyterLab is optional and is not included in those dependency files.

The generator in [`examples/synthetic_demo.py`](../examples/synthetic_demo.py) uses seed 7 and invented formulas. It creates two targets, four numeric features, and one categorical feature with generic A/B labels. Two feature names match the existing loader's ratio-feature switch; their values are synthetic and have no physical calibration.

The example checks the input shapes, the ratio-feature switch, and finite training results for two folds and five epochs. It writes only to a new requested directory or a temporary directory. It never reads original project data, notebook outputs, or model checkpoints.

## Known limitations

- The training routine normalizes all rows before splitting folds. Its fold metrics are not leakage-free estimates of generalization.
- Saved normalization metadata contains statistics for only the first target. Evaluation and surface code primarily concern the first output; do not interpret the second output using those saved statistics.
- Evaluation derives one-hot categories from its own input rows. Category sets and order must match training.
- Some legacy surface and sensitivity helpers expect array-shaped target statistics, while the loader returns scalars. Some helpers use `Total.Propellant`; others use `Total.Proppant.Volume`. Those interfaces need reconciliation before a compatible end-to-end analysis is claimed.
- The legacy surface helpers are not exercised by this small training example. Their code is retained without scientific-method changes.

No field-data accuracy, well-level behavior, treatment effect, or engineering recommendation is established by this example. Inspect and validate any new workflow on data you are authorized to use.

## Notebook

[`notebooks/synthetic_training.ipynb`](../notebooks/synthetic_training.ipynb) is an output-free entry point for the same generator. Start its kernel in the repository root or `notebooks/`. It uses temporary files, so it does not create a dataset inside the tracked tree. If you execute and save a notebook, check its outputs and metadata before committing it.
