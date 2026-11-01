# Eagle Ford neural-network examples

Python tools for two-output regression with a small neural network. This release includes reusable training, evaluation, and plotting code plus a synthetic example. It contains no original field dataset, trained field model, or saved field result.

## Try the synthetic example

Tested with Python 3.12.14 on Windows.

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python examples/synthetic_demo.py --output-dir artifacts/synthetic-demo
```

On PowerShell, activate with `.venv\Scripts\Activate.ps1`; on macOS/Linux, use `source .venv/bin/activate`. Other platforms have not been tested.

The example creates 48 invented rows, runs two training folds for five epochs each, and saves a synthetic CSV, diagnostic plot, and short report in a new directory. It does not read field data or saved checkpoints. Use a different output directory for another saved run, or omit `--output-dir` to use temporary files. No network or external service is needed after installation.

For a notebook entry point, install JupyterLab separately and open [`notebooks/synthetic_training.ipynb`](notebooks/synthetic_training.ipynb). It runs the same small example and has no saved outputs.

## Code

- [`src/my_package/models.py`](src/my_package/models.py): configurable MLP.
- [`src/my_package/utils.py`](src/my_package/utils.py): loading, normalization, and loss functions.
- [`src/my_package/training.py`](src/my_package/training.py): cross-validation training and diagnostics.
- `evaluation.py`, `io_surfaces.py`, `pf_ratio_surfaces.py`, `interactive_surfaces.py`, and `sensitivity.py`: existing evaluation and response-surface helpers for user-supplied compatible inputs.

The synthetic training path is tested. The legacy evaluation and surface helpers are retained for inspection and adaptation; they are not a fully validated end-to-end workflow. In particular, normalization is fitted before the fold split, saved target statistics describe only the first output, and some legacy helpers expect a different normalization shape or column naming. These limitations are unchanged. See [reproducibility and limitations](docs/REPRODUCIBILITY.md).

Synthetic outputs check software behavior. They are not Eagle Ford measurements, evidence of field prediction quality, or causal treatment effects. Original datasets, notebooks, model parameters, and derived results are excluded; see [data scope](docs/DATA_PROVENANCE.md).
