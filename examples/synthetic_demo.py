"""Exercise the existing two-output training path using invented numeric data."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.my_package.training import cross_validate_mlp
from src.my_package.utils import load_dataset


def run(output: Path) -> dict:
    rng = np.random.default_rng(7)
    torch.set_num_threads(1)
    n = 48
    a, b = rng.uniform(1, 3, (2, n))
    # Existing loader schema: two targets, numeric features, one categorical column.
    frame = pd.DataFrame({"synthetic_target_1": 2 * a + b,
                          "synthetic_target_2": a - 2 * b,
                          "synthetic_a": a, "synthetic_b": b,
                          "Proppant.per.GPI..lb.ft.": a / b,
                          "Fluid.per.GPI..gal.ft.": b / a,
                          "synthetic_group": np.where(np.arange(n) % 2, "A", "B")})
    csv = output / "synthetic-data.csv"
    frame.to_csv(csv, index=False)
    x_with, y, _, features_with = load_dataset(str(csv), True)
    x_without, _, _, features_without = load_dataset(str(csv), False)
    assert x_with.shape == (n, 6) and x_without.shape == (n, 4)
    assert y.shape == (n, 2) and np.isfinite(x_with).all() and np.isfinite(y).all()
    assert len(features_with) - len(features_without) == 2
    folds, histories, models, norms, features = cross_validate_mlp(
        csv_path=str(csv), k_folds=2, layer_dims=[8], activations=["relu"],
        epochs=5, lr=0.01, weight_decay=0.0, batch_size=12, random_state=7,
        save_results=False, show_plots=False,
    )
    assert len(folds) == len(histories) == len(models) == 2
    for fold, history in zip(folds, histories):
        assert len(fold["val_pred"]) == n // 2
        assert np.isfinite(fold["val_pred"]).all()
        assert all(len(series) == 5 and np.isfinite(series).all() for series in history)
    plt.gcf().suptitle("Synthetic smoke test only; not field validation")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output / "synthetic-diagnostics.png", dpi=100)
    plt.close("all")
    report = {"synthetic": True, "seed": 7, "rows": n, "folds": 2, "epochs_per_fold": 5,
              "input_features": features, "checks": "CSV schema; ratio feature switch; two finite training folds",
              "scope": "Tests the existing training path, not the saved field models or response surfaces.",
              "known_limitations": ["The existing training routine normalizes all rows before splitting folds.",
                                    "Saved normalization metadata describes only the first target.",
                                    "Evaluation builds categories from its own input rows; category sets must match training."]}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory for synthetic data, figure and report.")
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = run(args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="eagle-synthetic-") as folder:
            report = run(Path(folder))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
