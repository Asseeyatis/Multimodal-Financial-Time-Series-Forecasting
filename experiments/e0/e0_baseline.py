"""E0 baselines for the HS300 forecasting task.

The baselines predict all 24 future standardized closes with either the last
standardized close in the input window or a recent moving average.  When
requested, the official validation split is divided chronologically into an
earlier tuning portion and a later pseudo-test portion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hs300_dataset import HS300Dataset


HORIZONS = (1, 6, 12, 24)


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Return aggregate and selected-horizon MAE/MSE for [N, 24] arrays."""
    difference = prediction - target
    result = {
        "samples": int(target.shape[0]),
        "mae": float(np.abs(difference).mean()),
        "mse": float(np.square(difference).mean()),
    }
    for horizon in HORIZONS:
        error = difference[:, horizon - 1]
        result[f"mae_h{horizon}"] = float(np.abs(error).mean())
        result[f"mse_h{horizon}"] = float(np.square(error).mean())
    return result


def _collect(dataset: HS300Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build last-value, moving-average inputs and targets from a split."""
    n = len(dataset)
    last = np.empty(n, dtype=np.float32)
    close_means = np.empty((n, 20), dtype=np.float32)
    targets = np.empty((n, 24), dtype=np.float32)
    target_dates = np.empty(n, dtype=np.int32)

    for index in range(n):
        sample = dataset[index]
        close = sample["numerical"][:, 3].astype(np.float32)
        mean = np.float32(sample["close_mean"])
        std = max(np.float32(sample["close_std"]), np.float32(1e-6))
        standardized_close = (close - mean) / std
        last[index] = standardized_close[-1]
        for column, window in enumerate(range(1, 21)):
            close_means[index, column] = standardized_close[-window:].mean()
        targets[index] = np.asarray(sample["y"], dtype=np.float32)
        target_dates[index] = np.int32(sample["target_start_date"])

    return last, close_means, targets, target_dates


def _evaluate(
    last: np.ndarray,
    recent_means: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Evaluate last-value and moving-average baselines on a boolean mask."""
    target = target[mask]
    result = {"last_value": _metrics(np.repeat(last[mask, None], 24, axis=1), target)}
    for window in (5, 10, 20):
        baseline = recent_means[mask, window - 1]
        prediction = np.repeat(baseline[:, None], 24, axis=1)
        result[f"moving_average_{window}"] = _metrics(prediction, target)
    return result


def run(package_dir: Path, split: str, pseudo_test_fraction: float) -> dict:
    if not 0.0 <= pseudo_test_fraction < 1.0:
        raise ValueError("pseudo-test fraction must be in [0, 1)")

    dataset = HS300Dataset(package_dir, split)
    try:
        last, recent_means, target, target_dates = _collect(dataset)
    finally:
        dataset.close()

    unique_dates = np.unique(target_dates)
    if pseudo_test_fraction == 0.0:
        tuning_mask = np.ones(len(target_dates), dtype=bool)
        pseudo_test_mask = np.zeros(len(target_dates), dtype=bool)
        cutoff = None
    else:
        test_dates = max(1, int(np.ceil(len(unique_dates) * pseudo_test_fraction)))
        cutoff = int(unique_dates[-test_dates])
        pseudo_test_mask = target_dates >= cutoff
        tuning_mask = ~pseudo_test_mask

    result = {
        "split": split,
        "pseudo_test_fraction": pseudo_test_fraction,
        "pseudo_test_cutoff_target_start_date": cutoff,
        "tuning_target_start_date_min": int(target_dates[tuning_mask].min()),
        "tuning_target_start_date_max": int(target_dates[tuning_mask].max()),
        "pseudo_test_target_start_date_min": (
            int(target_dates[pseudo_test_mask].min()) if pseudo_test_mask.any() else None
        ),
        "pseudo_test_target_start_date_max": (
            int(target_dates[pseudo_test_mask].max()) if pseudo_test_mask.any() else None
        ),
        "tuning": _evaluate(last, recent_means, target, tuning_mask),
        "pseudo_test": _evaluate(last, recent_means, target, pseudo_test_mask)
        if pseudo_test_mask.any()
        else {},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root containing data/ (defaults to the repository root)",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--pseudo-test-fraction",
        type=float,
        default=0.2,
        help="fraction of the latest dates reserved as chronological pseudo-test data",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run(args.package_dir, args.split, args.pseudo_test_fraction)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
