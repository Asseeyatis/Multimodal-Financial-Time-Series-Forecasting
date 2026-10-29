"""Reusable chronological split of the official validation set.

The training set remains untouched.  The official validation set is divided
by target start date into an earlier development-validation portion and a
later holdout portion.  The holdout must not be used for early stopping or
hyperparameter search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hs300_dataset import HS300Dataset


@dataclass(frozen=True)
class ValidationSplit:
    dev_indices: np.ndarray
    holdout_indices: np.ndarray
    target_dates: np.ndarray
    holdout_start_date: int

    @property
    def dev_start_date(self) -> int:
        return int(self.target_dates[self.dev_indices].min())

    @property
    def dev_end_date(self) -> int:
        return int(self.target_dates[self.dev_indices].max())

    @property
    def holdout_end_date(self) -> int:
        return int(self.target_dates[self.holdout_indices].max())


def make_validation_split(
    dataset: HS300Dataset,
    holdout_fraction: float = 0.2,
) -> ValidationSplit:
    """Split a validation dataset chronologically by target start date."""
    if dataset.split != "val":
        raise ValueError("make_validation_split expects HS300Dataset(..., 'val')")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")

    target_dates = np.asarray(dataset.target_start_date, dtype=np.int32)
    unique_dates = np.unique(target_dates)
    holdout_date_count = max(1, int(np.ceil(len(unique_dates) * holdout_fraction)))
    holdout_start_date = int(unique_dates[-holdout_date_count])

    holdout_indices = np.flatnonzero(target_dates >= holdout_start_date)
    dev_indices = np.flatnonzero(target_dates < holdout_start_date)
    if len(dev_indices) == 0 or len(holdout_indices) == 0:
        raise ValueError("split produced an empty development or holdout portion")

    return ValidationSplit(
        dev_indices=dev_indices,
        holdout_indices=holdout_indices,
        target_dates=target_dates,
        holdout_start_date=holdout_start_date,
    )


def batches_for_indices(
    dataset: HS300Dataset,
    indices: np.ndarray,
    batch_size: int,
):
    """Yield sample lists for an exact split without copying the dataset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        yield [dataset[int(index)] for index in selected]


def save_split_metadata(
    package_dir: str | Path = ".",
    output: str | Path = "experiments/validation_split.json",
    holdout_fraction: float = 0.2,
) -> dict:
    """Create and save a small, reproducible description of the split."""
    dataset = HS300Dataset(package_dir, "val")
    try:
        split = make_validation_split(dataset, holdout_fraction)
    finally:
        dataset.close()

    metadata = {
        "source_split": "val",
        "holdout_fraction_by_unique_target_dates": holdout_fraction,
        "development_samples": int(len(split.dev_indices)),
        "holdout_samples": int(len(split.holdout_indices)),
        "development_target_start_date_min": split.dev_start_date,
        "development_target_start_date_max": split.dev_end_date,
        "holdout_target_start_date_min": split.holdout_start_date,
        "holdout_target_start_date_max": split.holdout_end_date,
        "rule": "target_start_date < holdout_start_date is development; otherwise holdout",
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing data/ (defaults to the repository root)",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/validation_split.json"))
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(save_split_metadata(args.package_dir, args.output, args.holdout_fraction), indent=2))
