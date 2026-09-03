"""Dataset loading and deterministic splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlservice.config.training import DataConfig, SplitConfig


@dataclass(frozen=True)
class Split:
    """A three-way partition of features and labels."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_validation: pd.DataFrame
    y_validation: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.x_train),
            "validation": len(self.x_validation),
            "test": len(self.x_test),
        }


def load_dataset(config: DataConfig) -> pd.DataFrame:
    """Read the training table and drop columns excluded by config."""
    path = Path(config.path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Training data not found at {path}. Run `make data` (scripts/make_dataset.py) first."
        )
    frame = pd.read_csv(path)
    if config.target not in frame.columns:
        raise KeyError(
            f"Target column '{config.target}' not in {path}; found {list(frame.columns)[:10]}..."
        )
    missing_drops = set(config.drop_columns) - set(frame.columns)
    if missing_drops:
        raise KeyError(f"drop_columns references absent column(s): {sorted(missing_drops)}")
    return frame.drop(columns=config.drop_columns)


def split_dataset(frame: pd.DataFrame, data_config: DataConfig, split_config: SplitConfig) -> Split:
    """Partition into train/validation/test.

    Two nested stratified splits driven by a single seed. Given the same input
    file and the same ``random_state``, the row membership of every partition is
    byte-for-byte identical across machines and runs — which is what makes a
    metric comparison between two model versions meaningful.
    """
    labels = frame[data_config.target]
    features = frame.drop(columns=[data_config.target])

    stratify_all = labels if split_config.stratify else None
    x_remaining, x_test, y_remaining, y_test = train_test_split(
        features,
        labels,
        test_size=split_config.test_size,
        random_state=split_config.random_state,
        stratify=stratify_all,
        shuffle=True,
    )

    # validation_size is expressed as a fraction of the *original* frame, so
    # rescale it against what is left after the test split was carved out.
    validation_fraction = split_config.validation_size / (1.0 - split_config.test_size)
    stratify_remaining = y_remaining if split_config.stratify else None
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_remaining,
        y_remaining,
        test_size=validation_fraction,
        random_state=split_config.random_state,
        stratify=stratify_remaining,
        shuffle=True,
    )

    return Split(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
    )


def class_balance(labels: pd.Series) -> dict[str, float]:
    """Fraction of rows per class, keyed by stringified label."""
    counts = labels.value_counts(normalize=True)
    return {str(key): float(value) for key, value in counts.items()}
