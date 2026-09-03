#!/usr/bin/env python3
"""Materialise the reference dataset.

The blueprint's example model is a binary classifier on the Wisconsin breast
cancer diagnostic dataset, which ships inside scikit-learn. Two properties make
it the right *infrastructure* fixture: it needs no network (so CI, containers,
and offline clones all behave identically), and it is small enough that a full
train/evaluate/register cycle finishes in seconds.

It is a real dataset with real column names. Replace this script and
``configs/training.yaml`` to point at your own table; nothing downstream of here
knows which dataset it is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

DEFAULT_OUTPUT = Path("data/raw/breast_cancer.csv")


def build_frame() -> pd.DataFrame:
    """Load the bundled dataset into a flat, CSV-friendly frame."""
    bundle = load_breast_cancer(as_frame=True)
    frame: pd.DataFrame = bundle.frame.copy()
    # sklearn's feature names contain spaces; keep them valid as JSON keys and
    # readable as column headers.
    frame.columns = [name.replace(" ", "_") for name in frame.columns]
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        print(f"{args.output} already exists; pass --force to regenerate.")
        return 0

    frame = build_frame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    positives = int(frame["target"].sum())
    print(
        f"wrote {args.output}: {len(frame)} rows x {frame.shape[1]} columns "
        f"({positives} positive, {len(frame) - positives} negative)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
