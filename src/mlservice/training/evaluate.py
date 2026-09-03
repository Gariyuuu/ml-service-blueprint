"""Evaluation: the metrics that decide whether an artifact is registrable."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_scores(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Threshold-free and threshold-dependent metrics for a binary classifier.

    Both families are recorded on purpose: ranking quality (``roc_auc``,
    ``average_precision``) tells you whether the model learned anything, while
    the thresholded metrics tell you how the deployed decision rule will behave.
    A model can improve on one and regress on the other.
    """
    truth = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "positive_rate": float(predicted.mean()),
        "base_rate": float(truth.mean()),
    }

    # These are undefined when the evaluation slice contains a single class,
    # which happens with very small holdouts; omit rather than emit a fake value.
    if len(np.unique(truth)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(truth, scores))
        metrics["average_precision"] = float(average_precision_score(truth, scores))
        metrics["log_loss"] = float(log_loss(truth, np.clip(scores, 1e-15, 1 - 1e-15)))
        metrics["brier_score"] = float(brier_score_loss(truth, scores))

    return {name: round(value, 6) for name, value in metrics.items()}


def score_distribution(scores: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Coarse summary of the score distribution, stored as a drift baseline."""
    scores = np.asarray(scores, dtype=float)
    counts, edges = np.histogram(scores, bins=bins, range=(0.0, 1.0))
    summary = {
        f"bin_{i:02d}_{edges[i]:.1f}-{edges[i + 1]:.1f}": float(count / max(len(scores), 1))
        for i, count in enumerate(counts)
    }
    summary["mean"] = float(scores.mean()) if len(scores) else 0.0
    summary["p50"] = float(np.percentile(scores, 50)) if len(scores) else 0.0
    summary["p95"] = float(np.percentile(scores, 95)) if len(scores) else 0.0
    return {name: round(value, 6) for name, value in summary.items()}


def choose_threshold(
    y_true: pd.Series | np.ndarray, scores: np.ndarray, *, objective: str = "f1"
) -> float:
    """Pick the decision threshold that maximises ``objective`` on a holdout.

    Call this on the *validation* split only. Tuning the threshold on test data
    leaks the test set into the deployed decision rule and makes the reported
    test metrics optimistic.
    """
    truth = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(np.round(np.linspace(0.05, 0.95, 91), 4))

    scorer = {"f1": f1_score, "precision": precision_score, "recall": recall_score}[objective]
    best_threshold, best_value = 0.5, -1.0
    for candidate in candidates:
        value = float(scorer(truth, (scores >= candidate).astype(int), zero_division=0))
        if value > best_value:
            best_threshold, best_value = float(candidate), value
    return best_threshold
