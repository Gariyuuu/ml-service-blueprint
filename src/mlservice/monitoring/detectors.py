"""A reference drift detector, to make the interface concrete.

One detector, one statistic: Population Stability Index on the predicted score
distribution. Score drift is the highest-value single signal because it needs no
labels and no feature store — the model tells you it is behaving differently
before anyone can tell you it is wrong.

Anything more (per-feature drift, label drift, delayed-outcome monitoring)
belongs in a dedicated system reading from a :class:`DriftSink`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlservice.artifacts.metadata import ArtifactMetadata
from mlservice.monitoring.base import DriftDetector, DriftSignal
from mlservice.monitoring.records import PredictionRecord

#: Conventional PSI reading: <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant.
PSI_WARN = 0.1
PSI_ALERT = 0.25

_EPSILON = 1e-6


def population_stability_index(baseline: np.ndarray, observed: np.ndarray, bins: int = 10) -> float:
    """PSI between two score distributions over ``bins`` equal-width [0,1] buckets.

    Both distributions are floored at ``_EPSILON`` before the log, because an
    empty baseline bucket would otherwise make PSI infinite the first time a new
    score range appears.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    baseline_share = _shares(baseline, edges)
    observed_share = _shares(observed, edges)
    ratio = observed_share / baseline_share
    return float(np.sum((observed_share - baseline_share) * np.log(ratio)))


def _shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), 1.0 / len(counts))
    shares: np.ndarray = np.maximum(counts / total, _EPSILON)
    return shares


class ScoreDistributionDrift(DriftDetector):
    """Compare live scores against the baseline stored in artifact metadata."""

    def __init__(
        self,
        baseline_scores: np.ndarray,
        *,
        warn: float = PSI_WARN,
        alert: float = PSI_ALERT,
        min_records: int = 200,
    ) -> None:
        self.baseline_scores = np.asarray(baseline_scores, dtype=float)
        self.warn = warn
        self.alert = alert
        self.min_records = min_records

    @classmethod
    def from_metadata(
        cls,
        metadata: ArtifactMetadata,
        *,
        warn: float = PSI_WARN,
        alert: float = PSI_ALERT,
        min_records: int = 200,
    ) -> ScoreDistributionDrift:
        """Rebuild the training-time score histogram recorded by the trainer.

        The artifact stores bucket *shares*, not raw scores, so this expands them
        back into a synthetic sample at bucket midpoints. That is exact for PSI,
        which only ever looks at bucket shares.
        """
        shares = {
            key.removeprefix("score_dist."): value
            for key, value in metadata.metrics.items()
            if key.startswith("score_dist.bin_")
        }
        if not shares:
            raise ValueError(
                f"{metadata.model_name}:{metadata.version} has no score_dist.* baseline; "
                "it was trained before score distributions were recorded."
            )
        sample: list[float] = []
        for key, share in sorted(shares.items()):
            low, high = key.rsplit("_", 1)[1].split("-")
            midpoint = (float(low) + float(high)) / 2.0
            sample.extend([midpoint] * int(round(share * 10_000)))
        return cls(
            np.array(sample or [0.5], dtype=float),
            warn=warn,
            alert=alert,
            min_records=min_records,
        )

    def evaluate(self, records: Sequence[PredictionRecord]) -> list[DriftSignal]:
        observed = np.array([record.score for record in records], dtype=float)
        if len(observed) < self.min_records:
            return [
                DriftSignal(
                    name="score_psi",
                    value=0.0,
                    threshold=self.warn,
                    severity="ok",
                    detail={
                        "reason": "insufficient_data",
                        "n": len(observed),
                        "min_records": self.min_records,
                    },
                )
            ]

        psi = population_stability_index(self.baseline_scores, observed)
        severity = "alert" if psi >= self.alert else "warn" if psi >= self.warn else "ok"
        return [
            DriftSignal(
                name="score_psi",
                value=round(psi, 6),
                threshold=self.warn,
                severity=severity,
                detail={
                    "n": len(observed),
                    "observed_mean": round(float(observed.mean()), 6),
                    "baseline_mean": round(float(self.baseline_scores.mean()), 6),
                    "alert_threshold": self.alert,
                },
            )
        ]


__all__ = ["PSI_ALERT", "PSI_WARN", "ScoreDistributionDrift", "population_stability_index"]
