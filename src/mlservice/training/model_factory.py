"""Estimator construction from a config name.

Config files name estimators; they never carry import paths. That keeps a YAML
file from being an arbitrary-code-execution vector and makes the supported set
of models explicit and reviewable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlservice.config.model import ModelConfig

_BUILDERS: dict[str, Callable[..., BaseEstimator]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

#: Estimators that accept a ``random_state``; used to inject the global seed.
_SEEDED = {"logistic_regression", "random_forest", "gradient_boosting"}


def build_estimator(config: ModelConfig, seed: int) -> BaseEstimator:
    """Instantiate the configured estimator with the run seed applied."""
    builder = _BUILDERS.get(config.estimator)
    if builder is None:
        raise ValueError(
            f"Unknown estimator '{config.estimator}'. "
            f"Register it in mlservice.training.model_factory. "
            f"Known: {sorted(_BUILDERS)}"
        )

    kwargs: dict[str, Any] = dict(config.hyperparameters)
    if config.estimator in _SEEDED:
        kwargs.setdefault("random_state", seed)

    try:
        estimator = builder(**kwargs)
    except TypeError as exc:
        raise ValueError(f"Invalid hyperparameters for '{config.estimator}': {exc}") from exc

    if config.calibrate:
        # cv=3 keeps calibration cheap; raise it for small datasets where the
        # calibration folds would otherwise be tiny.
        estimator = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
    return estimator


def available_estimators() -> list[str]:
    return sorted(_BUILDERS)
