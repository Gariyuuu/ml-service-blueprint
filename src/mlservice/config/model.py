"""Model configuration: which estimator to build and with what hyperparameters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from mlservice.config.base import YamlConfig

EstimatorName = Literal["logistic_regression", "random_forest", "gradient_boosting"]


class ModelConfig(YamlConfig):
    """Definition of the estimator at the end of the inference pipeline.

    ``estimator`` is a name resolved by :mod:`mlservice.training.model_factory`
    rather than an import path, so a config file can never be used to import
    arbitrary code.
    """

    name: str = Field(description="Logical model name; also the registry key.")
    estimator: EstimatorName = Field(description="Estimator implementation to build.")
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to the estimator constructor.",
    )
    calibrate: bool = Field(
        default=False,
        description="Wrap the estimator in sigmoid probability calibration.",
    )
