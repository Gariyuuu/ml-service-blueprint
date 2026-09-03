"""Shared fixtures.

Two datasets are used on purpose:

* a small synthetic frame with numeric, categorical, and boolean columns, for
  fast unit tests of the schema and preprocessing paths;
* the real reference dataset, for the slower end-to-end training tests.

Model training is session-scoped: fitting once and reusing the artifact keeps
the whole suite in the seconds range.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlservice.artifacts.artifact import ModelArtifact
from mlservice.config.model import ModelConfig
from mlservice.config.observability import ObservabilityConfig
from mlservice.config.service import ServiceConfig
from mlservice.config.training import DataConfig, EvaluationGates, SplitConfig, TrainingConfig
from mlservice.registry.local import LocalFilesystemRegistry
from mlservice.serving.app import create_app
from mlservice.training.pipeline import run_training

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "test-classifier"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    """A deterministic mixed-type frame with a genuinely learnable signal."""
    rng = np.random.default_rng(1234)
    n = 600
    age = rng.normal(45, 12, n).round(1)
    income = rng.lognormal(10.5, 0.4, n).round(2)
    region = rng.choice(["north", "south", "east", "west"], n, p=[0.4, 0.3, 0.2, 0.1])
    subscribed = rng.random(n) < 0.3

    logit = (
        -4.0
        + 0.05 * (age - 45)
        + 0.8 * subscribed
        + np.where(region == "north", 0.9, 0.0)
        + rng.normal(0, 0.5, n)
    )
    target = (1 / (1 + np.exp(-logit)) > 0.35).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "region": region,
            "subscribed": subscribed,
            "target": target,
        }
    )


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory, synthetic_frame: pd.DataFrame) -> Path:
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    synthetic_frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def training_config(
    synthetic_csv: Path, tmp_path_factory: pytest.TempPathFactory
) -> TrainingConfig:
    return TrainingConfig(
        seed=7,
        data=DataConfig(path=str(synthetic_csv), target="target"),
        split=SplitConfig(test_size=0.2, validation_size=0.2, random_state=7),
        # No gates: these fixtures exist to exercise plumbing, and a synthetic
        # dataset's metrics are not a quality bar worth enforcing.
        gates=EvaluationGates(),
        registry_root=str(tmp_path_factory.mktemp("registry-config")),
    )


@pytest.fixture(scope="session")
def model_config() -> ModelConfig:
    return ModelConfig(
        name=MODEL_NAME,
        estimator="logistic_regression",
        hyperparameters={"max_iter": 500},
    )


@pytest.fixture(scope="session")
def _fitted_artifact(training_config: TrainingConfig, model_config: ModelConfig) -> ModelArtifact:
    """Fit exactly once per session. Not handed to tests — see `trained_artifact`."""
    result = run_training(training_config, model_config, register=False, tune_threshold=True)
    return result.artifact


def _fresh_copy(artifact: ModelArtifact) -> ModelArtifact:
    """A new artifact object over the same fitted pipeline.

    `save()` and `register()` rewrite an artifact's metadata in place (version,
    model_file_sha256). A session-scoped artifact shared by many tests would
    therefore carry whatever version some earlier test happened to assign it,
    making results depend on collection order. The fitted pipeline is shared
    because refitting per test is pure waste; the metadata is not.
    """
    return ModelArtifact(
        pipeline=artifact.pipeline,
        metadata=artifact.metadata,
        model_card=artifact.model_card,
    )


@pytest.fixture
def trained_artifact(_fitted_artifact: ModelArtifact) -> ModelArtifact:
    """A fitted artifact whose metadata is private to this test."""
    return _fresh_copy(_fitted_artifact)


@pytest.fixture
def empty_registry(tmp_path: Path) -> LocalFilesystemRegistry:
    return LocalFilesystemRegistry(tmp_path / "registry")


@pytest.fixture(scope="session")
def served_registry(
    tmp_path_factory: pytest.TempPathFactory, _fitted_artifact: ModelArtifact
) -> LocalFilesystemRegistry:
    """A registry holding one registered, production-promoted model."""
    registry = LocalFilesystemRegistry(tmp_path_factory.mktemp("served-registry"))
    entry = registry.register(_fresh_copy(_fitted_artifact))
    registry.promote(MODEL_NAME, entry.version, "production", reason="test fixture")
    return registry


@pytest.fixture(scope="session")
def served_version(served_registry: LocalFilesystemRegistry) -> str:
    """The version the service fixtures actually resolve and serve."""
    return served_registry.resolve_stage(MODEL_NAME, "production").version


@pytest.fixture(scope="session")
def service_config(served_registry: LocalFilesystemRegistry) -> ServiceConfig:
    return ServiceConfig(
        registry_root=str(served_registry.root),
        model_name=MODEL_NAME,
        model_stage="production",
        environment="local",
        max_batch_size=100,
    )


@pytest.fixture
def observability_config() -> ObservabilityConfig:
    return ObservabilityConfig(log_format="console", drift_enabled=False)


@pytest.fixture
def client(
    service_config: ServiceConfig,
    observability_config: ObservabilityConfig,
    served_registry: LocalFilesystemRegistry,
) -> TestClient:
    app = create_app(
        service_config=service_config,
        observability_config=observability_config,
        registry=served_registry,
        configure_logs=False,
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def example_instance(_fitted_artifact: ModelArtifact) -> dict:
    return _fitted_artifact.feature_schema.example_row()
