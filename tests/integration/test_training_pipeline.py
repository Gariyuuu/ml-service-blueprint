"""End-to-end training: determinism, gates, provenance, and registration."""

from __future__ import annotations

import pytest

from mlservice.config.training import EvaluationGates
from mlservice.registry.base import RegistryError
from mlservice.training.pipeline import run_training


def test_training_produces_a_registrable_artifact(trained_artifact):
    metadata = trained_artifact.metadata
    assert metadata.metrics["roc_auc"] > 0.6, "the synthetic fixture has real signal"
    assert metadata.passed_gates
    assert metadata.training_duration_seconds > 0


def test_training_records_all_three_split_sizes(training_config, model_config):
    result = run_training(training_config, model_config, register=False)
    dataset = result.artifact.metadata.dataset
    assert dataset.n_train + dataset.n_validation + dataset.n_test == dataset.n_rows


def test_training_is_reproducible(training_config, model_config, synthetic_frame):
    """Same config, same data, same seed — the same scores, not merely similar."""
    rows = synthetic_frame.drop(columns=["target"]).head(30)
    first = run_training(training_config, model_config, register=False).artifact
    second = run_training(training_config, model_config, register=False).artifact
    assert (first.predict(rows).scores == second.predict(rows).scores).all()
    assert first.metadata.metrics == second.metadata.metrics


def test_schema_is_derived_from_training_rows_only(training_config, model_config):
    """Deriving it from the full frame would leak test-set ranges into the contract."""
    result = run_training(training_config, model_config, register=False)
    dataset = result.artifact.metadata.dataset
    assert dataset.n_train < dataset.n_rows
    assert len(result.artifact.feature_schema.features) == dataset.n_features


def test_threshold_tuning_can_be_turned_off(training_config, model_config):
    result = run_training(training_config, model_config, register=False, tune_threshold=False)
    assert result.artifact.metadata.decision_threshold == training_config.decision_threshold


def test_tuned_threshold_lands_inside_the_search_range(training_config, model_config):
    threshold = run_training(
        training_config, model_config, register=False
    ).artifact.metadata.decision_threshold
    assert 0.05 <= threshold <= 0.95


def test_metrics_are_recorded_for_both_holdouts(trained_artifact):
    assert trained_artifact.metadata.metrics
    assert trained_artifact.metadata.validation_metrics


def test_a_failed_gate_is_recorded_and_blocks_registration(
    training_config, model_config, empty_registry
):
    impossible = training_config.model_copy(update={"gates": EvaluationGates(min_roc_auc=0.999999)})
    result = run_training(impossible, model_config, register=False)
    assert not result.passed_gates
    assert result.artifact.metadata.gate_failures

    with pytest.raises(RegistryError, match="gates failed"):
        empty_registry.register(result.artifact)


def test_registration_assigns_a_version_and_rewrites_the_card(
    training_config, model_config, empty_registry
):
    result = run_training(training_config, model_config, registry=empty_registry, register=True)
    assert result.registered is not None
    assert result.artifact.metadata.version == result.registered.version

    card = (
        empty_registry.version_uri(model_config.name, result.registered.version) / "model_card.md"
    ).read_text()
    assert result.registered.version in card


def test_provenance_and_config_snapshots_are_captured(trained_artifact, repo_root):
    metadata = trained_artifact.metadata
    assert metadata.provenance.python_version
    assert "scikit-learn" in metadata.provenance.packages
    assert metadata.training_config["seed"] == 7
    assert metadata.model_config_snapshot["estimator"] == "logistic_regression"


def test_dataset_digest_is_recorded(trained_artifact):
    assert len(trained_artifact.metadata.dataset.content_sha256) == 64


def test_score_distribution_baseline_is_stored_for_drift(trained_artifact):
    assert any(key.startswith("score_dist.") for key in trained_artifact.metadata.metrics)


def test_missing_data_file_produces_an_actionable_error(training_config, model_config):
    broken = training_config.model_copy(
        update={"data": training_config.data.model_copy(update={"path": "nope.csv"})}
    )
    with pytest.raises(FileNotFoundError, match="make data"):
        run_training(broken, model_config, register=False)


def test_register_without_a_registry_is_a_programming_error(training_config, model_config):
    with pytest.raises(ValueError, match="requires a registry"):
        run_training(training_config, model_config, register=True, registry=None)


@pytest.mark.slow
def test_reference_dataset_trains_and_clears_its_shipped_gates(repo_root, tmp_path):
    """The configs in the repo must actually produce a registrable model."""
    from mlservice.config.model import ModelConfig
    from mlservice.config.training import TrainingConfig

    data_path = repo_root / "data" / "raw" / "breast_cancer.csv"
    if not data_path.is_file():
        pytest.skip("reference dataset not materialised; run `make data`")

    training = TrainingConfig.from_yaml(repo_root / "configs" / "training.yaml").model_copy(
        update={"registry_root": str(tmp_path / "registry")}
    )
    training = training.model_copy(
        update={"data": training.data.model_copy(update={"path": str(data_path)})}
    )
    model = ModelConfig.from_yaml(repo_root / "configs" / "model.yaml")

    result = run_training(training, model, register=False, repo_root=repo_root)
    assert result.passed_gates, result.artifact.metadata.gate_failures
