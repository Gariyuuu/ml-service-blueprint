"""Typed configuration: the failures should happen at load time, not at fit time."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlservice.config.model import ModelConfig
from mlservice.config.observability import ObservabilityConfig
from mlservice.config.service import ServiceConfig
from mlservice.config.training import EvaluationGates, SplitConfig, TrainingConfig


def test_yaml_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("name: m\nestimator: logistic_regression\nhyperparamters: {}\n")
    with pytest.raises(ValidationError, match="hyperparamters"):
        ModelConfig.from_yaml(path)


def test_yaml_config_rejects_unknown_estimator(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("name: m\nestimator: xgboost\n")
    with pytest.raises(ValidationError):
        ModelConfig.from_yaml(path)


def test_yaml_config_missing_file_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelConfig.from_yaml(tmp_path / "nope.yaml")


def test_yaml_config_rejects_non_mapping(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(TypeError, match="mapping"):
        ModelConfig.from_yaml(path)


def test_split_config_rejects_fractions_that_starve_training():
    with pytest.raises(ValidationError, match="at least 10%"):
        SplitConfig(test_size=0.5, validation_size=0.45)


def test_gates_report_every_failure_not_just_the_first():
    gates = EvaluationGates(min_roc_auc=0.9, min_f1=0.9)
    failures = gates.check({"roc_auc": 0.5, "f1": 0.1})
    assert len(failures) == 2
    assert any("roc_auc" in failure for failure in failures)


def test_gates_flag_a_metric_that_was_never_computed():
    gates = EvaluationGates(min_roc_auc=0.9)
    failures = gates.check({"f1": 0.99})
    assert failures and "was not computed" in failures[0]


def test_gates_pass_when_metrics_clear_thresholds():
    assert EvaluationGates(min_roc_auc=0.9).check({"roc_auc": 0.95}) == []


def test_service_config_reads_environment(monkeypatch):
    monkeypatch.setenv("MLSERVICE_MODEL_NAME", "from-env")
    monkeypatch.setenv("MLSERVICE_MAX_BATCH_SIZE", "7")
    config = ServiceConfig(_env_file=None)
    assert config.model_name == "from-env"
    assert config.max_batch_size == 7


def test_observability_config_requires_endpoint_when_tracing_on():
    with pytest.raises(ValidationError, match="otlp_endpoint"):
        ObservabilityConfig(tracing_enabled=True, otlp_endpoint=None)


def test_observability_config_accepts_tracing_with_endpoint():
    config = ObservabilityConfig(tracing_enabled=True, otlp_endpoint="http://x:4318/v1/traces")
    assert config.tracing_enabled


def test_training_config_serialises_to_json_for_artifact_metadata(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(f"data:\n  path: {tmp_path / 'x.csv'}\n  target: y\n")
    config = TrainingConfig.from_yaml(path)
    payload = config.to_dict()
    assert payload["data"]["target"] == "y"
    assert "split" in payload and "gates" in payload


def test_shipped_configs_are_valid(repo_root):
    """The configs in the repo must load; a broken default breaks `make train`."""
    training = TrainingConfig.from_yaml(repo_root / "configs" / "training.yaml")
    model = ModelConfig.from_yaml(repo_root / "configs" / "model.yaml")
    assert training.data.target
    assert model.name
