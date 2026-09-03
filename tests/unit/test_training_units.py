"""Unit tests for split determinism, preprocessing, estimator factory, metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mlservice.artifacts.schema import FeatureSchema
from mlservice.config.model import ModelConfig
from mlservice.config.training import DataConfig, PreprocessingConfig, SplitConfig
from mlservice.training.data import class_balance, split_dataset
from mlservice.training.evaluate import choose_threshold, evaluate_scores, score_distribution
from mlservice.training.model_factory import available_estimators, build_estimator
from mlservice.training.preprocessing import build_preprocessor


@pytest.fixture
def data_config() -> DataConfig:
    return DataConfig(path="unused.csv", target="target")


def test_split_is_reproducible(synthetic_frame, data_config):
    config = SplitConfig(random_state=99)
    first = split_dataset(synthetic_frame, data_config, config)
    second = split_dataset(synthetic_frame, data_config, config)
    assert first.x_train.index.equals(second.x_train.index)
    assert first.x_test.index.equals(second.x_test.index)


def test_a_different_seed_produces_a_different_split(synthetic_frame, data_config):
    a = split_dataset(synthetic_frame, data_config, SplitConfig(random_state=1))
    b = split_dataset(synthetic_frame, data_config, SplitConfig(random_state=2))
    assert not a.x_test.index.equals(b.x_test.index)


def test_split_partitions_are_disjoint_and_complete(synthetic_frame, data_config):
    split = split_dataset(synthetic_frame, data_config, SplitConfig())
    indices = [set(split.x_train.index), set(split.x_validation.index), set(split.x_test.index)]
    assert not (indices[0] & indices[1]) and not (indices[0] & indices[2])
    assert not (indices[1] & indices[2])
    assert set().union(*indices) == set(synthetic_frame.index)


def test_validation_size_is_a_fraction_of_the_whole_frame(synthetic_frame, data_config):
    """0.2 must mean 20% of the original rows, not 20% of what test left behind."""
    split = split_dataset(
        synthetic_frame, data_config, SplitConfig(test_size=0.2, validation_size=0.2)
    )
    total = len(synthetic_frame)
    assert split.sizes["validation"] == pytest.approx(0.2 * total, abs=2)
    assert split.sizes["test"] == pytest.approx(0.2 * total, abs=2)


def test_stratification_preserves_class_balance(synthetic_frame, data_config):
    split = split_dataset(synthetic_frame, data_config, SplitConfig(stratify=True))
    overall = synthetic_frame["target"].mean()
    assert split.y_test.mean() == pytest.approx(overall, abs=0.03)


def test_class_balance_sums_to_one(synthetic_frame):
    assert sum(class_balance(synthetic_frame["target"]).values()) == pytest.approx(1.0)


def test_preprocessor_covers_every_schema_column(synthetic_frame):
    schema = FeatureSchema.from_frame(synthetic_frame, target="target")
    preprocessor = build_preprocessor(schema, PreprocessingConfig())
    handled = {column for _, _, columns in preprocessor.transformers for column in columns}
    assert handled == set(schema.feature_names)


def test_preprocessor_output_is_finite_and_numeric(synthetic_frame):
    schema = FeatureSchema.from_frame(synthetic_frame, target="target")
    preprocessor = build_preprocessor(schema, PreprocessingConfig())
    features = schema.validate_frame(synthetic_frame.drop(columns=["target"]))
    transformed = preprocessor.fit_transform(features)
    assert np.isfinite(np.asarray(transformed, dtype=float)).all()


def test_preprocessor_survives_an_unseen_category(synthetic_frame):
    schema = FeatureSchema.from_frame(synthetic_frame, target="target")
    preprocessor = build_preprocessor(schema, PreprocessingConfig())
    features = schema.validate_frame(synthetic_frame.drop(columns=["target"]))
    preprocessor.fit(features)

    novel = features.head(1).copy()
    novel["region"] = "atlantis"
    known = preprocessor.transform(features.head(1))
    assert preprocessor.transform(novel).shape[1] == known.shape[1]


def test_build_estimator_injects_the_run_seed():
    estimator = build_estimator(ModelConfig(name="m", estimator="random_forest"), seed=42)
    assert estimator.random_state == 42


def test_explicit_hyperparameters_win_over_the_seed_default():
    config = ModelConfig(name="m", estimator="random_forest", hyperparameters={"random_state": 7})
    assert build_estimator(config, seed=42).random_state == 7


def test_bad_hyperparameters_fail_with_a_useful_message():
    config = ModelConfig(name="m", estimator="random_forest", hyperparameters={"nope": 1})
    with pytest.raises(ValueError, match="Invalid hyperparameters"):
        build_estimator(config, seed=1)


def test_every_advertised_estimator_can_be_built():
    for name in available_estimators():
        build_estimator(ModelConfig(name="m", estimator=name), seed=1)


def test_calibration_wraps_the_estimator():
    config = ModelConfig(name="m", estimator="logistic_regression", calibrate=True)
    assert type(build_estimator(config, seed=1)).__name__ == "CalibratedClassifierCV"


def test_evaluate_reports_both_ranking_and_threshold_metrics():
    truth = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    metrics = evaluate_scores(truth, scores, threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0 and metrics["recall"] == 1.0
    assert "brier_score" in metrics and "log_loss" in metrics


def test_evaluate_omits_undefined_metrics_on_a_single_class_slice():
    """A one-class holdout has no AUC; emitting 0.5 there would be a lie."""
    metrics = evaluate_scores(np.ones(5, dtype=int), np.full(5, 0.9), threshold=0.5)
    assert "roc_auc" not in metrics
    assert metrics["accuracy"] == 1.0


def test_threshold_selection_beats_the_naive_half():
    rng = np.random.default_rng(0)
    truth = (rng.random(400) < 0.1).astype(int)
    scores = np.clip(truth * 0.3 + rng.normal(0.2, 0.1, 400), 0, 1)
    chosen = choose_threshold(truth, scores)
    from sklearn.metrics import f1_score

    at_chosen = f1_score(truth, (scores >= chosen).astype(int), zero_division=0)
    at_half = f1_score(truth, (scores >= 0.5).astype(int), zero_division=0)
    assert at_chosen >= at_half


def test_score_distribution_bins_sum_to_one():
    scores = np.random.default_rng(0).random(1000)
    summary = score_distribution(scores)
    bins = [value for key, value in summary.items() if key.startswith("bin_")]
    assert sum(bins) == pytest.approx(1.0, abs=1e-4)


def test_score_distribution_handles_an_empty_batch():
    summary = score_distribution(np.array([]))
    assert summary["mean"] == 0.0
