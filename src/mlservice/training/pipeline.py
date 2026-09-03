"""The training entry point: data in, registered artifact out.

This module is the single place where the training steps are ordered. Anything
that needs to happen for every model — seeding, schema derivation, threshold
selection on validation, gate evaluation on test, provenance capture, model card
rendering — happens here, so a project adopting this template gets it by
default rather than by remembering.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mlservice.artifacts.artifact import ESTIMATOR_STEP, PREPROCESSOR_STEP, ModelArtifact
from mlservice.artifacts.digest import sha256_file
from mlservice.artifacts.metadata import ArtifactMetadata, DatasetInfo
from mlservice.artifacts.provenance import CodeProvenance
from mlservice.artifacts.schema import FeatureSchema
from mlservice.config.model import ModelConfig
from mlservice.config.training import TrainingConfig
from mlservice.registry.base import ModelRegistry, RegisteredVersion
from mlservice.training.data import Split, class_balance, load_dataset, split_dataset
from mlservice.training.evaluate import choose_threshold, evaluate_scores, score_distribution
from mlservice.training.model_card import render_model_card
from mlservice.training.model_factory import build_estimator
from mlservice.training.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

UNREGISTERED_VERSION = "v0-unregistered"


@dataclass
class TrainingResult:
    """Everything one training run produced."""

    artifact: ModelArtifact
    registered: RegisteredVersion | None
    split_sizes: dict[str, int]

    @property
    def metrics(self) -> dict[str, float]:
        return self.artifact.metadata.metrics

    @property
    def passed_gates(self) -> bool:
        return self.artifact.metadata.passed_gates


def run_training(
    training_config: TrainingConfig,
    model_config: ModelConfig,
    *,
    registry: ModelRegistry | None = None,
    register: bool = True,
    tune_threshold: bool = True,
    repo_root: Path | None = None,
) -> TrainingResult:
    """Train, evaluate, package, and optionally register a model."""
    started = time.perf_counter()
    _seed_everything(training_config.seed)

    frame = load_dataset(training_config.data)
    split = split_dataset(frame, training_config.data, training_config.split)
    logger.info("split sizes: %s", split.sizes)

    # The schema is derived from the training split only. Deriving it from the
    # full frame would bake test-set ranges and categories into the artifact's
    # declared contract and its drift baseline.
    train_frame = pd.concat([split.x_train, split.y_train], axis=1)
    schema = FeatureSchema.from_frame(train_frame, target=training_config.data.target)

    pipeline = Pipeline(
        [
            (PREPROCESSOR_STEP, build_preprocessor(schema, training_config.preprocessing)),
            (ESTIMATOR_STEP, build_estimator(model_config, training_config.seed)),
        ]
    )

    x_train = schema.validate_frame(split.x_train)
    y_train = _binarise(split.y_train, training_config.data.positive_label)
    pipeline.fit(x_train, y_train)
    logger.info("fitted %s on %d rows", model_config.estimator, len(x_train))

    validation_scores = _positive_scores(pipeline, schema.validate_frame(split.x_validation))
    y_validation = _binarise(split.y_validation, training_config.data.positive_label)

    threshold = training_config.decision_threshold
    if tune_threshold:
        threshold = choose_threshold(y_validation, validation_scores, objective="f1")
        logger.info("selected decision threshold %.3f on the validation split", threshold)

    validation_metrics = evaluate_scores(y_validation, validation_scores, threshold)

    test_scores = _positive_scores(pipeline, schema.validate_frame(split.x_test))
    y_test = _binarise(split.y_test, training_config.data.positive_label)
    test_metrics = evaluate_scores(y_test, test_scores, threshold)
    test_metrics |= {
        f"score_dist.{key}": value for key, value in score_distribution(test_scores).items()
    }
    logger.info("test metrics: %s", {k: v for k, v in test_metrics.items() if "." not in k})

    gate_failures = training_config.gates.check(test_metrics)
    if gate_failures:
        logger.warning("evaluation gates failed: %s", gate_failures)

    metadata = ArtifactMetadata(
        model_name=model_config.name,
        version=UNREGISTERED_VERSION,
        feature_schema=schema,
        decision_threshold=threshold,
        metrics=test_metrics,
        validation_metrics=validation_metrics,
        gate_failures=gate_failures,
        dataset=_dataset_info(training_config, frame, split),
        provenance=CodeProvenance.capture(repo_root),
        training_config=training_config.to_dict(),
        model_config_snapshot=model_config.to_dict(),
        training_duration_seconds=round(time.perf_counter() - started, 3),
    )
    card = render_model_card(metadata, training_config.model_card_template)
    artifact = ModelArtifact(pipeline=pipeline, metadata=metadata, model_card=card)

    registered: RegisteredVersion | None = None
    if register:
        if registry is None:
            raise ValueError("register=True requires a registry instance")
        registered = registry.register(artifact)
        # The card embeds the version, so re-render and rewrite it now that one exists.
        artifact.model_card = render_model_card(
            artifact.metadata, training_config.model_card_template
        )
        Path(registered.uri, artifact.metadata.model_card_file).write_text(
            artifact.model_card, encoding="utf-8"
        )
        logger.info(
            "registered %s %s at %s",
            registered.model_name,
            registered.version,
            registered.uri,
        )

    return TrainingResult(artifact=artifact, registered=registered, split_sizes=split.sizes)


def _dataset_info(config: TrainingConfig, frame: pd.DataFrame, split: Split) -> DatasetInfo:
    return DatasetInfo(
        source=config.data.path,
        content_sha256=sha256_file(config.data.path),
        n_rows=len(frame),
        n_features=frame.shape[1] - 1,
        n_train=len(split.x_train),
        n_validation=len(split.x_validation),
        n_test=len(split.x_test),
        class_balance=class_balance(frame[config.data.target]),
    )


def _binarise(labels: pd.Series, positive_label: int) -> np.ndarray:
    """Map the configured positive class to 1 and everything else to 0."""
    return (labels == positive_label).astype(int).to_numpy()


def _positive_scores(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(features))[:, 1].astype(float)
    return np.asarray(pipeline.predict(features), dtype=float)


def _seed_everything(seed: int) -> None:
    """Seed every global RNG the training path can reach."""
    random.seed(seed)
    np.random.seed(seed)
