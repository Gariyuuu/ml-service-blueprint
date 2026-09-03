"""The artifact contract: what must be inside, and that it survives a round trip."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mlservice.artifacts.artifact import (
    METADATA_FILE,
    MODEL_CARD_FILE,
    MODEL_FILE,
    ArtifactIntegrityError,
    ModelArtifact,
)
from mlservice.artifacts.metadata import ArtifactMetadata


def test_artifact_exposes_the_full_contract(trained_artifact):
    """Every element the artifact contract requires must be reachable."""
    metadata = trained_artifact.metadata
    assert trained_artifact.estimator is not None  # model
    assert trained_artifact.preprocessor is not None  # preprocessor
    assert metadata.feature_schema.features  # schema
    assert metadata.version  # version
    assert metadata.metrics  # metrics
    assert metadata.created_at  # training timestamp
    assert metadata.provenance.python_version  # code/version metadata
    assert metadata.dataset.content_sha256  # data provenance
    assert trained_artifact.model_card  # model card


def test_saved_artifact_has_all_three_files(trained_artifact, tmp_path):
    directory = trained_artifact.save(tmp_path / "artifact")
    assert (directory / MODEL_FILE).is_file()
    assert (directory / METADATA_FILE).is_file()
    assert (directory / MODEL_CARD_FILE).is_file()


def test_metadata_is_valid_json_matching_its_model(trained_artifact, tmp_path):
    directory = trained_artifact.save(tmp_path / "artifact")
    payload = json.loads((directory / METADATA_FILE).read_text())
    restored = ArtifactMetadata.model_validate(payload)
    assert restored.model_name == trained_artifact.metadata.model_name


def test_round_trip_preserves_predictions(trained_artifact, tmp_path, synthetic_frame):
    directory = trained_artifact.save(tmp_path / "artifact")
    reloaded = ModelArtifact.load(directory)

    rows = synthetic_frame.drop(columns=["target"]).head(25)
    original = trained_artifact.predict(rows)
    restored = reloaded.predict(rows)
    assert (original.scores == restored.scores).all()
    assert original.threshold == restored.threshold


def test_round_trip_preserves_the_schema(trained_artifact, tmp_path):
    directory = trained_artifact.save(tmp_path / "artifact")
    assert ModelArtifact.load(directory).feature_schema == trained_artifact.feature_schema


def test_save_records_a_digest_of_the_model_file(trained_artifact, tmp_path):
    directory = trained_artifact.save(tmp_path / "artifact")
    assert len(ModelArtifact.load(directory).metadata.model_file_sha256) == 64


def test_a_tampered_model_file_is_rejected(trained_artifact, tmp_path):
    directory = trained_artifact.save(tmp_path / "artifact")
    (directory / MODEL_FILE).write_bytes(b"not a model")
    with pytest.raises(ArtifactIntegrityError):
        ModelArtifact.load(directory)


def test_digest_check_can_be_skipped_deliberately(trained_artifact, tmp_path):
    directory = trained_artifact.save(tmp_path / "artifact")
    (directory / METADATA_FILE).write_text(
        trained_artifact.metadata.model_copy(update={"model_file_sha256": ""}).model_dump_json()
    )
    assert ModelArtifact.load(directory, verify_digest=False) is not None


def test_loading_a_directory_without_metadata_fails_clearly(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match=METADATA_FILE):
        ModelArtifact.load(tmp_path / "empty")


def test_predict_validates_by_default(trained_artifact):
    from mlservice.artifacts.schema import SchemaValidationError

    with pytest.raises(SchemaValidationError):
        trained_artifact.predict(pd.DataFrame([{"age": 30}]))


def test_predict_applies_the_supplied_threshold(trained_artifact, synthetic_frame):
    rows = synthetic_frame.drop(columns=["target"]).head(50)
    permissive = trained_artifact.predict(rows, threshold=0.01)
    strict = trained_artifact.predict(rows, threshold=0.99)
    assert permissive.labels.sum() >= strict.labels.sum()


def test_scores_are_probabilities(trained_artifact, synthetic_frame):
    result = trained_artifact.predict(synthetic_frame.drop(columns=["target"]).head(50))
    assert ((result.scores >= 0) & (result.scores <= 1)).all()


def test_batch_scoring_matches_row_by_row_scoring(trained_artifact, synthetic_frame):
    """Batching must be a throughput choice, never a behavioural one."""
    rows = synthetic_frame.drop(columns=["target"]).head(10)
    batched = trained_artifact.predict(rows).scores
    individually = [trained_artifact.predict(rows.iloc[[i]]).scores[0] for i in range(len(rows))]
    assert batched == pytest.approx(individually)


def test_model_card_names_the_version_and_metrics(trained_artifact):
    card = trained_artifact.model_card
    assert trained_artifact.metadata.model_name in card
    assert "## Metrics" in card
    assert "## Input schema" in card
