"""The model artifact: a self-describing, versioned, loadable unit of inference.

On disk an artifact is a directory::

    <artifact_dir>/
        model.joblib     fitted sklearn Pipeline (preprocessor -> estimator)
        metadata.json    ArtifactMetadata: schema, metrics, provenance, threshold
        model_card.md    human-readable model card

The pipeline is stored as a single object so that the preprocessor and the
estimator can never drift out of sync; they are still individually reachable
through the :attr:`ModelArtifact.preprocessor` and :attr:`ModelArtifact.estimator`
properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mlservice.artifacts.digest import sha256_file
from mlservice.artifacts.metadata import ArtifactMetadata
from mlservice.artifacts.schema import FeatureSchema

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
MODEL_CARD_FILE = "model_card.md"

PREPROCESSOR_STEP = "preprocessor"
ESTIMATOR_STEP = "estimator"


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact on disk does not match its recorded digest."""


@dataclass(frozen=True)
class PredictionResult:
    """Scores and derived labels for a batch of rows."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float

    def as_records(self) -> list[dict[str, Any]]:
        return [
            {"score": float(score), "label": int(label)}
            for score, label in zip(self.scores, self.labels, strict=True)
        ]


class ModelArtifact:
    """A fitted pipeline plus everything needed to serve and audit it."""

    def __init__(
        self, pipeline: Pipeline, metadata: ArtifactMetadata, model_card: str = ""
    ) -> None:
        self.pipeline = pipeline
        self.metadata = metadata
        self.model_card = model_card

    # ---- component access -------------------------------------------------

    @property
    def preprocessor(self) -> Any:
        """The fitted preprocessing stage of the pipeline."""
        return self.pipeline.named_steps[PREPROCESSOR_STEP]

    @property
    def estimator(self) -> Any:
        """The fitted estimator at the end of the pipeline."""
        return self.pipeline.named_steps[ESTIMATOR_STEP]

    @property
    def feature_schema(self) -> FeatureSchema:
        return self.metadata.feature_schema

    @property
    def version(self) -> str:
        return self.metadata.version

    # ---- inference --------------------------------------------------------

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        threshold: float | None = None,
        validate: bool = True,
    ) -> PredictionResult:
        """Score a frame of raw features.

        Schema validation runs by default; it is the same code path used by the
        API, so an offline batch job and the service cannot disagree about what
        constitutes a valid row.
        """
        prepared = self.feature_schema.validate_frame(frame) if validate else frame
        cutoff = threshold if threshold is not None else self.metadata.decision_threshold
        scores = self._score(prepared)
        labels = (scores >= cutoff).astype(int)
        return PredictionResult(scores=scores, labels=labels, threshold=cutoff)

    def _score(self, prepared: pd.DataFrame) -> np.ndarray:
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(prepared)
            # Column 1 is the positive class for the binary classifiers this
            # blueprint ships; a multiclass swap-in should override _score.
            return np.asarray(probabilities)[:, 1].astype(float)
        return np.asarray(self.pipeline.predict(prepared), dtype=float)

    # ---- persistence ------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Write the artifact directory and return its path."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)

        model_path = target / MODEL_FILE
        joblib.dump(self.pipeline, model_path, compress=3)

        metadata = self.metadata.model_copy(
            update={"model_file": MODEL_FILE, "model_file_sha256": sha256_file(model_path)}
        )
        (target / METADATA_FILE).write_text(
            metadata.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (target / MODEL_CARD_FILE).write_text(self.model_card, encoding="utf-8")

        self.metadata = metadata
        return target

    @classmethod
    def load(cls, directory: str | Path, *, verify_digest: bool = True) -> ModelArtifact:
        """Load an artifact directory, checking the model file digest by default."""
        source = Path(directory)
        metadata_path = source / METADATA_FILE
        if not metadata_path.is_file():
            raise FileNotFoundError(f"No {METADATA_FILE} in {source}")

        metadata = ArtifactMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        model_path = source / metadata.model_file
        if not model_path.is_file():
            raise FileNotFoundError(f"Artifact metadata references missing {model_path}")

        if verify_digest and metadata.model_file_sha256:
            observed = sha256_file(model_path)
            if observed != metadata.model_file_sha256:
                raise ArtifactIntegrityError(
                    f"{model_path} digest {observed[:12]}... does not match recorded "
                    f"{metadata.model_file_sha256[:12]}..."
                )

        pipeline = joblib.load(model_path)
        card_path = source / metadata.model_card_file
        card = card_path.read_text(encoding="utf-8") if card_path.is_file() else ""
        return cls(pipeline=pipeline, metadata=metadata, model_card=card)
