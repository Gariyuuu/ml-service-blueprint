"""Holds the loaded artifact for the process lifetime.

Kept separate from the FastAPI app so that loading, reloading, and readiness are
testable without HTTP, and so a future hot-reload endpoint has one obvious place
to live.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime

from mlservice.artifacts.artifact import ModelArtifact
from mlservice.config.service import ServiceConfig
from mlservice.observability import metrics
from mlservice.registry.base import ModelNotFoundError, ModelRegistry

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when an inference path runs before a model is available."""


class ModelHolder:
    """Thread-safe container for the currently served artifact."""

    def __init__(self, registry: ModelRegistry, config: ServiceConfig) -> None:
        self.registry = registry
        self.config = config
        self._artifact: ModelArtifact | None = None
        self._stage: str | None = None
        self._loaded_at: datetime | None = None
        self._load_error: str | None = None
        self._lock = threading.RLock()

    # ---- state ------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._artifact is not None

    @property
    def artifact(self) -> ModelArtifact:
        artifact = self._artifact
        if artifact is None:
            raise ModelNotLoadedError(self._load_error or "No model has been loaded")
        return artifact

    @property
    def stage(self) -> str | None:
        return self._stage

    @property
    def loaded_at(self) -> datetime | None:
        return self._loaded_at

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def threshold(self) -> float:
        """The threshold actually in force, honouring any service-level override."""
        if self.config.decision_threshold_override is not None:
            return self.config.decision_threshold_override
        return self.artifact.metadata.decision_threshold

    # ---- loading ----------------------------------------------------------

    def load(self) -> ModelArtifact:
        """Resolve and load the configured model.

        An exact ``model_version`` wins over ``model_stage``. Pinning a version
        is what you want for a canary or a reproducible investigation; stage
        resolution is what you want for a normal deploy.
        """
        with self._lock:
            started = time.perf_counter()
            name = self.config.model_name
            try:
                if self.config.model_version:
                    version = self.config.model_version
                    stage = None
                else:
                    entry = self.registry.resolve_stage(name, self.config.model_stage)
                    version = entry.version
                    stage = self.config.model_stage

                artifact = self.registry.load(name, version)
            except (ModelNotFoundError, FileNotFoundError, OSError) as exc:
                self._load_error = str(exc)
                self._artifact = None
                logger.error("model load failed: %s", exc, extra={"model_name": name})
                raise

            self._artifact = artifact
            self._stage = stage
            self._loaded_at = datetime.now(UTC)
            self._load_error = None

            metrics.set_model_loaded(name, artifact.version, stage or "pinned")
            logger.info(
                "model loaded",
                extra={
                    "model_name": name,
                    "model_version": artifact.version,
                    "stage": stage,
                    "load_ms": round((time.perf_counter() - started) * 1000, 2),
                    "n_features": len(artifact.feature_schema.features),
                    "threshold": self.threshold,
                },
            )
            return artifact

    def try_load(self) -> bool:
        """Load, converting failure into a False return.

        Used at startup when ``fail_fast_on_missing_model`` is disabled: the
        process comes up, /health passes, /ready fails, and the orchestrator
        keeps the pod out of the load balancer.
        """
        try:
            self.load()
        except Exception as exc:  # noqa: BLE001 - startup must not crash here
            self._load_error = str(exc)
            return False
        return True

    def unload(self) -> None:
        with self._lock:
            self._artifact = None
            self._loaded_at = None
