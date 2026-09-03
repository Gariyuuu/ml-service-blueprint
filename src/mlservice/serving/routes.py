"""API routes: /health, /ready, /predict, /model-info, /metrics."""

from __future__ import annotations

import time
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Response

from mlservice.observability import metrics
from mlservice.observability.context import get_request_id, set_model_version
from mlservice.observability.tracing import span
from mlservice.serving.dependencies import (
    DriftReporterDep,
    HolderDep,
    ObservabilityConfigDep,
    ServiceConfigDep,
)
from mlservice.serving.schemas import (
    FeatureInfo,
    HealthResponse,
    ModelInfoResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

router = APIRouter()

_STARTED_AT = time.monotonic()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["operations"],
    summary="Liveness probe",
)
def health(config: ServiceConfigDep) -> HealthResponse:
    """Is the process alive?

    Deliberately independent of model state. A liveness probe that fails when a
    model is missing causes the orchestrator to restart-loop a pod that a
    restart cannot fix.
    """
    from mlservice import __version__

    return HealthResponse(
        service=config.app_name,
        version=__version__,
        environment=config.environment,
        uptime_seconds=round(time.monotonic() - _STARTED_AT, 3),
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["operations"],
    summary="Readiness probe",
    responses={503: {"model": ReadyResponse, "description": "No model is loaded."}},
)
def ready(holder: HolderDep, response: Response) -> ReadyResponse:
    """Can this replica serve traffic right now?

    Returns 503 when no model is loaded, which is the signal that keeps a
    half-started pod out of the load balancer.
    """
    if not holder.is_loaded:
        response.status_code = 503  # Service Unavailable
        return ReadyResponse(
            status="not_ready",
            model_loaded=False,
            detail=holder.load_error or "No model loaded",
        )
    metadata = holder.artifact.metadata
    return ReadyResponse(
        status="ready",
        model_loaded=True,
        model_name=metadata.model_name,
        model_version=metadata.version,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Score one or more rows",
)
def predict(
    payload: PredictRequest,
    holder: HolderDep,
    config: ServiceConfigDep,
    observability: ObservabilityConfigDep,
    reporter: DriftReporterDep,
) -> PredictResponse:
    """Validate against the model's frozen feature schema, then score.

    A single row and a batch take the same code path; batching is purely a
    throughput optimisation, never a behavioural difference.
    """
    if len(payload.instances) > config.max_batch_size:
        raise HTTPException(
            status_code=413,  # Content Too Large
            detail=(
                f"Batch of {len(payload.instances)} exceeds max_batch_size {config.max_batch_size}"
            ),
        )

    artifact = holder.artifact
    metadata = artifact.metadata
    set_model_version(metadata.version)

    threshold = payload.threshold if payload.threshold is not None else holder.threshold

    started = time.perf_counter()
    with span("predict", batch_size=len(payload.instances), model_version=metadata.version):
        frame = pd.DataFrame(payload.instances)
        # validate=True here would re-run validation inside predict(); do it once
        # explicitly so the validated frame can be echoed back when asked for.
        validated = artifact.feature_schema.validate_frame(frame)
        result = artifact.predict(validated, threshold=threshold, validate=False)
    inference_ms = round((time.perf_counter() - started) * 1000, 3)

    scores = [float(score) for score in result.scores]
    labels = [int(label) for label in result.labels]

    metrics.record_predictions(
        metadata.model_name,
        metadata.version,
        scores,
        observe_scores=observability.prediction_histogram_enabled,
    )

    echoed: list[dict[str, Any]] | None = None
    if payload.return_features:
        # NaN is not valid JSON; imputed-away nulls come back as null, not NaN.
        echoed = [
            {str(key): value for key, value in row.items()}
            for row in validated.where(pd.notna(validated), None).to_dict(orient="records")
        ]
    reporter.report(
        model_name=metadata.model_name,
        model_version=metadata.version,
        scores=scores,
        labels=labels,
        threshold=threshold,
        event_ids=[uuid.uuid4().hex for _ in scores],
        latency_ms=inference_ms,
        features=echoed if observability.log_include_request_body else None,
    )

    predictions = [
        Prediction(
            score=score,
            label=label,
            features=echoed[index] if echoed is not None else None,
        )
        for index, (score, label) in enumerate(zip(scores, labels, strict=True))
    ]

    return PredictResponse(
        model_name=metadata.model_name,
        model_version=metadata.version,
        threshold=threshold,
        predictions=predictions,
        count=len(predictions),
        request_id=get_request_id() or "",
        inference_ms=inference_ms,
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["inference"],
    summary="Describe the loaded model",
)
def model_info(holder: HolderDep) -> ModelInfoResponse:
    """The artifact contract, as served. Use this to confirm what is deployed."""
    artifact = holder.artifact
    metadata = artifact.metadata
    schema = artifact.feature_schema

    return ModelInfoResponse(
        model_name=metadata.model_name,
        model_version=metadata.version,
        stage=holder.stage,
        loaded_at=holder.loaded_at.isoformat() if holder.loaded_at else "",
        decision_threshold=holder.threshold,
        trained_at=metadata.created_at.isoformat(),
        git_commit=metadata.provenance.git_commit,
        # Distribution baselines belong in /metrics, not in a client-facing payload.
        metrics={
            name: value
            for name, value in metadata.metrics.items()
            if not name.startswith("score_dist.")
        },
        dataset_sha256=metadata.dataset.content_sha256,
        n_features=len(schema.features),
        features=[
            FeatureInfo(
                name=feature.name,
                kind=feature.kind,
                required=feature.required,
                minimum=feature.minimum,
                maximum=feature.maximum,
                categories=feature.categories,
            )
            for feature in schema.features
        ],
        example_instance=schema.example_row(),
        model_card_available=bool(artifact.model_card),
    )


@router.get(
    "/model-card",
    tags=["inference"],
    summary="Model card for the loaded model",
    response_class=Response,
    responses={200: {"content": {"text/markdown": {}}}},
)
def model_card(holder: HolderDep) -> Response:
    """The generated model card, served as Markdown."""
    card = holder.artifact.model_card
    if not card:
        raise HTTPException(status_code=404, detail="This artifact has no model card")
    return Response(content=card, media_type="text/markdown; charset=utf-8")


def build_metrics_router(path: str) -> APIRouter:
    """Metrics endpoint, mounted at the configured path."""
    metrics_router = APIRouter()

    @metrics_router.get(path, tags=["operations"], summary="Prometheus scrape endpoint")
    def scrape() -> Response:
        payload, content_type = metrics.render()
        return Response(content=payload, media_type=content_type)

    return metrics_router
