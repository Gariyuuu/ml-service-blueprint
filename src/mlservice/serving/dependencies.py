"""Dependency providers.

Everything the routes need is stashed on ``app.state`` at startup and handed out
here, so tests can construct an app with a different registry or a stub reporter
without patching module globals.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from mlservice.config.observability import ObservabilityConfig
from mlservice.config.service import ServiceConfig
from mlservice.monitoring.reporter import DriftReporter
from mlservice.serving.model_holder import ModelHolder


def get_model_holder(request: Request) -> ModelHolder:
    holder: ModelHolder = request.app.state.model_holder
    return holder


def get_service_config(request: Request) -> ServiceConfig:
    config: ServiceConfig = request.app.state.service_config
    return config


def get_observability_config(request: Request) -> ObservabilityConfig:
    config: ObservabilityConfig = request.app.state.observability_config
    return config


def get_drift_reporter(request: Request) -> DriftReporter:
    reporter: DriftReporter = request.app.state.drift_reporter
    return reporter


HolderDep = Annotated[ModelHolder, Depends(get_model_holder)]
ServiceConfigDep = Annotated[ServiceConfig, Depends(get_service_config)]
ObservabilityConfigDep = Annotated[ObservabilityConfig, Depends(get_observability_config)]
DriftReporterDep = Annotated[DriftReporter, Depends(get_drift_reporter)]
