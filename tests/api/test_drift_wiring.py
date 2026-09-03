"""The serving path's drift seam, wired end to end."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mlservice.config.observability import ObservabilityConfig
from mlservice.monitoring.sinks import read_jsonl
from mlservice.serving.app import create_app


def _client(service_config, registry, observability):
    app = create_app(
        service_config=service_config,
        observability_config=observability,
        registry=registry,
        configure_logs=False,
    )
    return TestClient(app)


def test_predictions_reach_the_sink_when_drift_is_enabled(
    tmp_path, service_config, served_registry, served_version, example_instance
):
    sink_path = tmp_path / "drift.jsonl"
    observability = ObservabilityConfig(
        log_format="console",
        drift_enabled=True,
        drift_sink="jsonl",
        drift_sink_path=str(sink_path),
    )
    with _client(service_config, served_registry, observability) as client:
        client.post("/predict", json={"instances": [example_instance] * 3})

    records = read_jsonl(sink_path)
    assert len(records) == 3
    assert records[0].model_version == served_version
    assert records[0].request_id
    assert records[0].features is None, "features are withheld unless explicitly enabled"


def test_nothing_is_written_when_drift_is_disabled(
    tmp_path, service_config, served_registry, example_instance
):
    sink_path = tmp_path / "drift.jsonl"
    observability = ObservabilityConfig(
        log_format="console",
        drift_enabled=False,
        drift_sink="jsonl",
        drift_sink_path=str(sink_path),
    )
    with _client(service_config, served_registry, observability) as client:
        client.post("/predict", json={"instances": [example_instance]})
    assert not sink_path.exists()


def test_a_broken_sink_does_not_break_inference(
    service_config, served_registry, observability_config, example_instance
):
    from mlservice.monitoring.base import DriftSink
    from mlservice.monitoring.reporter import DriftReporter

    class Exploding(DriftSink):
        def emit(self, record):
            raise RuntimeError("sink down")

    app = create_app(
        service_config=service_config,
        observability_config=observability_config,
        registry=served_registry,
        drift_reporter=DriftReporter(Exploding(), enabled=True),
        configure_logs=False,
    )
    with TestClient(app) as client:
        assert client.post("/predict", json={"instances": [example_instance]}).status_code == 200
