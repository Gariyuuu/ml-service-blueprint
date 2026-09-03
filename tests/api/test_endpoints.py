"""HTTP surface: probes, prediction, schema enforcement, metrics, error shape."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlservice.config.service import ServiceConfig
from mlservice.serving.app import create_app
from mlservice.serving.middleware import REQUEST_ID_HEADER

# ---- probes ---------------------------------------------------------------


def test_health_is_independent_of_model_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["uptime_seconds"] >= 0


def test_ready_reports_the_loaded_model(client):
    body = client.get("/ready").json()
    assert body == {
        "status": "ready",
        "model_loaded": True,
        "model_name": "test-classifier",
        "model_version": body["model_version"],
        "detail": None,
    }


def test_ready_returns_503_when_no_model_is_loaded(tmp_path, observability_config):
    """A pod with no model must stay out of the load balancer, not crash-loop."""
    app = create_app(
        service_config=ServiceConfig(
            registry_root=str(tmp_path),
            model_name="absent",
            fail_fast_on_missing_model=False,
        ),
        observability_config=observability_config,
        configure_logs=False,
    )
    with TestClient(app) as unloaded:
        assert unloaded.get("/health").status_code == 200
        response = unloaded.get("/ready")
        assert response.status_code == 503
        assert response.json()["model_loaded"] is False


def test_startup_fails_fast_when_configured_to(tmp_path, observability_config):
    """A missing model is a deploy bug; the deploy should fail, loudly."""
    app = create_app(
        service_config=ServiceConfig(
            registry_root=str(tmp_path),
            model_name="absent",
            fail_fast_on_missing_model=True,
        ),
        observability_config=observability_config,
        configure_logs=False,
    )
    with pytest.raises(Exception, match="absent"), TestClient(app):
        pass


# ---- prediction -----------------------------------------------------------


def test_predict_scores_a_single_row(client, example_instance):
    body = client.post("/predict", json={"instances": [example_instance]}).json()
    assert body["count"] == 1
    prediction = body["predictions"][0]
    assert 0.0 <= prediction["score"] <= 1.0
    assert prediction["label"] in (0, 1)


def test_predict_scores_a_batch(client, example_instance):
    body = client.post("/predict", json={"instances": [example_instance] * 25}).json()
    assert body["count"] == 25
    assert len(body["predictions"]) == 25


def test_batching_does_not_change_scores(client, example_instance, synthetic_frame):
    rows = synthetic_frame.drop(columns=["target"]).head(8).to_dict(orient="records")
    batched = client.post("/predict", json={"instances": rows}).json()["predictions"]
    singles = [
        client.post("/predict", json={"instances": [row]}).json()["predictions"][0] for row in rows
    ]
    assert [p["score"] for p in batched] == pytest.approx([p["score"] for p in singles])


def test_response_names_the_exact_model_that_scored(client, example_instance):
    body = client.post("/predict", json={"instances": [example_instance]}).json()
    info = client.get("/model-info").json()
    assert body["model_name"] == info["model_name"]
    assert body["model_version"] == info["model_version"]


def test_predict_honours_a_per_request_threshold(client, synthetic_frame):
    rows = synthetic_frame.drop(columns=["target"]).head(40).to_dict(orient="records")
    permissive = client.post("/predict", json={"instances": rows, "threshold": 0.01}).json()
    strict = client.post("/predict", json={"instances": rows, "threshold": 0.99}).json()
    assert sum(p["label"] for p in permissive["predictions"]) >= sum(
        p["label"] for p in strict["predictions"]
    )


def test_features_are_not_echoed_unless_requested(client, example_instance):
    body = client.post("/predict", json={"instances": [example_instance]}).json()
    assert body["predictions"][0]["features"] is None


def test_features_are_echoed_when_requested(client, example_instance):
    body = client.post(
        "/predict", json={"instances": [example_instance], "return_features": True}
    ).json()
    assert body["predictions"][0]["features"] is not None


def test_column_order_does_not_matter(client, example_instance):
    reversed_row = dict(reversed(list(example_instance.items())))
    assert client.post("/predict", json={"instances": [reversed_row]}).status_code == 200


# ---- validation and errors ------------------------------------------------


def test_missing_features_return_422_listing_all_of_them(client):
    response = client.post("/predict", json={"instances": [{"age": 30}]})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "feature_schema_violation"
    assert body["details"]


def test_unexpected_features_are_rejected(client, example_instance):
    response = client.post("/predict", json={"instances": [example_instance | {"target": 1}]})
    assert response.status_code == 422
    assert "unexpected" in " ".join(response.json()["details"])


def test_wrong_type_is_rejected(client, example_instance):
    response = client.post("/predict", json={"instances": [example_instance | {"age": "old"}]})
    assert response.status_code == 422
    assert "non-numeric" in " ".join(response.json()["details"])


def test_empty_instance_list_is_rejected(client):
    response = client.post("/predict", json={"instances": []})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_oversized_batch_is_rejected_with_413(client, example_instance, service_config):
    response = client.post(
        "/predict", json={"instances": [example_instance] * (service_config.max_batch_size + 1)}
    )
    assert response.status_code == 413


def test_every_error_uses_the_same_body_shape(client):
    body = client.post("/predict", json={"instances": [{"age": 1}]}).json()
    assert set(body) == {"error", "message", "request_id", "details"}


def test_error_bodies_carry_the_request_id(client):
    body = client.post("/predict", json={"instances": [{"age": 1}]}).json()
    assert body["request_id"]


# ---- model info -----------------------------------------------------------


def test_model_info_exposes_the_served_contract(client):
    body = client.get("/model-info").json()
    assert body["model_version"]
    assert body["stage"] == "production"
    assert 0.0 < body["decision_threshold"] < 1.0
    assert body["n_features"] == len(body["features"])
    assert body["git_commit"] is None or isinstance(body["git_commit"], str)
    assert len(body["dataset_sha256"]) == 64


def test_model_info_example_instance_is_directly_usable(client):
    example = client.get("/model-info").json()["example_instance"]
    assert client.post("/predict", json={"instances": [example]}).status_code == 200


def test_model_info_hides_distribution_baselines(client):
    """Score histograms belong in /metrics, not in a client-facing payload."""
    metrics = client.get("/model-info").json()["metrics"]
    assert not any(name.startswith("score_dist.") for name in metrics)


def test_model_card_is_served_as_markdown(client):
    response = client.get("/model-card")
    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    assert "## Metrics" in response.text


# ---- observability --------------------------------------------------------


def test_every_response_carries_a_request_id(client):
    assert client.get("/health").headers[REQUEST_ID_HEADER]


def test_a_caller_supplied_request_id_is_preserved(client):
    response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-me"})
    assert response.headers[REQUEST_ID_HEADER] == "trace-me"


def test_metrics_endpoint_exposes_the_expected_series(client, example_instance):
    client.post("/predict", json={"instances": [example_instance]})
    text = client.get("/metrics").text
    for series in (
        "mlservice_requests_total",
        "mlservice_request_duration_seconds",
        "mlservice_predictions_total",
        "mlservice_prediction_score",
        "mlservice_model_loaded",
    ):
        assert series in text


def test_metrics_label_routes_by_template_not_by_path(client):
    """Raw-path labels are how a metrics backend gets killed by a scanner."""
    client.get("/does-not-exist")
    text = client.get("/metrics").text
    assert 'route="unmatched"' in text
    assert "does-not-exist" not in text


def test_prediction_counter_advances_by_batch_size(client, example_instance):
    def count(text: str) -> float:
        for line in text.splitlines():
            if line.startswith("mlservice_predictions_total{"):
                return float(line.rsplit(" ", 1)[1])
        return 0.0

    before = count(client.get("/metrics").text)
    client.post("/predict", json={"instances": [example_instance] * 5})
    assert count(client.get("/metrics").text) - before == 5.0


def test_openapi_document_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/model-info" in schema["paths"]
