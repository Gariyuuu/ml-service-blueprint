"""Drift hooks: interfaces, sinks, sampling, and fail-open behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from mlservice.config.observability import ObservabilityConfig
from mlservice.monitoring.base import DriftSink
from mlservice.monitoring.detectors import ScoreDistributionDrift, population_stability_index
from mlservice.monitoring.records import PredictionRecord
from mlservice.monitoring.reporter import DriftReporter, build_sink
from mlservice.monitoring.sinks import JsonlSink, NullSink, read_jsonl


def _records(scores, model="m", version="v1"):
    return [
        PredictionRecord(
            event_id=f"e{index}",
            model_name=model,
            model_version=version,
            score=float(score),
            label=int(score >= 0.5),
            threshold=0.5,
        )
        for index, score in enumerate(scores)
    ]


class ExplodingSink(DriftSink):
    """A sink that always fails, to prove monitoring cannot take down inference."""

    def emit(self, record: PredictionRecord) -> None:
        raise RuntimeError("sink is down")


def test_null_sink_swallows_everything():
    NullSink().emit_batch(_records([0.1, 0.2]))


def test_jsonl_sink_round_trips(tmp_path):
    path = tmp_path / "nested" / "drift.jsonl"
    sink = JsonlSink(path)
    sink.emit_batch(_records([0.1, 0.9]))
    sink.flush()
    sink.close()

    loaded = read_jsonl(path)
    assert [record.score for record in loaded] == [0.1, 0.9]


def test_jsonl_sink_drops_features_unless_asked(tmp_path):
    """Feature echo is off by default; inference inputs are frequently PII."""
    path = tmp_path / "drift.jsonl"
    sink = JsonlSink(path)
    record = _records([0.5])[0].model_copy(update={"features": {"ssn": "123"}})
    sink.emit(record)
    sink.close()
    assert "123" not in path.read_text()


def test_jsonl_sink_keeps_features_when_enabled(tmp_path):
    path = tmp_path / "drift.jsonl"
    sink = JsonlSink(path, include_features=True)
    sink.emit(_records([0.5])[0].model_copy(update={"features": {"age": 30}}))
    sink.close()
    assert read_jsonl(path)[0].features == {"age": 30}


def test_reporter_is_a_no_op_when_disabled(tmp_path):
    reporter = DriftReporter(JsonlSink(tmp_path / "d.jsonl"), enabled=False)
    assert (
        reporter.report(
            model_name="m",
            model_version="v1",
            scores=[0.5],
            labels=[1],
            threshold=0.5,
            event_ids=["e"],
        )
        == 0
    )


def test_reporter_emits_when_enabled(tmp_path):
    path = tmp_path / "d.jsonl"
    reporter = DriftReporter(JsonlSink(path), enabled=True)
    count = reporter.report(
        model_name="m",
        model_version="v1",
        scores=[0.1, 0.9],
        labels=[0, 1],
        threshold=0.5,
        event_ids=["a", "b"],
    )
    reporter.shutdown()
    assert count == 2
    assert len(read_jsonl(path)) == 2


def test_reporter_fails_open_when_the_sink_raises():
    """A broken drift pipeline must degrade to 'unmonitored', never to 'down'."""
    reporter = DriftReporter(ExplodingSink(), enabled=True)
    assert (
        reporter.report(
            model_name="m",
            model_version="v1",
            scores=[0.5],
            labels=[1],
            threshold=0.5,
            event_ids=["e"],
        )
        == 0
    )


def test_reporter_samples(tmp_path, monkeypatch):
    monkeypatch.setattr("mlservice.monitoring.reporter.random.random", lambda: 0.9)
    reporter = DriftReporter(JsonlSink(tmp_path / "d.jsonl"), sample_ratio=0.5, enabled=True)
    assert (
        reporter.report(
            model_name="m",
            model_version="v1",
            scores=[0.1] * 10,
            labels=[0] * 10,
            threshold=0.5,
            event_ids=[str(i) for i in range(10)],
        )
        == 0
    )


def test_build_sink_honours_configuration(tmp_path):
    assert isinstance(build_sink(ObservabilityConfig(drift_enabled=False)), NullSink)
    sink = build_sink(
        ObservabilityConfig(
            drift_enabled=True, drift_sink="jsonl", drift_sink_path=str(tmp_path / "d.jsonl")
        )
    )
    assert isinstance(sink, JsonlSink)
    sink.close()


def test_psi_of_a_distribution_against_itself_is_zero():
    sample = np.random.default_rng(0).random(5000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_as_distributions_separate():
    rng = np.random.default_rng(0)
    baseline = np.clip(rng.normal(0.3, 0.1, 5000), 0, 1)
    near = np.clip(rng.normal(0.35, 0.1, 5000), 0, 1)
    far = np.clip(rng.normal(0.8, 0.1, 5000), 0, 1)
    assert population_stability_index(baseline, near) < population_stability_index(baseline, far)


def test_psi_is_finite_when_a_baseline_bucket_is_empty():
    """An unseen score range must not make PSI infinite."""
    value = population_stability_index(np.zeros(100), np.ones(100))
    assert np.isfinite(value)


def test_score_drift_detector_holds_fire_below_the_sample_floor():
    detector = ScoreDistributionDrift(np.random.default_rng(0).random(1000), min_records=200)
    signal = detector.evaluate(_records([0.5] * 10))[0]
    assert signal.severity == "ok"
    assert signal.detail["reason"] == "insufficient_data"


def test_score_drift_detector_alerts_on_a_shifted_batch():
    rng = np.random.default_rng(0)
    detector = ScoreDistributionDrift(np.clip(rng.normal(0.2, 0.05, 5000), 0, 1), min_records=100)
    shifted = np.clip(rng.normal(0.9, 0.05, 500), 0, 1)
    signal = detector.evaluate(_records(shifted))[0]
    assert signal.severity == "alert"
    assert signal.is_drifting


def test_score_drift_detector_stays_quiet_on_a_stable_batch():
    rng = np.random.default_rng(0)
    baseline = np.clip(rng.normal(0.4, 0.15, 5000), 0, 1)
    detector = ScoreDistributionDrift(baseline, min_records=100)
    signal = detector.evaluate(_records(np.clip(rng.normal(0.4, 0.15, 2000), 0, 1)))[0]
    assert signal.severity == "ok"


def test_detector_can_be_rebuilt_from_artifact_metadata(trained_artifact):
    detector = ScoreDistributionDrift.from_metadata(trained_artifact.metadata, min_records=10)
    assert len(detector.baseline_scores) > 0
    assert detector.evaluate(_records([0.5] * 50))[0].name == "score_psi"
