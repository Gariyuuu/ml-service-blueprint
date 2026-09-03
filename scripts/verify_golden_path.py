#!/usr/bin/env python3
"""Execute the entire local golden path and report exactly what succeeded.

    data → training → validation → artifact → version → inference service
         → Docker → CI checks → monitoring hooks

Every step reports PASS, FAIL, or SKIP with the reason. Nothing is inferred:
a step is PASS only if this script ran it and observed the result. Steps whose
prerequisites are absent (no Docker daemon, no k6 binary) report SKIP with the
missing prerequisite named — they are never silently counted as successes.

    make golden-path
    python scripts/verify_golden_path.py --json report.json
    python scripts/verify_golden_path.py --skip-quality     # faster, no lint/mypy/pytest
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_BIN = REPO_ROOT / ".venv" / "bin"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m",
    "\033[31m",
    "\033[33m",
    "\033[2m",
    "\033[1m",
    "\033[0m",
)


class StepSkipped(Exception):
    """Raised by a step whose prerequisites are absent."""


@dataclass
class StepResult:
    name: str
    stage: str
    status: str
    duration_seconds: float
    detail: str = ""
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """State threaded between steps: each one builds on the last."""

    workdir: Path
    registry_root: Path
    dataset_path: Path
    use_repo_registry: bool
    artifact: Any = None
    registered_version: str | None = None
    model_name: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


class Runner:
    def __init__(self, context: Context) -> None:
        self.context = context
        self.results: list[StepResult] = []

    def step(self, stage: str, name: str, fn: Callable[[Context], dict[str, Any] | None]) -> None:
        started = time.perf_counter()
        print(f"{DIM}·{RESET} {name} ", end="", flush=True)
        try:
            facts = fn(self.context) or {}
        except StepSkipped as exc:
            self._record(StepResult(name, stage, SKIP, time.perf_counter() - started, str(exc)))
        except Exception as exc:  # noqa: BLE001 - the report is the product
            detail = f"{type(exc).__name__}: {exc}"
            self._record(
                StepResult(
                    name,
                    stage,
                    FAIL,
                    time.perf_counter() - started,
                    detail,
                    {"traceback": traceback.format_exc()[-2000:]},
                )
            )
        else:
            self._record(StepResult(name, stage, PASS, time.perf_counter() - started, facts=facts))

    def _record(self, result: StepResult) -> None:
        colour = {PASS: GREEN, FAIL: RED, SKIP: YELLOW}[result.status]
        elapsed = f"{DIM}({result.duration_seconds:.2f}s){RESET}"
        print(f"\r{colour}{result.status:<4}{RESET} {result.name} {elapsed}")
        if result.detail:
            print(f"     {DIM}{result.detail}{RESET}")
        for key, value in result.facts.items():
            if key != "traceback":
                print(f"     {DIM}{key}: {value}{RESET}")
        self.results.append(result)

    @property
    def failed(self) -> list[StepResult]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def skipped(self) -> list[StepResult]:
        return [r for r in self.results if r.status == SKIP]


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


def step_environment(ctx: Context) -> dict[str, Any]:
    import fastapi
    import pandas
    import sklearn

    import mlservice

    if sys.version_info < (3, 11):  # noqa: UP036 - a 3.10 clone must get a clear message
        raise RuntimeError(f"Python 3.11+ required, running {sys.version.split()[0]}")
    return {
        "python": sys.version.split()[0],
        "mlservice": mlservice.__version__,
        "versions": (
            f"sklearn={sklearn.__version__} pandas={pandas.__version__} "
            f"fastapi={fastapi.__version__}"
        ),
    }


def step_config(ctx: Context) -> dict[str, Any]:
    from mlservice.config.model import ModelConfig
    from mlservice.config.observability import ObservabilityConfig
    from mlservice.config.service import ServiceConfig
    from mlservice.config.training import TrainingConfig

    training = TrainingConfig.from_yaml(REPO_ROOT / "configs" / "training.yaml")
    model = ModelConfig.from_yaml(REPO_ROOT / "configs" / "model.yaml")
    ServiceConfig(_env_file=None)
    ObservabilityConfig(_env_file=None)

    ctx.model_name = model.name
    return {
        "training config": f"seed={training.seed} target={training.data.target}",
        "model config": f"{model.name} / {model.estimator}",
        "gates": ", ".join(
            f"{k}={v}" for k, v in training.gates.model_dump().items() if v is not None
        )
        or "none configured",
    }


def step_data(ctx: Context) -> dict[str, Any]:
    import pandas as pd

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from make_dataset import main as make_dataset

    ctx.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if make_dataset(["--output", str(ctx.dataset_path), "--force"]) != 0:
        raise RuntimeError("make_dataset.py returned non-zero")

    frame = pd.read_csv(ctx.dataset_path)
    shown = (
        ctx.dataset_path.relative_to(REPO_ROOT)
        if ctx.dataset_path.is_relative_to(REPO_ROOT)
        else ctx.dataset_path
    )
    return {
        "path": str(shown),
        "shape": f"{len(frame)} rows x {frame.shape[1]} columns",
        "class balance": f"{frame['target'].mean():.3f} positive",
    }


def step_training(ctx: Context) -> dict[str, Any]:
    from mlservice.config.model import ModelConfig
    from mlservice.config.training import TrainingConfig
    from mlservice.training.pipeline import run_training

    training = TrainingConfig.from_yaml(REPO_ROOT / "configs" / "training.yaml")
    training = training.model_copy(
        update={"data": training.data.model_copy(update={"path": str(ctx.dataset_path)})}
    )
    model = ModelConfig.from_yaml(REPO_ROOT / "configs" / "model.yaml")

    result = run_training(training, model, register=False, repo_root=REPO_ROOT)
    ctx.artifact = result.artifact
    ctx.metrics = result.metrics
    ctx.model_name = model.name

    return {
        "estimator": model.estimator,
        "splits": " / ".join(f"{k}={v}" for k, v in result.split_sizes.items()),
        "duration": f"{result.artifact.metadata.training_duration_seconds:.2f}s",
        "threshold": f"{result.artifact.metadata.decision_threshold:.3f} (tuned on validation)",
    }


def step_validation(ctx: Context) -> dict[str, Any]:
    metadata = ctx.artifact.metadata
    if metadata.gate_failures:
        raise RuntimeError("gates failed: " + "; ".join(metadata.gate_failures))

    headline = {
        name: round(value, 4)
        for name, value in metadata.metrics.items()
        if name in {"roc_auc", "average_precision", "f1", "precision", "recall", "accuracy"}
    }
    return {
        "test metrics": json.dumps(headline),
        "validation f1": round(metadata.validation_metrics.get("f1", 0.0), 4),
        "gates": "all passed",
    }


def step_artifact(ctx: Context) -> dict[str, Any]:
    import pandas as pd

    from mlservice.artifacts.artifact import METADATA_FILE, MODEL_FILE, ModelArtifact

    directory = ctx.workdir / "artifact-roundtrip"
    saved = ctx.artifact.save(directory)

    missing = [f for f in (MODEL_FILE, METADATA_FILE, "model_card.md") if not (saved / f).is_file()]
    if missing:
        raise RuntimeError(f"artifact is missing {missing}")

    reloaded = ModelArtifact.load(saved)
    rows = pd.read_csv(ctx.dataset_path).drop(columns=["target"]).head(20)
    original = ctx.artifact.predict(rows).scores
    restored = reloaded.predict(rows).scores
    if not (original == restored).all():
        raise RuntimeError("reloaded artifact produced different scores")

    metadata = reloaded.metadata
    contract = {
        "model": reloaded.estimator is not None,
        "preprocessor": reloaded.preprocessor is not None,
        "schema": bool(metadata.feature_schema.features),
        "version": bool(metadata.version),
        "metrics": bool(metadata.metrics),
        "training timestamp": metadata.created_at is not None,
        "code/version metadata": bool(metadata.provenance.python_version),
        "model card": bool(reloaded.model_card),
    }
    absent = [name for name, present in contract.items() if not present]
    if absent:
        raise RuntimeError(f"artifact contract incomplete: {absent}")

    return {
        "contract": "all 8 required elements present",
        "size": f"{(saved / MODEL_FILE).stat().st_size / 1024:.0f} KiB",
        "digest verified": metadata.model_file_sha256[:16] + "…",
        "round trip": "scores identical after reload",
    }


def step_integrity(ctx: Context) -> dict[str, Any]:
    from mlservice.artifacts.artifact import MODEL_FILE, ArtifactIntegrityError, ModelArtifact

    directory = ctx.workdir / "artifact-tamper"
    ctx.artifact.save(directory)
    (directory / MODEL_FILE).write_bytes(b"tampered")
    try:
        ModelArtifact.load(directory)
    except ArtifactIntegrityError:
        return {"tampered artifact": "rejected, as expected"}
    raise RuntimeError("a tampered model file was loaded without complaint")


def step_register(ctx: Context) -> dict[str, Any]:
    from mlservice.registry.local import LocalFilesystemRegistry

    registry = LocalFilesystemRegistry(ctx.registry_root)
    entry = registry.register(ctx.artifact)
    ctx.registered_version = entry.version
    return {
        "version": entry.version,
        "uri": str(entry.uri),
        "git commit": entry.git_commit[:8] if entry.git_commit else "not a git checkout",
    }


def step_registry_operations(ctx: Context) -> dict[str, Any]:
    from mlservice.registry.local import LocalFilesystemRegistry

    registry = LocalFilesystemRegistry(ctx.registry_root)
    name = ctx.model_name
    first = ctx.registered_version
    if first is None:
        raise RuntimeError("no version was registered by the previous step")

    second = registry.register(ctx.artifact)
    if second.version == first:
        raise RuntimeError("registering twice reused a version")

    registry.promote(name, first, "production", reason="golden path")
    registry.promote(name, second.version, "production", reason="golden path")
    if registry.resolve_stage(name, "production").version != second.version:
        raise RuntimeError("promotion did not move the stage pointer")

    rolled = registry.rollback(name, "production", reason="golden path rollback")
    if registry.resolve_stage(name, "production").version != first:
        raise RuntimeError("rollback did not restore the previous version")
    if not rolled.is_rollback:
        raise RuntimeError("rollback was not recorded as a rollback")

    # Leave production pointing at the newest version for the serving steps.
    registry.promote(name, second.version, "production", reason="golden path restore")
    ctx.registered_version = second.version

    return {
        "register": f"{ctx.registered_version} allocated without collision",
        "list": f"{registry.list_models()} / {[v.version for v in registry.list_versions(name)]}",
        "promote": f"production -> {ctx.registered_version}",
        "rollback": "verified: pointer returned to the previous version, then restored",
        "history": f"{len(registry.transitions(name, 'production'))} transitions recorded",
    }


def step_service(ctx: Context) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from mlservice.config.observability import ObservabilityConfig
    from mlservice.config.service import ServiceConfig
    from mlservice.serving.app import create_app

    app = create_app(
        service_config=ServiceConfig(
            _env_file=None,
            registry_root=str(ctx.registry_root),
            model_name=ctx.model_name,
            model_stage="production",
        ),
        observability_config=ObservabilityConfig(_env_file=None, log_format="json"),
        configure_logs=False,
    )

    with TestClient(app) as client:
        checks: dict[str, str] = {}

        health = client.get("/health")
        _expect(health.status_code == 200, f"/health returned {health.status_code}")
        checks["/health"] = f"200 {health.json()['status']}"

        ready = client.get("/ready")
        _expect(ready.status_code == 200 and ready.json()["model_loaded"], "/ready is not ready")
        checks["/ready"] = f"200 serving {ready.json()['model_version']}"

        info = client.get("/model-info")
        _expect(info.status_code == 200, f"/model-info returned {info.status_code}")
        body = info.json()
        checks["/model-info"] = (
            f"200 {body['n_features']} features, threshold {body['decision_threshold']}"
        )

        example = body["example_instance"]
        single = client.post("/predict", json={"instances": [example]})
        _expect(
            single.status_code == 200,
            f"/predict returned {single.status_code}: {single.text[:200]}",
        )
        checks["/predict (1 row)"] = f"200 score={single.json()['predictions'][0]['score']:.4f}"

        batch = client.post("/predict", json={"instances": [example] * 50})
        _expect(batch.status_code == 200 and batch.json()["count"] == 50, "batch prediction failed")
        checks["/predict (50 rows)"] = f"200 {batch.json()['inference_ms']}ms server-side"

        bad = client.post("/predict", json={"instances": [{"not_a_feature": 1}]})
        _expect(
            bad.status_code == 422,
            f"schema violation returned {bad.status_code}, expected 422",
        )
        checks["schema enforcement"] = f"422 {bad.json()['error']}"

        limit = ServiceConfig(_env_file=None).max_batch_size
        oversized = client.post("/predict", json={"instances": [example] * (limit + 1)})
        _expect(oversized.status_code == 413, f"oversized batch returned {oversized.status_code}")
        checks["batch limit"] = f"413 at {limit + 1} rows, as expected"

        card = client.get("/model-card")
        _expect(card.status_code == 200, "/model-card failed")
        checks["/model-card"] = f"200 {len(card.text)} bytes of markdown"

        openapi = client.get("/openapi.json")
        _expect(openapi.status_code == 200, "/openapi.json failed")
        checks["/openapi.json"] = f"200 {len(openapi.json()['paths'])} paths"

    return checks


def step_observability(ctx: Context) -> dict[str, Any]:
    import logging

    from fastapi.testclient import TestClient

    from mlservice.config.observability import ObservabilityConfig
    from mlservice.config.service import ServiceConfig
    from mlservice.observability.logging import JsonFormatter
    from mlservice.serving.app import create_app
    from mlservice.serving.middleware import REQUEST_ID_HEADER

    record = logging.LogRecord("t", logging.INFO, __file__, 1, "probe", None, None)
    parsed = json.loads(JsonFormatter().format(record))
    _expect(parsed["message"] == "probe", "JSON formatter did not emit a parseable line")

    app = create_app(
        service_config=ServiceConfig(
            _env_file=None, registry_root=str(ctx.registry_root), model_name=ctx.model_name
        ),
        observability_config=ObservabilityConfig(_env_file=None),
        configure_logs=False,
    )
    with TestClient(app) as client:
        example = client.get("/model-info").json()["example_instance"]
        response = client.post("/predict", json={"instances": [example] * 3})
        request_id = response.headers.get(REQUEST_ID_HEADER)
        _expect(bool(request_id), "no request id header on the response")

        text = client.get("/metrics").text
        expected = [
            "mlservice_requests_total",
            "mlservice_request_duration_seconds",
            "mlservice_errors_total",
            "mlservice_predictions_total",
            "mlservice_prediction_score",
            "mlservice_batch_size",
            "mlservice_model_loaded",
        ]
        missing = [series for series in expected if series not in text]
        _expect(not missing, f"/metrics is missing {missing}")
        _expect("does-not-exist" not in text, "metrics are labelled with raw paths")

    return {
        "structured logging": "JSON lines parse, request context included",
        "request id": f"propagated ({request_id[:8]}…)",
        "metrics": f"all {len(expected)} series exposed at /metrics",
        "tracing": "optional; enable with the 'otel' extra and MLSERVICE_OBS_TRACING_ENABLED",
    }


def step_drift_hooks(ctx: Context) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from mlservice.config.observability import ObservabilityConfig
    from mlservice.config.service import ServiceConfig
    from mlservice.monitoring.base import DriftSink
    from mlservice.monitoring.detectors import ScoreDistributionDrift
    from mlservice.monitoring.records import PredictionRecord
    from mlservice.monitoring.reporter import DriftReporter
    from mlservice.monitoring.sinks import read_jsonl
    from mlservice.serving.app import create_app

    sink_path = ctx.workdir / "drift" / "predictions.jsonl"
    service = ServiceConfig(
        _env_file=None, registry_root=str(ctx.registry_root), model_name=ctx.model_name
    )
    observability = ObservabilityConfig(
        _env_file=None, drift_enabled=True, drift_sink="jsonl", drift_sink_path=str(sink_path)
    )

    # Real rows, not 25 copies of one row: a degenerate input distribution makes
    # the detector's output meaningless as a demonstration.
    import pandas as pd

    rows = pd.read_csv(ctx.dataset_path).drop(columns=["target"]).head(25).to_dict(orient="records")

    app = create_app(
        service_config=service, observability_config=observability, configure_logs=False
    )
    with TestClient(app) as client:
        client.post("/predict", json={"instances": rows})

    records = read_jsonl(sink_path)
    _expect(len(records) == 25, f"sink captured {len(records)} of 25 predictions")
    _expect(records[0].features is None, "features were written to the sink without being enabled")

    detector = ScoreDistributionDrift.from_metadata(ctx.artifact.metadata, min_records=10)
    signals = detector.evaluate(records)
    _expect(bool(signals), "detector produced no signal")

    class Exploding(DriftSink):
        def emit(self, record: PredictionRecord) -> None:
            raise RuntimeError("sink down")

    app = create_app(
        service_config=service,
        observability_config=observability,
        drift_reporter=DriftReporter(Exploding(), enabled=True),
        configure_logs=False,
    )
    with TestClient(app) as client:
        example = client.get("/model-info").json()["example_instance"]
        resilient = client.post("/predict", json={"instances": [example]})
    _expect(resilient.status_code == 200, "a failing drift sink broke inference")

    return {
        "sink": f"{len(records)} records written to {sink_path.name}",
        "PII default": "feature values withheld unless explicitly enabled",
        "detector": f"{signals[0].name}={signals[0].value} severity={signals[0].severity}",
        "fail-open": "verified: a broken sink leaves /predict at 200",
    }


def step_quality(ctx: Context) -> dict[str, Any]:
    checks = {
        "ruff check": [str(VENV_BIN / "ruff"), "check", "."],
        "ruff format --check": [str(VENV_BIN / "ruff"), "format", "--check", "."],
        "mypy": [str(VENV_BIN / "mypy")],
        "pytest": [str(VENV_BIN / "pytest"), "-q", "-m", "not container"],
    }
    missing = [name for name, cmd in checks.items() if not Path(cmd[0]).exists()]
    if missing:
        raise StepSkipped(f"tooling not installed in .venv ({missing}); run `make install`")

    facts: dict[str, Any] = {}
    failures: list[str] = []
    for name, command in checks.items():
        result = subprocess.run(
            command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800, check=False
        )
        if result.returncode == 0:
            summary = (result.stdout.strip().splitlines() or ["ok"])[-1]
            facts[name] = summary[:110]
        else:
            failures.append(name)
            facts[name] = f"FAILED: {(result.stdout + result.stderr).strip()[-300:]}"
    if failures:
        raise RuntimeError(f"quality checks failed: {failures}")
    return facts


def step_package_build(ctx: Context) -> dict[str, Any]:
    if not (VENV_BIN / "python").exists():
        raise StepSkipped("no .venv; run `make install`")
    try:
        import build  # noqa: F401
    except ImportError:
        raise StepSkipped("the 'build' package is not installed; run `make install`") from None

    outdir = ctx.workdir / "dist"
    result = subprocess.run(
        [str(VENV_BIN / "python"), "-m", "build", "--outdir", str(outdir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stdout + result.stderr)[-800:])
    produced = sorted(path.name for path in outdir.iterdir())
    _expect(any(name.endswith(".whl") for name in produced), "no wheel was produced")
    return {"artifacts": ", ".join(produced)}


def step_docker(ctx: Context) -> dict[str, Any]:
    if shutil.which("docker") is None:
        raise StepSkipped("docker is not installed on this machine")
    probe = subprocess.run(["docker", "info"], capture_output=True, timeout=60, check=False)
    if probe.returncode != 0:
        raise StepSkipped("the docker CLI is present but no daemon is reachable")

    if not ctx.use_repo_registry:
        raise StepSkipped(
            "the image bakes in ./registry, which this run did not write to "
            "(pass --use-repo-registry to build against it)"
        )

    tag = "ml-service-blueprint:golden-path"
    build = subprocess.run(
        ["docker", "build", "-t", tag, "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    if build.returncode != 0:
        raise RuntimeError(f"docker build failed: {build.stderr[-800:]}")

    size = subprocess.run(
        ["docker", "image", "inspect", tag, "-f", "{{.Size}}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    import socket
    import urllib.request

    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        port = probe_socket.getsockname()[1]

    name = "ml-service-blueprint-golden-path"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "-p", f"127.0.0.1:{port}:8000", tag],
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 120
        ready = False
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{base}/ready", timeout=2) as response:  # noqa: S310
                    ready = response.status == 200
                    if ready:
                        break
            except OSError:
                time.sleep(1)
        if not ready:
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True, check=False
            )
            tail = logs.stdout[-600:] + logs.stderr[-600:]
            raise RuntimeError(f"container never became ready: {tail}")

        uid = subprocess.run(
            ["docker", "exec", name, "id", "-u"], capture_output=True, text=True, check=True
        ).stdout.strip()
        _expect(uid != "0", "the container runs as root")

        with urllib.request.urlopen(f"{base}/model-info", timeout=5) as response:  # noqa: S310
            example = json.loads(response.read())["example_instance"]
        request = urllib.request.Request(  # noqa: S310
            f"{base}/predict",
            data=json.dumps({"instances": [example]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            predicted = json.loads(response.read())
        _expect(predicted["count"] == 1, "the container did not return a prediction")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)

    return {
        "image": f"{tag} ({int(size) / 1_000_000:.0f} MB)" if size.isdigit() else tag,
        "user": f"non-root (uid {uid})",
        "boot": "/ready returned 200",
        "prediction": f"served version {predicted['model_version']}",
    }


def step_loadtest_tooling(ctx: Context) -> dict[str, Any]:
    """Confirm the load-test configuration exists. Deliberately runs no load."""
    k6_script = REPO_ROOT / "loadtest" / "k6_smoke.js"
    locustfile = REPO_ROOT / "loadtest" / "locustfile.py"
    _expect(k6_script.is_file() and locustfile.is_file(), "load-test configuration is missing")

    available = []
    if shutil.which("k6"):
        available.append("k6")
    if (VENV_BIN / "locust").exists():
        available.append("locust")

    return {
        "configs": "k6_smoke.js and locustfile.py present",
        "runners installed": (
            ", ".join(available) or "none (install k6, or `pip install '.[loadtest]'`)"
        ),
        "note": "no throughput is measured or claimed here; see loadtest/README.md",
    }


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

STAGES = [
    ("environment", "Environment"),
    ("config", "Configuration"),
    ("data", "Data"),
    ("training", "Training"),
    ("validation", "Validation"),
    ("artifact", "Artifact"),
    ("version", "Version / registry"),
    ("service", "Inference service"),
    ("observability", "Observability"),
    ("monitoring", "Drift hooks"),
    ("ci", "CI checks"),
    ("container", "Container"),
    ("loadtest", "Load testing"),
]


def print_report(runner: Runner, elapsed: float) -> None:
    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}GOLDEN PATH REPORT{RESET}")
    print("=" * 78)

    width = max(len(result.name) for result in runner.results) + 2
    for _, label in STAGES:
        stage_results = [r for r in runner.results if r.stage == label]
        if not stage_results:
            continue
        print(f"\n{BOLD}{label}{RESET}")
        for result in stage_results:
            colour = {PASS: GREEN, FAIL: RED, SKIP: YELLOW}[result.status]
            line = (
                f"  {colour}{result.status:<4}{RESET} "
                f"{result.name:<{width}} {result.duration_seconds:>6.2f}s"
            )
            if result.detail:
                line += f"  {DIM}{result.detail}{RESET}"
            print(line)

    passed = sum(1 for r in runner.results if r.status == PASS)
    print(f"\n{'-' * 78}")
    print(
        f"{GREEN}{passed} passed{RESET}, "
        f"{RED if runner.failed else DIM}{len(runner.failed)} failed{RESET}, "
        f"{YELLOW if runner.skipped else DIM}{len(runner.skipped)} skipped{RESET}"
        f"  in {elapsed:.1f}s"
    )

    if runner.skipped:
        print(
            f"\n{YELLOW}Not verified on this machine{RESET} (prerequisites absent, not failures):"
        )
        for result in runner.skipped:
            print(f"  - {result.name}: {result.detail}")

    if runner.failed:
        print(f"\n{RED}Failures{RESET}:")
        for result in runner.failed:
            print(f"  - {result.name}: {result.detail}")
        print(f"\n{RED}GOLDEN PATH FAILED{RESET}")
    else:
        verified = " → ".join(
            label
            for _, label in STAGES
            if any(r.stage == label and r.status == PASS for r in runner.results)
        )
        print(f"\n{GREEN}GOLDEN PATH VERIFIED{RESET}: {verified}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", type=Path, help="Write a machine-readable report here.")
    parser.add_argument("--skip-quality", action="store_true", help="Skip lint/typecheck/pytest.")
    parser.add_argument("--skip-docker", action="store_true", help="Skip the container step.")
    parser.add_argument(
        "--use-repo-registry",
        action="store_true",
        help="Write to ./registry instead of a temp directory (required for the Docker step).",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mlservice-golden-") as temp:
        workdir = Path(temp)
        context = Context(
            workdir=workdir,
            registry_root=(
                (REPO_ROOT / "registry") if args.use_repo_registry else workdir / "registry"
            ),
            dataset_path=REPO_ROOT / "data" / "raw" / "breast_cancer.csv",
            use_repo_registry=args.use_repo_registry,
        )

        print(f"{BOLD}ML Service Blueprint — golden path{RESET}")
        print(f"{DIM}repo:     {REPO_ROOT}{RESET}")
        print(f"{DIM}registry: {context.registry_root}{RESET}\n")

        runner = Runner(context)
        runner.step("Environment", "package imports and Python version", step_environment)
        runner.step("Configuration", "typed configs load and validate", step_config)
        runner.step("Data", "dataset materialises", step_data)
        runner.step("Training", "pipeline trains a model", step_training)
        runner.step("Validation", "metrics clear the configured gates", step_validation)
        runner.step("Artifact", "artifact contract is complete and round-trips", step_artifact)
        runner.step("Artifact", "tampered artifacts are rejected", step_integrity)
        runner.step("Version / registry", "artifact registers with a version", step_register)
        runner.step(
            "Version / registry", "list / promote / rollback / history", step_registry_operations
        )
        runner.step("Inference service", "all endpoints respond correctly", step_service)
        runner.step("Observability", "logs, request ids, and metrics", step_observability)
        runner.step("Drift hooks", "sink, detector, and fail-open behaviour", step_drift_hooks)

        if args.skip_quality:
            runner.step("CI checks", "lint, typecheck, tests", lambda _: _skip("--skip-quality"))
            runner.step("CI checks", "package builds", lambda _: _skip("--skip-quality"))
        else:
            runner.step("CI checks", "lint, typecheck, tests", step_quality)
            runner.step("CI checks", "package builds", step_package_build)

        if args.skip_docker:
            runner.step(
                "Container",
                "image builds, boots, and serves",
                lambda _: _skip("--skip-docker"),
            )
        else:
            runner.step("Container", "image builds, boots, and serves", step_docker)

        runner.step("Load testing", "load-test configuration present", step_loadtest_tooling)

        elapsed = time.perf_counter() - started
        print_report(runner, elapsed)

        if args.json:
            args.json.write_text(
                json.dumps(
                    {
                        "elapsed_seconds": round(elapsed, 3),
                        "passed": sum(1 for r in runner.results if r.status == PASS),
                        "failed": len(runner.failed),
                        "skipped": len(runner.skipped),
                        "steps": [
                            {
                                "stage": r.stage,
                                "name": r.name,
                                "status": r.status,
                                "duration_seconds": round(r.duration_seconds, 3),
                                "detail": r.detail,
                                "facts": {k: v for k, v in r.facts.items() if k != "traceback"},
                            }
                            for r in runner.results
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"\n{DIM}report written to {args.json}{RESET}")

        return 1 if runner.failed else 0


def _skip(reason: str) -> dict[str, Any]:
    raise StepSkipped(f"skipped by {reason}")


if __name__ == "__main__":
    sys.exit(main())
