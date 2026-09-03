"""``mlservice`` command line: train, inspect, promote, roll back, and serve.

Everything here is a thin shell over the library. Nothing in the CLI implements
behaviour the library does not already expose, so automation can call either.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from mlservice.config.model import ModelConfig
from mlservice.config.observability import ObservabilityConfig
from mlservice.config.service import ServiceConfig
from mlservice.config.training import TrainingConfig
from mlservice.observability.logging import configure_logging
from mlservice.registry.base import RegistryError, StageTransition
from mlservice.registry.local import LocalFilesystemRegistry

app = typer.Typer(
    name="mlservice",
    help="Train, register, promote, and serve tabular models.",
    no_args_is_help=True,
    add_completion=False,
)
registry_app = typer.Typer(help="Inspect and manage the model registry.", no_args_is_help=True)
app.add_typer(registry_app, name="registry")

DEFAULT_TRAINING_CONFIG = "configs/training.yaml"
DEFAULT_MODEL_CONFIG = "configs/model.yaml"


def _registry(root: str) -> LocalFilesystemRegistry:
    return LocalFilesystemRegistry(root)


def _fail(message: str) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command()
def train(
    training_config: Annotated[
        Path, typer.Option("--training-config", "-t", help="Path to the training YAML.")
    ] = Path(DEFAULT_TRAINING_CONFIG),
    model_config: Annotated[
        Path, typer.Option("--model-config", "-m", help="Path to the model YAML.")
    ] = Path(DEFAULT_MODEL_CONFIG),
    register: Annotated[
        bool, typer.Option("--register/--no-register", help="Write the artifact to the registry.")
    ] = True,
    promote_to: Annotated[
        str | None,
        typer.Option("--promote-to", help="Promote the new version to this stage on success."),
    ] = None,
    tune_threshold: Annotated[
        bool,
        typer.Option(
            "--tune-threshold/--fixed-threshold",
            help="Select the decision threshold on validation, or keep the configured one.",
        ),
    ] = True,
    log_format: Annotated[str, typer.Option("--log-format", help="json or console")] = "console",
) -> None:
    """Run the training pipeline end to end."""
    from mlservice.training.pipeline import run_training

    if log_format not in ("json", "console"):
        _fail(f"--log-format must be 'json' or 'console', got {log_format!r}")
    configure_logging(ObservabilityConfig(log_format=log_format))

    training = TrainingConfig.from_yaml(training_config)
    model = ModelConfig.from_yaml(model_config)
    registry = _registry(training.registry_root)

    result = run_training(
        training,
        model,
        registry=registry,
        register=register,
        tune_threshold=tune_threshold,
    )

    headline = {
        key: value for key, value in result.metrics.items() if not key.startswith("score_dist.")
    }
    typer.echo(json.dumps({"split_sizes": result.split_sizes, "metrics": headline}, indent=2))

    if not result.passed_gates:
        _fail("evaluation gates failed: " + "; ".join(result.artifact.metadata.gate_failures))

    if result.registered is None:
        typer.secho("trained but not registered (--no-register)", fg=typer.colors.YELLOW)
        return

    typer.secho(
        f"registered {result.registered.model_name} {result.registered.version} "
        f"-> {result.registered.uri}",
        fg=typer.colors.GREEN,
    )

    if promote_to:
        transition = registry.promote(
            result.registered.model_name,
            result.registered.version,
            promote_to,
            reason="promoted by `mlservice train --promote-to`",
        )
        _echo_transition(transition)


@registry_app.command("list")
def registry_list(
    root: Annotated[str, typer.Option("--root", help="Registry root directory.")] = "registry",
) -> None:
    """List registered models and their stage pointers."""
    registry = _registry(root)
    models = registry.list_models()
    if not models:
        typer.secho(f"no models registered under {root}/", fg=typer.colors.YELLOW)
        return
    for name in models:
        stages = registry.stages(name)
        versions = registry.list_versions(name)
        typer.secho(f"{name}", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"  versions: {', '.join(v.version for v in versions) or 'none'}")
        rendered = ", ".join(f"{stage}={version}" for stage, version in sorted(stages.items()))
        typer.echo(f"  stages:   {rendered or 'none'}")


@registry_app.command("versions")
def registry_versions(
    model_name: Annotated[str, typer.Argument(help="Model name.")],
    root: Annotated[str, typer.Option("--root")] = "registry",
) -> None:
    """Show every version of a model with its headline metrics."""
    registry = _registry(root)
    versions = registry.list_versions(model_name)
    if not versions:
        _fail(f"no versions for '{model_name}' under {root}/")
    for entry in versions:
        stages = f" [{', '.join(entry.stages)}]" if entry.stages else ""
        gates = "" if entry.passed_gates else " GATES-FAILED"
        headline = {
            key: round(value, 4)
            for key, value in entry.metrics.items()
            if key in {"roc_auc", "f1", "precision", "recall"}
        }
        typer.echo(
            f"{entry.version:<6}{stages}{gates}  {entry.created_at.isoformat()}  "
            f"{entry.git_commit[:8] if entry.git_commit else 'nogit':<8}  {headline}"
        )


@registry_app.command("show")
def registry_show(
    model_name: Annotated[str, typer.Argument()],
    version: Annotated[str, typer.Argument()],
    root: Annotated[str, typer.Option("--root")] = "registry",
) -> None:
    """Print an artifact's full metadata as JSON."""
    registry = _registry(root)
    try:
        metadata = registry.read_metadata(model_name, version)
    except RegistryError as exc:
        _fail(str(exc))
    typer.echo(metadata.model_dump_json(indent=2))


@registry_app.command("promote")
def registry_promote(
    model_name: Annotated[str, typer.Argument()],
    version: Annotated[str, typer.Argument()],
    stage: Annotated[str, typer.Argument(help="Target stage, e.g. production.")],
    reason: Annotated[str, typer.Option("--reason")] = "",
    root: Annotated[str, typer.Option("--root")] = "registry",
) -> None:
    """Point a stage at a version."""
    try:
        transition = _registry(root).promote(model_name, version, stage, reason=reason)
    except RegistryError as exc:
        _fail(str(exc))
    _echo_transition(transition)


@registry_app.command("rollback")
def registry_rollback(
    model_name: Annotated[str, typer.Argument()],
    stage: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option("--reason")] = "",
    root: Annotated[str, typer.Option("--root")] = "registry",
) -> None:
    """Return a stage to the version it pointed at before the last change."""
    try:
        transition = _registry(root).rollback(model_name, stage, reason=reason)
    except RegistryError as exc:
        _fail(str(exc))
    _echo_transition(transition)


@registry_app.command("history")
def registry_history(
    model_name: Annotated[str, typer.Argument()],
    stage: Annotated[str | None, typer.Option("--stage")] = None,
    root: Annotated[str, typer.Option("--root")] = "registry",
) -> None:
    """Print the stage-change audit log."""
    for transition in _registry(root).transitions(model_name, stage):
        arrow = f"{transition.from_version or '-'} -> {transition.to_version}"
        tag = " (rollback)" if transition.is_rollback else ""
        typer.echo(
            f"{transition.at.isoformat()}  {transition.stage:<12} {arrow}{tag}  "
            f"{transition.actor}  {transition.reason}"
        )


@app.command()
def predict(
    input_csv: Annotated[Path, typer.Argument(help="CSV of raw feature rows.")],
    model_name: Annotated[str, typer.Option("--model")] = "tabular-classifier",
    stage: Annotated[str, typer.Option("--stage")] = "production",
    version: Annotated[str | None, typer.Option("--version", help="Pin an exact version.")] = None,
    root: Annotated[str, typer.Option("--root")] = "registry",
    output_csv: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    """Score a CSV offline, through the same artifact the service loads."""
    import pandas as pd

    registry = _registry(root)
    try:
        artifact = (
            registry.load(model_name, version)
            if version
            else registry.load_stage(model_name, stage)
        )
    except RegistryError as exc:
        _fail(str(exc))

    frame = pd.read_csv(input_csv)
    known = set(artifact.feature_schema.feature_names)
    result = artifact.predict(frame[[c for c in frame.columns if c in known]])

    scored = frame.copy()
    scored["score"] = result.scores
    scored["label"] = result.labels
    if output_csv:
        scored.to_csv(output_csv, index=False)
        typer.secho(f"wrote {len(scored)} rows to {output_csv}", fg=typer.colors.GREEN)
    else:
        typer.echo(scored.to_csv(index=False))


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Development autoreload.")] = False,
) -> None:
    """Run the inference service with uvicorn."""
    import uvicorn

    config = ServiceConfig()
    uvicorn.run(
        "mlservice.serving.app:build_app",
        factory=True,
        host=host or config.host,
        port=port or config.port,
        reload=reload,
        access_log=False,  # our middleware emits structured access logs instead
    )


@app.command("estimators")
def list_estimators() -> None:
    """List estimator names accepted by configs/model.yaml."""
    from mlservice.training.model_factory import available_estimators

    for name in available_estimators():
        typer.echo(name)


def _echo_transition(transition: StageTransition) -> None:
    verb = "rolled back" if transition.is_rollback else "promoted"
    typer.secho(
        f"{verb} {transition.model_name}:{transition.stage} "
        f"{transition.from_version or '-'} -> {transition.to_version}",
        fg=typer.colors.GREEN,
    )


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
