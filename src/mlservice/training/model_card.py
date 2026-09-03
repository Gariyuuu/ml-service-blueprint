"""Model card rendering.

The card is generated from artifact metadata rather than hand-written, so it
cannot go stale relative to the model it describes. Sections that require human
judgement (intended use, limitations, ethical considerations) come from a
template that the template's adopter edits.
"""

from __future__ import annotations

from pathlib import Path

from mlservice.artifacts.metadata import ArtifactMetadata

_FALLBACK_TEMPLATE = """## Intended use

_Describe the decisions this model is allowed to inform, and the ones it is not._

## Limitations

_Known failure modes, populations the training data under-represents, and the
conditions under which this model should be retrained._

## Ethical considerations

_Fairness, privacy, and contestability notes._
"""


def render_model_card(metadata: ArtifactMetadata, template_path: str | Path | None = None) -> str:
    """Render a Markdown model card for one artifact version."""
    human_sections = _FALLBACK_TEMPLATE
    if template_path is not None:
        candidate = Path(template_path)
        if candidate.is_file():
            human_sections = candidate.read_text(encoding="utf-8")

    dataset = metadata.dataset
    schema = metadata.feature_schema
    numeric = sum(1 for f in schema.features if f.kind == "numeric")
    categorical = sum(1 for f in schema.features if f.kind == "categorical")
    boolean = sum(1 for f in schema.features if f.kind == "boolean")

    lines = [
        f"# Model card — {metadata.model_name} {metadata.version}",
        "",
        f"Generated {metadata.created_at.isoformat()} by the ML Service Blueprint "
        "training pipeline.",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Model name | `{metadata.model_name}` |",
        f"| Version | `{metadata.version}` |",
        f"| Trained at | {metadata.created_at.isoformat()} |",
        f"| Training duration | {metadata.training_duration_seconds:.2f}s |",
        f"| Git commit | `{metadata.provenance.git_commit or 'unknown'}` |",
        f"| Git branch | `{metadata.provenance.git_branch or 'unknown'}` |",
        f"| Working tree clean | {metadata.provenance.git_dirty is False} |",
        f"| Python | {metadata.provenance.python_version} |",
        f"| Platform | {metadata.provenance.platform} |",
        "",
        "Package versions: "
        + ", ".join(
            f"`{name}=={version}`" for name, version in sorted(metadata.provenance.packages.items())
        ),
        "",
        "## Data",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Source | `{dataset.source}` |",
        f"| SHA-256 | `{dataset.content_sha256}` |",
        f"| Rows | {dataset.n_rows} |",
        f"| Features | {dataset.n_features} ({numeric} numeric, "
        f"{categorical} categorical, {boolean} boolean) |",
        f"| Train / validation / test | {dataset.n_train} / "
        f"{dataset.n_validation} / {dataset.n_test} |",
        f"| Class balance | {_format_balance(dataset.class_balance)} |",
        "",
        "## Decision rule",
        "",
        "Scores are the predicted probability of the positive class. A row is "
        f"labelled positive when `score >= {metadata.decision_threshold}`.",
        "",
        "## Metrics",
        "",
        "Test-set metrics (the split the model never saw during fitting or threshold selection):",
        "",
        _metrics_table(metadata.metrics),
        "",
        "Validation-set metrics (used for threshold selection):",
        "",
        _metrics_table(metadata.validation_metrics),
        "",
        "## Evaluation gates",
        "",
        (
            "All configured gates passed."
            if metadata.passed_gates
            else "**Gates FAILED:**\n\n"
            + "\n".join(f"- {failure}" for failure in metadata.gate_failures)
        ),
        "",
        "## Input schema",
        "",
        _schema_table(metadata),
        "",
        "---",
        "",
        human_sections.strip(),
        "",
    ]
    return "\n".join(lines)


def _metrics_table(metrics: dict[str, float]) -> str:
    if not metrics:
        return "_No metrics recorded._"
    rows = ["| Metric | Value |", "| --- | --- |"]
    rows += [f"| {name} | {value:.4f} |" for name, value in sorted(metrics.items())]
    return "\n".join(rows)


def _schema_table(metadata: ArtifactMetadata, limit: int = 40) -> str:
    features = metadata.feature_schema.features
    rows = ["| Feature | Kind | Required | Range / categories |", "| --- | --- | --- | --- |"]
    for feature in features[:limit]:
        if feature.kind == "numeric":
            detail = f"[{feature.minimum}, {feature.maximum}]"
        elif feature.categories:
            shown = ", ".join(feature.categories[:6])
            detail = shown + (" …" if len(feature.categories) > 6 else "")
        else:
            detail = "—"
        rows.append(f"| `{feature.name}` | {feature.kind} | {feature.required} | {detail} |")
    if len(features) > limit:
        rows.append(f"| _… {len(features) - limit} more_ | | | |")
    return "\n".join(rows)


def _format_balance(balance: dict[str, float]) -> str:
    if not balance:
        return "—"
    return ", ".join(f"`{label}`: {share:.1%}" for label, share in sorted(balance.items()))
