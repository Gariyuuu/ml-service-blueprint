"""Training configuration: data source, deterministic split, preprocessing, gates."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from mlservice.config.base import YamlConfig


class DataConfig(YamlConfig):
    """Where the training table lives and which column is the label."""

    path: str = Field(description="Path to the training CSV, relative to the repo root.")
    target: str = Field(description="Name of the label column.")
    positive_label: int = Field(default=1, description="Class treated as positive for metrics.")
    drop_columns: list[str] = Field(
        default_factory=list, description="Columns excluded from features entirely."
    )


class SplitConfig(YamlConfig):
    """Deterministic train/validation/test partitioning."""

    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    validation_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    stratify: bool = Field(default=True, description="Preserve label balance across splits.")
    random_state: int = Field(
        default=20240101, description="Seed for the split; never change silently."
    )

    @model_validator(mode="after")
    def _fractions_leave_training_data(self) -> SplitConfig:
        if self.test_size + self.validation_size >= 0.9:
            raise ValueError(
                "test_size + validation_size must leave at least 10% of rows for training"
            )
        return self


class PreprocessingConfig(YamlConfig):
    """Feature handling applied inside the fitted pipeline."""

    numeric_imputation: str = Field(default="median", pattern="^(mean|median|most_frequent)$")
    categorical_imputation: str = Field(
        default="most_frequent", pattern="^(most_frequent|constant)$"
    )
    scale_numeric: bool = Field(default=True)
    one_hot_min_frequency: float | None = Field(
        default=None,
        description="Collapse categories rarer than this fraction into an 'infrequent' bucket.",
    )


class EvaluationGates(YamlConfig):
    """Minimum test-set metrics a run must clear to produce a registrable artifact.

    A run that misses a gate still writes its metrics; it just refuses to register.
    """

    min_roc_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    min_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    min_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    min_recall: float | None = Field(default=None, ge=0.0, le=1.0)

    def check(self, metrics: dict[str, float]) -> list[str]:
        """Return a list of human-readable gate failures (empty means the run passed)."""
        failures: list[str] = []
        for gate_name, threshold in self.model_dump().items():
            if threshold is None:
                continue
            metric_name = gate_name.removeprefix("min_")
            observed = metrics.get(metric_name)
            if observed is None:
                failures.append(f"gate {gate_name} set but metric '{metric_name}' was not computed")
            elif observed < threshold:
                failures.append(f"{metric_name}={observed:.4f} below required {threshold:.4f}")
        return failures


class TrainingConfig(YamlConfig):
    """Top-level training experiment configuration."""

    seed: int = Field(
        default=20240101, description="Global seed applied to numpy and the estimator."
    )
    data: DataConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    gates: EvaluationGates = Field(default_factory=EvaluationGates)
    decision_threshold: float = Field(
        default=0.5, gt=0.0, lt=1.0, description="Probability cutoff baked into the artifact."
    )
    registry_root: str = Field(default="registry", description="Filesystem registry location.")
    model_card_template: str = Field(default="model_cards/TEMPLATE.md")

    @field_validator("registry_root")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("registry_root must not be empty")
        return value
