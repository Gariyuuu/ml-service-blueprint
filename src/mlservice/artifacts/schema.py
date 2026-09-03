"""The feature schema: the contract between training data and inference requests.

The schema is derived from the training frame, frozen into the artifact, and
re-applied to every inference request. Serving a model whose caller sends
columns in a different order, with a different dtype, or with a category the
model never saw is the single most common silent production failure; this
module turns each of those into an explicit error.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

FeatureKind = Literal["numeric", "categorical", "boolean"]


class SchemaValidationError(ValueError):
    """Raised when an inference payload does not satisfy the frozen feature schema."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


class FeatureSpec(BaseModel):
    """Everything the service needs to know about one input column."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: FeatureKind
    required: bool = True
    dtype: str = Field(description="Pandas dtype string observed during training.")
    # Numeric summary, used for range checks and as a drift baseline.
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    std: float | None = None
    # Categorical summary.
    categories: list[str] | None = None
    null_fraction: float = 0.0

    def coerce(self, values: pd.Series) -> pd.Series:
        """Cast an incoming column to the dtype family the pipeline was fitted on."""
        if self.kind == "numeric":
            return pd.to_numeric(values, errors="coerce")
        if self.kind == "boolean":
            return values.astype("boolean")
        return values.astype("string")


class FeatureSchema(BaseModel):
    """Ordered, frozen description of the model's input columns."""

    model_config = ConfigDict(frozen=True)

    features: list[FeatureSpec]
    target: str
    target_classes: list[str] = Field(default_factory=list)
    # A checksum-ish identifier so a service can report which schema it is enforcing.
    schema_version: int = 1

    @property
    def feature_names(self) -> list[str]:
        """Column order the fitted pipeline expects."""
        return [feature.name for feature in self.features]

    @property
    def required_names(self) -> list[str]:
        return [feature.name for feature in self.features if feature.required]

    def spec(self, name: str) -> FeatureSpec:
        for feature in self.features:
            if feature.name == name:
                return feature
        raise KeyError(name)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        target: str,
        *,
        drop_columns: list[str] | None = None,
    ) -> FeatureSchema:
        """Infer the schema from a training frame.

        Called once, during training. The result is serialized into the artifact
        and is never re-inferred at serving time.
        """
        dropped = set(drop_columns or []) | {target}
        specs: list[FeatureSpec] = []
        for name in frame.columns:
            if name in dropped:
                continue
            column = frame[name]
            kind = _infer_kind(column)
            spec_kwargs: dict[str, Any] = {
                "name": name,
                "kind": kind,
                "dtype": str(column.dtype),
                "null_fraction": float(column.isna().mean()),
            }
            if kind == "numeric":
                numeric = pd.to_numeric(column, errors="coerce")
                spec_kwargs |= {
                    "minimum": _finite(numeric.min()),
                    "maximum": _finite(numeric.max()),
                    "mean": _finite(numeric.mean()),
                    "std": _finite(numeric.std()),
                }
            elif kind == "categorical":
                spec_kwargs["categories"] = sorted(str(value) for value in column.dropna().unique())
            specs.append(FeatureSpec(**spec_kwargs))

        target_values = sorted(str(value) for value in frame[target].dropna().unique())
        return cls(features=specs, target=target, target_classes=target_values)

    def validate_frame(
        self, frame: pd.DataFrame, *, strict_categories: bool = False
    ) -> pd.DataFrame:
        """Validate and normalise an inference frame.

        Returns a new frame with exactly the schema's columns, in the schema's
        order, coerced to the training dtypes. Raises
        :class:`SchemaValidationError` listing *every* problem rather than
        failing on the first one, so a caller can fix their payload in one pass.

        ``strict_categories`` rejects unseen category values. It defaults to
        False because most encoders are configured to handle unknowns, and
        rejecting a live request for a new category is usually worse than
        scoring it; enable it when an unknown category means a broken upstream.
        """
        errors: list[str] = []

        missing = [name for name in self.required_names if name not in frame.columns]
        if missing:
            errors.append(f"missing required feature(s): {', '.join(sorted(missing))}")

        unexpected = [name for name in frame.columns if name not in set(self.feature_names)]
        if unexpected:
            errors.append(f"unexpected feature(s): {', '.join(sorted(unexpected))}")

        if errors:
            raise SchemaValidationError(errors)

        normalised: dict[str, pd.Series] = {}
        for feature in self.features:
            if feature.name not in frame.columns:
                normalised[feature.name] = pd.Series(
                    [None] * len(frame), index=frame.index, dtype="object"
                )
                continue

            column = frame[feature.name]
            try:
                coerced = feature.coerce(column)
            except (TypeError, ValueError) as exc:
                errors.append(f"{feature.name}: cannot coerce to {feature.kind} ({exc})")
                continue

            if feature.kind == "numeric":
                became_null = coerced.isna() & column.notna()
                if became_null.any():
                    bad = column[became_null].head(3).tolist()
                    errors.append(
                        f"{feature.name}: {int(became_null.sum())} non-numeric value(s), e.g. {bad}"
                    )
                infinite = np.isinf(coerced.to_numpy(dtype="float64", na_value=np.nan))
                if infinite.any():
                    errors.append(f"{feature.name}: {int(infinite.sum())} infinite value(s)")
            elif feature.kind == "categorical" and strict_categories and feature.categories:
                known = set(feature.categories)
                unseen = sorted({str(value) for value in coerced.dropna().unique()} - known)
                if unseen:
                    errors.append(f"{feature.name}: unseen categor(y|ies) {unseen[:5]}")

            normalised[feature.name] = coerced

        if errors:
            raise SchemaValidationError(errors)

        return pd.DataFrame(normalised, columns=self.feature_names, index=frame.index)

    def example_row(self) -> dict[str, Any]:
        """A representative payload, used in OpenAPI examples and smoke tests."""
        row: dict[str, Any] = {}
        for feature in self.features:
            if feature.kind == "numeric":
                row[feature.name] = round(feature.mean, 4) if feature.mean is not None else 0.0
            elif feature.kind == "boolean":
                row[feature.name] = False
            else:
                row[feature.name] = feature.categories[0] if feature.categories else "unknown"
        return row


def _infer_kind(column: pd.Series) -> FeatureKind:
    if pd.api.types.is_bool_dtype(column):
        return "boolean"
    if pd.api.types.is_numeric_dtype(column):
        return "numeric"
    return "categorical"


def _finite(value: Any) -> float | None:
    """Convert a numpy scalar to a JSON-safe float, dropping NaN/inf."""
    if value is None:
        return None
    as_float = float(value)
    return as_float if np.isfinite(as_float) else None
