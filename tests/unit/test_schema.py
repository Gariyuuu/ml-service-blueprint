"""Feature schema: the contract that stops silent production failures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlservice.artifacts.schema import FeatureSchema, SchemaValidationError


@pytest.fixture
def schema(synthetic_frame: pd.DataFrame) -> FeatureSchema:
    return FeatureSchema.from_frame(synthetic_frame, target="target")


def test_infers_kinds_from_dtypes(schema: FeatureSchema):
    kinds = {feature.name: feature.kind for feature in schema.features}
    assert kinds == {
        "age": "numeric",
        "income": "numeric",
        "region": "categorical",
        "subscribed": "boolean",
    }


def test_target_is_not_a_feature(schema: FeatureSchema):
    assert "target" not in schema.feature_names


def test_drop_columns_are_excluded(synthetic_frame: pd.DataFrame):
    schema = FeatureSchema.from_frame(synthetic_frame, target="target", drop_columns=["income"])
    assert "income" not in schema.feature_names


def test_numeric_specs_carry_a_drift_baseline(schema: FeatureSchema):
    age = schema.spec("age")
    assert age.minimum is not None and age.maximum is not None
    assert age.minimum < age.mean < age.maximum


def test_categorical_specs_record_observed_categories(schema: FeatureSchema):
    assert set(schema.spec("region").categories) == {"north", "south", "east", "west"}


def test_validate_reorders_columns_to_training_order(schema: FeatureSchema):
    scrambled = pd.DataFrame(
        [{"subscribed": True, "region": "north", "income": 100.0, "age": 30.0}]
    )
    assert schema.validate_frame(scrambled).columns.tolist() == schema.feature_names


def test_validate_reports_every_missing_column_at_once(schema: FeatureSchema):
    with pytest.raises(SchemaValidationError) as excinfo:
        schema.validate_frame(pd.DataFrame([{"age": 30.0}]))
    assert "income" in str(excinfo.value)
    assert "region" in str(excinfo.value)


def test_validate_rejects_unexpected_columns(schema: FeatureSchema):
    row = dict.fromkeys(schema.feature_names, 1) | {"leaked_target": 1}
    with pytest.raises(SchemaValidationError, match="unexpected"):
        schema.validate_frame(pd.DataFrame([row]))


def test_validate_rejects_non_numeric_values_in_numeric_columns(schema: FeatureSchema):
    row = schema.example_row() | {"age": "not-a-number"}
    with pytest.raises(SchemaValidationError, match="non-numeric"):
        schema.validate_frame(pd.DataFrame([row]))


def test_validate_rejects_infinities(schema: FeatureSchema):
    row = schema.example_row() | {"income": np.inf}
    with pytest.raises(SchemaValidationError, match="infinite"):
        schema.validate_frame(pd.DataFrame([row]))


def test_numeric_strings_are_accepted_after_coercion(schema: FeatureSchema):
    """JSON clients routinely send numbers as strings; that is not an error."""
    row = schema.example_row() | {"age": "42"}
    validated = schema.validate_frame(pd.DataFrame([row]))
    assert validated["age"].iloc[0] == 42


def test_unseen_categories_pass_by_default_and_fail_under_strict(schema: FeatureSchema):
    row = schema.example_row() | {"region": "atlantis"}
    frame = pd.DataFrame([row])
    schema.validate_frame(frame)
    with pytest.raises(SchemaValidationError, match="unseen"):
        schema.validate_frame(frame, strict_categories=True)


def test_example_row_satisfies_its_own_schema(schema: FeatureSchema):
    validated = schema.validate_frame(pd.DataFrame([schema.example_row()]))
    assert len(validated) == 1


def test_schema_round_trips_through_json(schema: FeatureSchema):
    restored = FeatureSchema.model_validate_json(schema.model_dump_json())
    assert restored == schema
